"""Phase 3 tests: boundaries, missing classes, duplicate timestamps, unseen ids, delayed labels, cache."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_evaluation import (
    evaluate_weekly, recall_at_fpr, window_metrics,
)
from backend.src.fraud_detection.fd_feature_engineering import (
    add_customer_features, add_terminal_features, build_features, features_fingerprint,
    load_or_build_features,
)
from backend.src.fraud_detection.fd_preprocessing import matured_mask


def _frame(rows):
    """rows: (day, hour, customer, terminal, amount, fraud)."""
    df = pd.DataFrame(rows, columns=["day", "hour", "CUSTOMER_ID", "TERMINAL_ID",
                                     "TX_AMOUNT", "TX_FRAUD"])
    df["TX_DATETIME"] = pd.Timestamp("2018-04-01") + pd.to_timedelta(df.day, "D") \
        + pd.to_timedelta(df.hour, "h")
    df["TX_TIME_DAYS"] = df.day
    return df.drop(columns=["day", "hour"]).sort_values("TX_DATETIME").reset_index(drop=True)


# ----------------------------------------------------------------- delayed-label boundaries
def test_terminal_label_matures_exactly_at_delay():
    # one fraud on terminal 1 at day 10; delay 7 -> visible from day 17, not day 16
    df = _frame([(10, 12, 1, 1, 10, 1), (16, 12, 2, 1, 10, 0), (17, 12, 3, 1, 10, 0),
                 (17, 13, 4, 2, 10, 0)])
    out = add_terminal_features(df, delay=7, window=28)
    d16 = out[out.TX_TIME_DAYS == 16].iloc[0]
    d17 = out[(out.TX_TIME_DAYS == 17) & (out.TERMINAL_ID == 1)].iloc[0]
    assert (d16.t_vol, d16.t_risk) == (0, 0)
    assert d17.t_vol == 1 and d17.t_risk == 1.0
    assert out[out.TERMINAL_ID == 2].iloc[0].t_vol == 0   # other terminal unaffected


def test_terminal_window_lower_boundary():
    # window = 3 days, delay 2. For D=20: hi=18, lo=15 -> days 16..18 included, day 15 excluded
    df = _frame([(15, 1, 1, 1, 5, 1), (16, 1, 2, 1, 5, 0), (18, 1, 3, 1, 5, 0),
                 (19, 1, 4, 1, 5, 1), (20, 1, 5, 1, 5, 0)])
    out = add_terminal_features(df, delay=2, window=3)
    r = out[out.TX_TIME_DAYS == 20].iloc[0]
    assert r.t_vol == 2 and r.t_risk == 0.0      # day 15 fraud excluded, day 19 not yet matured


def test_label_flip_on_day_d_minus_delay_changes_feature_but_one_day_later_does_not():
    df = _frame([(10, 1, 1, 1, 5, 0), (11, 1, 2, 1, 5, 0), (17, 1, 3, 1, 5, 0)])
    base = add_terminal_features(df, 7)
    last = df.index[df.TX_TIME_DAYS == 17][0]
    flip_matured = df.copy(); flip_matured.loc[df.TX_TIME_DAYS == 10, "TX_FRAUD"] = 1
    flip_future = df.copy(); flip_future.loc[df.TX_TIME_DAYS == 11, "TX_FRAUD"] = 1
    assert add_terminal_features(flip_matured, 7).loc[last, "t_risk"] != base.loc[last, "t_risk"]
    assert add_terminal_features(flip_future, 7).loc[last, "t_risk"] == base.loc[last, "t_risk"]


def test_matured_mask_equivalence_with_terminal_features_random():
    rng = np.random.default_rng(3)
    rows = [(int(d), 1, int(c), int(t), 5.0, int(f)) for d, c, t, f in zip(
        rng.integers(0, 30, 300), rng.integers(0, 4, 300), rng.integers(0, 3, 300),
        rng.random(300) < 0.2)]
    df = _frame(rows)
    out = add_terminal_features(df, 5, 10)
    for i in range(0, len(df), 17):
        r = df.iloc[i]; D = int(r.TX_TIME_DAYS)
        sub = df[matured_mask(df, D, 5) & (df.TX_TIME_DAYS > D - 5 - 10)
                 & (df.TERMINAL_ID == r.TERMINAL_ID)]
        assert out.loc[i, "t_vol"] == len(sub)


# ----------------------------------------------------------------- customer windows
def test_duplicate_timestamps_do_not_see_each_other():
    df = _frame([(5, 1, 1, 1, 10, 0), (5, 1, 1, 1, 30, 0), (5, 2, 1, 1, 50, 0)])
    out = add_customer_features(df, ("1D",))
    assert list(out.c_cnt_1D) == [0, 0, 2]            # same-timestamp rows are excluded (conservative)
    assert out.c_mean_1D.iloc[2] == pytest.approx(20.0)
    assert np.isnan(out.c_mean_1D.iloc[0])


def test_customer_window_left_boundary_included_current_excluded():
    # pandas closed='left' window is [t-w, t): a tx exactly w earlier is included
    df = _frame([(1, 0, 1, 1, 10, 0), (2, 0, 1, 1, 20, 0)])
    out = add_customer_features(df, ("1D",))
    assert out.c_cnt_1D.iloc[1] == 1 and out.c_mean_1D.iloc[1] == 10.0
    assert out.c_cnt_1D.iloc[0] == 0


def test_unseen_customer_and_terminal_get_safe_defaults():
    df = _frame([(40, 1, 7, 99, 25.0, 0)])
    out = build_features(df, 7)
    r = out.iloc[0]
    assert r.c_cnt_1D == 0 and r.c_cnt_30D == 0 and np.isnan(r.c_mean_30D)
    assert r.amt_ratio == pytest.approx(1.0, abs=1e-5)   # falls back to the amount itself
    assert r.t_vol == 0 and r.t_risk == 0


def test_sparse_terminal_ids_ok():
    df = _frame([(1, 1, 1, 0, 5, 0), (30, 1, 2, 5000, 5, 0)])
    out = build_features(df, 7)
    assert len(out) == 2 and out.t_vol.iloc[1] == 0


def test_build_features_handles_unsorted_input_and_does_not_mutate():
    rng = np.random.default_rng(0)
    rows = [(int(d), 1, int(c), int(t), float(a), int(f)) for d, c, t, a, f in zip(
        rng.integers(0, 40, 300), rng.integers(0, 5, 300), rng.integers(0, 5, 300),
        rng.uniform(1, 100, 300).round(2), rng.random(300) < 0.1)]
    df = _frame(rows)
    ref = build_features(df, 7)
    shuf = df.sample(frac=1, random_state=1).reset_index(drop=True)
    snap = shuf.copy()
    out = build_features(shuf, 7)
    pd.testing.assert_frame_equal(shuf, snap)
    key = ["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]
    m = ref.merge(out, on=key, suffixes=("_r", "_s"))
    for c in ("c_cnt_7D", "c_mean_7D", "t_risk", "t_vol"):
        np.testing.assert_allclose(m[c + "_r"], m[c + "_s"], equal_nan=True)


# ----------------------------------------------------------------- metrics edge cases
def test_missing_classes_give_nan_not_crash():
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(recall_at_fpr(np.zeros(3), s))
    assert np.isnan(recall_at_fpr(np.ones(3), s))
    m = window_metrics(np.zeros(3), s)
    assert np.isnan(m["ap"]) and np.isnan(m["recall_at_1fpr"]) and np.isnan(m["ap_over_prevalence"])


def test_weekly_eval_with_week_without_fraud():
    day = np.repeat(np.arange(14), 50)
    y = np.zeros(len(day), dtype=int); y[:5] = 1           # frauds only in the first week
    df = pd.DataFrame({"TX_TIME_DAYS": day, "TX_FRAUD": y})
    res = evaluate_weekly(df, np.random.default_rng(0).random(len(df)), 0, 13)
    assert len(res) == 2 and np.isnan(res.recall_at_1fpr.iloc[1])


def test_recall_threshold_ties_use_strict_greater():
    y = np.array([0] * 100 + [1] * 10)
    assert recall_at_fpr(y, np.full(110, 0.5)) == 0.0       # nothing strictly above the threshold


def test_actual_fpr_matches_target_on_large_sample():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.01).astype(int)
    p = rng.random(200_000) + 0.6 * y
    thr = np.quantile(p[y == 0], 0.99)
    assert (p[y == 0] > thr).mean() == pytest.approx(0.01, abs=1e-4)


# ----------------------------------------------------------------- stale-cache protection
def test_fingerprint_changes_with_labels_amounts_delay_and_times():
    df = _frame([(1, 1, 1, 1, 5, 0), (3, 1, 2, 1, 6, 1), (9, 1, 1, 2, 7, 0)])
    base = features_fingerprint(df, 7)
    assert features_fingerprint(df.copy(), 7) == base
    lab = df.copy(); lab.loc[0, "TX_FRAUD"] = 1
    amt = df.copy(); amt.loc[0, "TX_AMOUNT"] = 99
    tim = df.copy(); tim["TX_DATETIME"] = tim.TX_DATETIME + pd.Timedelta("6h")
    assert len({base, features_fingerprint(lab, 7), features_fingerprint(amt, 7),
                features_fingerprint(tim, 7), features_fingerprint(df, 14)}) == 5


def test_cache_not_reused_after_data_change(tmp_path):
    df = _frame([(1, 1, 1, 1, 5, 0), (30, 1, 2, 1, 6, 1), (40, 1, 1, 2, 7, 0)])
    a = load_or_build_features(df, 7, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 1
    changed = df.copy(); changed.loc[1, "TX_AMOUNT"] = 500.0
    b = load_or_build_features(changed, 7, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 2
    assert not np.allclose(a.TX_AMOUNT, b.TX_AMOUNT)
    c = load_or_build_features(df, 7, cache_dir=tmp_path)           # same data -> cache hit
    assert len(list(tmp_path.glob("*.parquet"))) == 2
    pd.testing.assert_frame_equal(a, c)
