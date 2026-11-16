"""Tests for features, training guard and metrics."""

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_evaluation import (
    average_precision, recall_at_fpr, window_metrics,
)
from backend.src.fraud_detection.fd_feature_engineering import (
    add_customer_features, add_terminal_features, build_features,
)
from backend.src.fraud_detection.fd_preprocessing import matured_mask
from backend.src.fraud_detection.fd_training import train_model


def _toy(n=600, days=40, seed=0):
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(0, days, n))
    dt = pd.Timestamp("2018-04-01") + pd.to_timedelta(day, "D") + pd.to_timedelta(
        rng.integers(0, 86400, n), "s")
    df = pd.DataFrame({
        "TX_DATETIME": dt,
        "CUSTOMER_ID": rng.integers(0, 6, n),
        "TERMINAL_ID": rng.integers(0, 5, n),
        "TX_AMOUNT": rng.uniform(1, 100, n).round(2),
        "TX_TIME_DAYS": day,
        "TX_FRAUD": (rng.random(n) < 0.1).astype(int),
    })
    return df.sort_values("TX_DATETIME").reset_index(drop=True)


def test_customer_features_exclude_current_row():
    df = _toy()
    out = add_customer_features(df, windows=("1D", "7D"))
    for w, delta in (("1D", pd.Timedelta("1D")), ("7D", pd.Timedelta("7D"))):
        for i in range(0, len(df), 37):
            r = df.iloc[i]
            prev = df[(df.CUSTOMER_ID == r.CUSTOMER_ID)
                      & (df.TX_DATETIME < r.TX_DATETIME)
                      & (df.TX_DATETIME >= r.TX_DATETIME - delta)]
            # strictly-previous rows only (window is (t-w, t) with closed='left'); same-time rows excluded
            prev = prev[prev.TX_DATETIME > r.TX_DATETIME - delta]
            assert out.loc[i, f"c_cnt_{w}"] == len(prev)
            if len(prev):
                assert out.loc[i, f"c_mean_{w}"] == pytest.approx(prev.TX_AMOUNT.mean(), rel=1e-4)
            else:
                assert np.isnan(out.loc[i, f"c_mean_{w}"])


def test_first_transaction_of_customer_has_no_history():
    df = _toy()
    out = add_customer_features(df, windows=("30D",))
    first = df.groupby("CUSTOMER_ID").head(1).index
    assert (out.loc[first, "c_cnt_30D"] == 0).all()


@pytest.mark.parametrize("delay,window", [(2, 5), (7, 10)])
def test_terminal_features_match_manual_toy_computation(delay, window):
    df = _toy()
    out = add_terminal_features(df, delay=delay, window=window)
    for i in range(0, len(df), 23):
        r = df.iloc[i]
        D = int(r.TX_TIME_DAYS)
        m = matured_mask(df, D, delay) & (df.TX_TIME_DAYS > D - delay - window) \
            & (df.TERMINAL_ID == r.TERMINAL_ID)
        sub = df[m]
        assert out.loc[i, "t_vol"] == len(sub)
        exp = sub.TX_FRAUD.sum() / len(sub) if len(sub) else 0.0
        assert out.loc[i, "t_risk"] == pytest.approx(exp, abs=1e-6)


@pytest.mark.parametrize("delay", [7, 14])
def test_leakage_future_labels_do_not_change_features(delay):
    df = _toy(n=900, days=60)
    base = build_features(df, delay)
    rng = np.random.default_rng(1)
    for t in rng.integers(20, 55, size=5):
        mod = df.copy()
        future = mod.TX_TIME_DAYS > t - delay      # labels NOT matured at time t
        mod.loc[future, "TX_FRAUD"] = 1 - mod.loc[future, "TX_FRAUD"]
        alt = build_features(mod, delay)
        seen = base.TX_TIME_DAYS <= t              # rows already observed at time t
        cols = ["t_risk", "t_vol", "c_mean_30D", "c_cnt_30D", "amt_ratio"]
        pd.testing.assert_frame_equal(base.loc[seen, cols], alt.loc[seen, cols])


def test_build_features_has_all_model_features_and_does_not_mutate():
    df = _toy()
    cols_before = list(df.columns)
    out = build_features(df, 7)
    from backend.src.fraud_detection.fd_config import MODEL_FEATURES
    assert all(f in out.columns for f in MODEL_FEATURES)
    assert list(df.columns) == cols_before


def test_training_rejects_forbidden_features():
    df = build_features(_toy(), 2)
    with pytest.raises(ValueError, match="Forbidden"):
        train_model(df, seed=1, features=["TX_AMOUNT", "TX_FRAUD_SCENARIO"])


def test_recall_at_fpr_known_answer():
    y = np.array([0] * 100 + [1] * 4)
    p = np.concatenate([np.linspace(0, 0.99, 100), [0.995, 0.5, 0.999, 0.1]])
    # 99th percentile of negatives ~0.98 -> only 0.995 and 0.999 exceed it
    assert recall_at_fpr(y, p, 0.01) == pytest.approx(0.5)
    assert np.isnan(recall_at_fpr(np.zeros(5), np.arange(5)))


def test_ap_over_prevalence_perfect_ranking():
    y = np.array([0, 0, 0, 1])
    m = window_metrics(y, np.array([0.1, 0.2, 0.3, 0.9]))
    assert m["ap"] == pytest.approx(1.0)
    assert m["ap_over_prevalence"] == pytest.approx(4.0)
    assert np.isnan(average_precision(np.zeros(3), np.arange(3)))
