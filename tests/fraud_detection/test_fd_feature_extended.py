"""Tests for extended features and leakage safety."""

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_feature_extended import (
    ALL_EXTENDED_FEATURES,
    EXTENDED_FEATURE_NAMES,
    add_extended_customer_features,
    add_smoothed_terminal_risk,
    build_extended_features,
    extended_features_fingerprint,
    load_or_build_extended_features,
)
from backend.src.fraud_detection.fd_preprocessing import matured_mask


def _toy(n=600, days=40, seed=0):
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(0, days, n))
    dt = pd.Timestamp("2018-04-01") + pd.to_timedelta(day, "D") + pd.to_timedelta(
        rng.integers(0, 86400, n), "s"
    )
    df = pd.DataFrame({
        "TX_DATETIME": dt,
        "CUSTOMER_ID": rng.integers(0, 6, n),
        "TERMINAL_ID": rng.integers(0, 5, n),
        "TX_AMOUNT": rng.uniform(1, 100, n).round(2),
        "TX_TIME_DAYS": day,
        "TX_FRAUD": (rng.random(n) < 0.1).astype(int),
    })
    return df.sort_values("TX_DATETIME").reset_index(drop=True)


def test_extended_features_contain_all_expected_columns():
    df = _toy()
    out = build_extended_features(df, delay=7)
    for col in ALL_EXTENDED_FEATURES:
        assert col in out.columns, f"Missing feature: {col}"
        assert not np.isinf(out[col]).any(), f"Infinite values in {col}"


def test_customer_extended_features_exclude_current_row():
    df = _toy()
    out = build_extended_features(df, delay=7)
    for i in range(0, len(df), 41):
        r = df.iloc[i]
        prev = df[
            (df.CUSTOMER_ID == r.CUSTOMER_ID)
            & (df.TX_DATETIME < r.TX_DATETIME)
            & (df.TX_DATETIME >= r.TX_DATETIME - pd.Timedelta("30D"))
        ]
        prev = prev[prev.TX_DATETIME > r.TX_DATETIME - pd.Timedelta("30D")]
        if len(prev) >= 2:
            assert out.loc[i, "c_std_30D"] == pytest.approx(prev.TX_AMOUNT.std(), rel=1e-3)
        elif len(prev) < 2:
            assert out.loc[i, "c_std_30D"] == 0.0


def test_extended_features_leakage_safety_test_period():
    # Test stream occurs at t >= 70 (after training days 35..62 have fully matured)
    df = _toy(n=1500, days=150)
    delay = 7
    t = 100  # Evaluation during test period
    base = build_extended_features(df, delay=delay)

    # Flip future labels (not yet matured at t)
    mod = df.copy()
    future = mod.TX_TIME_DAYS > t - delay
    mod.loc[future, "TX_FRAUD"] = 1 - mod.loc[future, "TX_FRAUD"]
    alt = build_extended_features(mod, delay=delay)

    seen = base.TX_TIME_DAYS <= t
    for col in ALL_EXTENDED_FEATURES:
        a = base.loc[seen, col].to_numpy(dtype=np.float64)
        b = alt.loc[seen, col].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(a, b, equal_nan=True, err_msg=f"Leakage detected in {col}")


def test_extended_features_leakage_safety_explicit_prior():
    # When prior is explicitly passed from training data
    df = _toy(n=800, days=50)
    delay = 7
    t = 30
    fixed_prior = 0.05
    base = build_extended_features(df, delay=delay, prior_fraud_rate=fixed_prior)

    mod = df.copy()
    future = mod.TX_TIME_DAYS > t - delay
    mod.loc[future, "TX_FRAUD"] = 1 - mod.loc[future, "TX_FRAUD"]
    alt = build_extended_features(mod, delay=delay, prior_fraud_rate=fixed_prior)

    seen = base.TX_TIME_DAYS <= t
    for col in ALL_EXTENDED_FEATURES:
        a = base.loc[seen, col].to_numpy(dtype=np.float64)
        b = alt.loc[seen, col].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(a, b, equal_nan=True, err_msg=f"Leakage detected in {col}")


def test_compute_training_prior_strictly_ignores_validation_and_test_labels():
    from backend.src.fraud_detection.fd_feature_extended import (
        compute_expanding_delay_aware_prior,
        compute_pretraining_warmup_prior,
        compute_training_prior,
    )

    df = _toy(n=1200, days=150)
    delay = 7

    # Expanding delay-aware prior
    base_expanding = compute_expanding_delay_aware_prior(df, delay=delay, train_end=62)
    # Warmup prior
    base_warmup = compute_pretraining_warmup_prior(df, delay=delay, train_start=35)

    # Change all test labels (days 70..181)
    mod_test = df.copy()
    test_mask = mod_test.TX_TIME_DAYS >= 70
    mod_test.loc[test_mask, "TX_FRAUD"] = 1 - mod_test.loc[test_mask, "TX_FRAUD"]
    np.testing.assert_allclose(
        compute_expanding_delay_aware_prior(mod_test, delay=delay, train_end=62),
        base_expanding,
        err_msg="Expanding prior leaked test labels!",
    )
    assert compute_pretraining_warmup_prior(mod_test, delay=delay, train_start=35) == pytest.approx(base_warmup)

    # Change gap labels (days 63..69)
    mod_gap = df.copy()
    gap_mask = (mod_gap.TX_TIME_DAYS >= 63) & (mod_gap.TX_TIME_DAYS <= 69)
    mod_gap.loc[gap_mask, "TX_FRAUD"] = 1 - mod_gap.loc[gap_mask, "TX_FRAUD"]
    np.testing.assert_allclose(
        compute_expanding_delay_aware_prior(mod_gap, delay=delay, train_end=62),
        base_expanding,
        err_msg="Expanding prior leaked gap labels!",
    )
    assert compute_pretraining_warmup_prior(mod_gap, delay=delay, train_start=35) == pytest.approx(base_warmup)


def test_intra_training_label_changes_do_not_leak_to_earlier_rows():
    """Regression test: changing later training labels must leave earlier rows' features unchanged."""
    df = _toy(n=2000, days=70, seed=42)
    delay = 7
    d_early = 40
    d_late = 55  # Later training day, unmatured on Day 40 (40 - 7 = 33 < 55)

    base = build_extended_features(df, delay=delay)

    # Perturb labels on later training day
    mod = df.copy()
    late_mask = mod.TX_TIME_DAYS == d_late
    mod.loc[late_mask, "TX_FRAUD"] = 1 - mod.loc[late_mask, "TX_FRAUD"]
    alt = build_extended_features(mod, delay=delay)

    early_mask = base.TX_TIME_DAYS <= d_early
    assert early_mask.sum() > 0, "Must have earlier rows to test"

    for col in ALL_EXTENDED_FEATURES:
        a = base.loc[early_mask, col].to_numpy(dtype=np.float64)
        b = alt.loc[early_mask, col].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(
            a,
            b,
            equal_nan=True,
            err_msg=f"Intra-training leakage in {col}: earlier rows (<= Day {d_early}) changed when Day {d_late} labels modified!",
        )


def test_add_smoothed_terminal_risk_validates_prior():
    df = _toy(n=200, days=40)
    with pytest.raises(ValueError, match="prior_fraud_rate must be in"):
        add_smoothed_terminal_risk(df, delay=7, prior_fraud_rate=0.0)
    with pytest.raises(ValueError, match="prior_fraud_rate must be in"):
        add_smoothed_terminal_risk(df, delay=7, prior_fraud_rate=1.5)
    with pytest.raises(ValueError, match="prior_fraud_rate array must be in"):
        add_smoothed_terminal_risk(df, delay=7, prior_fraud_rate=np.zeros(41))


def test_extended_features_fingerprint_changes_with_base_feature_code(tmp_path):
    """Verify cache fingerprint changes when base feature engineering code changes, using a safe mocked file."""
    df = _toy(n=200, days=40)
    delay = 7

    # Create two temporary source code versions for base feature engineering
    base_v1 = tmp_path / "base_feat_v1.py"
    base_v2 = tmp_path / "base_feat_v2.py"
    base_v1.write_text("# Base feature engineering version 1\ndef build_features(): pass\n")
    base_v2.write_text("# Base feature engineering version 2 (modified logic)\ndef build_features(): pass\n")

    fp_v1 = extended_features_fingerprint(df, delay=delay, base_module_file=base_v1)
    fp_v2 = extended_features_fingerprint(df, delay=delay, base_module_file=base_v2)

    assert fp_v1 != fp_v2, "Cache fingerprint must change when base feature engineering code changes!"
    assert len(fp_v1) == 16
    assert len(fp_v2) == 16


def test_extended_features_fingerprint_hashes_both_files():
    """Verify default fingerprinting succeeds and hashes both files without error."""
    df = _toy(n=200, days=40)
    fp = extended_features_fingerprint(df, delay=7)
    assert isinstance(fp, str) and len(fp) == 16



