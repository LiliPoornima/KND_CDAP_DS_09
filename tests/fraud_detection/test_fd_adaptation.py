"""Tests for Attribution-Guided Adaptation (AGA) and baselines (Spec Section 11 & 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_adaptation import (
    CANDIDATE_DROP_D,
    CANDIDATE_FULL,
    CANDIDATE_RECENCY,
    build_matured_slices,
    compute_recency_weights,
    evaluate_candidates,
    run_aga,
    select_top_k_features,
    train_candidates,
)
from backend.src.fraud_detection.fd_baselines import (
    CBPEstimator,
    METHOD_B0,
    METHOD_B1,
    METHOD_B2,
    METHOD_B3,
    METHOD_B4,
    METHOD_B5,
    METHOD_P1,
    ProdemMetaModel,
    run_benchmark_method,
)
from backend.src.fraud_detection.fd_config import COL_DAY, COL_FRAUD, MODEL_FEATURES


@pytest.fixture
def sample_matured_df() -> pd.DataFrame:
    """Create miniature DataFrame with days 80..130."""
    records = []
    base_time = pd.Timestamp("2018-04-01")
    rng = np.random.default_rng(42)

    for day in range(80, 131):
        for i in range(40):
            dt = base_time + pd.Timedelta(days=day, hours=i % 24)
            amt = float(50.0 + rng.exponential(scale=30.0))
            is_fraud = int(rng.uniform() < (0.05 if day >= 105 else 0.01))
            row = {
                "TRANSACTION_ID": len(records) + 1,
                "TX_DATETIME": dt,
                "CUSTOMER_ID": (i % 10) + 1,
                "TERMINAL_ID": (i % 5) + 1,
                "TX_AMOUNT": amt,
                "TX_TIME_SECONDS": int((dt - base_time).total_seconds()),
                COL_DAY: day,
                COL_FRAUD: is_fraud,
                "TX_FRAUD_SCENARIO": 1 if is_fraud else 0,
                "hour": float(dt.hour),
                "dow": float(dt.day_of_week),
                "amt_ratio": 1.0,
                "t_risk": 0.01,
                "t_vol": 10.0,
                "c_mean_1D": 50.0,
                "c_cnt_1D": 2.0,
                "c_mean_7D": 50.0,
                "c_cnt_7D": 10.0,
                "c_mean_30D": 50.0,
                "c_cnt_30D": 30.0,
            }
            records.append(row)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- Matured Labels Leakage Proof
def test_aga_strictly_obeys_matured_label_boundary(sample_matured_df: pd.DataFrame):
    decision_day = 126
    delay = 7
    cutoff = decision_day - delay  # Day 119

    df_train, df_val, tr_start, tr_end, val_start, val_end = build_matured_slices(
        sample_matured_df, decision_day=decision_day, delay=delay
    )

    # Prove all rows strictly satisfy day <= cutoff
    assert df_train[COL_DAY].max() <= cutoff
    assert df_val[COL_DAY].max() <= cutoff
    assert tr_end < val_start
    assert val_end == cutoff

    # Perturbation test: change unmatured labels on Day 120 and Day 125
    perturbed_df = sample_matured_df.copy(deep=True)
    unmatured_mask = perturbed_df[COL_DAY] > cutoff
    perturbed_df.loc[unmatured_mask, COL_FRAUD] = 1 - perturbed_df.loc[unmatured_mask, COL_FRAUD]

    # Re-slice and prove training & validation sets are 100% byte-for-byte identical
    df_train_p, df_val_p, _, _, _, _ = build_matured_slices(
        perturbed_df, decision_day=decision_day, delay=delay
    )
    pd.testing.assert_frame_equal(df_train, df_train_p)
    pd.testing.assert_frame_equal(df_val, df_val_p)


# --------------------------------------------------------------------------- Candidates & Selection
def test_candidates_training_and_selection(sample_matured_df: pd.DataFrame):
    decision_day = 126
    delay = 7
    D = ["c_mean_30D", "c_mean_7D", "TX_AMOUNT"]

    df_train, df_val, _, _, _, _ = build_matured_slices(
        sample_matured_df, decision_day=decision_day, delay=delay
    )

    candidates = train_candidates(df_train, D=D, seed=1)
    assert CANDIDATE_FULL in candidates
    assert CANDIDATE_DROP_D in candidates
    assert CANDIDATE_RECENCY in candidates

    # Check dropped features in DROP_D
    model_drop, drop_feats = candidates[CANDIDATE_DROP_D]
    for dropped in D:
        assert dropped not in drop_feats

    # Selection
    winner, val_scores = evaluate_candidates(candidates, df_val)
    assert winner in (CANDIDATE_FULL, CANDIDATE_DROP_D, CANDIDATE_RECENCY)
    assert len(val_scores) == 3


# --------------------------------------------------------------------------- Feature Selection Modes
def test_feature_selection_modes():
    deltas = {
        "delta_TX_AMOUNT": 0.05,
        "delta_hour": 0.01,
        "delta_c_mean_30D": 0.08,
        "psi_TX_AMOUNT": 0.12,
        "psi_hour": 0.02,
        "psi_c_mean_30D": 0.25,
    }

    # SHAP mode
    top_shap = select_top_k_features(deltas, k=2, mode="shap")
    assert top_shap == ["c_mean_30D", "TX_AMOUNT"]

    # PSI mode
    top_psi = select_top_k_features(deltas, k=2, mode="psi")
    assert top_psi == ["c_mean_30D", "TX_AMOUNT"]

    # Random mode
    top_rnd1 = select_top_k_features(deltas, k=2, mode="random", seed=10)
    top_rnd2 = select_top_k_features(deltas, k=2, mode="random", seed=10)
    assert top_rnd1 == top_rnd2
    assert len(top_rnd1) == 2


# --------------------------------------------------------------------------- Recency Weights Decay
def test_recency_weights_decay():
    days = np.array([100, 105, 110, 115, 120])
    w = compute_recency_weights(days, half_life_days=7.0)

    # Weights must be strictly monotonically increasing with day
    assert (np.diff(w) > 0).all()
    # Mean weight must be 1.0
    assert pytest.approx(1.0, abs=1e-5) == np.mean(w)


# --------------------------------------------------------------------------- Baselines Smoke Test
def test_cbpe_and_prodem_estimators(sample_matured_df: pd.DataFrame):
    cbpe = CBPEstimator(target_fpr=0.01)
    probs = np.linspace(0.01, 0.99, 100)
    rec = cbpe.estimate_recall(probs)
    assert 0.0 <= rec <= 1.0

    prodem = ProdemMetaModel(seed=42)
    # Fit mock model
    from lightgbm import LGBMClassifier
    base = LGBMClassifier(n_estimators=10, verbose=-1, random_state=42)
    X = sample_matured_df[list(MODEL_FEATURES)]
    y = sample_matured_df[COL_FRAUD]
    base.fit(X, y)
    base.fd_features_ = list(MODEL_FEATURES)

    prodem.fit(base, sample_matured_df)
    assert prodem.is_fitted is True
    err = prodem.predict_error_rate(base, sample_matured_df)
    assert 0.0 <= err <= 1.0
