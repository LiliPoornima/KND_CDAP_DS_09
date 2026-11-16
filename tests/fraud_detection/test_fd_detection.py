"""Tests for early-warning score, locked thresholds, and decision policy (Spec Section 10 & 15)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from backend.src.fraud_detection.fd_detection import (
    DECISION_ADAPT,
    DECISION_MONITOR,
    DECISION_WAIT,
    EarlyWarningDetector,
    LockedDetectionConfig,
    create_calibrated_config,
    evaluate_stream,
    load_locked_thresholds,
    save_locked_thresholds,
)
from backend.src.fraud_detection.fd_preprocessing import matured_mask


# --------------------------------------------------------------------------- Threshold Locking & Immutability
def test_thresholds_cannot_change_during_evaluation(tmp_path: Path):
    config = create_calibrated_config()
    json_path = tmp_path / "locked.json"
    save_locked_thresholds(config, json_path)

    detector = EarlyWarningDetector(config_path=json_path)
    orig_theta1 = detector.theta1
    orig_theta2 = detector.theta2
    orig_weights = dict(detector.config.weights)

    # Evaluate multiple mock windows
    for day in range(70, 140, 7):
        detector.evaluate_window(
            signals={"mean_psi": 0.05, "score_js": 0.15, "shap_shift": 0.08},
            current_day=day,
        )

    # Prove thresholds and parameters are 100% unchanged
    assert detector.theta1 == orig_theta1
    assert detector.theta2 == orig_theta2
    assert detector.config.weights == orig_weights

    # Prove LockedDetectionConfig is immutable (frozen dataclass)
    with pytest.raises(dataclasses.FrozenInstanceError):
        detector.config.theta1 = 0.99  # type: ignore


# --------------------------------------------------------------------------- Matured Labels Rule
def test_only_matured_labels_used():
    """Verify that decisions at time t only ever reference matured labels (label_day <= t - delay)."""
    df = pd.DataFrame({
        "TX_TIME_DAYS": [10, 15, 20, 25, 30],
        "TX_FRAUD": [0, 1, 0, 1, 0],
    })
    delay = 7
    decision_time = 25  # Day 25

    # Any label visible must be <= 25 - 7 = 18
    m = matured_mask(df, t=decision_time, d=delay)
    visible_days = df.loc[m, "TX_TIME_DAYS"].to_list()
    assert visible_days == [10, 15]
    assert 20 not in visible_days
    assert 25 not in visible_days


# --------------------------------------------------------------------------- Decision Thresholds
def test_decision_boundaries():
    config = create_calibrated_config()
    detector = EarlyWarningDetector(config=config)

    # 1. EWS < theta1 (0.22) -> WAIT
    dec1 = detector.evaluate_window(
        signals={"mean_psi": 0.0, "score_js": 0.0, "shap_shift": 0.0, "entropy_rel_change": 0.0, "max_psi": 0.0},
        current_day=70,
    )
    assert dec1["ews"] < config.theta1
    assert dec1["decision"] == DECISION_WAIT

    # 2. theta1 <= EWS < theta2 (0.22 <= EWS < 0.35) -> MONITOR
    dec2 = detector.evaluate_window(
        signals={"mean_psi": 0.0004, "score_js": 0.035, "shap_shift": 0.035, "entropy_rel_change": 0.06, "max_psi": 0.002},
        current_day=77,
    )
    assert config.theta1 <= dec2["ews"] < config.theta2
    assert dec2["decision"] == DECISION_MONITOR

    # 3. EWS >= theta2 (0.35) -> ADAPT
    dec3 = detector.evaluate_window(
        signals={"mean_psi": 0.002, "score_js": 0.15, "shap_shift": 0.10, "entropy_rel_change": 0.35, "max_psi": 0.008},
        current_day=84,
    )
    assert dec3["ews"] >= config.theta2
    assert dec3["decision"] == DECISION_ADAPT


# --------------------------------------------------------------------------- Cooldown Enforcement
def test_14_day_cooldown_enforcement():
    config = create_calibrated_config()
    detector = EarlyWarningDetector(config=config)

    high_drift = {
        "mean_psi": 0.002, "score_js": 0.25, "shap_shift": 0.15, "entropy_rel_change": 0.50, "max_psi": 0.01,
    }

    # Day 105: triggers ADAPT
    d1 = detector.evaluate_window(high_drift, current_day=105)
    assert d1["decision"] == DECISION_ADAPT
    assert d1["cooldown_active"] is False

    # Day 112 (7 days later): high drift still present, but cooldown MUST throttle to MONITOR
    d2 = detector.evaluate_window(high_drift, current_day=112)
    assert d2["raw_decision"] == DECISION_ADAPT
    assert d2["decision"] == DECISION_MONITOR
    assert d2["cooldown_active"] is True
    assert d2["days_since_adapt"] == 7

    # Day 119 (14 days later): cooldown has elapsed, can trigger ADAPT again
    d3 = detector.evaluate_window(high_drift, current_day=119)
    assert d3["decision"] == DECISION_ADAPT
    assert d3["cooldown_active"] is False
    assert d3["days_since_adapt"] == 14


# --------------------------------------------------------------------------- Fallback Calculation
def test_fallback_weighted_sum():
    config = create_calibrated_config()
    detector = EarlyWarningDetector(config=config)

    # Zero drift signals -> z-scores near 0 -> fallback probability near 0.5
    zero_signals = {s: config.standardization_stats[s]["mean"] for s in config.signals}
    prob_fallback = detector.compute_ews(zero_signals, use_fallback=True)
    assert pytest.approx(0.5, abs=0.05) == prob_fallback


# --------------------------------------------------------------------------- JSON Roundtrip
def test_locked_json_roundtrip(tmp_path: Path):
    config = create_calibrated_config()
    save_path = tmp_path / "locked_thresholds.json"
    save_locked_thresholds(config, save_path)

    loaded = load_locked_thresholds(save_path)
    assert loaded.theta1 == config.theta1
    assert loaded.theta2 == config.theta2
    assert loaded.weights == config.weights
    assert loaded.is_locked is True
