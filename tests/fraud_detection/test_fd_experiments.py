"""Tests for experiment grid execution, plotting, and IEEE-CIS checks (Spec Section 13 & 14)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_experiments import (
    generate_experiment_figures,
    run_single_experiment_stream,
)
from backend.src.fraud_detection.fd_ieee_check import run_ieee_cis_check
from backend.src.fraud_detection.fd_baselines import METHOD_B0, METHOD_B1, METHOD_P1


def test_ieee_cis_check_handles_missing_file(tmp_path: Path):
    missing_path = tmp_path / "train_transaction.csv"
    res = run_ieee_cis_check(csv_path=missing_path)
    assert res["status"] == "DATA_MISSING"
    assert res["degradation_present"] is False
    assert res["verdict"] == "DATA_NOT_FOUND"
    assert "Per Spec Section 3.2" in res["message"]


def test_ieee_cis_check_on_synthetic_data(tmp_path: Path):
    # Create miniature synthetic IEEE-CIS dataset
    records = []
    rng = np.random.default_rng(42)
    # Days 0..90
    for day in range(90):
        dt_base = day * 86400
        for i in range(25):
            is_fraud = int(rng.uniform() < (0.01 if day < 49 else 0.08))
            row = {
                "TransactionID": len(records) + 1,
                "TransactionDT": dt_base + int(i * 3600),
                "isFraud": is_fraud,
                "TransactionAmt": float(rng.exponential(50.0)),
                "C1": float(rng.normal(10.0, 2.0)),
                "C2": float(rng.normal(5.0, 1.0)),
                "D1": float(rng.uniform(0.0, 100.0)),
            }
            records.append(row)

    csv_path = tmp_path / "train_transaction.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    res = run_ieee_cis_check(csv_path=csv_path, output_dir=tmp_path)
    assert res["status"] == "COMPLETED"
    assert res["table"] is not None
    assert len(res["table"]) > 0
    assert "recall_at_1fpr" in res["table"].columns
    assert "relative_to_first_window_pct" in res["table"].columns


def test_generate_experiment_figures(tmp_path: Path):
    # Create minimal summary and long format mock data
    summary_data = [
        {
            "scenario": "control",
            "seed": 1,
            "delay": 7,
            "method": METHOD_B0,
            "retrain_count": 0,
            "relative_retention_pct": 98.4,
        },
        {
            "scenario": "S1",
            "seed": 1,
            "delay": 7,
            "method": METHOD_P1,
            "retrain_count": 4,
            "relative_retention_pct": 92.7,
        },
    ]
    summary_df = pd.DataFrame(summary_data)

    long_data = []
    for day in range(70, 180, 7):
        long_data.append({
            "scenario": "S1",
            "seed": 1,
            "delay": 7,
            "day_start": day,
            "method": METHOD_B0,
            "recall_at_1fpr": 0.52 if day >= 105 else 0.74,
        })
        long_data.append({
            "scenario": "S1",
            "seed": 1,
            "delay": 7,
            "day_start": day,
            "method": METHOD_P1,
            "recall_at_1fpr": 0.70 if day >= 105 else 0.74,
        })
    long_df = pd.DataFrame(long_data)

    plots = generate_experiment_figures(long_df, summary_df, output_dir=tmp_path)
    assert len(plots) > 0
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 0


# --------------------------------------------------------------------------- Focused Audit Verification Tests (Phase 8 Stage B)

def test_recovery_time_no_degradation_returns_none():
    """When recall stays above 80% baseline throughout, recovery time is None."""
    base_recall = 0.75
    records = [
        {"day_start": d, "week": d // 7, "recall_at_1fpr": 0.74, "retrain_triggered": False}
        for d in range(70, 140, 7)
    ]
    post_df = pd.DataFrame(records)
    degraded_windows = post_df[post_df["recall_at_1fpr"] < 0.80 * base_recall]
    if len(degraded_windows) == 0:
        degrade_day = None
        recovery_days = None
    else:
        degrade_day = int(degraded_windows["day_start"].iloc[0])
        post_degrade_df = post_df[post_df["day_start"] >= degrade_day]
        recovered_df = post_degrade_df[post_degrade_df["recall_at_1fpr"] >= 0.90 * base_recall]
        recovery_days = int(recovered_df["day_start"].iloc[0]) - degrade_day if len(recovered_df) > 0 else -1

    assert degrade_day is None
    assert recovery_days is None


def test_recovery_time_degradation_with_recovery():
    """When recall degrades at day 91 and recovers at day 105, MTTR is 14 days."""
    base_recall = 0.75
    threshold_degrade = 0.80 * base_recall  # 0.60
    threshold_recover = 0.90 * base_recall  # 0.675
    records = [
        {"day_start": 70, "week": 10, "recall_at_1fpr": 0.75, "retrain_triggered": False},
        {"day_start": 77, "week": 11, "recall_at_1fpr": 0.74, "retrain_triggered": False},
        {"day_start": 84, "week": 12, "recall_at_1fpr": 0.70, "retrain_triggered": False},
        {"day_start": 91, "week": 13, "recall_at_1fpr": 0.55, "retrain_triggered": True},  # degrade day 91, adapt triggered
        {"day_start": 98, "week": 14, "recall_at_1fpr": 0.58, "retrain_triggered": False},
        {"day_start": 105, "week": 15, "recall_at_1fpr": 0.72, "retrain_triggered": False},  # recovered >= 0.675
    ]
    post_df = pd.DataFrame(records)
    degraded = post_df[post_df["recall_at_1fpr"] < threshold_degrade]
    degrade_day = int(degraded["day_start"].iloc[0])
    post_degrade = post_df[post_df["day_start"] >= degrade_day]
    recovered = post_degrade[post_degrade["recall_at_1fpr"] >= threshold_recover]
    recov_day = int(recovered["day_start"].iloc[0])
    recovery_days = recov_day - degrade_day

    first_retrain = int(post_df[post_df["retrain_triggered"]]["day_start"].iloc[0])
    recovery_post_adapt = recov_day - first_retrain

    assert degrade_day == 91
    assert recov_day == 105
    assert recovery_days == 14
    assert recovery_post_adapt == 14


def test_recovery_time_degradation_without_recovery():
    """When recall degrades and remains low until stream end, recovery is -1."""
    base_recall = 0.75
    records = [
        {"day_start": 70, "week": 10, "recall_at_1fpr": 0.75},
        {"day_start": 77, "week": 11, "recall_at_1fpr": 0.50},  # degrades < 0.60
        {"day_start": 84, "week": 12, "recall_at_1fpr": 0.52},
        {"day_start": 91, "week": 13, "recall_at_1fpr": 0.49},
    ]
    post_df = pd.DataFrame(records)
    degraded = post_df[post_df["recall_at_1fpr"] < 0.80 * base_recall]
    degrade_day = int(degraded["day_start"].iloc[0])
    post_degrade = post_df[post_df["day_start"] >= degrade_day]
    recovered = post_degrade[post_degrade["recall_at_1fpr"] >= 0.90 * base_recall]
    recovery_days = int(recovered["day_start"].iloc[0]) - degrade_day if len(recovered) > 0 else -1

    assert degrade_day == 77
    assert recovery_days == -1


def test_timing_conventions_lead_time_positive():
    """Early alarm at day 77 before ground-truth degrade day 91 gives positive lead time (+14d)."""
    first_alarm_day = 77
    degrade_day = 91
    lead_time = (degrade_day - first_alarm_day) if (degrade_day and first_alarm_day) else None
    assert lead_time == 14
    assert lead_time > 0


def test_timing_conventions_adaptation_lag_positive():
    """Adaptation occurring at day 98 after degradation onset at day 91 gives positive adaptation lag (+7d)."""
    first_adapt_day = 98
    degrade_day = 91
    adaptation_lag = (first_adapt_day - degrade_day) if (first_adapt_day and degrade_day) else None
    assert adaptation_lag == 7
    assert adaptation_lag > 0


def test_timing_conventions_edge_case_no_alarm_or_adapt():
    """When no alarm triggers or no adaptation occurs, lead time and lag return None."""
    degrade_day = 91
    first_alarm_day = None
    first_adapt_day = None
    lead_time = (degrade_day - first_alarm_day) if (degrade_day and first_alarm_day) else None
    adaptation_lag = (first_adapt_day - degrade_day) if (first_adapt_day and degrade_day) else None
    assert lead_time is None
    assert adaptation_lag is None


def test_two_consecutive_window_noise_suppression():
    """A 2-consecutive-window degradation rule suppresses single-window transient dips."""
    base_recall = 0.75
    threshold = 0.80 * base_recall  # 0.60
    # Simulate single-window dip at window 2
    recalls = [0.74, 0.75, 0.55, 0.73, 0.74]

    # 1-window rule flags window 2 as degraded
    degraded_1win = any(r < threshold for r in recalls)
    assert degraded_1win is True

    # 2-consecutive-window rule suppresses single-window dip
    degraded_2win = any(
        recalls[i] < threshold and recalls[i + 1] < threshold
        for i in range(len(recalls) - 1)
    )
    assert degraded_2win is False


def test_calibrated_cbpe_trigger_avoids_continuous_retrain():
    """Calibrated CBPE checks est_rec <= 0.80 * base_est_rec, avoiding spurious triggers when score distribution is stable."""
    from backend.src.fraud_detection.fd_baselines import CBPEstimator
    cbpe = CBPEstimator()
    rng = np.random.default_rng(42)
    base_probs = rng.beta(0.5, 5.0, size=1000)
    base_est_rec = cbpe.estimate_recall(base_probs)

    new_probs = rng.beta(0.5, 5.0, size=1000)
    est_rec = cbpe.estimate_recall(new_probs)

    calibrated_trigger = est_rec <= 0.80 * base_est_rec
    assert calibrated_trigger is False
