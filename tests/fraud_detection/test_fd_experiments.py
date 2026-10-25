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
