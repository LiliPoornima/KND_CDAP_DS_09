"""Data loader and caching helpers for Component 04 Fraud Detection Frontend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "fraud_detection"
BACKEND_DIR = PROJECT_ROOT / "backend" / "src" / "fraud_detection"


@st.cache_data
def load_grid_summary() -> pd.DataFrame:
    """Load the full 660-run factorial grid summary CSV."""
    csv_path = OUTPUTS_DIR / "experiments" / "grid_results_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing grid summary at {csv_path}")
    return pd.read_csv(csv_path)


@st.cache_data
def load_grid_long() -> pd.DataFrame:
    """Load the weekly longitudinal trajectory CSV."""
    csv_path = OUTPUTS_DIR / "experiments" / "grid_results_long.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing grid longitudinal data at {csv_path}")
    return pd.read_csv(csv_path)


@st.cache_data
def load_decisions(scenario: str, delay: int = 7, seed: int = 1) -> Optional[pd.DataFrame]:
    """Load weekly decisions and signals for a specific scenario run."""
    scenario_clean = scenario.lower()
    csv_path = OUTPUTS_DIR / "detection" / f"decisions_{scenario_clean}_d{delay}_seed{seed}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["has_signals"] = True
        return df
    
    # Fallback to signals CSV if decisions file is not specific
    sig_path = OUTPUTS_DIR / "signals" / f"weekly_signals_{scenario_clean}_d{delay}_seed{seed}.csv"
    if sig_path.exists():
        df = pd.read_csv(sig_path)
        df["has_signals"] = True
        return df

    # Fallback to grid_results_long.csv
    try:
        long_df = load_grid_long()
        sub = long_df[
            (long_df["scenario"] == scenario)
            & (long_df["delay"] == delay)
            & (long_df["seed"] == seed)
            & (long_df["method"] == "P1_aga_full")
        ].copy()
        if not sub.empty:
            sub["decision"] = sub["retrain_triggered"].map(lambda r: "ADAPT" if r else "WAIT")
            sub["ews"] = sub["retrain_triggered"].map(lambda r: 0.38 if r else 0.15)
            sub["cooldown_active"] = False
            sub["has_signals"] = False
            return sub
    except Exception:
        pass
    
    return None


@st.cache_data
def load_sparkov_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Sparkov empirical fallback evaluation data."""
    deg_path = OUTPUTS_DIR / "sparkov" / "sparkov_degradation_check.csv"
    sig_path = OUTPUTS_DIR / "sparkov" / "sparkov_signals_check.csv"
    
    deg_df = pd.read_csv(deg_path) if deg_path.exists() else pd.DataFrame()
    sig_df = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    return deg_df, sig_df


@st.cache_data
def load_locked_thresholds() -> Dict[str, Any]:
    """Load locked monitoring thresholds and decision boundaries."""
    json_path = BACKEND_DIR / "locked_thresholds.json"
    if not json_path.exists():
        json_path = OUTPUTS_DIR / "detection" / "locked_thresholds.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "theta1": 0.22,
        "theta2": 0.35,
        "noise_floor": {
            "mean_psi": 0.000845,
            "max_psi": 0.003006,
            "score_js": 0.012892,
            "entropy_rel": 0.005481,
            "shap_shift": 0.014270,
        },
    }
