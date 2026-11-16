"""Exploratory analysis of the default (no injected drift) Handbook data.

Run: ``python -m backend.src.fraud_detection.fd_eda``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .fd_config import (
    COL_AMOUNT, COL_DAY, COL_FRAUD, COL_SCENARIO, LAST_TEST_DAY, OUTPUT_DIR, get_logger,
)
from .fd_data_loader import load_handbook

logger = get_logger(__name__)


def weekly_fraud_table(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly transaction count, fraud count and fraud rate."""
    week = df[COL_DAY] // 7
    g = df.groupby(week)[COL_FRAUD].agg(["size", "sum"])
    g.columns = ["n_tx", "n_fraud"]
    g["fraud_rate"] = g["n_fraud"] / g["n_tx"]
    g.index.name = "week"
    return g.reset_index()


def weekly_scenario_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud counts per scenario per week (scenario 0 = genuine)."""
    week = df[COL_DAY] // 7
    fr = df[df[COL_FRAUD] == 1]
    return (fr.groupby([week[fr.index], COL_SCENARIO]).size()
            .unstack(fill_value=0).rename_axis("week").reset_index())


def amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Amount distribution summary for fraud vs genuine."""
    return df.groupby(COL_FRAUD)[COL_AMOUNT].describe(percentiles=[.5, .9, .99])


def drift_summary(weekly: pd.DataFrame, first_week: int = 10, last_week: int = 25) -> Dict[str, float]:
    """Plain numbers describing stability of the weekly fraud rate (no verdict is forced)."""
    w = weekly[(weekly["week"] >= first_week) & (weekly["week"] <= last_week)]
    slope = float(np.polyfit(w["week"], w["fraud_rate"], 1)[0])
    return {
        "mean_rate": float(w["fraud_rate"].mean()),
        "std_rate": float(w["fraud_rate"].std()),
        "cv": float(w["fraud_rate"].std() / w["fraud_rate"].mean()),
        "slope_per_week": slope,
        "first_half_mean": float(w.head(len(w) // 2)["fraud_rate"].mean()),
        "second_half_mean": float(w.tail(len(w) // 2)["fraud_rate"].mean()),
    }


def run_eda(output_dir: Path = OUTPUT_DIR / "eda") -> Dict[str, float]:
    """Compute EDA tables/figures and save them under ``outputs/fraud_detection/eda``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_handbook()
    df = df[df[COL_DAY] <= LAST_TEST_DAY + 1]

    weekly = weekly_fraud_table(df)
    mix = weekly_scenario_mix(df)
    amt = amount_summary(df)
    summ = drift_summary(weekly)
    weekly.to_csv(output_dir / "weekly_fraud_rate.csv", index=False)
    mix.to_csv(output_dir / "weekly_scenario_mix.csv", index=False)
    amt.to_csv(output_dir / "amount_summary_by_label.csv")
    pd.Series(summ).to_csv(output_dir / "drift_summary.csv", header=["value"])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(weekly["week"], weekly["fraud_rate"] * 100, marker="o")
    ax.set(xlabel="week", ylabel="fraud rate (%)", title="Weekly fraud rate (default data)")
    fig.tight_layout(); fig.savefig(output_dir / "weekly_fraud_rate.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    mix.set_index("week").plot.bar(stacked=True, ax=ax)
    ax.set(ylabel="fraud count", title="Fraud scenario mix per week")
    fig.tight_layout(); fig.savefig(output_dir / "weekly_scenario_mix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bins = np.linspace(0, df[COL_AMOUNT].quantile(0.995), 60)
    for lab, name in ((0, "genuine"), (1, "fraud")):
        ax.hist(df.loc[df[COL_FRAUD] == lab, COL_AMOUNT], bins=bins, alpha=0.6,
                density=True, label=name)
    ax.set(xlabel="TX_AMOUNT", title="Amount distribution: fraud vs genuine"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "amount_distribution.png", dpi=120); plt.close(fig)

    logger.info("EDA saved to %s; weekly-rate summary (weeks 10-25): %s", output_dir, summ)
    return summ


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for k, v in run_eda().items():
        logger.info("%s = %.6f", k, v)
