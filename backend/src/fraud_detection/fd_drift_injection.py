"""Fraud-detection drift injection generators (Spec Section 8).

Generators create simulated concept and covariate drift streams:
- Control: Identical copy of original data (zero drift).
- S1 Concept: Label rule change post-onset (amounts > 180 become fraud, scenario 1).
  Input distributions are completely untouched.
- S2 Mixed: New fraud pattern post-onset where 12 customers per day are compromised for 14 days.
  Transactions receive 3x amount, fraud=1, scenario=4. Post-onset fraud rate rises to ~4%.
- S3 Virtual Covariate: 50% of customers post-onset receive a +6 hour datetime shift.
  Labels are untouched. Timestamps and days are recomputed and re-sorted.

Each generator returns a NEW dataframe without mutating the input.
Features must be rebuilt after any drift injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .fd_config import (
    COL_AMOUNT,
    COL_CUSTOMER,
    COL_DATETIME,
    COL_DAY,
    COL_FRAUD,
    COL_ID,
    COL_SCENARIO,
    COL_SECONDS,
    SCENARIO_CONTROL,
    SCENARIO_S1,
    SCENARIO_S2,
    SCENARIO_S3,
    get_logger,
)

logger = get_logger(__name__)


def inject_control(
    raw_df: pd.DataFrame,
    onset_day: int = 105,
    seed: int = 1,
) -> pd.DataFrame:
    """Control stream: returns an identical copy of the raw data without any drift."""
    logger.debug("Generating Control stream (seed=%d, onset=%d)", seed, onset_day)
    return raw_df.copy()


def inject_s1_concept(
    raw_df: pd.DataFrame,
    onset_day: int = 105,
    seed: int = 1,
) -> pd.DataFrame:
    """S1 Concept drift: label rule change, inputs completely unchanged.

    Rows with day >= onset, TX_AMOUNT > 180 and TX_FRAUD == 0 become:
    TX_FRAUD = 1, TX_FRAUD_SCENARIO = 1.
    """
    logger.debug("Generating S1 Concept drift (seed=%d, onset=%d)", seed, onset_day)
    out = raw_df.copy()
    mask = (out[COL_DAY] >= onset_day) & (out[COL_AMOUNT] > 180.0) & (out[COL_FRAUD] == 0)
    out.loc[mask, COL_FRAUD] = 1
    out.loc[mask, COL_SCENARIO] = 1
    return out


def inject_s2_mixed(
    raw_df: pd.DataFrame,
    onset_day: int = 105,
    seed: int = 1,
) -> pd.DataFrame:
    """S2 Mixed drift: new fraud pattern where inputs shift for fraud transactions.

    Each day D >= onset, pick 12 customers (rng with seed) and mark them compromised
    for days [D, D+14). Their transactions in that period receive:
    TX_AMOUNT *= 3, TX_FRAUD = 1, TX_FRAUD_SCENARIO = 4.
    """
    logger.debug("Generating S2 Mixed drift (seed=%d, onset=%d)", seed, onset_day)
    out = raw_df.copy()
    max_day = int(out[COL_DAY].max())
    unique_customers = np.sort(out[COL_CUSTOMER].unique())
    rng = np.random.default_rng(seed)

    compromised_dict: Dict[int, set] = {}
    for D in range(onset_day, max_day + 1):
        chosen = rng.choice(unique_customers, size=12, replace=False)
        for c in chosen:
            if c not in compromised_dict:
                compromised_dict[c] = set()
            for d in range(D, min(D + 14, max_day + 1)):
                compromised_dict[c].add(d)

    cust_arr = out[COL_CUSTOMER].to_numpy()
    day_arr = out[COL_DAY].to_numpy()

    mask = np.zeros(len(out), dtype=bool)
    post_idx = np.where(day_arr >= onset_day)[0]
    for idx in post_idx:
        c = cust_arr[idx]
        d = day_arr[idx]
        if c in compromised_dict and d in compromised_dict[c]:
            mask[idx] = True

    out.loc[mask, COL_AMOUNT] = out.loc[mask, COL_AMOUNT] * 3.0
    out.loc[mask, COL_FRAUD] = 1
    out.loc[mask, COL_SCENARIO] = 4
    return out


def inject_s3_virtual_covariate(
    raw_df: pd.DataFrame,
    onset_day: int = 105,
    seed: int = 1,
) -> pd.DataFrame:
    """S3 Virtual Covariate drift: inputs shift, labels untouched.

    Pick 50% of customers (seed). For rows with day >= onset, shift TX_DATETIME
    by +6 hours. Recompute TX_TIME_DAYS and TX_TIME_SECONDS, then re-sort by TX_DATETIME.
    """
    logger.debug("Generating S3 Virtual Covariate drift (seed=%d, onset=%d)", seed, onset_day)
    out = raw_df.copy()
    unique_customers = np.sort(out[COL_CUSTOMER].unique())
    rng = np.random.default_rng(seed)
    n_pick = len(unique_customers) // 2
    picked = set(rng.choice(unique_customers, size=n_pick, replace=False))

    cust_arr = out[COL_CUSTOMER].to_numpy()
    day_arr = out[COL_DAY].to_numpy()

    mask = (day_arr >= onset_day) & np.isin(cust_arr, list(picked))
    out.loc[mask, COL_DATETIME] = out.loc[mask, COL_DATETIME] + pd.Timedelta(hours=6)

    # Recompute calendar offset relative to 2018-04-01
    base_timestamp = pd.Timestamp("2018-04-01")
    out[COL_DAY] = (out[COL_DATETIME] - base_timestamp).dt.days
    out[COL_SECONDS] = (out[COL_DATETIME] - base_timestamp).dt.total_seconds().astype(int)

    # Deterministic stable sort by new timestamp and transaction id
    out = out.sort_values([COL_DATETIME, COL_ID], kind="stable").reset_index(drop=True)
    return out


GENERATOR_MAP: Dict[str, Callable[[pd.DataFrame, int, int], pd.DataFrame]] = {
    SCENARIO_CONTROL: inject_control,
    SCENARIO_S1: inject_s1_concept,
    SCENARIO_S2: inject_s2_mixed,
    SCENARIO_S3: inject_s3_virtual_covariate,
}


def get_scenario_generator(
    scenario: str,
) -> Callable[[pd.DataFrame, int, int], pd.DataFrame]:
    """Retrieve generator function by scenario name (case-insensitive)."""
    norm = scenario.strip()
    if norm.lower() == SCENARIO_CONTROL.lower():
        return inject_control
    if norm.upper() == SCENARIO_S1.upper():
        return inject_s1_concept
    if norm.upper() == SCENARIO_S2.upper():
        return inject_s2_mixed
    if norm.upper() in (SCENARIO_S3.upper(), "S3_VIRTUAL"):
        return inject_s3_virtual_covariate
    raise ValueError(
        f"Unknown drift scenario '{scenario}'. Expected one of: "
        f"{', '.join(GENERATOR_MAP.keys())}"
    )


def inject_drift(
    raw_df: pd.DataFrame,
    scenario: str,
    onset_day: int = 105,
    seed: int = 1,
) -> pd.DataFrame:
    """Convenience dispatcher to inject drift by scenario name."""
    gen = get_scenario_generator(scenario)
    return gen(raw_df, onset_day, seed)


def run_drift_experiment(
    delay: int = 7,
    onset_day: int = 105,
    seed: int = 1,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Reproduce the verified experiment comparing Control, S1, S2, and S3 (Spec Section 15).

    Evaluates Recall@1%FPR, AP, prevalence, and relative degradation pre- vs post-onset.
    Saves results to outputs/fraud_detection/drift_reproduction/.
    """
    from .fd_config import OUTPUT_DIR
    from .fd_data_loader import load_handbook
    from .fd_evaluation import evaluate_weekly
    from .fd_feature_engineering import build_features
    from .fd_preprocessing import build_splits
    from .fd_training import predict_scores, train_model

    target_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "drift_reproduction"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw Handbook dataset...")
    raw = load_handbook()

    # Train base model once on uncorrupted training days 35..62
    logger.info("Building base features and training B0 model on days 35..62 (seed=%d)...", seed)
    ctrl_df = inject_control(raw, onset_day=onset_day, seed=seed)
    ctrl_feat = build_features(ctrl_df, delay=delay)
    sp = build_splits(ctrl_feat, delay)
    base_model = train_model(ctrl_feat[sp["train"]], seed=seed)

    summary_rows = []
    onset_week = onset_day // 7

    for sc in (SCENARIO_CONTROL, SCENARIO_S1, SCENARIO_S2, SCENARIO_S3):
        logger.info("Running drift experiment for scenario: %s", sc)
        injected = inject_drift(raw, sc, onset_day=onset_day, seed=seed)
        feat = build_features(injected, delay=delay)
        test = feat[sp["test"]].reset_index(drop=True)
        scores = predict_scores(base_model, test)
        res = evaluate_weekly(test, scores, sp["config"].test_start, sp["config"].test_end)
        res["scenario"] = sc
        res["delay"] = delay
        res["seed"] = seed

        # Save weekly CSV
        res.to_csv(target_dir / f"weekly_{sc.lower()}_d{delay}_seed{seed}.csv", index=False)

        pre_mask = res["week"] < onset_week
        post_mask = res["week"] >= onset_week
        pre_rec = float(res.loc[pre_mask, "recall_at_1fpr"].mean())
        post_rec = float(res.loc[post_mask, "recall_at_1fpr"].mean())
        rel_change = float((post_rec - pre_rec) / pre_rec * 100.0) if pre_rec > 0 else 0.0

        pre_ap = float(res.loc[pre_mask, "ap"].mean())
        post_ap = float(res.loc[post_mask, "ap"].mean())

        summary_rows.append({
            "scenario": sc,
            "delay": delay,
            "onset_day": onset_day,
            "seed": seed,
            "recall_pre_onset": pre_rec,
            "recall_post_onset": post_rec,
            "relative_change_pct": rel_change,
            "ap_pre_onset": pre_ap,
            "ap_post_onset": post_ap,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(target_dir / f"reproduction_summary_d{delay}_seed{seed}.csv", index=False)
    logger.info("Reproduction experiment complete. Results saved to %s", target_dir)
    return summary_df


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run drift injection experiment reproduction.")
    parser.add_argument("--delay", type=int, default=7, help="Label delay in days (default: 7)")
    parser.add_argument("--onset", type=int, default=105, help="Drift onset day (default: 105)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    args = parser.parse_args()

    df_res = run_drift_experiment(delay=args.delay, onset_day=args.onset, seed=args.seed)
    print("\n" + df_res.to_string(index=False))
