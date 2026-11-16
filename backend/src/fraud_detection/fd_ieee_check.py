"""IEEE-CIS real-data degradation check (Spec Section 14).

Protocol:
1. Load train_transaction.csv from data/fraud_detection/raw/ieee_cis/.
2. day = TransactionDT // 86400; numeric columns only (fillna not needed for LightGBM).
3. Train model on days < 42.
4. Evaluate 14-day windows starting at day 49 using Recall@1%FPR.
5. Print table and report verdict:
   - Degradation present if recall falls >= 20% relative to the first window and stays lower.
   - If not present (or data missing), report honestly that IEEE-CIS cannot serve as real-degradation validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fd_config import DATA_DIR, OUTPUT_DIR, get_logger
from .fd_evaluation import recall_at_fpr

logger = get_logger(__name__)


def run_ieee_cis_check(
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    target_col: str = "isFraud",
    time_col: str = "TransactionDT",
    train_cutoff_day: int = 42,
    eval_start_day: int = 49,
    window_days: int = 14,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the IEEE-CIS degradation check per Spec Section 14."""
    import lightgbm as lgb

    path = Path(csv_path) if csv_path else DATA_DIR / "raw" / "ieee_cis" / "train_transaction.csv"
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "ieee_cis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        msg = (
            f"IEEE-CIS data file not found at: {path}\n"
            "Per Spec Section 3.2, train_transaction.csv must be manually downloaded from Kaggle.\n"
            "Verdict: IEEE-CIS data is absent; cannot serve as real-degradation validation.\n"
            "Handbook simulated drift streams remain the primary benchmark."
        )
        logger.warning(msg)
        return {
            "status": "DATA_MISSING",
            "file_path": str(path),
            "degradation_present": False,
            "verdict": "DATA_NOT_FOUND",
            "message": msg,
            "table": None,
        }

    logger.info("Loading IEEE-CIS transaction data from %s...", path)
    # Read numeric columns only
    # Sample or read full
    df = pd.read_csv(path)
    if target_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Required columns '{target_col}' or '{time_col}' missing from IEEE-CIS data")

    df["day"] = (df[time_col] // 86400).astype(int)

    # Select numeric features excluding target and raw identifiers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in (target_col, time_col, "day", "TransactionID")]

    logger.info("IEEE-CIS loaded: %d rows, %d numeric features", len(df), len(feature_cols))

    # Train on days < 42
    train_mask = df["day"] < train_cutoff_day
    X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    y_train = df.loc[train_mask, target_col].to_numpy(dtype=int)

    logger.info("Training initial LightGBM model on days < %d (%d rows, %d frauds)...",
                train_cutoff_day, len(X_train), int(y_train.sum()))

    pos_cnt = int(y_train.sum())
    neg_cnt = len(y_train) - pos_cnt
    spw = float(neg_cnt / max(pos_cnt, 1))

    clf = lgb.LGBMClassifier(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=100,
        scale_pos_weight=min(spw, 20.0),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate 14-day windows starting at eval_start_day
    max_day = int(df["day"].max())
    windows = []
    curr = eval_start_day
    while curr + window_days - 1 <= max_day:
        windows.append((curr, curr + window_days - 1))
        curr += window_days

    eval_records = []
    for w_idx, (w_start, w_end) in enumerate(windows):
        w_mask = (df["day"] >= w_start) & (df["day"] <= w_end)
        w_df = df[w_mask]
        if len(w_df) == 0:
            continue

        X_val = w_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = w_df[target_col].to_numpy(dtype=int)
        scores = clf.predict_proba(X_val)[:, 1]

        rec = recall_at_fpr(y_val, scores, fpr=0.01)
        frauds = int(y_val.sum())
        total = len(y_val)
        prev = float(frauds / max(total, 1))

        eval_records.append({
            "window_idx": w_idx,
            "day_start": w_start,
            "day_end": w_end,
            "total_transactions": total,
            "frauds": frauds,
            "prevalence": prev,
            "recall_at_1fpr": rec,
        })

    eval_df = pd.DataFrame(eval_records)
    if len(eval_df) == 0:
        return {
            "status": "NO_EVAL_WINDOWS",
            "degradation_present": False,
            "verdict": "INSUFFICIENT_DATA",
            "table": eval_df,
        }

    first_window_rec = eval_df["recall_at_1fpr"].iloc[0]
    eval_df["relative_to_first_window_pct"] = (
        (eval_df["recall_at_1fpr"] / first_window_rec * 100.0) if first_window_rec > 0 else 0.0
    )

    # Verdict: degradation present if recall falls >= 20% relative to the first window and stays lower
    drops_20 = eval_df["relative_to_first_window_pct"] <= 80.0
    first_drop_idx = drops_20.idxmax() if drops_20.any() else -1

    if first_drop_idx != -1:
        # Check if it stays lower
        subsequent = eval_df.loc[first_drop_idx:, "relative_to_first_window_pct"]
        stays_lower = bool((subsequent <= 85.0).all())
    else:
        stays_lower = False

    degradation_present = bool(first_drop_idx != -1 and stays_lower)

    if degradation_present:
        verdict = "DEGRADATION_PRESENT"
        summary_msg = "Degradation observed: Recall@1%FPR dropped >= 20% and remained suppressed."
    else:
        verdict = "NO_PERSISTENT_DEGRADATION"
        summary_msg = (
            "No persistent degradation observed: recall does not fall and stay >= 20% below initial window. "
            "Per Spec Section 14, IEEE-CIS cannot serve as real-degradation validation."
        )

    eval_df.to_csv(out_dir / "ieee_cis_degradation_check.csv", index=False)
    logger.info("IEEE-CIS check completed: %s", summary_msg)

    return {
        "status": "COMPLETED",
        "file_path": str(path),
        "degradation_present": degradation_present,
        "verdict": verdict,
        "message": summary_msg,
        "table": eval_df,
    }


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run IEEE-CIS real-data degradation check (Spec Section 14)")
    parser.add_argument("--path", type=str, default=None, help="Path to train_transaction.csv")
    args = parser.parse_args()

    res = run_ieee_cis_check(csv_path=Path(args.path) if args.path else None)
    print("\n=== IEEE-CIS Degradation Check Verdict ===")
    print(f"Status: {res['status']}")
    print(f"Verdict: {res['verdict']}")
    print(f"Details: {res['message']}")
    if res.get("table") is not None:
        print("\nEvaluation Windows:")
        print(res["table"].to_string(index=False))
