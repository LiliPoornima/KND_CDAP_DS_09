"""Fraud-detection model evaluation: metrics and the B0 (never retrain) baseline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .fd_config import (
    COL_DAY, COL_FRAUD, MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR, TARGET_FPR, get_logger,
)
from .fd_preprocessing import make_windows, window_mask

logger = get_logger(__name__)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float = TARGET_FPR) -> float:
    """Recall at a fixed false-positive rate.

    thr = quantile(scores[y==0], 1 - fpr); recall = mean(scores[y==1] > thr).
    Returns NaN when a class is missing.
    """
    y = np.asarray(y_true)
    p = np.asarray(scores)
    neg, pos = p[y == 0], p[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(pos > thr))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (NaN if no positives)."""
    from sklearn.metrics import average_precision_score

    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def window_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Recall@1%FPR, AP, prevalence and AP/prevalence for one window.

    AP alone must never be used to judge degradation: it rises with the fraud rate.
    """
    prev = float(np.mean(y_true))
    ap = average_precision(y_true, scores)
    return {
        "n_tx": int(len(y_true)),
        "n_fraud": int(np.sum(y_true)),
        "prevalence": prev,
        "recall_at_1fpr": recall_at_fpr(y_true, scores),
        "ap": ap,
        "ap_over_prevalence": ap / prev if prev > 0 else float("nan"),
    }


def evaluate_weekly(
    df: pd.DataFrame, scores: np.ndarray, start: int, end: int, size: int = 7
) -> pd.DataFrame:
    """Per-window metrics for ``df`` (scores aligned positionally with ``df``)."""
    rows: List[dict] = []
    for win in make_windows(start, end, size):
        m = window_mask(df, win).to_numpy()
        rec = window_metrics(df[COL_FRAUD].to_numpy()[m], scores[m])
        rec.update({"week": win[0] // 7, "day_start": win[0], "day_end": win[1]})
        rows.append(rec)
    cols = ["week", "day_start", "day_end"]
    out = pd.DataFrame(rows)
    return out[cols + [c for c in out.columns if c not in cols]]


def run_b0(
    delay: int = 7,
    seed: int = 1,
    df: Optional[pd.DataFrame] = None,
    output_dir: Path = OUTPUT_DIR / "b0",
    use_feature_cache: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """B0: train once on days 35..62 (default, no-drift data), evaluate weekly, never retrain."""
    from .fd_data_loader import load_handbook
    from .fd_feature_engineering import build_features
    from .fd_preprocessing import build_splits
    from .fd_training import predict_scores, save_model, train_model

    from .fd_feature_engineering import load_or_build_features

    if df is None:
        raw = load_handbook()
        df = (load_or_build_features(raw, delay) if use_feature_cache
              else build_features(raw, delay))

    sp = build_splits(df, delay)
    cfg = sp["config"]
    model = train_model(df[sp["train"]], seed=seed)
    test = df[sp["test"]].reset_index(drop=True)
    scores = predict_scores(model, test)
    res = evaluate_weekly(test, scores, cfg.test_start, cfg.test_end)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(output_dir / f"b0_weekly_d{delay}_seed{seed}.csv", index=False)
        save_model(model, MODEL_DIR / f"b0_d{delay}_seed{seed}.joblib")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = run_b0(delay=7, seed=1)
    logger.info("\n%s", r.round(4).to_string(index=False))
    logger.info("recall@1%%FPR mean=%.4f std=%.4f min=%.4f max=%.4f | AP mean=%.4f",
                r.recall_at_1fpr.mean(), r.recall_at_1fpr.std(),
                r.recall_at_1fpr.min(), r.recall_at_1fpr.max(), r.ap.mean())
