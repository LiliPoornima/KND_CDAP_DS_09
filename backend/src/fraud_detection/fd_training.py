"""Fraud-detection model training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .fd_config import (
    FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURES, MODEL_PARAMS, ModelParams, get_logger,
)

logger = get_logger(__name__)


def _check_features(features: Sequence[str]) -> list:
    bad = [f for f in features if f in FORBIDDEN_MODEL_COLUMNS]
    if bad:
        raise ValueError(f"Forbidden columns used as model features (leakage): {bad}")
    return list(features)


def train_model(
    df: pd.DataFrame,
    seed: int,
    features: Sequence[str] = MODEL_FEATURES,
    sample_weight: Optional[np.ndarray] = None,
    params: ModelParams = MODEL_PARAMS,
    label_col: str = "TX_FRAUD",
):
    """Train a LightGBM binary classifier on ``df`` (spec section 7)."""
    from lightgbm import LGBMClassifier

    feats = _check_features(features)
    model = LGBMClassifier(
        objective="binary",
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        n_estimators=params.n_estimators,
        subsample=params.subsample,
        subsample_freq=params.subsample_freq,
        colsample_bytree=params.colsample_bytree,
        scale_pos_weight=params.scale_pos_weight,
        random_state=seed,
        n_jobs=params.n_jobs,
        verbose=-1,
    )
    model.fit(df[feats], df[label_col], sample_weight=sample_weight)
    model.fd_features_ = feats
    logger.info("Trained LightGBM on %d rows (%d frauds), %d features",
                len(df), int(df[label_col].sum()), len(feats))
    return model


def predict_scores(model, df: pd.DataFrame) -> np.ndarray:
    """Fraud probability for each row of ``df``."""
    return model.predict_proba(df[model.fd_features_])[:, 1]


def save_model(model, path: Path) -> None:
    """Persist a trained model with joblib."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
