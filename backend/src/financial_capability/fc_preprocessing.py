"""Preprocessing pipeline and train/test split.

All *fitted* steps (imputers, encoders) live inside a scikit-learn Pipeline so
they are fitted on training data only (inside every CV fold as well).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .fc_config import (
    CLASS_NAMES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TEST_SIZE,
)


def build_preprocessor(features: Optional[Sequence[str]] = None) -> ColumnTransformer:
    """Median-impute ordinal/binary answers; mode-impute + one-hot encode nominal ones.

    Why: ordinal Likert answers keep their order as numbers; nominal answers
    (employment, housing, marital status) have no order so they are one-hot
    encoded. ``handle_unknown='ignore'`` keeps prediction safe for unseen codes.
    ``features`` optionally restricts the inputs (used for the ablation study).
    """
    chosen = set(features) if features is not None else None
    numeric_cols = [f for f in NUMERIC_FEATURES if chosen is None or f in chosen]
    nominal_cols = [f for f in NOMINAL_FEATURES if chosen is None or f in chosen]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric, numeric_cols))
    if nominal_cols:
        transformers.append(("cat", nominal, nominal_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with a fixed seed (test set stays untouched until the end)."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def class_distribution(y: pd.Series) -> Dict[str, Dict[str, float]]:
    """Counts and percentages per class name."""
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    return {
        CLASS_NAMES[int(k)]: {"count": int(v), "percent": round(100.0 * v / total, 2)}
        for k, v in counts.items()
    }
