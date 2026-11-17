"""Training: Random Forest and XGBoost on the same features and split.

Hyper-parameters are chosen by stratified cross-validation on the TRAINING set
only (macro-F1). The test set is only used once, for the final comparison.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .fc_config import CV_FOLDS, RANDOM_SEED, SEARCH_CONFIG
from .fc_preprocessing import build_preprocessor


def _make_rf(seed: int, **params: Any) -> RandomForestClassifier:
    # class_weight="balanced" because the Medium class is smaller than Low/High.
    return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight="balanced", **params)


def _make_xgb(seed: int, **params: Any) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
        random_state=seed, n_jobs=-1, **params,
    )


def build_pipeline(estimator, features: Optional[Sequence[str]] = None) -> Pipeline:
    """Preprocessor + model in one object (so saving it saves the preprocessing too)."""
    return Pipeline([("prep", build_preprocessor(features)), ("model", estimator)])


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit a pipeline; XGBoost gets balanced sample weights (RF uses class_weight)."""
    if isinstance(pipeline.named_steps["model"], XGBClassifier):
        weights = compute_sample_weight("balanced", y)
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None, seed: int = RANDOM_SEED,
) -> Pipeline:
    """Train a Random Forest pipeline with the given (or default) parameters."""
    params = params or {"n_estimators": 300}
    return fit_pipeline(build_pipeline(_make_rf(seed, **params)), X_train, y_train)


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None, seed: int = RANDOM_SEED,
) -> Pipeline:
    """Train an XGBoost pipeline with the given (or default) parameters."""
    params = params or {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}
    return fit_pipeline(build_pipeline(_make_xgb(seed, **params)), X_train, y_train)


def _param_combinations(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def cross_validate_grid(
    model_name: str,
    factory: Callable[..., Any],
    grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Stratified-CV grid search on training data; returns one row per parameter set.

    The whole pipeline (imputer, encoder, model) is re-fitted inside each fold,
    so the validation fold never influences preprocessing.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for params in _param_combinations(grid):
        f1s, accs = [], []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            pipe = fit_pipeline(
                build_pipeline(factory(seed, **params)),
                X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            )
            pred = pipe.predict(X_train.iloc[va_idx])
            f1s.append(f1_score(y_train.iloc[va_idx], pred, average="macro"))
            accs.append(accuracy_score(y_train.iloc[va_idx], pred))
        rows.append({
            "model": model_name,
            "params": params,
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s)),
            "cv_accuracy_mean": float(np.mean(accs)),
        })
    return (pd.DataFrame(rows)
            .sort_values("cv_macro_f1_mean", ascending=False)
            .reset_index(drop=True))


def tune_random_forest(X_train, y_train, folds: int = CV_FOLDS, seed: int = RANDOM_SEED,
                       grid: Optional[Dict[str, list]] = None) -> pd.DataFrame:
    """CV grid search for Random Forest."""
    return cross_validate_grid("Random Forest", _make_rf, grid or SEARCH_CONFIG.rf_grid,
                               X_train, y_train, folds, seed)


def tune_xgboost(X_train, y_train, folds: int = CV_FOLDS, seed: int = RANDOM_SEED,
                 grid: Optional[Dict[str, list]] = None) -> pd.DataFrame:
    """CV grid search for XGBoost."""
    return cross_validate_grid("XGBoost", _make_xgb, grid or SEARCH_CONFIG.xgb_grid,
                               X_train, y_train, folds, seed)


def select_best_model(rf_cv: pd.DataFrame, xgb_cv: pd.DataFrame) -> str:
    """Pick the model family with the higher best CV macro-F1 (never the test set)."""
    return ("Random Forest"
            if rf_cv.iloc[0]["cv_macro_f1_mean"] >= xgb_cv.iloc[0]["cv_macro_f1_mean"]
            else "XGBoost")
