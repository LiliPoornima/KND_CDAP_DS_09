"""Fraud-detection model explainability via TreeSHAP (Spec Section 9).

Label-free attribution profiles computed per window:
- shap.TreeExplainer(model)
- Sample up to 10,000 rows of the window (fixed seed) from ALL traffic (labels unavailable).
- Handles both list and ndarray outputs from SHAP explainer.
- Profile = mean |SHAP| per feature, normalized to sum to 1.0.
- Shift = Jensen-Shannon distance to the trusted reference profile.
- Per-feature delta = profile_w - profile_ref.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import shap

from .fd_config import MODEL_FEATURES, get_logger

logger = get_logger(__name__)


def build_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Initialize a TreeExplainer for the trained tree model."""
    logger.debug("Initializing TreeExplainer for model %s", type(model).__name__)
    return shap.TreeExplainer(model)


def calculate_shap_profile(
    explainer: shap.TreeExplainer,
    df_window: pd.DataFrame,
    features: Sequence[str] = MODEL_FEATURES,
    max_samples: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Calculate the normalized mean |SHAP| attribution profile for a window.

    1. Subsamples up to ``max_samples`` rows using a fixed seed.
    2. Handles list and ndarray outputs from ``explainer.shap_values``.
    3. Computes mean |SHAP| per feature.
    4. Normalizes to sum to 1.0 (probability distribution over feature importances).
    """
    feat_list = list(features)
    missing = [f for f in feat_list if f not in df_window.columns]
    if missing:
        raise ValueError(f"Window DataFrame is missing required features: {missing}")

    if len(df_window) > max_samples:
        sample_df = df_window.sample(n=max_samples, random_state=seed)
    else:
        sample_df = df_window

    X = sample_df[feat_list]
    vals = explainer.shap_values(X)

    # Handle diverse SHAP output formats (binary classification)
    if isinstance(vals, list):
        # List of [class_0, class_1] -> use class_1 (positive fraud class)
        matrix = vals[1] if len(vals) > 1 else vals[0]
    elif isinstance(vals, np.ndarray) and vals.ndim == 3:
        # Array of shape (N, n_features, 2)
        matrix = vals[:, :, 1]
    else:
        matrix = np.asarray(vals)

    mean_abs = np.mean(np.abs(matrix), axis=0)
    total_abs = float(np.sum(mean_abs))

    if total_abs > 0.0:
        profile = mean_abs / total_abs
    else:
        # Uniform fallback if all attributions are zero
        profile = np.ones(len(feat_list), dtype=np.float64) / len(feat_list)

    return profile.astype(np.float64)


def calculate_shap_shift(
    win_profile: np.ndarray,
    ref_profile: np.ndarray,
) -> float:
    """Calculate the Jensen-Shannon distance between window and reference SHAP profiles."""
    p_win = np.asarray(win_profile, dtype=np.float64)
    p_ref = np.asarray(ref_profile, dtype=np.float64)
    return float(jensenshannon(p_win, p_ref))


def calculate_per_feature_delta(
    win_profile: np.ndarray,
    ref_profile: np.ndarray,
    features: Sequence[str] = MODEL_FEATURES,
) -> Dict[str, float]:
    """Calculate per-feature attribution delta: profile_w - profile_ref."""
    feat_list = list(features)
    deltas = np.asarray(win_profile) - np.asarray(ref_profile)
    return {f"delta_{f}": float(d) for f, d in zip(feat_list, deltas)}


def compute_explainability_signals(
    explainer: shap.TreeExplainer,
    win_df: pd.DataFrame,
    ref_profile: np.ndarray,
    features: Sequence[str] = MODEL_FEATURES,
    max_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute SHAP profile, shift against reference, and per-feature attribution deltas."""
    win_profile = calculate_shap_profile(
        explainer,
        win_df,
        features=features,
        max_samples=max_samples,
        seed=seed,
    )
    shap_shift = calculate_shap_shift(win_profile, ref_profile)
    deltas = calculate_per_feature_delta(win_profile, ref_profile, features=features)

    res: Dict[str, Any] = {
        "shap_shift": shap_shift,
        "shap_profile": win_profile,
    }
    res.update(deltas)
    return res
