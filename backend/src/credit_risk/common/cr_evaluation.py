"""Reusable evaluation helpers for binary credit-risk models."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.src.credit_risk.common.cr_calibration import (
    cr_calibration_summary,
)


DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def _validate_binary_probabilities(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        truth_values = np.asarray(list(y_true), dtype=float)
        probability = np.asarray(list(y_probability), dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Targets and probabilities must be numeric iterables.") from exc
    if truth_values.ndim != 1 or probability.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional.")
    if truth_values.size == 0 or truth_values.size != probability.size:
        raise ValueError("Targets and probabilities must have equal non-zero length.")
    if not np.isfinite(truth_values).all() or set(np.unique(truth_values)) != {0.0, 1.0}:
        raise ValueError("y_true must contain both binary classes: 0 and 1.")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return truth_values.astype(int), probability


def cr_analyze_probability_thresholds(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return precision, recall, and F1 over validation probabilities."""

    truth, probability = _validate_binary_probabilities(y_true, y_probability)
    candidates = list(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if not candidates:
        raise ValueError("At least one threshold must be supplied.")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 < value < 1.0
        for value in candidates
    ):
        raise ValueError("Every threshold must be strictly between 0 and 1.")

    rows = []
    for threshold in candidates:
        predicted = (probability >= threshold).astype(int)
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": precision_score(truth, predicted, zero_division=0),
                "Recall": recall_score(truth, predicted, zero_division=0),
                "F1-score": f1_score(truth, predicted, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def cr_evaluate_binary_probabilities(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    *,
    threshold: float = 0.5,
    target_names: tuple[str, str] = ("Class 0", "Class 1"),
) -> dict[str, Any]:
    """Evaluate binary probabilities at a fixed threshold.

    Threshold-independent ranking and calibration metrics are reported
    alongside threshold-dependent outcomes and a confusion matrix.
    """

    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError("threshold must be strictly between 0 and 1.")
    if (
        not isinstance(target_names, tuple)
        or len(target_names) != 2
        or not all(isinstance(name, str) and name for name in target_names)
    ):
        raise ValueError("target_names must contain two non-empty strings.")

    truth, probability = _validate_binary_probabilities(y_true, y_probability)
    predicted = (probability >= threshold).astype(int)
    calibration = cr_calibration_summary(truth, probability)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1_score": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc_average_precision": float(
            average_precision_score(truth, probability)
        ),
        "brier_score": calibration["brier_score"],
        "log_loss": calibration["log_loss"],
        "expected_calibration_error": calibration[
            "expected_calibration_error"
        ],
        "confusion_matrix": confusion_matrix(
            truth,
            predicted,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            truth,
            predicted,
            labels=[0, 1],
            target_names=list(target_names),
            zero_division=0,
            output_dict=True,
        ),
        "predictions": predicted.tolist(),
        "calibration_bins": calibration["calibration_bins"].to_dict(
            orient="records"
        ),
    }
