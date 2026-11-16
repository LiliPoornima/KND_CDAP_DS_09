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
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.src.credit_risk.common.cr_calibration import (
    cr_calibration_summary,
)


DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def cr_bootstrap_binary_performance(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate test-metric percentile intervals with a stratified bootstrap.

    Rows are resampled with replacement within each observed target class, so
    every replicate contains both classes and preserves the test set's class
    counts. Intervals condition on the fitted model and frozen threshold; they
    do not include model-refit or threshold-selection uncertainty.
    """
    truth, probability = _validate_binary_probabilities(y_true, y_probability)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError("threshold must be strictly between 0 and 1.")
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not isinstance(confidence_level, (int, float)) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError("random_state must be an integer.")

    def calculate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        predicted = (p >= threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1_score": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc_average_precision": float(average_precision_score(y, p)),
            "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }

    point = calculate(truth, probability)
    class_indices = [np.flatnonzero(truth == label) for label in (0, 1)]
    rng = np.random.default_rng(random_state)
    bootstrap_values = {name: [] for name in point}
    for _ in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        replicate = calculate(truth[sampled], probability[sampled])
        for name, value in replicate.items():
            bootstrap_values[name].append(value)

    alpha = (1.0 - float(confidence_level)) / 2.0
    return pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": estimate,
                "confidence_level": float(confidence_level),
                "lower": float(np.quantile(bootstrap_values[name], alpha)),
                "upper": float(np.quantile(bootstrap_values[name], 1.0 - alpha)),
                "valid_replicates": n_bootstrap,
                "n": int(len(truth)),
                "positive_support": int((truth == 1).sum()),
                "negative_support": int((truth == 0).sum()),
                "threshold": float(threshold),
                "bootstrap_method": "stratified_by_target_class",
            }
            for name, estimate in point.items()
        ]
    )


def cr_bootstrap_paired_binary_performance(
    y_true: Iterable[Any],
    probability_a: Iterable[Any],
    probability_b: Iterable[Any],
    *,
    threshold_a: float,
    threshold_b: float,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate paired test-metric differences with a shared stratified bootstrap.

    Each replicate resamples the same test-row indexes for both models within
    each observed target class. The reported difference is model B minus model
    A, with both validation-selected thresholds held fixed.
    """

    truth, probability_left = _validate_binary_probabilities(y_true, probability_a)
    truth_right, probability_right = _validate_binary_probabilities(
        truth, probability_b
    )
    if not np.array_equal(truth, truth_right):
        raise ValueError("Both model probabilities must correspond to the same y_true rows.")
    for label, threshold in (("threshold_a", threshold_a), ("threshold_b", threshold_b)):
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 < threshold < 1.0
        ):
            raise ValueError(f"{label} must be strictly between 0 and 1.")
    if (
        not isinstance(model_a_name, str)
        or not model_a_name.strip()
        or not isinstance(model_b_name, str)
        or not model_b_name.strip()
        or model_a_name == model_b_name
    ):
        raise ValueError("Model names must be distinct, non-empty strings.")
    if (
        not isinstance(n_bootstrap, int)
        or isinstance(n_bootstrap, bool)
        or n_bootstrap < 1
    ):
        raise ValueError("n_bootstrap must be a positive integer.")
    if (
        not isinstance(confidence_level, (int, float))
        or isinstance(confidence_level, bool)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError("random_state must be an integer.")

    def calculate(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
        predicted = (probability >= threshold).astype(int)
        calibration = cr_calibration_summary(y, probability)
        return {
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1_score": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc_average_precision": float(average_precision_score(y, probability)),
            "brier_score": float(calibration["brier_score"]),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "expected_calibration_error": float(
                calibration["expected_calibration_error"]
            ),
        }

    metrics_left = calculate(truth, probability_left, float(threshold_a))
    metrics_right = calculate(truth, probability_right, float(threshold_b))
    point_differences = {
        metric: metrics_right[metric] - metrics_left[metric]
        for metric in metrics_left
    }
    class_indices = [np.flatnonzero(truth == label) for label in (0, 1)]
    rng = np.random.default_rng(random_state)
    bootstrap_differences = {metric: [] for metric in point_differences}
    for _ in range(n_bootstrap):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        y_sampled = truth[sampled]
        sampled_left = calculate(
            y_sampled, probability_left[sampled], float(threshold_a)
        )
        sampled_right = calculate(
            y_sampled, probability_right[sampled], float(threshold_b)
        )
        for metric in point_differences:
            bootstrap_differences[metric].append(
                sampled_right[metric] - sampled_left[metric]
            )

    alpha = (1.0 - float(confidence_level)) / 2.0
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "model_a": model_a_name,
                "model_b": model_b_name,
                "estimate_a": metrics_left[metric],
                "estimate_b": metrics_right[metric],
                "difference_b_minus_a": point_differences[metric],
                "confidence_level": float(confidence_level),
                "lower": float(np.quantile(bootstrap_differences[metric], alpha)),
                "upper": float(
                    np.quantile(bootstrap_differences[metric], 1.0 - alpha)
                ),
                "valid_replicates": n_bootstrap,
                "n": int(len(truth)),
                "positive_support": int((truth == 1).sum()),
                "negative_support": int((truth == 0).sum()),
                "threshold_a": float(threshold_a),
                "threshold_b": float(threshold_b),
                "bootstrap_method": "paired_stratified_by_target_class",
            }
            for metric in point_differences
        ]
    )


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
