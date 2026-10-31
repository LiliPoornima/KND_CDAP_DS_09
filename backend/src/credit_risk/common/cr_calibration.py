"""Calibration metrics and reliability-bin summaries."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def cr_calibration_summary(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Calculate Brier score, log loss, ECE, and reliability-bin values.

    This function evaluates supplied probabilities; it does not fit or
    calibrate a model. Callers are responsible for passing predictions from
    a validation or held-out test partition rather than training data.
    ``strategy`` may be ``"uniform"`` or ``"quantile"``.
    """

    if (
        not isinstance(n_bins, int)
        or isinstance(n_bins, bool)
        or n_bins < 2
    ):
        raise ValueError("n_bins must be an integer of at least 2.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    try:
        truth = np.asarray(list(y_true), dtype=float)
        probability = np.asarray(list(y_probability), dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Targets and probabilities must be numeric iterables.") from exc

    if truth.ndim != 1 or probability.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional.")
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("Targets and probabilities must have equal non-zero length.")
    if not np.isfinite(truth).all() or not np.isfinite(probability).all():
        raise ValueError("Targets and probabilities must be finite.")
    if not set(np.unique(truth)).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only binary labels: 0 and 1.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if strategy == "uniform":
        bin_ids = pd.cut(
            probability,
            bins=np.linspace(0.0, 1.0, n_bins + 1),
            labels=False,
            include_lowest=True,
        )
        expected_bins = list(range(n_bins))
    elif np.unique(probability).size < 2:
        bin_ids = np.zeros(probability.size, dtype=int)
        expected_bins = [0]
    else:
        bin_ids = pd.qcut(
            probability,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        expected_bins = sorted(int(value) for value in pd.Series(bin_ids).dropna().unique())

    frame = pd.DataFrame(
        {
            "bin": np.asarray(bin_ids, dtype=float),
            "y_true": truth,
            "y_probability": probability,
        }
    )
    binned = (
        frame.groupby("bin", observed=True)
        .agg(
            count=("y_true", "size"),
            mean_predicted_probability=("y_probability", "mean"),
            observed_rate=("y_true", "mean"),
        )
        .reindex(expected_bins)
    )
    binned.index.name = "bin"
    binned["count"] = binned["count"].fillna(0).astype(int)
    binned["absolute_calibration_gap"] = (
        binned["mean_predicted_probability"] - binned["observed_rate"]
    ).abs()
    occupied = binned["count"] > 0
    ece = float(
        (
            binned.loc[occupied, "count"]
            / truth.size
            * binned.loc[occupied, "absolute_calibration_gap"]
        ).sum()
    )
    clipped = np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    return {
        "brier_score": float(np.mean((probability - truth) ** 2)),
        "log_loss": float(
            -np.mean(
                truth * np.log(clipped)
                + (1.0 - truth) * np.log1p(-clipped)
            )
        ),
        "expected_calibration_error": ece,
        "calibration_bins": binned.reset_index(),
        "n_bins": int(n_bins),
        "binning_strategy": strategy,
    }


def cr_group_calibration_summary(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    groups: Iterable[Any],
    *,
    attribute_name: str = "group",
    min_group_size: int = 30,
    min_class_count: int = 5,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Calculate calibration metrics and reliability bins within audit groups.

    Metrics are retained for every observed group. The returned support flags
    identify groups meeting the project reporting rule (minimum group size and
    minimum positive and negative outcome counts); they do not imply that
    smaller groups are calibrated or that larger groups are statistically
    precise.
    """

    if not isinstance(attribute_name, str) or not attribute_name.strip():
        raise ValueError("attribute_name must be a non-empty string.")
    if (
        not isinstance(min_group_size, int)
        or isinstance(min_group_size, bool)
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer.")
    if (
        not isinstance(min_class_count, int)
        or isinstance(min_class_count, bool)
        or min_class_count < 1
    ):
        raise ValueError("min_class_count must be a positive integer.")

    try:
        truth = pd.Series(list(y_true), name="y_true")
        probability = pd.Series(list(y_probability), name="y_probability")
        group_values = pd.Series(list(groups), name="group")
    except TypeError as exc:
        raise TypeError("Inputs must be iterable one-dimensional values.") from exc
    if truth.empty or not (len(truth) == len(probability) == len(group_values)):
        raise ValueError("Outcome, probability, and group lengths must match and be non-zero.")

    frame = pd.DataFrame(
        {
            "y_true": truth,
            "y_probability": probability,
            "group": group_values.astype("string").fillna("[missing]"),
        }
    )
    summary_rows = []
    bin_tables = []
    for group, subset in frame.groupby("group", sort=True):
        calibration = cr_calibration_summary(
            subset["y_true"],
            subset["y_probability"],
            n_bins=n_bins,
            strategy="uniform",
        )
        positives = int(subset["y_true"].sum())
        negatives = int(len(subset) - positives)
        sample_sufficient = len(subset) >= min_group_size
        positive_sufficient = positives >= min_class_count
        negative_sufficient = negatives >= min_class_count
        summary_rows.append(
            {
                "attribute": attribute_name,
                "group": str(group),
                "n": int(len(subset)),
                "positive_support": positives,
                "negative_support": negatives,
                "outcome_rate": float(subset["y_true"].mean()),
                "mean_predicted_probability": float(
                    subset["y_probability"].mean()
                ),
                "calibration_in_the_large": float(
                    subset["y_probability"].mean() - subset["y_true"].mean()
                ),
                "brier_score": calibration["brier_score"],
                "log_loss": calibration["log_loss"],
                "expected_calibration_error": calibration[
                    "expected_calibration_error"
                ],
                "sample_sufficient": sample_sufficient,
                "positive_support_sufficient": positive_sufficient,
                "negative_support_sufficient": negative_sufficient,
                "calibration_reportable": (
                    sample_sufficient and positive_sufficient and negative_sufficient
                ),
            }
        )
        bins = calibration["calibration_bins"].copy()
        bins.insert(0, "group", str(group))
        bins.insert(0, "attribute", attribute_name)
        bin_tables.append(bins)

    return {
        "group_metrics": pd.DataFrame(summary_rows),
        "calibration_bins": pd.concat(bin_tables, ignore_index=True),
    }


def cr_bootstrap_group_calibration(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    groups: Iterable[Any],
    *,
    attribute_name: str = "group",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    min_group_size: int = 30,
    min_class_count: int = 5,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Estimate within-group calibration intervals by stratified bootstrap.

    Rows are resampled separately within each audit group and observed target
    class. Intervals therefore condition on each group's observed class counts,
    the fitted model, and its supplied probabilities. Groups below the project
    support rules remain in the output with missing bounds and zero valid
    replicates. ECE uses fixed, equal-width probability bins.
    """

    if not isinstance(attribute_name, str) or not attribute_name.strip():
        raise ValueError("attribute_name must be a non-empty string.")
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if (
        not isinstance(confidence_level, (int, float))
        or isinstance(confidence_level, bool)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError("random_state must be an integer.")
    if (
        not isinstance(min_group_size, int)
        or isinstance(min_group_size, bool)
        or min_group_size < 1
        or not isinstance(min_class_count, int)
        or isinstance(min_class_count, bool)
        or min_class_count < 1
    ):
        raise ValueError("Group size and class support thresholds must be positive integers.")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2.")

    try:
        truth = np.asarray(list(y_true), dtype=float)
        probability = np.asarray(list(y_probability), dtype=float)
        group_values = pd.Series(list(groups), dtype="string").fillna("[missing]")
    except (TypeError, ValueError) as exc:
        raise TypeError("Targets, probabilities, and groups must be numeric or iterable values.") from exc
    if truth.ndim != 1 or probability.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional.")
    if truth.size == 0 or not (truth.size == probability.size == len(group_values)):
        raise ValueError("Targets, probabilities, and groups must have equal non-zero length.")
    if not np.isfinite(truth).all() or not set(np.unique(truth)).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only finite binary labels: 0 and 1.")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1.")

    def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        clipped = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        bin_ids = np.searchsorted(
            np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
            p,
            side="left",
        )
        counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
        probability_sums = np.bincount(bin_ids, weights=p, minlength=n_bins)
        outcome_sums = np.bincount(bin_ids, weights=y, minlength=n_bins)
        occupied = counts > 0
        ece = np.sum(
            counts[occupied]
            / len(y)
            * np.abs(
                probability_sums[occupied] / counts[occupied]
                - outcome_sums[occupied] / counts[occupied]
            )
        )
        return {
            "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(
                -np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
            ),
            "expected_calibration_error": float(ece),
            "calibration_in_the_large": float(np.mean(p) - np.mean(y)),
        }

    rng = np.random.default_rng(random_state)
    alpha = (1.0 - float(confidence_level)) / 2.0
    rows = []
    group_array = group_values.to_numpy(dtype=str)
    for group in sorted(np.unique(group_array)):
        positions = np.flatnonzero(group_array == group)
        y_group = truth[positions].astype(int)
        p_group = probability[positions]
        negative_positions = positions[y_group == 0]
        positive_positions = positions[y_group == 1]
        n = len(positions)
        positives = len(positive_positions)
        negatives = len(negative_positions)
        reportable = (
            n >= min_group_size
            and positives >= min_class_count
            and negatives >= min_class_count
        )
        point_estimates = _metrics(y_group, p_group)
        distributions = {metric: [] for metric in point_estimates}

        if reportable:
            for _ in range(n_bootstrap):
                sampled_positions = np.concatenate(
                    [
                        rng.choice(negative_positions, size=negatives, replace=True),
                        rng.choice(positive_positions, size=positives, replace=True),
                    ]
                )
                sampled = _metrics(
                    truth[sampled_positions].astype(int),
                    probability[sampled_positions],
                )
                for metric, value in sampled.items():
                    distributions[metric].append(value)

        for metric, estimate in point_estimates.items():
            values = distributions[metric]
            rows.append(
                {
                    "attribute": attribute_name,
                    "group": group,
                    "metric": metric,
                    "estimate": estimate,
                    "confidence_level": float(confidence_level),
                    "lower": float(np.quantile(values, alpha)) if values else np.nan,
                    "upper": float(np.quantile(values, 1.0 - alpha)) if values else np.nan,
                    "valid_replicates": len(values),
                    "n": n,
                    "positive_support": positives,
                    "negative_support": negatives,
                    "calibration_reportable": reportable,
                    "bootstrap_method": "within_group_stratified_by_target_class",
                    "n_bins": n_bins,
                    "random_state": random_state,
                }
            )
    return pd.DataFrame(rows)
