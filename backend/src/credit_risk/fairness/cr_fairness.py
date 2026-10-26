"""Group-level performance and error-rate disparity analysis."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def cr_bin_age_groups(
    ages: Iterable[Any],
) -> pd.Series:
    """Map ages to fixed ten-year bands for subgroup reporting.

    Values below 18 or above 120 are labeled ``[invalid]`` rather than
    silently merged with an adult group. Missing values remain visible.
    """

    try:
        age_values = pd.to_numeric(pd.Series(list(ages)), errors="coerce")
    except TypeError as exc:
        raise TypeError("ages must be an iterable of numeric values.") from exc
    valid = age_values.where(age_values.between(18, 120, inclusive="both"))
    bands = pd.cut(
        valid,
        bins=[18, 25, 35, 45, 55, 65, 121],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=False,
        include_lowest=True,
    ).astype("string")
    bands = bands.fillna("[missing]")
    bands.loc[age_values.notna() & ~age_values.between(18, 120, inclusive="both")] = (
        "[invalid]"
    )
    return bands.rename("age_group")


def cr_analyze_group_fairness(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    sensitive_groups: Iterable[Any],
    *,
    attribute_name: str = "group",
    min_group_size: int = 30,
    min_class_count: int = 5,
) -> dict[str, Any]:
    """Summarize binary outcome and prediction rates by an audit attribute.

    The sensitive attribute is used only to group the supplied outcomes and
    predictions. It is not used to fit or alter a model. Small groups remain
    visible in ``group_metrics`` but are excluded from disparity gaps.
    TPR/FPR gaps also require at least ``min_class_count`` observations of
    the corresponding true class in each group used for that gap.

    Returns a mapping with a per-group DataFrame and max-minus-min gaps for
    selection rate, true-positive rate, false-positive rate, and equalized
    odds. Rates requiring a missing class are reported as NaN.
    """

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
    if not isinstance(attribute_name, str) or not attribute_name.strip():
        raise ValueError("attribute_name must be a non-empty string.")

    try:
        truth = pd.Series(list(y_true), name="y_true")
        predictions = pd.Series(list(y_pred), name="y_pred")
        groups = pd.Series(list(sensitive_groups), name="group")
    except TypeError as exc:
        raise TypeError("Inputs must be iterable one-dimensional values.") from exc

    if truth.empty:
        raise ValueError("Inputs must contain at least one observation.")
    if not (len(truth) == len(predictions) == len(groups)):
        raise ValueError("Outcome, prediction, and group lengths must match.")
    if truth.isna().any() or predictions.isna().any():
        raise ValueError("Outcomes and predictions must not contain missing values.")
    if set(truth.unique()) != {0, 1}:
        raise ValueError("y_true must contain both binary classes: 0 and 1.")
    if not set(predictions.unique()).issubset({0, 1}):
        raise ValueError("y_pred must contain only binary labels: 0 and 1.")

    frame = pd.DataFrame(
        {
            "y_true": truth.astype(int),
            "y_pred": predictions.astype(int),
            "group": groups.astype("string").fillna("[missing]"),
        }
    )
    rows = []
    for group_value, subset in frame.groupby("group", sort=True):
        actual = subset["y_true"]
        predicted = subset["y_pred"]
        positives = int((actual == 1).sum())
        negatives = int((actual == 0).sum())
        true_positives = int(((actual == 1) & (predicted == 1)).sum())
        false_positives = int(((actual == 0) & (predicted == 1)).sum())
        false_negatives = int(((actual == 1) & (predicted == 0)).sum())
        true_negatives = int(((actual == 0) & (predicted == 0)).sum())
        n = len(subset)

        rows.append(
            {
                "attribute": attribute_name,
                "group": str(group_value),
                "n": int(n),
                "positive_support": positives,
                "negative_support": negatives,
                "outcome_rate": positives / n,
                "selection_rate": float(predicted.mean()),
                "true_positive_rate": (
                    true_positives / positives if positives else np.nan
                ),
                "false_positive_rate": (
                    false_positives / negatives if negatives else np.nan
                ),
                "precision": (
                    true_positives / (true_positives + false_positives)
                    if true_positives + false_positives
                    else np.nan
                ),
                "sample_sufficient": bool(n >= min_group_size),
                "positive_support_sufficient": bool(
                    positives >= min_class_count
                ),
                "negative_support_sufficient": bool(
                    negatives >= min_class_count
                ),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
            }
        )

    group_metrics = pd.DataFrame(rows)
    sufficient = group_metrics[group_metrics["sample_sufficient"]]
    tpr_sufficient = group_metrics[
        group_metrics["sample_sufficient"]
        & group_metrics["positive_support_sufficient"]
    ]
    fpr_sufficient = group_metrics[
        group_metrics["sample_sufficient"]
        & group_metrics["negative_support_sufficient"]
    ]

    def gap(frame: pd.DataFrame, column: str) -> float | None:
        values = frame[column].dropna()
        if len(values) < 2:
            return None
        return float(values.max() - values.min())

    selection_gap = gap(sufficient, "selection_rate")
    tpr_gap = gap(tpr_sufficient, "true_positive_rate")
    fpr_gap = gap(fpr_sufficient, "false_positive_rate")
    odds_components = [value for value in (tpr_gap, fpr_gap) if value is not None]

    disparities = {
        "min_group_size": min_group_size,
        "min_class_count": min_class_count,
        "selection_rate_groups": int(len(sufficient)),
        "true_positive_rate_groups": int(len(tpr_sufficient)),
        "false_positive_rate_groups": int(len(fpr_sufficient)),
        "selection_rate_difference": selection_gap,
        "true_positive_rate_difference": tpr_gap,
        "false_positive_rate_difference": fpr_gap,
        "equalized_odds_difference": (
            max(odds_components) if len(odds_components) == 2 else None
        ),
    }

    return {"group_metrics": group_metrics, "disparities": disparities}


def cr_bootstrap_group_fairness(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    sensitive_groups: Iterable[Any],
    *,
    attribute_name: str = "group",
    min_group_size: int = 30,
    min_class_count: int = 5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Estimate percentile intervals by bootstrapping within audit groups.

    Each group's observed ``(true label, predicted label)`` outcomes are
    resampled with replacement while preserving its observed size. This
    estimates sampling uncertainty conditional on the fitted model, threshold,
    and observed group sizes; it does not include model-refit uncertainty.
    Disparity intervals recompute max-minus-min gaps in each replicate while
    applying the same minimum group and class-support rules as the point
    estimates.
    """

    if (
        not isinstance(n_bootstrap, int)
        or isinstance(n_bootstrap, bool)
        or n_bootstrap < 100
    ):
        raise ValueError("n_bootstrap must be an integer of at least 100.")
    if (
        not isinstance(confidence_level, (int, float))
        or isinstance(confidence_level, bool)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be between 0 and 1.")

    truth_values = list(y_true)
    prediction_values = list(y_pred)
    group_values = list(sensitive_groups)
    point = cr_analyze_group_fairness(
        truth_values,
        prediction_values,
        group_values,
        attribute_name=attribute_name,
        min_group_size=min_group_size,
        min_class_count=min_class_count,
    )
    truth = pd.Series(truth_values).astype(int).to_numpy()
    prediction = pd.Series(prediction_values).astype(int).to_numpy()
    groups = pd.Series(group_values, dtype="string").fillna(
        "[missing]"
    ).to_numpy()
    group_metrics = point["group_metrics"].set_index("group")
    rng = np.random.default_rng(random_state)
    metrics = (
        "outcome_rate",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
    )
    group_distributions: dict[str, dict[str, np.ndarray]] = {}

    for group in group_metrics.index:
        mask = groups == group
        actual = truth[mask]
        predicted = prediction[mask]
        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        size = len(actual)
        probabilities = np.array([tp, fp, tn, fn], dtype=float) / size
        draws = rng.multinomial(size, probabilities, size=n_bootstrap)
        draw_tp, draw_fp, draw_tn, draw_fn = draws.T
        positive = draw_tp + draw_fn
        negative = draw_fp + draw_tn
        selected = draw_tp + draw_fp

        group_distributions[group] = {
            "outcome_rate": positive / size,
            "selection_rate": selected / size,
            "true_positive_rate": np.divide(
                draw_tp,
                positive,
                out=np.full(n_bootstrap, np.nan),
                where=positive > 0,
            ),
            "false_positive_rate": np.divide(
                draw_fp,
                negative,
                out=np.full(n_bootstrap, np.nan),
                where=negative > 0,
            ),
            "precision": np.divide(
                draw_tp,
                selected,
                out=np.full(n_bootstrap, np.nan),
                where=selected > 0,
            ),
            "positive_support": positive,
            "negative_support": negative,
        }

    tail = (1 - confidence_level) / 2

    def interval(values: np.ndarray) -> tuple[float, float, int]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan, np.nan, 0
        lower, upper = np.quantile(finite, [tail, 1 - tail])
        return float(lower), float(upper), int(len(finite))

    group_rows = []
    for group, distributions in group_distributions.items():
        for metric in metrics:
            lower, upper, valid = interval(distributions[metric])
            group_rows.append(
                {
                    "attribute": attribute_name,
                    "group": group,
                    "metric": metric,
                    "estimate": float(group_metrics.loc[group, metric]),
                    "confidence_level": confidence_level,
                    "lower": lower,
                    "upper": upper,
                    "valid_replicates": valid,
                    "n": int(group_metrics.loc[group, "n"]),
                }
            )

    sufficient_groups = [
        group
        for group in group_metrics.index
        if bool(group_metrics.loc[group, "sample_sufficient"])
    ]
    gap_distributions = {
        "selection_rate_difference": np.full(n_bootstrap, np.nan),
        "true_positive_rate_difference": np.full(n_bootstrap, np.nan),
        "false_positive_rate_difference": np.full(n_bootstrap, np.nan),
        "equalized_odds_difference": np.full(n_bootstrap, np.nan),
    }
    for replicate in range(n_bootstrap):
        selected_rates = [
            group_distributions[group]["selection_rate"][replicate]
            for group in sufficient_groups
        ]
        if len(selected_rates) >= 2:
            gap_distributions["selection_rate_difference"][replicate] = (
                max(selected_rates) - min(selected_rates)
            )

        tpr_rates = [
            group_distributions[group]["true_positive_rate"][replicate]
            for group in sufficient_groups
            if group_distributions[group]["positive_support"][replicate]
            >= min_class_count
        ]
        fpr_rates = [
            group_distributions[group]["false_positive_rate"][replicate]
            for group in sufficient_groups
            if group_distributions[group]["negative_support"][replicate]
            >= min_class_count
        ]
        tpr_gap = max(tpr_rates) - min(tpr_rates) if len(tpr_rates) >= 2 else np.nan
        fpr_gap = max(fpr_rates) - min(fpr_rates) if len(fpr_rates) >= 2 else np.nan
        gap_distributions["true_positive_rate_difference"][replicate] = tpr_gap
        gap_distributions["false_positive_rate_difference"][replicate] = fpr_gap
        if np.isfinite(tpr_gap) and np.isfinite(fpr_gap):
            gap_distributions["equalized_odds_difference"][replicate] = max(
                tpr_gap,
                fpr_gap,
            )

    disparity_rows = []
    for metric, distribution in gap_distributions.items():
        lower, upper, valid = interval(distribution)
        estimate = point["disparities"][metric]
        disparity_rows.append(
            {
                "attribute": attribute_name,
                "metric": metric,
                "estimate": estimate,
                "confidence_level": confidence_level,
                "lower": lower,
                "upper": upper,
                "valid_replicates": valid,
            }
        )

    return {
        "group_confidence_intervals": pd.DataFrame(group_rows),
        "disparity_confidence_intervals": pd.DataFrame(disparity_rows),
    }
