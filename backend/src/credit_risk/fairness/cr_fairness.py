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


def cr_build_subgroup_support_coverage(
    output_root: str,
    output_path: str | None = None,
    *,
    min_group_size: int = 30,
    min_class_count: int = 5,
) -> pd.DataFrame:
    """Collect per-group audit support and eligibility from saved fairness CSVs.

    The output keeps every observed group and makes the existing audit
    thresholds explicit, so small groups are reported as coverage limits
    rather than silently omitted.
    """
    from pathlib import Path

    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Credit-risk output root not found: {root}")
    if min_group_size < 1 or min_class_count < 1:
        raise ValueError("Support thresholds must be positive integers.")

    rows: list[dict[str, Any]] = []
    for source in sorted(root.rglob("*_fairness.csv")):
        try:
            relative = source.relative_to(root)
            dataset = relative.parts[0]
            table = pd.read_csv(source)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        required = {"group", "n", "positive_support", "negative_support"}
        if not required.issubset(table.columns) or table.empty:
            continue
        stem = source.stem
        model = next(
            (name for name in ("logistic_regression", "random_forest") if stem.startswith(name)),
            "unknown",
        )
        attribute = str(table.iloc[0].get("attribute", "unknown"))
        for record in table.to_dict(orient="records"):
            n = int(record["n"])
            positives = int(record["positive_support"])
            negatives = int(record["negative_support"])
            eligible_sample = n >= min_group_size
            eligible_tpr = eligible_sample and positives >= min_class_count
            eligible_fpr = eligible_sample and negatives >= min_class_count
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "attribute": attribute,
                    "group": str(record["group"]),
                    "n": n,
                    "positive_support": positives,
                    "negative_support": negatives,
                    "min_group_size": min_group_size,
                    "min_class_count": min_class_count,
                    "selection_rate_reportable": eligible_sample,
                    "tpr_reportable": eligible_tpr,
                    "fpr_reportable": eligible_fpr,
                    "coverage_note": (
                        "meets all rate support rules"
                        if eligible_tpr and eligible_fpr
                        else "limited: " + ", ".join(
                            label
                            for condition, label in (
                                (not eligible_sample, "group below minimum size"),
                                (eligible_sample and not eligible_tpr, "positive class below minimum"),
                                (eligible_sample and not eligible_fpr, "negative class below minimum"),
                            )
                            if condition
                        )
                    ),
                    "source_file": str(relative).replace("\\", "/"),
                }
            )

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        raise FileNotFoundError(f"No saved subgroup fairness tables found under {root}")
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(destination, index=False)
    return coverage


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


def cr_bootstrap_paired_group_fairness_difference(
    y_true: Iterable[Any],
    y_pred_a: Iterable[Any],
    y_pred_b: Iterable[Any],
    sensitive_groups: Iterable[Any],
    *,
    attribute_name: str = "group",
    min_group_size: int = 30,
    min_class_count: int = 5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate paired intervals for changes in subgroup fairness gaps.

    Each replicate resamples the same test-row indices for both prediction
    vectors, separately within every observed group and target class. The
    returned difference is model B's max-minus-min disparity gap minus model
    A's gap. Intervals condition on both fitted models, their frozen
    validation thresholds, and observed group-by-outcome counts.
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
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError("random_state must be an integer.")

    try:
        truth = np.asarray(list(y_true), dtype=float)
        prediction_a = np.asarray(list(y_pred_a), dtype=float)
        prediction_b = np.asarray(list(y_pred_b), dtype=float)
        groups = pd.Series(list(sensitive_groups), dtype="string").fillna(
            "[missing]"
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Targets, predictions, and groups must be iterable values.") from exc
    if (
        truth.ndim != 1
        or prediction_a.ndim != 1
        or prediction_b.ndim != 1
    ):
        raise ValueError("Targets and predictions must be one-dimensional.")
    if truth.size == 0 or not (
        len(truth) == len(prediction_a) == len(prediction_b) == len(groups)
    ):
        raise ValueError("Targets, predictions, and groups must have equal non-zero length.")
    if not np.isfinite(truth).all() or set(np.unique(truth)) != {0.0, 1.0}:
        raise ValueError("y_true must contain both binary classes: 0 and 1.")
    if (
        not np.isfinite(prediction_a).all()
        or not np.isfinite(prediction_b).all()
        or not set(np.unique(prediction_a)).issubset({0.0, 1.0})
        or not set(np.unique(prediction_b)).issubset({0.0, 1.0})
    ):
        raise ValueError("Predictions must contain only finite binary labels: 0 and 1.")
    if not isinstance(attribute_name, str) or not attribute_name.strip():
        raise ValueError("attribute_name must be a non-empty string.")

    truth = truth.astype(int)
    prediction_a = prediction_a.astype(int)
    prediction_b = prediction_b.astype(int)
    group_array = groups.to_numpy(dtype=str)
    point_a = cr_analyze_group_fairness(
        truth,
        prediction_a,
        group_array,
        attribute_name=attribute_name,
        min_group_size=min_group_size,
        min_class_count=min_class_count,
    )["disparities"]
    point_b = cr_analyze_group_fairness(
        truth,
        prediction_b,
        group_array,
        attribute_name=attribute_name,
        min_group_size=min_group_size,
        min_class_count=min_class_count,
    )["disparities"]

    strata: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    group_sizes: dict[str, int] = {}
    for group in sorted(np.unique(group_array)):
        indexes = np.flatnonzero(group_array == group)
        strata[group] = (indexes[truth[indexes] == 0], indexes[truth[indexes] == 1])
        group_sizes[group] = len(indexes)
    selection_groups = [
        group for group, size in group_sizes.items() if size >= min_group_size
    ]
    tpr_groups = [
        group
        for group, (negative, positive) in strata.items()
        if group_sizes[group] >= min_group_size and len(positive) >= min_class_count
    ]
    fpr_groups = [
        group
        for group, (negative, positive) in strata.items()
        if group_sizes[group] >= min_group_size and len(negative) >= min_class_count
    ]
    metrics = (
        "selection_rate_difference",
        "true_positive_rate_difference",
        "false_positive_rate_difference",
        "equalized_odds_difference",
    )
    differences = {metric: np.full(n_bootstrap, np.nan) for metric in metrics}
    rng = np.random.default_rng(random_state)

    for replicate in range(n_bootstrap):
        group_rates_a: dict[str, tuple[float, float | None, float | None]] = {}
        group_rates_b: dict[str, tuple[float, float | None, float | None]] = {}
        for group, (negative_indexes, positive_indexes) in strata.items():
            sampled_negative = (
                rng.choice(negative_indexes, size=len(negative_indexes), replace=True)
                if len(negative_indexes)
                else negative_indexes
            )
            sampled_positive = (
                rng.choice(positive_indexes, size=len(positive_indexes), replace=True)
                if len(positive_indexes)
                else positive_indexes
            )
            sampled_group = np.concatenate((sampled_negative, sampled_positive))
            rates = []
            for prediction in (prediction_a, prediction_b):
                selection = float(prediction[sampled_group].mean())
                tpr = (
                    float(prediction[sampled_positive].mean())
                    if len(positive_indexes)
                    else None
                )
                fpr = (
                    float(prediction[sampled_negative].mean())
                    if len(negative_indexes)
                    else None
                )
                rates.append((selection, tpr, fpr))
            group_rates_a[group], group_rates_b[group] = rates

        def _gap(
            values: dict[str, tuple[float, float | None, float | None]],
            eligible: list[str],
            rate_index: int,
        ) -> float | None:
            rates = [values[group][rate_index] for group in eligible]
            rates = [rate for rate in rates if rate is not None]
            return float(max(rates) - min(rates)) if len(rates) >= 2 else None

        gap_a = {
            "selection_rate_difference": _gap(
                group_rates_a, selection_groups, 0
            ),
            "true_positive_rate_difference": _gap(group_rates_a, tpr_groups, 1),
            "false_positive_rate_difference": _gap(group_rates_a, fpr_groups, 2),
        }
        gap_b = {
            "selection_rate_difference": _gap(
                group_rates_b, selection_groups, 0
            ),
            "true_positive_rate_difference": _gap(group_rates_b, tpr_groups, 1),
            "false_positive_rate_difference": _gap(group_rates_b, fpr_groups, 2),
        }
        for gaps in (gap_a, gap_b):
            components = [
                gaps[key]
                for key in (
                    "true_positive_rate_difference",
                    "false_positive_rate_difference",
                )
                if gaps[key] is not None
            ]
            gaps["equalized_odds_difference"] = (
                max(components) if len(components) == 2 else None
            )
        for metric in metrics:
            if gap_a[metric] is not None and gap_b[metric] is not None:
                differences[metric][replicate] = gap_b[metric] - gap_a[metric]

    tail = (1.0 - float(confidence_level)) / 2.0
    rows = []
    for metric in metrics:
        values = differences[metric]
        finite = values[np.isfinite(values)]
        estimate_a = point_a[metric]
        estimate_b = point_b[metric]
        rows.append(
            {
                "attribute": attribute_name,
                "metric": metric,
                "estimate_a": estimate_a,
                "estimate_b": estimate_b,
                "difference_b_minus_a": (
                    float(estimate_b - estimate_a)
                    if estimate_a is not None and estimate_b is not None
                    else np.nan
                ),
                "confidence_level": float(confidence_level),
                "lower": float(np.quantile(finite, tail)) if len(finite) else np.nan,
                "upper": float(np.quantile(finite, 1.0 - tail)) if len(finite) else np.nan,
                "valid_replicates": int(len(finite)),
                "n": int(len(truth)),
                "eligible_selection_groups": len(selection_groups),
                "eligible_tpr_groups": len(tpr_groups),
                "eligible_fpr_groups": len(fpr_groups),
                "bootstrap_method": "paired_stratified_by_group_and_target_class",
                "random_state": random_state,
            }
        )
    return pd.DataFrame(rows)
