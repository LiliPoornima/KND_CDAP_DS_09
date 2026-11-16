"""Run controlled feature-group ablations on numeric credit-risk data."""

from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from backend.src.credit_risk.give_me_some_credit.model import (
    TARGET_COLUMN,
    cr_evaluate_model,
    cr_train_logistic_regression,
    cr_train_random_forest,
)
from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
    cr_bootstrap_paired_binary_performance,
    cr_evaluate_binary_probabilities,
)
from backend.src.credit_risk.common.cr_calibration import (
    cr_group_calibration_summary,
)
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_analyze_group_fairness,
    cr_bin_age_groups,
    cr_bootstrap_paired_group_fairness_difference,
)


def cr_run_feature_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    model_name: str = "Random Forest",
    target_column: str = TARGET_COLUMN,
    threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare a full feature set with one-group-at-a-time removals.

    The fixed threshold should be selected on validation data before calling
    this function. The supplied test set is used to compare the predefined
    feature configurations and must not be used for further tuning.
    """

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise TypeError("train_df and test_df must be pandas DataFrames.")
    if train_df.empty or test_df.empty:
        raise ValueError("Training and test data must not be empty.")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError("threshold must be strictly between 0 and 1.")
    if not train_df.columns.is_unique or not test_df.columns.is_unique:
        raise ValueError("Training and test data must have unique columns.")
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' is required in both splits.")
    train_features = [c for c in train_df.columns if c != target_column]
    test_features = [c for c in test_df.columns if c != target_column]
    if train_features != test_features:
        raise ValueError("Training and test feature columns must match in order.")
    if not isinstance(feature_groups, dict) or not feature_groups:
        raise ValueError("feature_groups must be a non-empty mapping.")

    used_columns: dict[str, str] = {}
    for group_name, columns in feature_groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError("Feature-group names must be non-empty strings.")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"Feature group '{group_name}' must contain columns.")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Feature group '{group_name}' repeats columns.")
        for column in columns:
            if column == target_column or column not in train_features:
                raise ValueError(
                    f"Feature group '{group_name}' contains an unavailable predictor: {column}"
                )
            if column in used_columns:
                raise ValueError(
                    f"Feature '{column}' appears in both '{used_columns[column]}' "
                    f"and '{group_name}'."
                )
            used_columns[column] = group_name

    trainers = {
        "Logistic Regression": cr_train_logistic_regression,
        "Random Forest": cr_train_random_forest,
    }
    if model_name not in trainers:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from {list(trainers)}."
        )
    scenarios = [("full", None, train_features)]
    for group_name, columns in feature_groups.items():
        remaining = [column for column in train_features if column not in columns]
        if not remaining:
            raise ValueError(
                f"Removing '{group_name}' would leave no predictor columns."
            )
        scenarios.append((f"without_{group_name}", group_name, remaining))

    rows = []
    for scenario, excluded_group, columns in scenarios:
        train_subset = train_df[columns + [target_column]]
        test_subset = test_df[columns + [target_column]]
        model = trainers[model_name](
            train_subset,
            target_column=target_column,
        )
        metrics = cr_evaluate_model(
            model,
            test_subset,
            target_column=target_column,
            threshold=threshold,
        )
        rows.append(
            {
                "Model": model_name,
                "Scenario": scenario,
                "Excluded feature group": excluded_group,
                "Excluded features": (
                    ";".join(feature_groups[excluded_group])
                    if excluded_group is not None
                    else ""
                ),
                "Predictor count": len(columns),
                "Threshold": threshold,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc_average_precision"],
                "Brier score": metrics["brier_score"],
                "Log loss": metrics["log_loss"],
                "Expected calibration error": metrics[
                    "expected_calibration_error"
                ],
            }
        )

    result = pd.DataFrame(rows)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    return result


def cr_run_validation_selected_feature_ablation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    trainer: Callable[..., Any],
    model_name: str,
    target_column: str,
    thresholds: Iterable[float] | None = None,
    output_path: str | Path | None = None,
    test_groups: pd.DataFrame | None = None,
    subgroup_output_path: str | Path | None = None,
    subgroup_calibration_output_path: str | Path | None = None,
    subgroup_calibration_bins_output_path: str | Path | None = None,
    paired_comparison_output_path: str | Path | None = None,
    paired_fairness_comparison_output_path: str | Path | None = None,
    paired_bootstrap_replicates: int = 1000,
    paired_bootstrap_random_state: int = 42,
) -> pd.DataFrame:
    """Run mixed-type feature-group ablations with validation-only thresholds.

    The full feature set and each one-group-removed configuration are fitted
    on the same training split. Each configuration selects its threshold on
    the validation split; the test split is used once for final comparison.
    ``trainer`` must fit the supplied DataFrame using training-only
    preprocessing and accept ``target_column`` as a keyword argument.
    """

    datasets = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    for split_name, data in datasets.items():
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{split_name}_df must be a pandas DataFrame.")
        if data.empty:
            raise ValueError(f"{split_name}_df must not be empty.")
        if not data.columns.is_unique:
            raise ValueError(f"{split_name}_df must have unique columns.")
        if target_column not in data.columns:
            raise ValueError(f"{target_column} is required in {split_name}_df.")

    features = [column for column in train_df.columns if column != target_column]
    for split_name, data in (("validation", validation_df), ("test", test_df)):
        current = [column for column in data.columns if column != target_column]
        if current != features:
            raise ValueError(
                f"{split_name} predictors must match training columns and order."
            )
    if not callable(trainer):
        raise TypeError("trainer must be callable.")
    if test_groups is not None:
        if not isinstance(test_groups, pd.DataFrame) or test_groups.empty:
            raise ValueError("test_groups must be a non-empty DataFrame.")
        if (
            not test_groups.columns.is_unique
            or len(test_groups) != len(test_df)
            or not test_groups.index.equals(test_df.index)
        ):
            raise ValueError("test_groups rows and index must align with test_df.")
    if any(
        path is not None
        for path in (
            subgroup_output_path,
            subgroup_calibration_output_path,
            subgroup_calibration_bins_output_path,
            paired_fairness_comparison_output_path,
        )
    ) and test_groups is None:
        raise ValueError("test_groups are required when saving subgroup results or fairness intervals.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(feature_groups, dict) or not feature_groups:
        raise ValueError("feature_groups must be a non-empty mapping.")

    used_columns: dict[str, str] = {}
    for group_name, columns in feature_groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError("Feature-group names must be non-empty strings.")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"Feature group '{group_name}' must contain columns.")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Feature group '{group_name}' repeats columns.")
        for column in columns:
            if column == target_column or column not in features:
                raise ValueError(
                    f"Feature group '{group_name}' contains an unavailable predictor: {column}"
                )
            if column in used_columns:
                raise ValueError(
                    f"Feature '{column}' appears in both '{used_columns[column]}' "
                    f"and '{group_name}'."
                )
            used_columns[column] = group_name

    scenarios = [("full", None, features)]
    for group_name, columns in feature_groups.items():
        remaining = [column for column in features if column not in columns]
        if not remaining:
            raise ValueError(
                f"Removing '{group_name}' would leave no predictor columns."
            )
        scenarios.append((f"without_{group_name}", group_name, remaining))

    threshold_candidates = None if thresholds is None else list(thresholds)
    rows = []
    probabilities_by_scenario: dict[str, np.ndarray] = {}
    thresholds_by_scenario: dict[str, float] = {}
    predictions_by_scenario: dict[str, np.ndarray] = {}
    subgroup_rows: list[dict[str, Any]] = []
    subgroup_calibration_rows: list[pd.DataFrame] = []
    subgroup_calibration_bin_rows: list[pd.DataFrame] = []
    for scenario, excluded_group, columns in scenarios:
        train_subset = train_df[columns + [target_column]]
        validation_subset = validation_df[columns + [target_column]]
        test_subset = test_df[columns + [target_column]]
        model = trainer(train_subset, target_column=target_column)
        positive_index = list(model.classes_).index(1)
        validation_probability = model.predict_proba(
            validation_subset[columns]
        )[:, positive_index]
        threshold_table = cr_analyze_probability_thresholds(
            validation_subset[target_column],
            validation_probability,
            thresholds=threshold_candidates,
        )
        selected = threshold_table.sort_values(
            ["F1-score", "Precision", "Threshold"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        threshold = float(selected["Threshold"])
        test_probability = model.predict_proba(test_subset[columns])[:, positive_index]
        probabilities_by_scenario[scenario] = test_probability
        thresholds_by_scenario[scenario] = threshold
        predictions_by_scenario[scenario] = (
            test_probability >= threshold
        ).astype(int)
        metrics = cr_evaluate_binary_probabilities(
            test_subset[target_column],
            test_probability,
            threshold=threshold,
        )
        if test_groups is not None:
            predictions = (test_probability >= threshold).astype(int)
            for attribute in test_groups.columns:
                group_values = test_groups[attribute]
                if str(attribute).lower() in {"age", "age_years"}:
                    group_values = cr_bin_age_groups(group_values)
                fairness = cr_analyze_group_fairness(
                    test_subset[target_column].reset_index(drop=True),
                    predictions,
                    group_values,
                    attribute_name=str(attribute),
                )
                calibration = cr_group_calibration_summary(
                    test_subset[target_column],
                    test_probability,
                    group_values,
                    attribute_name=str(attribute),
                )
                calibration_metrics = calibration["group_metrics"].copy()
                calibration_metrics.insert(0, "Model", model_name)
                calibration_metrics.insert(1, "Scenario", scenario)
                calibration_metrics.insert(
                    2, "Excluded feature group", excluded_group
                )
                calibration_metrics.insert(
                    3, "Validation-selected threshold", threshold
                )
                subgroup_calibration_rows.append(calibration_metrics)
                calibration_bins = calibration["calibration_bins"].copy()
                calibration_bins.insert(0, "Model", model_name)
                calibration_bins.insert(1, "Scenario", scenario)
                calibration_bins.insert(
                    2, "Excluded feature group", excluded_group
                )
                calibration_bins.insert(
                    3, "Validation-selected threshold", threshold
                )
                subgroup_calibration_bin_rows.append(calibration_bins)
                disparities = fairness["disparities"]
                for record in fairness["group_metrics"].to_dict(orient="records"):
                    subgroup_rows.append(
                        {
                            "Model": model_name,
                            "Scenario": scenario,
                            "Excluded feature group": excluded_group,
                            "Validation-selected threshold": threshold,
                            **record,
                            **{
                                f"{key}_gap": value
                                for key, value in disparities.items()
                                if key.endswith("_difference")
                            },
                        }
                    )
        rows.append(
            {
                "Model": model_name,
                "Scenario": scenario,
                "Excluded feature group": excluded_group,
                "Excluded features": (
                    ";".join(feature_groups[excluded_group])
                    if excluded_group is not None
                    else ""
                ),
                "Predictor count": len(columns),
                "Validation-selected threshold": threshold,
                "Validation F1-score": float(selected["F1-score"]),
                "Test accuracy": metrics["accuracy"],
                "Test precision": metrics["precision"],
                "Test recall": metrics["recall"],
                "Test F1-score": metrics["f1_score"],
                "Test ROC-AUC": metrics["roc_auc"],
                "Test PR-AUC": metrics["pr_auc_average_precision"],
                "Test Brier score": metrics["brier_score"],
                "Test log loss": metrics["log_loss"],
                "Test expected calibration error": metrics[
                    "expected_calibration_error"
                ],
            }
        )

    result = pd.DataFrame(rows)
    if paired_comparison_output_path is not None:
        paired_tables = []
        for scenario, excluded_group, _ in scenarios[1:]:
            paired = cr_bootstrap_paired_binary_performance(
                test_df[target_column],
                probabilities_by_scenario["full"],
                probabilities_by_scenario[scenario],
                threshold_a=thresholds_by_scenario["full"],
                threshold_b=thresholds_by_scenario[scenario],
                model_a_name="full",
                model_b_name=scenario,
                n_bootstrap=paired_bootstrap_replicates,
                random_state=paired_bootstrap_random_state,
            )
            paired.insert(0, "excluded_feature_group", excluded_group)
            paired.insert(0, "comparison", f"{scenario}_minus_full")
            paired.insert(0, "scenario", scenario)
            paired.insert(0, "model", model_name)
            paired_tables.append(paired)
        destination = Path(paired_comparison_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if paired_tables:
            pd.concat(paired_tables, ignore_index=True).to_csv(
                destination,
                index=False,
            )
        else:
            pd.DataFrame().to_csv(destination, index=False)
    if paired_fairness_comparison_output_path is not None:
        paired_fairness_tables = []
        for scenario, excluded_group, _ in scenarios[1:]:
            for attribute in test_groups.columns:
                group_values = test_groups[attribute]
                if str(attribute).strip().lower() in {"age", "age_years"}:
                    group_values = cr_bin_age_groups(group_values)
                paired_fairness = cr_bootstrap_paired_group_fairness_difference(
                    test_df[target_column],
                    predictions_by_scenario["full"],
                    predictions_by_scenario[scenario],
                    group_values,
                    attribute_name=str(attribute),
                    n_bootstrap=paired_bootstrap_replicates,
                    random_state=paired_bootstrap_random_state,
                )
                paired_fairness.insert(0, "excluded_feature_group", excluded_group)
                paired_fairness.insert(0, "comparison", f"{scenario}_minus_full")
                paired_fairness.insert(0, "scenario", scenario)
                paired_fairness.insert(0, "model", model_name)
                paired_fairness_tables.append(paired_fairness)
        destination = Path(paired_fairness_comparison_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if paired_fairness_tables:
            pd.concat(paired_fairness_tables, ignore_index=True).to_csv(
                destination,
                index=False,
            )
        else:
            pd.DataFrame().to_csv(destination, index=False)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    if subgroup_output_path is not None:
        destination = Path(subgroup_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(subgroup_rows).to_csv(destination, index=False)
    if subgroup_calibration_output_path is not None:
        destination = Path(subgroup_calibration_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(subgroup_calibration_rows, ignore_index=True).to_csv(
            destination, index=False
        )
    if subgroup_calibration_bins_output_path is not None:
        destination = Path(subgroup_calibration_bins_output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(subgroup_calibration_bin_rows, ignore_index=True).to_csv(
            destination, index=False
        )
    return result
