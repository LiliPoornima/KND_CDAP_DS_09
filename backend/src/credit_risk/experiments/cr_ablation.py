"""Run controlled feature-group ablations on numeric credit-risk data."""

from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any
import pandas as pd

from backend.src.credit_risk.give_me_some_credit.model import (
    TARGET_COLUMN,
    cr_evaluate_model,
    cr_train_logistic_regression,
    cr_train_random_forest,
)
from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
    cr_evaluate_binary_probabilities,
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
        metrics = cr_evaluate_binary_probabilities(
            test_subset[target_column],
            test_probability,
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
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    return result
