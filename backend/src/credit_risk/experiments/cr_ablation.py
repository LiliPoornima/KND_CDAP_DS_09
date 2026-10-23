"""Run controlled feature-group ablations on numeric credit-risk data."""

from pathlib import Path
import pandas as pd

from backend.src.credit_risk.give_me_some_credit.model import (
    TARGET_COLUMN,
    cr_evaluate_model,
    cr_train_logistic_regression,
    cr_train_random_forest,
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
