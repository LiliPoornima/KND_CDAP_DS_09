"""Preparation and controlled baseline experiments for Home Credit."""

from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
    cr_evaluate_binary_probabilities,
)
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_analyze_group_fairness,
    cr_bootstrap_group_fairness,
    cr_bin_age_groups,
)
from backend.src.credit_risk.home_credit.data_loader import (
    cr_load_home_credit_table,
)
from backend.src.credit_risk.home_credit.feature_engineering import (
    cr_aggregate_home_credit_histories,
)
from backend.src.credit_risk.home_credit.preprocessing import (
    TARGET_COLUMN,
    cr_prepare_home_credit_dataset,
)
from backend.src.credit_risk.home_credit.training import (
    cr_train_home_credit_logistic_regression,
    cr_train_home_credit_random_forest,
)


def cr_prepare_home_credit_dataset_from_archive(
    archive_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    include_history: bool = True,
    history_chunksize: int = 100_000,
    **split_options: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load labeled applications, attach safe history aggregates, and split.

    Set ``include_history=False`` to run the application-only baseline.
    Linked tables are read in chunks and never extracted from the archive.
    """

    application_train = cr_load_home_credit_table(
        archive_path,
        table_name="application_train.csv",
    )
    if include_history:
        history_features = cr_aggregate_home_credit_histories(
            archive_path,
            chunksize=history_chunksize,
        )
        application_train = application_train.merge(
            history_features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )
        history_columns = [
            column
            for column in history_features.columns
            if column != "SK_ID_CURR"
        ]
        application_train[history_columns] = application_train[
            history_columns
        ].fillna(0)

    train_df, test_df, metadata = cr_prepare_home_credit_dataset(
        application_train,
        output_dir=output_dir,
        **split_options,
    )
    metadata["includes_prior_history"] = include_history
    metadata["history_chunksize"] = history_chunksize if include_history else None
    return train_df, test_df, metadata


def cr_run_home_credit_baselines(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    thresholds: list[float] | None = None,
    test_groups: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train baselines, select thresholds on validation, then test once.

    Each model's threshold is selected by maximum validation F1; ties are
    broken by higher validation precision, then the higher threshold. The held-out
    test partition is used only after that threshold is fixed. If audit-group
    data are supplied, group metrics are computed on the test predictions and
    the audit attributes are never passed to the estimators.
    """

    datasets = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    predictor_columns: list[str] | None = None
    for split_name, data in datasets.items():
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{split_name} data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError(f"{split_name} data must not be empty.")
        if not data.columns.is_unique:
            raise ValueError(f"{split_name} data must have unique columns.")
        if target_column not in data.columns:
            raise ValueError(f"{target_column} is missing from {split_name} data.")
        current_columns = [column for column in data.columns if column != target_column]
        if predictor_columns is None:
            predictor_columns = current_columns
        elif current_columns != predictor_columns:
            raise ValueError(
                "Train, validation, and test predictor columns must match "
                "in the same order."
            )

    models = {
        "Logistic Regression": cr_train_home_credit_logistic_regression(
            train_df,
            target_column=target_column,
        ),
        "Random Forest": cr_train_home_credit_random_forest(
            train_df,
            target_column=target_column,
        ),
    }
    validation_truth = validation_df[target_column]
    test_truth = test_df[target_column]
    validation_X = validation_df[predictor_columns]
    test_X = test_df[predictor_columns]
    model_metrics: dict[str, dict[str, Any]] = {}
    training_diagnostics: list[dict[str, Any]] = []
    threshold_tables = []
    comparison_rows = []

    for model_name, model in models.items():
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(estimator, "n_iter_"):
            iterations = [int(value) for value in estimator.n_iter_]
            max_iterations = int(estimator.max_iter)
            training_diagnostics.append(
                {
                    "Model": model_name,
                    "Iterations": ";".join(map(str, iterations)),
                    "Maximum iterations": max_iterations,
                    "Converged": all(value < max_iterations for value in iterations),
                }
            )
        positive_class_index = list(model.classes_).index(1)
        validation_probability = model.predict_proba(validation_X)[
            :, positive_class_index
        ]
        threshold_table = cr_analyze_probability_thresholds(
            validation_truth,
            validation_probability,
            thresholds=thresholds,
        )
        threshold_table.insert(0, "Model", model_name)
        threshold_tables.append(threshold_table)
        best_row = threshold_table.sort_values(
            ["F1-score", "Precision", "Threshold"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        selected_threshold = float(best_row["Threshold"])

        test_probability = model.predict_proba(test_X)[:, positive_class_index]
        metrics = cr_evaluate_binary_probabilities(
            test_truth,
            test_probability,
            threshold=selected_threshold,
            target_names=("No payment difficulty", "Payment difficulty"),
        )
        model_metrics[model_name] = metrics
        comparison_rows.append(
            {
                "Model": model_name,
                "Validation-selected threshold": selected_threshold,
                "Validation F1-score": float(best_row["F1-score"]),
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

    result: dict[str, Any] = {
        "models": models,
        "threshold_analysis": pd.concat(threshold_tables, ignore_index=True),
        "comparison": pd.DataFrame(comparison_rows).set_index("Model"),
        "test_metrics": model_metrics,
        "training_diagnostics": pd.DataFrame(training_diagnostics),
    }

    if test_groups is not None:
        if not isinstance(test_groups, pd.DataFrame):
            raise TypeError("test_groups must be a pandas DataFrame.")
        if len(test_groups) != len(test_df):
            raise ValueError("test_groups and test_df must have matching row counts.")
        if not test_groups.index.equals(test_df.index):
            raise ValueError("test_groups must be row-aligned with test_df.")
        fairness_results = {}
        fairness_uncertainty = {}
        for model_name, metrics in model_metrics.items():
            per_attribute = {}
            uncertainty_by_attribute = {}
            for attribute in test_groups.columns:
                group_values = test_groups[attribute]
                if str(attribute).strip().lower() in {"age", "age_years"}:
                    group_values = cr_bin_age_groups(group_values)
                group_fairness = cr_analyze_group_fairness(
                    test_truth,
                    metrics["predictions"],
                    group_values,
                    attribute_name=attribute,
                )
                per_attribute[attribute] = group_fairness
                uncertainty_by_attribute[attribute] = cr_bootstrap_group_fairness(
                    test_truth,
                    metrics["predictions"],
                    group_values,
                    attribute_name=attribute,
                    random_state=42,
                )
            fairness_results[model_name] = per_attribute
            fairness_uncertainty[model_name] = uncertainty_by_attribute
        result["fairness"] = fairness_results
        result["fairness_uncertainty"] = fairness_uncertainty

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        result["threshold_analysis"].to_csv(
            destination / "threshold_analysis.csv",
            index=False,
        )
        result["comparison"].to_csv(
            destination / "model_comparison.csv",
        )
        result["training_diagnostics"].to_csv(
            destination / "training_diagnostics.csv",
            index=False,
        )
        for model_name, metrics in model_metrics.items():
            safe_name = model_name.lower().replace(" ", "_")
            pd.DataFrame(metrics["calibration_bins"]).to_csv(
                destination / f"{safe_name}_calibration_bins.csv",
                index=False,
            )
            if "fairness" in result:
                disparity_rows = []
                for attribute, fairness in result["fairness"][model_name].items():
                    attribute_name = str(attribute).lower().replace(" ", "_")
                    fairness["group_metrics"].to_csv(
                        destination
                        / f"{safe_name}_{attribute_name}_fairness.csv",
                        index=False,
                    )
                    uncertainty = result["fairness_uncertainty"][model_name][
                        attribute
                    ]
                    uncertainty["group_confidence_intervals"].to_csv(
                        destination
                        / f"{safe_name}_{attribute_name}_group_confidence_intervals.csv",
                        index=False,
                    )
                    uncertainty["disparity_confidence_intervals"].to_csv(
                        destination
                        / f"{safe_name}_{attribute_name}_gap_confidence_intervals.csv",
                        index=False,
                    )
                    disparity_rows.append(
                        {
                            "attribute": attribute,
                            **fairness["disparities"],
                        }
                    )
                pd.DataFrame(disparity_rows).to_csv(
                    destination / f"{safe_name}_fairness_gaps.csv",
                    index=False,
                )

    return result
