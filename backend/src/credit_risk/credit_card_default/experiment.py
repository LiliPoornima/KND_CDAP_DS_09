"""Data loading and experiment preparation for credit-card default research."""

from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import pandas as pd

from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
    cr_bootstrap_binary_performance,
    cr_evaluate_binary_probabilities,
)
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_analyze_group_fairness,
    cr_bootstrap_group_fairness,
    cr_bin_age_groups,
)
from backend.src.credit_risk.credit_card_default.preprocessing import (
    cr_prepare_credit_card_dataset,
)
from backend.src.credit_risk.credit_card_default.training import (
    cr_train_credit_card_logistic_regression,
    cr_train_credit_card_random_forest,
)

TARGET_COLUMN = "default.payment.next.month"
TARGET_COLUMN_ALIASES = {"default payment next month": TARGET_COLUMN}


def _normalize_credit_card_workbook(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the UCI workbook variant with labels stored in row one."""

    if TARGET_COLUMN not in data.columns and "default payment next month" not in {
        str(column).strip().lower() for column in data.columns
    }:
        if data.empty:
            raise ValueError("The credit-card workbook contains no header row.")
        first_row = data.iloc[0]
        candidate_headers = [
            str(value).strip() if pd.notna(value) else ""
            for value in first_row.tolist()
        ]
        normalized_headers = [value.lower() for value in candidate_headers]
        if (
            "default payment next month" in normalized_headers
            and all(candidate_headers)
            and len(set(candidate_headers)) == len(candidate_headers)
        ):
            data = data.iloc[1:].copy()
            data.columns = candidate_headers
            data = data.reset_index(drop=True)

    # The source is numeric after its embedded header row. Convert columns
    # where every non-missing cell is parseable, leaving other datasets intact.
    for column in data.columns:
        converted = pd.to_numeric(data[column], errors="coerce")
        if converted.notna().sum() == data[column].notna().sum():
            data[column] = converted
    return data


def cr_load_credit_card_default_data(
    file_path: str | Path,
    *,
    sheet_name: int | str = 0,
) -> pd.DataFrame:
    """Load the Taiwan credit-card workbook or its ZIP archive.

    The supplied archive contains one legacy ``.xls`` workbook. When a ZIP
    is given, this function reads that workbook directly from the archive
    and does not extract or modify any files.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Credit-card dataset not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(
            f"Expected an Excel workbook or ZIP archive: {path}"
        )

    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                workbooks = [
                    item.filename
                    for item in archive.infolist()
                    if PurePosixPath(item.filename).suffix.lower()
                    in {".xls", ".xlsx"}
                    and not item.is_dir()
                ]
                if len(workbooks) != 1:
                    raise ValueError(
                        "ZIP archive must contain exactly one Excel workbook; "
                        f"found {len(workbooks)}."
                    )
                with archive.open(workbooks[0]) as stream:
                    workbook_bytes = BytesIO(stream.read())
            data = pd.read_excel(
                workbook_bytes,
                sheet_name=sheet_name,
            )
        elif path.suffix.lower() in {".xls", ".xlsx"}:
            data = pd.read_excel(path, sheet_name=sheet_name)
        else:
            raise ValueError(
                "Input must be an .xls, .xlsx, or .zip file containing one workbook."
            )
    except BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {path}") from exc
    except ImportError as exc:
        raise ImportError(
            "Reading this legacy .xls workbook requires xlrd. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    if not isinstance(data, pd.DataFrame):
        raise ValueError("The selected workbook sheet did not return a table.")
    if data.empty:
        raise ValueError("The credit-card dataset contains no data rows.")
    data = _normalize_credit_card_workbook(data)
    rename = {
        column: TARGET_COLUMN_ALIASES[column.strip().lower()]
        for column in data.columns
        if isinstance(column, str)
        and column.strip().lower() in TARGET_COLUMN_ALIASES
    }
    if TARGET_COLUMN in data.columns and TARGET_COLUMN in rename.values():
        raise ValueError("Workbook contains ambiguous default target columns.")
    if rename:
        data = data.rename(columns=rename)
    if not data.columns.is_unique:
        raise ValueError("Credit-card dataset columns must be unique.")

    return data


def cr_prepare_credit_card_dataset_from_file(
    file_path: str | Path,
    output_dir: str | Path | None = None,
    **split_options: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the source workbook/archive and prepare independent data splits."""

    data = cr_load_credit_card_default_data(file_path)
    return cr_prepare_credit_card_dataset(
        data,
        output_dir=output_dir,
        **split_options,
    )


def cr_run_credit_card_baselines(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    thresholds: list[float] | None = None,
    test_groups: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Select thresholds on validation data and evaluate baselines on test."""

    if not isinstance(train_df, pd.DataFrame) or train_df.empty:
        raise ValueError("train_df must be a non-empty pandas DataFrame.")
    if not train_df.columns.is_unique or target_column not in train_df.columns:
        raise ValueError("train_df must have unique columns and the target.")
    predictor_columns = [
        column for column in train_df.columns if column != target_column
    ]
    for split_name, data in (
        ("validation", validation_df),
        ("test", test_df),
    ):
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError(f"{split_name} data must be a non-empty DataFrame.")
        if not data.columns.is_unique or target_column not in data.columns:
            raise ValueError(f"{target_column} is missing from {split_name} data.")
        columns = [column for column in data.columns if column != target_column]
        if columns != predictor_columns:
            raise ValueError("Train, validation, and test features must match in order.")

    models = {
        "Logistic Regression": cr_train_credit_card_logistic_regression(
            train_df,
            target_column=target_column,
        ),
        "Random Forest": cr_train_credit_card_random_forest(
            train_df,
            target_column=target_column,
        ),
    }
    threshold_tables = []
    comparison_rows = []
    model_metrics = {}
    performance_intervals = {}

    for model_name, model in models.items():
        positive_index = list(model.classes_).index(1)
        validation_probability = model.predict_proba(
            validation_df[predictor_columns]
        )[:, positive_index]
        threshold_table = cr_analyze_probability_thresholds(
            validation_df[target_column],
            validation_probability,
            thresholds=thresholds,
        )
        threshold_table.insert(0, "Model", model_name)
        threshold_tables.append(threshold_table)
        selected = threshold_table.sort_values(
            ["F1-score", "Precision", "Threshold"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        selected_threshold = float(selected["Threshold"])
        test_probability = model.predict_proba(
            test_df[predictor_columns]
        )[:, positive_index]
        metrics = cr_evaluate_binary_probabilities(
            test_df[target_column],
            test_probability,
            threshold=selected_threshold,
            target_names=("No default", "Default"),
        )
        performance_intervals[model_name] = cr_bootstrap_binary_performance(
            test_df[target_column],
            test_probability,
            threshold=selected_threshold,
            random_state=42,
        )
        model_metrics[model_name] = metrics
        comparison_rows.append(
            {
                "Model": model_name,
                "Validation-selected threshold": selected_threshold,
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

    result: dict[str, Any] = {
        "models": models,
        "threshold_analysis": pd.concat(threshold_tables, ignore_index=True),
        "comparison": pd.DataFrame(comparison_rows).set_index("Model"),
        "test_metrics": model_metrics,
    }

    if test_groups is not None:
        if not isinstance(test_groups, pd.DataFrame):
            raise TypeError("test_groups must be a pandas DataFrame.")
        if len(test_groups) != len(test_df) or not test_groups.index.equals(test_df.index):
            raise ValueError("test_groups must be row-aligned with test_df.")
        fairness_results = {}
        fairness_uncertainty = {}
        for model_name, metrics in model_metrics.items():
            fairness_results[model_name] = {}
            fairness_uncertainty[model_name] = {}
            for attribute in test_groups.columns:
                group_values = test_groups[attribute]
                if str(attribute).strip().lower() == "age":
                    group_values = cr_bin_age_groups(group_values)
                group_fairness = cr_analyze_group_fairness(
                    test_df[target_column],
                    metrics["predictions"],
                    group_values,
                    attribute_name=str(attribute),
                )
                fairness_results[model_name][attribute] = group_fairness
                fairness_uncertainty[model_name][attribute] = (
                    cr_bootstrap_group_fairness(
                        test_df[target_column],
                        metrics["predictions"],
                        group_values,
                        attribute_name=str(attribute),
                        random_state=42,
                    )
                )
        result["fairness"] = fairness_results
        result["fairness_uncertainty"] = fairness_uncertainty

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        result["threshold_analysis"].to_csv(
            destination / "threshold_analysis.csv",
            index=False,
        )
        result["comparison"].to_csv(destination / "model_comparison.csv")
        for model_name, metrics in model_metrics.items():
            safe_name = model_name.lower().replace(" ", "_")
            pd.DataFrame(metrics["calibration_bins"]).to_csv(
                destination / f"{safe_name}_calibration_bins.csv",
                index=False,
            )
            performance_intervals[model_name].to_csv(
                destination / f"{safe_name}_performance_confidence_intervals.csv",
                index=False,
            )
            if "fairness" in result:
                gap_rows = []
                for attribute, fairness in result["fairness"][model_name].items():
                    attribute_name = str(attribute).lower().replace(" ", "_")
                    fairness["group_metrics"].to_csv(
                        destination / f"{safe_name}_{attribute_name}_fairness.csv",
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
                    gap_rows.append({"attribute": attribute, **fairness["disparities"]})
                pd.DataFrame(gap_rows).to_csv(
                    destination / f"{safe_name}_fairness_gaps.csv",
                    index=False,
                )

    return result
