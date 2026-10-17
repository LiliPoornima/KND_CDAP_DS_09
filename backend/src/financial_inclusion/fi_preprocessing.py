
"""Validate and prepare Global Findex 2025 microdata."""

import json
from pathlib import Path

import pandas as pd

from fi_data_loader import load_microdata


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "financial_inclusion" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"

REQUIRED_COLUMNS = [
    "year",
    "economy",
    "economycode",
    "wpid_random",
    "wgt",
    "account_fin",
    "internet_use",
    "con1",
]


def preprocess_microdata() -> dict:
    """Validate source data and save an unchanged-response working copy."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_microdata()

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the dataset: {missing_columns}"
        )

    # Do not drop records or recode survey responses at this stage.
    processed_path = PROCESSED_DIR / "fi_microdata_preprocessed.csv"
    df.to_csv(processed_path, index=False)

    missing_counts = df.isna().sum()
    missing_report = (
        missing_counts[missing_counts > 0]
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )
    missing_report["missing_percent"] = (
        missing_report["missing_count"] / len(df) * 100
    ).round(2)

    missing_report_path = REPORT_DIR / "fi_missing_value_report.csv"
    missing_report.to_csv(missing_report_path, index_label="variable")

    id_duplicates = int(df["wpid_random"].duplicated().sum())
    missing_ids = int(df["wpid_random"].isna().sum())

    report = {
        "source_file": "findex_microdata_2025_labelled_update112425.csv",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_column_count": int(df.columns.duplicated().sum()),
        "duplicate_wpid_random_count": id_duplicates,
        "missing_wpid_random_count": missing_ids,
        "required_columns": REQUIRED_COLUMNS,
        "missing_cells_total": int(df.isna().sum().sum()),
        "missing_value_report": str(missing_report_path.relative_to(PROJECT_ROOT)),
        "processed_file": str(processed_path.relative_to(PROJECT_ROOT)),
        "response_codes_changed": False,
        "rows_dropped": 0,
        "missing_values_imputed": False,
    }

    report_path = REPORT_DIR / "fi_preprocessing_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    preprocess_microdata()

