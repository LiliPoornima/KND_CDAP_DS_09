
"""Validate binary indicators and create a separate analysis dataset."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "financial_inclusion"
    / "processed" / "fi_microdata_preprocessed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "financial_inclusion" / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"
)

BINARY_VARIABLES = [
    "internet_use",
    "account",
    "account_fin",
    "saved",
    "borrowed",
    "anydigpayment",
    "merchantpay_dig",
]


def prepare_analysis_dataset() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    missing_columns = [
        col for col in BINARY_VARIABLES if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Preserve every original column and value.
    analysis_df = df.copy()

    report = {
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "rows_dropped": 0,
        "missing_values_imputed": False,
        "original_values_modified": False,
        "binary_variable_audit": {},
    }

    for variable in BINARY_VARIABLES:
        series = df[variable]
        unexpected = series.notna() & ~series.isin([0, 1])

        report["binary_variable_audit"][variable] = {
            "zero_count": int((series == 0).sum()),
            "one_count": int((series == 1).sum()),
            "missing_count": int(series.isna().sum()),
            "unexpected_code_count": int(unexpected.sum()),
            "unexpected_codes": {
                str(code): int(count)
                for code, count in series[unexpected]
                .value_counts().items()
            },
        }

    output_path = OUTPUT_DIR / "fi_analysis_dataset.csv"
    analysis_df.to_csv(output_path, index=False)

    report["output_file"] = str(output_path.relative_to(PROJECT_ROOT))
    report["output_rows"] = int(len(analysis_df))
    report["output_columns"] = int(len(analysis_df.columns))
    report["row_count_preserved"] = len(df) == len(analysis_df)
    report["column_count_preserved"] = (
        len(df.columns) == len(analysis_df.columns)
    )

    report_path = REPORT_DIR / "fi_analysis_preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    prepare_analysis_dataset()
