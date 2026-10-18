
"""Audit categorical response codes before recoding."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "financial_inclusion"
    / "processed" / "fi_analysis_dataset.csv"
)

REPORT_DIR = (
    PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"
)

CATEGORICAL_VARIABLES = [
    "con1", "con9", "con12", "con13",
    "con18", "con19", "con21", "con22", "con23",
    "con24", "con25", "con26", "fin2",
    "fin3", "fin26a", "fin26b",
    "pay_utilities", "domestic_remittances",
    "receive_wages", "receive_transfers",
    "receive_pensions", "receive_agriculture",
    "fin24", "fin24a", "fin24b",
]


def audit_categorical_variables() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    missing_columns = [
        col for col in CATEGORICAL_VARIABLES if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing variables: {missing_columns}")

    details = []

    for variable in CATEGORICAL_VARIABLES:
        counts = df[variable].value_counts(dropna=False)

        for code, count in counts.items():
            details.append({
                "variable": variable,
                "response_code": (
                    "MISSING" if pd.isna(code) else str(code)
                ),
                "count": int(count),
                "percentage": round(count / len(df) * 100, 2),
            })

    audit_df = pd.DataFrame(details)

    output_path = REPORT_DIR / "fi_categorical_response_audit.csv"
    audit_df.to_csv(output_path, index=False)

    summary = {
        "row_count": int(len(df)),
        "variables_audited": CATEGORICAL_VARIABLES,
        "variable_count": len(CATEGORICAL_VARIABLES),
        "raw_values_modified": False,
        "recoding_performed": False,
        "report_file": str(output_path.relative_to(PROJECT_ROOT)),
    }

    summary_path = REPORT_DIR / "fi_categorical_response_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nFirst 40 audit rows:")
    print(audit_df.head(40).to_string(index=False))

    return summary


if __name__ == "__main__":
    audit_categorical_variables()
