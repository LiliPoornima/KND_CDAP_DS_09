
"""Prepare a separate three-dimension clustering input."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILE = (
    ROOT / "data" / "financial_inclusion" / "processed"
    / "fi_dimension_dataset.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "financial_inclusion" / "processed"
    / "fi_clustering_input_3d.csv"
)

REPORT_FILE = (
    ROOT / "outputs" / "financial_inclusion" / "reports"
    / "fi_clustering_input_3d_report.json"
)

ID_COLUMN = "wpid_random"
COUNTRY_COLUMN = "economy"

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
]


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    df = pd.read_csv(SOURCE_FILE, low_memory=False)

    required = [ID_COLUMN, COUNTRY_COLUMN, *FEATURES]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise ValueError("Respondent IDs must be non-missing and unique.")

    # Use only respondents with complete scores for all three dimensions.
    eligible = df[required].dropna(subset=FEATURES).copy()

    if eligible.empty:
        raise ValueError("No eligible respondents found.")

    # Validate score ranges.
    scores = eligible[FEATURES]

    if not scores.apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("All dimension scores must be within [0, 1].")

    if not scores.notna().all().all():
        raise ValueError("Missing feature values remain in the output.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    eligible.to_csv(OUTPUT_FILE, index=False)

    report = {
        "source_file": str(SOURCE_FILE.relative_to(ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "source_respondents": int(len(df)),
        "eligible_respondents": int(len(eligible)),
        "excluded_respondents": int(len(df) - len(eligible)),
        "eligible_percent": round(len(eligible) / len(df) * 100, 2),
        "total_economies_in_source": int(df[COUNTRY_COLUMN].nunique()),
        "economies_in_output": int(eligible[COUNTRY_COLUMN].nunique()),
        "features": FEATURES,
        "missing_feature_values_in_output": int(
            eligible[FEATURES].isna().sum().sum()
        ),
        "respondent_ids_unique": bool(eligible[ID_COLUMN].is_unique),
        "source_data_modified": False,
    }

    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Three-dimension clustering input created.")
    print(json.dumps(report, indent=2))
    print(f"\nInput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
