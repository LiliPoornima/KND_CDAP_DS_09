
"""Prepare and validate the five-dimensional clustering input."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_dimension_dataset.csv"
)

OUTPUT_DIR = ROOT / "data" / "financial_inclusion" / "processed"
REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports"

CLUSTERING_FILE = OUTPUT_DIR / "fi_clustering_input.csv"
REPORT_FILE = REPORT_DIR / "fi_clustering_input_report.json"

ID_COLUMN = "wpid_random"
COUNTRY_COLUMN = "economy"

SCORE_COLUMNS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [ID_COLUMN, COUNTRY_COLUMN, *SCORE_COLUMNS]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[ID_COLUMN].isna().any():
        raise ValueError("Missing respondent IDs found.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate respondent IDs found.")

    # Only respondents with all five scores are eligible for this input.
    complete_mask = df[SCORE_COLUMNS].notna().all(axis=1)

    # Preserve identifiers and economy for interpreting cluster results.
    # They are metadata, NOT clustering features.
    clustering_input = df.loc[
        complete_mask, [ID_COLUMN, COUNTRY_COLUMN, *SCORE_COLUMNS]
    ].copy()

    # Check numerical validity of the five score features.
    for column in SCORE_COLUMNS:
        clustering_input[column] = pd.to_numeric(
            clustering_input[column], errors="raise"
        )

        if not clustering_input[column].between(0, 1).all():
            raise ValueError(
                f"Out-of-range score detected in {column}; expected [0, 1]."
            )

    if clustering_input[SCORE_COLUMNS].isna().any().any():
        raise ValueError("Missing values remain in clustering features.")

    if clustering_input.empty:
        raise ValueError("No complete respondents available for clustering.")

    clustering_input.to_csv(CLUSTERING_FILE, index=False)

    report = {
        "source_file": str(INPUT_FILE.relative_to(ROOT)),
        "output_file": str(CLUSTERING_FILE.relative_to(ROOT)),
        "source_respondents": int(len(df)),
        "eligible_respondents": int(len(clustering_input)),
        "excluded_for_missing_scores": int((~complete_mask).sum()),
        "eligible_percent": round(
            float(len(clustering_input) / len(df) * 100), 2
        ),
        "source_economies": int(df[COUNTRY_COLUMN].nunique()),
        "eligible_economies": int(
            clustering_input[COUNTRY_COLUMN].nunique()
        ),
        "feature_columns": SCORE_COLUMNS,
        "metadata_columns": [ID_COLUMN, COUNTRY_COLUMN],
        "missing_values_in_features": int(
            clustering_input[SCORE_COLUMNS].isna().sum().sum()
        ),
        "all_features_within_zero_one": True,
        "rows_dropped_from_source": 0,
        "imputation_performed": False,
        "source_data_modified": False,
        "standardisation_applied": False,
        "note": (
            "This is a separate complete-case clustering input. "
            "IDs and economy are metadata only. Standardisation and "
            "clustering have not yet been performed."
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Clustering input preparation completed.")
    print(json.dumps(report, indent=2))
    print(f"\nInput CSV: {CLUSTERING_FILE}")
    print(f"Report JSON: {REPORT_FILE}")


if __name__ == "__main__":
    main()

