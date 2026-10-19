
"""Audit combinations of available dimension scores."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_dimension_dataset.csv"
)

REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports" / "02_dimension_creation"

SCORE_COLUMNS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    missing_columns = [
        column for column in SCORE_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing score columns: {missing_columns}")

    availability = df[SCORE_COLUMNS].notna()

    # Count respondents who have each exact combination of available scores.
    patterns = availability.astype(int).astype(str).agg("".join, axis=1)

    pattern_report = (
        patterns.value_counts()
        .rename_axis("availability_pattern")
        .reset_index(name="respondents")
    )

    pattern_report["percent"] = (
        100 * pattern_report["respondents"] / len(df)
    ).round(2)

    pattern_report.to_csv(
        REPORT_DIR / "fi_dimension_availability_patterns.csv",
        index=False,
    )

    # Evaluate sample sizes for several possible dimension sets.
    candidate_sets = {
        "all_five_dimensions": SCORE_COLUMNS,
        "without_digital_safety": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_financial_resilience",
        ],
        "connectivity_access_usage": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
        ],
        "connectivity_and_access": [
            "score_connectivity",
            "score_financial_access",
        ],
    }

    comparisons = []

    for name, columns in candidate_sets.items():
        complete = df[columns].notna().all(axis=1)

        # Count economies with at least one complete respondent.
        complete_by_economy = (
            df.assign(_complete=complete)
            .groupby("economy")["_complete"]
            .sum()
        )

        comparisons.append(
            {
                "candidate_set": name,
                "dimensions_used": len(columns),
                "respondents_available": int(complete.sum()),
                "percent_of_full_sample": round(
                    float(complete.mean() * 100), 2
                ),
                "economies_with_respondents": int(
                    (complete_by_economy > 0).sum()
                ),
                "economies_with_zero_respondents": int(
                    (complete_by_economy == 0).sum()
                ),
            }
        )

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(
        REPORT_DIR / "fi_dimension_set_comparison.csv",
        index=False,
    )

    summary = {
        "total_respondents": int(len(df)),
        "total_economies": int(df["economy"].nunique()),
        "score_missing_percent": {
            col: round(float(df[col].isna().mean() * 100), 2)
            for col in SCORE_COLUMNS
        },
        "availability_patterns_file": (
            "outputs/financial_inclusion/reports/02_dimension_creation/"
            "fi_dimension_availability_patterns.csv"
        ),
        "dimension_set_comparison_file": (
            "outputs/financial_inclusion/reports/02_dimension_creation/"
            "fi_dimension_set_comparison.csv"
        ),
        "rows_dropped": 0,
        "imputation_performed": False,
        "input_data_modified": False,
    }

    (REPORT_DIR / "fi_dimension_availability_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Dimension availability audit completed.")
    print("\n=== DIMENSION SET COMPARISON ===")
    print(comparison_df.to_string(index=False))
    print("\n=== SCORE MISSINGNESS ===")
    print(json.dumps(summary["score_missing_percent"], indent=2))
    print("\nNo rows dropped; no imputation; input unchanged.")


if __name__ == "__main__":
    main()
