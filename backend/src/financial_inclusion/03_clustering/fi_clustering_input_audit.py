
"""Audit dimension-score availability before clustering."""

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

REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports" / "03_clustering"

ECONOMY_REPORT = REPORT_DIR / "fi_clustering_input_by_economy.csv"
SUMMARY_REPORT = REPORT_DIR / "fi_clustering_input_audit_summary.json"

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

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = ["wpid_random", "economy", *SCORE_COLUMNS]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["wpid_random"].isna().any():
        raise ValueError("Missing respondent IDs found.")

    if df["wpid_random"].duplicated().any():
        raise ValueError("Duplicate respondent IDs found.")

    # A respondent is complete only if all five dimension scores exist.
    complete_mask = df[SCORE_COLUMNS].notna().all(axis=1)

    # Summarise the usable sample separately for each economy.
    economy_report = (
        df.assign(all_five_scores_available=complete_mask)
        .groupby("economy", dropna=False)
        .agg(
            respondents=("wpid_random", "size"),
            respondents_with_all_five_scores=(
                "all_five_scores_available", "sum"
            ),
        )
        .reset_index()
    )

    economy_report["complete_percent"] = (
        100
        * economy_report["respondents_with_all_five_scores"]
        / economy_report["respondents"]
    ).round(2)

    economy_report["incomplete_respondents"] = (
        economy_report["respondents"]
        - economy_report["respondents_with_all_five_scores"]
    )

    economy_report.to_csv(ECONOMY_REPORT, index=False)

    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "total_respondents": int(len(df)),
        "total_economies": int(df["economy"].nunique()),
        "score_columns": SCORE_COLUMNS,
        "respondents_with_all_five_scores": int(complete_mask.sum()),
        "respondents_missing_one_or_more_scores": int(
            (~complete_mask).sum()
        ),
        "complete_case_percent": round(
            float(complete_mask.mean() * 100), 2
        ),
        "incomplete_case_percent": round(
            float((~complete_mask).mean() * 100), 2
        ),
        "missing_percent_by_score": {
            column: round(float(df[column].isna().mean() * 100), 2)
            for column in SCORE_COLUMNS
        },
        "economies_with_zero_complete_cases": int(
            (economy_report["respondents_with_all_five_scores"] == 0).sum()
        ),
        "rows_dropped": 0,
        "imputation_performed": False,
        "input_data_modified": False,
        "economy_report": str(ECONOMY_REPORT.relative_to(ROOT)),
        "note": (
            "Audit only. No rows were removed and no values were imputed. "
            "Complete-case counts are descriptive and do not determine "
            "the final clustering missing-data strategy."
        ),
    }

    SUMMARY_REPORT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Clustering input audit completed.")
    print(json.dumps(summary, indent=2))
    print(f"\nEconomy report: {ECONOMY_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    main()
