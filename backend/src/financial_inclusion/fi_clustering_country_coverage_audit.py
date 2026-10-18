
"""Audit economy-level representation in the clustering sample."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

FULL_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_dimension_dataset.csv"
)

CLUSTERING_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_clustering_input.csv"
)

REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports"

COUNTRY_REPORT = REPORT_DIR / "fi_clustering_country_coverage.csv"
SUMMARY_REPORT = REPORT_DIR / "fi_clustering_country_coverage_summary.json"

SCORE_COLUMNS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def main():
    for path in [FULL_FILE, CLUSTERING_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(FULL_FILE, low_memory=False)
    eligible = pd.read_csv(CLUSTERING_FILE, low_memory=False)

    required = ["wpid_random", "economy", *SCORE_COLUMNS]

    for name, df in [("full", full), ("eligible", eligible)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

    if full["wpid_random"].duplicated().any():
        raise ValueError("Duplicate IDs in full dataset.")

    if eligible["wpid_random"].duplicated().any():
        raise ValueError("Duplicate IDs in clustering dataset.")

    # Verify that clustering respondents are a subset of the full dataset.
    eligible_ids = set(eligible["wpid_random"])
    if not eligible_ids.issubset(set(full["wpid_random"])):
        raise ValueError("Clustering dataset contains unknown respondent IDs.")

    full_counts = (
        full.groupby("economy")
        .agg(total_respondents=("wpid_random", "size"))
    )

    eligible_counts = (
        eligible.groupby("economy")
        .agg(eligible_respondents=("wpid_random", "size"))
    )

    report = full_counts.join(eligible_counts, how="left").fillna(
        {"eligible_respondents": 0}
    )

    report["eligible_respondents"] = (
        report["eligible_respondents"].astype(int)
    )

    report["excluded_respondents"] = (
        report["total_respondents"] - report["eligible_respondents"]
    )

    report["eligible_percent"] = (
        100
        * report["eligible_respondents"]
        / report["total_respondents"]
    ).round(2)

    report = report.reset_index().sort_values("eligible_percent")

    report.to_csv(COUNTRY_REPORT, index=False)

    summary = {
        "total_respondents": int(len(full)),
        "eligible_respondents": int(len(eligible)),
        "overall_eligible_percent": round(
            100 * len(eligible) / len(full), 2
        ),
        "total_economies": int(full["economy"].nunique()),
        "economies_with_eligible_respondents": int(
            (report["eligible_respondents"] > 0).sum()
        ),
        "economies_with_zero_eligible_respondents": int(
            (report["eligible_respondents"] == 0).sum()
        ),
        "economies_with_under_10_percent_eligible": int(
            (report["eligible_percent"] < 10).sum()
        ),
        "economies_with_under_25_percent_eligible": int(
            (report["eligible_percent"] < 25).sum()
        ),
        "economies_with_at_least_50_percent_eligible": int(
            (report["eligible_percent"] >= 50).sum()
        ),
        "lowest_eligible_percent": float(report["eligible_percent"].min()),
        "median_economy_eligible_percent": float(
            report["eligible_percent"].median()
        ),
        "highest_eligible_percent": float(report["eligible_percent"].max()),
        "country_report": str(COUNTRY_REPORT.relative_to(ROOT)),
        "source_data_modified": False,
    }

    SUMMARY_REPORT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Country coverage audit completed.")
    print("\n=== COVERAGE SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== 15 ECONOMIES WITH LOWEST ELIGIBILITY ===")
    print(report.head(15).to_string(index=False))

    print("\n=== 10 ECONOMIES WITH HIGHEST ELIGIBILITY ===")
    print(report.tail(10).sort_values(
        "eligible_percent", ascending=False
    ).to_string(index=False))

    print(f"\nCountry report: {COUNTRY_REPORT}")
    print(f"Summary report: {SUMMARY_REPORT}")


if __name__ == "__main__":
    main()
