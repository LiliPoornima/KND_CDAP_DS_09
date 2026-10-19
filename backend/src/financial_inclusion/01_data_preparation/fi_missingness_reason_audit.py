
"""Investigate missingness patterns without changing source data."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_PATH = (
    PROJECT_ROOT / "data" / "financial_inclusion" / "raw"
    / "findex_microdata_2025_labelled_update112425.csv"
)
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"

VARIABLES = [
    "anydigpayment", "con18", "con21",
    "saved", "fin24a", "fin24b",
]


def main():
    columns = [
        "economy", "economycode", "con1", "account",
        "internet_use", *VARIABLES,
    ]

    df = pd.read_csv(RAW_PATH, usecols=columns, low_memory=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Missingness by economy
    economy_rows = []

    for (economy, code), group in df.groupby(
        ["economy", "economycode"], dropna=False
    ):
        for variable in VARIABLES:
            economy_rows.append({
                "economy": economy,
                "economycode": code,
                "variable": variable,
                "total_rows": len(group),
                "missing_count": int(group[variable].isna().sum()),
                "missing_percent": round(
                    group[variable].isna().mean() * 100, 2
                ),
            })

    by_economy = pd.DataFrame(economy_rows)
    by_economy.to_csv(
        REPORT_DIR / "fi_missingness_reason_by_economy.csv",
        index=False,
    )

    # 2. Compare missingness with phone ownership and account status
    group_rows = []

    for variable in VARIABLES:
        for group_variable in ["con1", "account", "internet_use"]:
            summary = (
                df.groupby(group_variable, dropna=False)[variable]
                .agg(
                    total_rows="size",
                    missing_count=lambda s: int(s.isna().sum()),
                )
                .reset_index()
            )
            summary["variable"] = variable
            summary["group_variable"] = group_variable
            summary["missing_percent"] = (
                summary["missing_count"] / summary["total_rows"] * 100
            ).round(2)
            group_rows.append(summary)

    by_group = pd.concat(group_rows, ignore_index=True)
    by_group.to_csv(
        REPORT_DIR / "fi_missingness_reason_by_group.csv",
        index=False,
    )

    print("=== MISSINGNESS BY ECONOMY ===")
    print(
        by_economy.sort_values(
            ["variable", "missing_percent"],
            ascending=[True, False],
        ).groupby("variable", sort=False).head(5).to_string(index=False)
    )

    print("\n=== MISSINGNESS BY RESPONDENT GROUP ===")
    print(by_group.to_string(index=False))

    print(f"\nReports saved in: {REPORT_DIR}")
    print("Raw data modified: False")


if __name__ == "__main__":
    main()
