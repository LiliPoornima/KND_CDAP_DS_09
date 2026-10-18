
"""Compare overlapping missingness patterns in candidate indicators."""

from pathlib import Path
from itertools import combinations

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_PATH = (
    PROJECT_ROOT / "data" / "financial_inclusion" / "raw"
    / "findex_microdata_2025_labelled_update112425.csv"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"

VARIABLES = [
    "anydigpayment",
    "con18",
    "con21",
    "saved",
    "fin24a",
    "fin24b",
]


def main():
    columns = ["economy", "economycode", *VARIABLES]
    df = pd.read_csv(RAW_PATH, usecols=columns, low_memory=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Compare which rows are missing from both variables.
    pair_rows = []

    for first, second in combinations(VARIABLES, 2):
        first_missing = df[first].isna()
        second_missing = df[second].isna()

        pair_rows.append({
            "variable_1": first,
            "variable_2": second,
            "both_missing": int((first_missing & second_missing).sum()),
            "first_missing_second_present": int(
                (first_missing & ~second_missing).sum()
            ),
            "first_present_second_missing": int(
                (~first_missing & second_missing).sum()
            ),
            "both_present": int(
                (~first_missing & ~second_missing).sum()
            ),
        })

    pair_df = pd.DataFrame(pair_rows)
    pair_path = REPORT_DIR / "fi_missingness_overlap_pairs.csv"
    pair_df.to_csv(pair_path, index=False)

    # Identify economies where each variable is entirely missing.
    economy_rows = []

    for (economy, code), group in df.groupby(
        ["economy", "economycode"], dropna=False
    ):
        for variable in VARIABLES:
            economy_rows.append({
                "economy": economy,
                "economycode": code,
                "variable": variable,
                "row_count": len(group),
                "missing_percent": round(
                    group[variable].isna().mean() * 100, 2
                ),
            })

    economy_df = pd.DataFrame(economy_rows)
    all_missing = economy_df[economy_df["missing_percent"] == 100]
    country_path = REPORT_DIR / "fi_missingness_all_missing_economies.csv"
    all_missing.to_csv(country_path, index=False)

    print("=== PAIRWISE MISSINGNESS OVERLAP ===")
    print(pair_df.to_string(index=False))

    print("\n=== ECONOMIES WITH 100% MISSINGNESS ===")
    print(all_missing.to_string(index=False))

    print(f"\nPair report: {pair_path}")
    print(f"Economy report: {country_path}")
    print("Raw data modified: False")


if __name__ == "__main__":
    main()
