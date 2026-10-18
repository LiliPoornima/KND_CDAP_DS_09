
"""Investigate unexpected con24 response codes without modifying raw data."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = (
    PROJECT_ROOT / "data" / "financial_inclusion" / "raw"
    / "findex_microdata_2025_labelled_update112425.csv"
)
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"


def main():
    df = pd.read_csv(
        RAW_PATH,
        usecols=["economy", "economycode", "con1", "con24", "internet_use"],
        low_memory=False,
    )

    # Inspect only the unexpected con24 codes; do not change the data.
    issues = df[df["con24"].isin([3, 4])].copy()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    by_economy = (
        issues.groupby(["economy", "economycode", "con24"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["con24", "row_count"], ascending=[True, False])
    )

    output_path = REPORT_DIR / "fi_con24_unexpected_codes_by_economy.csv"
    by_economy.to_csv(output_path, index=False)

    print("Unexpected con24 code counts:")
    print(issues["con24"].value_counts().sort_index().to_string())

    print("\nCounts by economy and code:")
    print(by_economy.to_string(index=False))

    print("\nExamples for manual review:")
    print(
        issues[
            ["economy", "economycode", "con1", "con24", "internet_use"]
        ].head(15).to_string(index=False)
    )

    print(f"\nReport saved to: {output_path}")
    print("Raw data modified: False")


if __name__ == "__main__":
    main()

