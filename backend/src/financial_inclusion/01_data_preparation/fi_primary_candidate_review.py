
"""Review missingness and observed codes for primary indicator candidates."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_PATH = (
    PROJECT_ROOT / "data" / "financial_inclusion" / "raw"
    / "findex_microdata_2025_labelled_update112425.csv"
)
REGISTER_PATH = (
    PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"
    / "fi_indicator_selection_register.csv"
)
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"


def main():
    register = pd.read_csv(REGISTER_PATH)

    primary_vars = register.loc[
        register["provisional_priority"] == "PRIMARY_CANDIDATE",
        "variable",
    ].tolist()

    df = pd.read_csv(
        RAW_PATH,
        usecols=primary_vars,
        low_memory=False,
    )

    rows = []

    for variable in primary_vars:
        series = df[variable]
        missing_count = int(series.isna().sum())
        missing_percent = round(series.isna().mean() * 100, 2)

        rows.append({
            "variable": variable,
            "total_rows": len(series),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "observed_unique_values": int(series.nunique(dropna=True)),
            "observed_values": str(
                series.value_counts(dropna=False).to_dict()
            ),
        })

    result = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "fi_primary_candidate_review.csv"
    result.to_csv(output_path, index=False)

    print("PRIMARY CANDIDATE MISSINGNESS REVIEW")
    print(result.to_string(index=False))
    print(f"\nReport saved to: {output_path}")
    print("Raw data modified: False")


if __name__ == "__main__":
    main()
