
"""Validate connectivity response codes for Component 02."""

import json
from pathlib import Path

import pandas as pd

from fi_data_loader import load_microdata

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"

# Codes documented in the supplied Global Findex 2025 codebook.
EXPECTED_CODES = {
    "con24": {1, 2, 8, 9},
    "con25": {1, 2, 8, 9},
    "con26": {1, 2, 3, 4, 7, 8, 9},
    "internet_use": {0, 1},
}


def validate_connectivity() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_microdata()

    required = [
        "economy", "economycode", "wpid_random",
        "internet_use", "con24", "con25", "con26",
    ]
    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Required connectivity columns are missing: {missing_columns}"
        )

    # Keep source values unchanged; only identify unexpected codes.
    unexpected_frames = []

    for variable, expected in EXPECTED_CODES.items():
        unexpected_mask = (
            df[variable].notna()
            & ~df[variable].isin(expected)
        )

        if unexpected_mask.any():
            flagged = df.loc[
                unexpected_mask,
                ["economy", "economycode", "wpid_random", variable],
            ].copy()

            flagged = flagged.rename(
                columns={variable: "observed_value"}
            )
            flagged.insert(2, "variable", variable)
            flagged["issue"] = "CODE_NOT_LISTED_IN_CODEBOOK"
            unexpected_frames.append(flagged)

    if unexpected_frames:
        unexpected = pd.concat(
            unexpected_frames, ignore_index=True
        )
    else:
        unexpected = pd.DataFrame(
            columns=[
                "economy", "economycode", "variable",
                "observed_value", "wpid_random", "issue",
            ]
        )

    unexpected_path = REPORT_DIR / "fi_connectivity_unexpected_codes.csv"
    unexpected.to_csv(unexpected_path, index=False)

    # Record each variable's observed values, including missing values.
    distribution_rows = []

    for variable in EXPECTED_CODES:
        counts = df[variable].value_counts(dropna=False)

        for value, count in counts.items():
            distribution_rows.append({
                "variable": variable,
                "observed_value": (
                    "<MISSING>" if pd.isna(value) else str(value)
                ),
                "count": int(count),
                "percent_of_all_rows": round(
                    int(count) / len(df) * 100, 2
                ),
            })

    distribution = pd.DataFrame(distribution_rows)
    distribution_path = REPORT_DIR / "fi_connectivity_distributions.csv"
    distribution.to_csv(distribution_path, index=False)

    # Compare the two internet indicators without assuming they are equivalent.
    comparison = pd.crosstab(
        df["internet_use"],
        df["con25"],
        dropna=False,
    )
    comparison_path = REPORT_DIR / "fi_internet_use_con25_crosstab.csv"
    comparison.to_csv(comparison_path)

    summary = {
        "row_count": int(len(df)),
        "codebook_expected_codes": {
            variable: sorted(codes)
            for variable, codes in EXPECTED_CODES.items()
        },
        "unexpected_code_row_count": int(len(unexpected)),
        "unexpected_code_counts_by_variable": {
            variable: int(
                (
                    df[variable].notna()
                    & ~df[variable].isin(expected)
                ).sum()
            )
            for variable, expected in EXPECTED_CODES.items()
        },
        "internet_use_values": {
            str(value): int(count)
            for value, count in df["internet_use"]
            .value_counts(dropna=False).items()
        },
        "unexpected_codes_file": str(
            unexpected_path.relative_to(PROJECT_ROOT)
        ),
        "distribution_file": str(
            distribution_path.relative_to(PROJECT_ROOT)
        ),
        "comparison_file": str(
            comparison_path.relative_to(PROJECT_ROOT)
        ),
        "raw_data_modified": False,
        "note": (
            "Unexpected codes are flagged for investigation, not recoded. "
            "The internet_use and con25 variables are compared but are "
            "not assumed to be equivalent."
        ),
    }

    summary_path = REPORT_DIR / "fi_connectivity_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    validate_connectivity()
