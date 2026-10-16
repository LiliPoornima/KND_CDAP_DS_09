
from pathlib import Path
import argparse
import json

import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

# This script is located inside research/component_02/src/
COMPONENT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    COMPONENT_DIR
    / "data"
    / "raw"
    / "findex_microdata_2025_labelled_update112425.csv"
)

OUTPUT_DIR = COMPONENT_DIR / "reports" / "generated"


# ---------------------------------------------------------
# 2. VARIABLES TO AUDIT
# These are candidates, not final approved model features.
# ---------------------------------------------------------

DIMENSION_CANDIDATES = {
    "D1_Digital_Access_Connectivity": [
        "internet",
        "con1",
        "con9",
        "con11",
        "con12",
        "con13",
        "con14",
        "con16",
    ],
    "D2_Financial_Access": [
        "account_fin",
        "account_mob",
        "account",
        "dig_account",
        "fin2",
        "fin11a",
        "fin11b",
        "fin11c",
        "fin11d",
        "fin11e",
        "fin11f",
    ],
    "D3_Digital_Financial_Usage": [
        "anydigpayment",
        "merchantpay_dig",
        "fin3",
        "fin9a",
        "fin9b",
        "fin13b",
        "fin13d",
        "fin25e2",
        "fin26a",
        "fin26b",
        "fin28",
        "fin29",
        "fin31a",
        "fin31b",
    ],
    "D4_Financial_Resilience": [
        "saved",
        "borrowed",
        "fin17a",
        "fin17b",
        "fin17c",
        "fin24",
        "fin24a",
        "fin24b",
        "fin22a",
        "fin22b",
    ],
    "D5_Financial_Digital_Safety": [
        "fin13e",
        "fin13f",
        "fin13f_1",
        "fin14e",
        "fin25e4",
        "fin47",
        "fin48f",
    ],
}

# Useful context variables for later profiling.
CONTEXT_VARIABLES = [
    "economy",
    "economycode",
    "regionwb",
    "wgt",
    "female",
    "age",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
]


# ---------------------------------------------------------
# 3. LOAD RAW DATA
# ---------------------------------------------------------

def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            "Check that the CSV is inside data/raw/."
        )

    # Read without changing or recoding survey values.
    return pd.read_csv(input_path, low_memory=False)


# ---------------------------------------------------------
# 4. AUDIT EACH CANDIDATE VARIABLE
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit Component 02 dimension candidate variables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the original Global Findex CSV file.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)

    # Standardize column names only for matching.
    # Original dataset column names/data are not modified.
    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    audit_rows = []
    value_rows = []

    for dimension, variables in DIMENSION_CANDIDATES.items():
        for variable in variables:
            actual_column = column_lookup.get(variable.lower())

            if actual_column is None:
                audit_rows.append({
                    "dimension_candidate": dimension,
                    "variable": variable,
                    "status": "NOT FOUND",
                    "actual_column_name": None,
                    "missing_count": None,
                    "missing_pct": None,
                    "unique_nonmissing_values": None,
                })
                continue

            series = df[actual_column]
            observed_values = series.dropna().unique().tolist()

            audit_rows.append({
                "dimension_candidate": dimension,
                "variable": variable,
                "status": "FOUND",
                "actual_column_name": str(actual_column),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(
                    series.isna().mean() * 100, 2
                ),
                "unique_nonmissing_values": len(observed_values),
            })

            # Save every observed response value and its frequency.
            counts = series.value_counts(dropna=False)

            for value, count in counts.items():
                value_rows.append({
                    "dimension_candidate": dimension,
                    "variable": variable,
                    "observed_value": (
                        "<MISSING>" if pd.isna(value) else str(value)
                    ),
                    "count": int(count),
                    "percentage": round(
                        count / len(df) * 100, 3
                    ),
                })

    # Audit the context variables separately.
    context_rows = []

    for variable in CONTEXT_VARIABLES:
        actual_column = column_lookup.get(variable.lower())

        if actual_column is None:
            context_rows.append({
                "variable": variable,
                "status": "NOT FOUND",
            })
            continue

        series = df[actual_column]

        context_rows.append({
            "variable": variable,
            "status": "FOUND",
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "unique_nonmissing_values": int(series.nunique()),
        })

    audit_df = pd.DataFrame(audit_rows)
    values_df = pd.DataFrame(value_rows)
    context_df = pd.DataFrame(context_rows)

    audit_path = OUTPUT_DIR / "dimension_variable_audit.csv"
    values_path = OUTPUT_DIR / "dimension_variable_value_counts.csv"
    context_path = OUTPUT_DIR / "context_variable_audit.csv"
    summary_path = OUTPUT_DIR / "dimension_variable_audit_summary.json"

    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    values_df.to_csv(values_path, index=False, encoding="utf-8-sig")
    context_df.to_csv(context_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_file": args.input.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dimension_candidates_checked": int(len(audit_df)),
        "candidates_found": int((audit_df["status"] == "FOUND").sum()),
        "candidates_not_found": int(
            (audit_df["status"] == "NOT FOUND").sum()
        ),
        "note": (
            "Audit only. No recoding, imputation, row filtering, "
            "or dimension scoring was performed."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== COMPONENT 02: DIMENSION VARIABLE AUDIT ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Candidates checked: {len(audit_df)}")
    print(
        "Candidates found:",
        int((audit_df["status"] == "FOUND").sum()),
    )
    print(
        "Candidates not found:",
        int((audit_df["status"] == "NOT FOUND").sum()),
    )

    print("\n=== VARIABLE STATUS BY DIMENSION ===")
    print(
        audit_df.groupby(
            ["dimension_candidate", "status"]
        ).size().to_string()
    )

    print("\n=== OUTPUT FILES ===")
    print(audit_path)
    print(values_path)
    print(context_path)
    print(summary_path)


if __name__ == "__main__":
    main()
