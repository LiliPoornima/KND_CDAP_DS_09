
"""Audit candidate Global Findex indicators before dimension construction."""

import json
from pathlib import Path

import pandas as pd

from fi_data_loader import load_microdata


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"


# Preliminary codebook-informed candidates.
# Inclusion in a dimension score remains pending until approved.

CANDIDATE_VARIABLES = {
    "Connectivity": [
        "con1", "con9", "con12", "con13",
        "con24", "con25", "con26", "internet_use"
    ],

    "Financial Access": [
        "account_fin", "account_mob", "account",
        "dig_account", "fin2"
    ],

    "Digital Financial Usage": [
        "anydigpayment", "merchantpay_dig",
        "pay_utilities", "domestic_remittances",
        "receive_wages", "receive_transfers",
        "receive_pensions", "receive_agriculture",
        "fin3", "fin26a", "fin26b"
    ],

    "Digital Safety": [
        "con18", "con19", "con21", "con22", "con23"
    ],

    "Financial Resilience": [
        "saved", "borrowed", "fin24", "fin24a", "fin24b"
    ],
}

IDENTIFIER_AND_CONTEXT_VARIABLES = [
    "wpid_random",
    "economy",
    "economycode",
    "regionwb",
    "year",
    "wgt",
    "female",
    "age",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
]



def audit_variables() -> tuple[pd.DataFrame, dict]:
    """Summarise availability, missingness and observed codes for candidates."""
    df = load_microdata()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for dimension, variables in CANDIDATE_VARIABLES.items():
        for variable in variables:
            if variable not in df.columns:
                records.append(
                    {
                        "variable": variable,
                        "proposed_dimension": dimension,
                        "available": False,
                        "non_missing_count": 0,
                        "missing_count": None,
                        "missing_percent": None,
                        "observed_values": None,
                        "codebook_verified": False,
                        "include_in_primary_score": "Pending",
                        "review_note": "Variable not found in source CSV.",
                    }
                )
                continue

            series = df[variable]
            counts = series.value_counts(dropna=False)
            observed_values = sorted(
                str(value)
                for value in series.dropna().unique().tolist()
            )

            records.append(
                {
                    "variable": variable,
                    "proposed_dimension": dimension,
                    "available": True,
                    "non_missing_count": int(series.notna().sum()),
                    "missing_count": int(series.isna().sum()),
                    "missing_percent": round(
                        float(series.isna().mean() * 100), 2
                    ),
                    "observed_values": json.dumps(observed_values),
                    "codebook_verified": False,
                    "include_in_primary_score": "Pending",
                    "review_note": (
                        "Candidate only: verify official definition, "
                        "valid codes, direction and skip pattern."
                    ),
                }
            )

    audit_df = pd.DataFrame(records)
    audit_path = REPORT_DIR / "fi_variable_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    context_records = []
    for variable in IDENTIFIER_AND_CONTEXT_VARIABLES:
        if variable in df.columns:
            context_records.append(
                {
                    "variable": variable,
                    "available": True,
                    "missing_count": int(df[variable].isna().sum()),
                    "unique_non_missing_values": int(
                        df[variable].nunique(dropna=True)
                    ),
                }
            )
        else:
            context_records.append(
                {
                    "variable": variable,
                    "available": False,
                    "missing_count": None,
                    "unique_non_missing_values": None,
                }
            )

    context_df = pd.DataFrame(context_records)
    context_path = REPORT_DIR / "fi_context_variable_audit.csv"
    context_df.to_csv(context_path, index=False)

    summary = {
        "row_count": int(len(df)),
        "source_column_count": int(len(df.columns)),
        "candidate_count": int(len(audit_df)),
        "candidate_variables_found": int(audit_df["available"].sum()),
        "candidate_variables_not_found": int((~audit_df["available"]).sum()),
        "codebook_verified_count": 0,
        "primary_score_variables_approved": 0,
        "candidate_audit_file": str(audit_path.relative_to(PROJECT_ROOT)),
        "context_audit_file": str(context_path.relative_to(PROJECT_ROOT)),
        "note": (
            "Candidate variables are not approved for scoring until "
            "their official definitions, codes, direction and skip "
            "patterns are reviewed."
        ),
    }

    summary_path = REPORT_DIR / "fi_variable_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nCandidate variables by proposed dimension:")
    print(
        audit_df.groupby("proposed_dimension")["available"]
        .agg(["count", "sum"])
        .to_string()
    )

    return audit_df, summary


if __name__ == "__main__":
    audit_variables()
