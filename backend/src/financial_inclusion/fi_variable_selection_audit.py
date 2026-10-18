
"""Combine codebook review and missingness evidence for variable selection."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"

REGISTER_PATH = REPORT_DIR / "fi_codebook_verification_register.csv"
MISSINGNESS_PATH = REPORT_DIR / "fi_missingness_by_variable.csv"
OUTPUT_PATH = REPORT_DIR / "fi_variable_selection_audit.csv"
SUMMARY_PATH = REPORT_DIR / "fi_variable_selection_audit_summary.json"


def run_variable_selection_audit() -> dict:
    register = pd.read_csv(REGISTER_PATH)
    missingness = pd.read_csv(MISSINGNESS_PATH)

    audit = register.merge(
        missingness[
            ["variable", "missing_count", "non_missing_count", "missing_percent"]
        ],
        on="variable",
        how="left",
        validate="one_to_one",
    )

    if audit["missing_percent"].isna().any():
        missing_vars = audit.loc[
            audit["missing_percent"].isna(), "variable"
        ].tolist()
        raise ValueError(f"Missingness results unavailable for: {missing_vars}")

    def classify(row):
        variable = row["variable"]
        status = row["verification_status"]
        missing_percent = float(row["missing_percent"])

        # These decisions are provisional; no variable is approved for scoring.
        if status != "VERIFIED":
            return "REVIEW_REQUIRED"

        if variable in {
            "con9", "con12", "con13", "con18", "con19",
            "con21", "con22", "con23", "con25", "con26",
            "account_mob", "dig_account", "anydigpayment",
            "merchantpay_dig", "fin3", "fin26a", "fin26b",
        }:
            return "CONDITIONAL_CANDIDATE"

        if missing_percent >= 50:
            return "COVERAGE_REVIEW"

        return "PRIMARY_CANDIDATE"

    audit["preliminary_selection"] = audit.apply(classify, axis=1)
    audit["approved_for_primary_model"] = False
    audit["decision_note"] = (
        "Provisional only: verify codebook wording, skip logic, "
        "response codes, and cross-economy comparability before selection."
    )

    audit.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "candidate_variable_count": int(len(audit)),
        "preliminary_selection_counts": {
            str(key): int(value)
            for key, value in audit["preliminary_selection"]
            .value_counts()
            .items()
        },
        "approved_for_primary_model_count": int(
            audit["approved_for_primary_model"].sum()
        ),
        "outputs": {
            "selection_audit": str(OUTPUT_PATH.relative_to(PROJECT_ROOT))
        },
        "raw_data_modified": False,
        "imputation_performed": False,
        "note": (
            "Categories are preliminary review priorities, not final "
            "scientific decisions. High missingness alone does not justify "
            "excluding a variable."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nVariables requiring review:")
    print(
        audit[
            [
                "variable",
                "dimension",
                "verification_status",
                "missing_percent",
                "preliminary_selection",
            ]
        ]
        .sort_values(["preliminary_selection", "dimension", "variable"])
        .to_string(index=False)
    )

    return summary


if __name__ == "__main__":
    run_variable_selection_audit()
