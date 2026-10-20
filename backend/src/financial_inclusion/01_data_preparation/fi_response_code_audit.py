
"""Audit observed response codes for Component 02 candidate variables."""

from pathlib import Path
import json

import pandas as pd

from fi_data_loader import load_microdata

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"

CANDIDATE_VARIABLES = [
    "con1", "con9", "con12", "con13", "con24", "con25",
    "con26", "internet_use",
    "account_fin", "account_mob", "account", "dig_account", "fin2",
    "anydigpayment", "merchantpay_dig", "pay_utilities",
    "domestic_remittances", "receive_wages", "receive_transfers",
    "receive_pensions", "receive_agriculture", "fin3", "fin26a", "fin26b",
    "con18", "con19", "con21", "con22", "con23",
    "saved", "borrowed", "fin24", "fin24a", "fin24b",
]


def audit_response_codes() -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_microdata()
    rows = []

    for variable in CANDIDATE_VARIABLES:
        if variable not in df.columns:
            rows.append({
                "variable": variable,
                "observed_value": None,
                "count": None,
                "percent_of_all_rows": None,
                "status": "VARIABLE_NOT_FOUND",
            })
            continue

        counts = df[variable].value_counts(dropna=False, sort=False)

        for value, count in counts.items():
            is_missing = pd.isna(value)

            rows.append({
                "variable": variable,
                "observed_value": (
                    "<MISSING>" if is_missing else str(value)
                ),
                "count": int(count),
                "percent_of_all_rows": round(
                    int(count) / len(df) * 100, 2
                ),
                "status": "MISSING" if is_missing else "REVIEW_CODEBOOK",
            })

    result = pd.DataFrame(rows)
    output_path = REPORT_DIR / "fi_response_code_audit.csv"
    result.to_csv(output_path, index=False)

    summary = {
        "row_count": int(len(df)),
        "candidate_count": len(CANDIDATE_VARIABLES),
        "variables_found": int(
            result.loc[
                result["status"] != "VARIABLE_NOT_FOUND", "variable"
            ].nunique()
        ),
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "note": (
            "Observed values are not automatically classified as valid, "
            "invalid, positive, negative, or missing-by-design."
        ),
    }

    summary_path = REPORT_DIR / "fi_response_code_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nResponse-code audit saved to:", output_path)

    return result


if __name__ == "__main__":
    audit_response_codes()
