
"""Audit overlap between candidate indicators for Component 02."""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from fi_data_loader import load_microdata

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"

OUTPUT_PATH = REPORT_DIR / "fi_variable_overlap_audit.csv"
SUMMARY_PATH = REPORT_DIR / "fi_variable_overlap_audit_summary.json"

# Conceptually related groups to review.
VARIABLE_GROUPS = {
    "Connectivity": ["con1", "internet_use"],
    "Financial_Access": [
        "account_fin", "account", "account_mob", "dig_account", "fin2"
    ],
    "Digital_Usage": [
        "anydigpayment", "merchantpay_dig", "fin3", "fin26a", "fin26b",
        "pay_utilities", "domestic_remittances", "receive_wages",
        "receive_transfers", "receive_pensions", "receive_agriculture",
    ],
    "Digital_Safety": ["con18", "con19", "con21", "con22", "con23"],
    "Financial_Resilience": ["saved", "borrowed", "fin24", "fin24a", "fin24b"],
}


def run_overlap_audit() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_microdata()

    rows = []
    missing_variables = []

    for group_name, variables in VARIABLE_GROUPS.items():
        absent = [v for v in variables if v not in df.columns]
        missing_variables.extend(absent)

        available = [v for v in variables if v in df.columns]

        for var_a, var_b in combinations(available, 2):
            pair = df[[var_a, var_b]].dropna()
            n = len(pair)

            result = {
                "group": group_name,
                "variable_a": var_a,
                "variable_b": var_b,
                "complete_pair_count": int(n),
                "excluded_pair_rows_due_to_missing": int(len(df) - n),
                "exact_match_percent": None,
                "pearson_correlation": None,
                "association_note": "",
            }

            if n > 0:
                result["exact_match_percent"] = round(
                    float((pair[var_a] == pair[var_b]).mean() * 100), 2
                )

                # Correlation is meaningful here only as a descriptive
                # comparison for numeric-coded variables; not proof of equivalence.
                a_numeric = pd.to_numeric(pair[var_a], errors="coerce")
                b_numeric = pd.to_numeric(pair[var_b], errors="coerce")
                valid = a_numeric.notna() & b_numeric.notna()

                if valid.sum() > 1 and a_numeric[valid].nunique() > 1 \
                        and b_numeric[valid].nunique() > 1:
                    result["pearson_correlation"] = round(
                        float(a_numeric[valid].corr(b_numeric[valid])), 4
                    )

                result["association_note"] = (
                    "Descriptive only. Different code meanings or skip "
                    "patterns may make direct comparison inappropriate."
                )

            rows.append(result)

    if missing_variables:
        raise ValueError(
            f"Variables missing from the source dataset: {sorted(set(missing_variables))}"
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "row_count": int(len(df)),
        "group_count": int(len(VARIABLE_GROUPS)),
        "variable_pair_count": int(len(audit)),
        "pairs_with_comparable_rows": int(
            (audit["complete_pair_count"] > 0).sum()
        ),
        "outputs": {
            "pair_audit": str(OUTPUT_PATH.relative_to(PROJECT_ROOT))
        },
        "raw_data_modified": False,
        "variables_removed": False,
        "variables_approved_for_model": 0,
        "note": (
            "Exact-match percentages and correlations are descriptive "
            "screening results, not automatic grounds for dropping or "
            "approving variables. Review codebook meanings and skip logic."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nHighest exact-match pairs:")
    print(
        audit.dropna(subset=["exact_match_percent"])
        .sort_values("exact_match_percent", ascending=False)
        .head(15)[
            [
                "group", "variable_a", "variable_b",
                "complete_pair_count", "exact_match_percent",
                "pearson_correlation",
            ]
        ]
        .to_string(index=False)
    )

    return summary


if __name__ == "__main__":
    run_overlap_audit()
