
"""Create a transparent, provisional indicator-selection register."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"

CODEBOOK_PATH = REPORT_DIR / "fi_codebook_verification_register.csv"
OVERLAP_PATH = REPORT_DIR / "fi_variable_overlap_audit.csv"

OUTPUT_PATH = REPORT_DIR / "fi_indicator_selection_register.csv"
SUMMARY_PATH = REPORT_DIR / "fi_indicator_selection_summary.json"

# Provisional review priorities, not final model approvals.

# Provisional priorities based on codebook verification,
# missingness, country coverage, and indicator overlap.
# These are not final model approvals.

SELECTION_PLAN = {
    "con1": {
        "priority": "PRIMARY_CANDIDATE",
        "reason": "Direct indicator of personal mobile-phone access; low missingness and suitable for broad country coverage.",
        "overlap_note": "Complementary to internet use; not interchangeable.",
    },
    "internet_use": {
        "priority": "PRIMARY_CANDIDATE",
        "reason": "Constructed indicator of internet use in the past three months; complete coverage across the 140 economies.",
        "overlap_note": "Complementary to phone ownership; verify comparability across economies.",
    },
    "account": {
        "priority": "PRIMARY_CANDIDATE",
        "reason": "Broad account ownership measure covering financial institution and mobile money accounts; complete coverage across the 140 economies.",
        "overlap_note": "Overlaps with account_fin and related account indicators; prefer one broad access measure to avoid double weighting.",
    },
    "account_fin": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Constructed financial-institution account/debit-card-related measure for comparison with the broader account indicator.",
        "overlap_note": "High overlap with account; do not include both automatically.",
    },
    "dig_account": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Digitally enabled account measure for supplementary analysis.",
        "overlap_note": "Related to account ownership and digital payment indicators; inspect skip patterns and missingness.",
    },
    "account_mob": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Mobile money account/service indicator for supplementary analysis.",
        "overlap_note": "Related to dig_account and account; missingness and applicability require consideration.",
    },
    "fin2": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Debit-card ownership indicator for supplementary financial-access analysis.",
        "overlap_note": "Related to financial access; orient response codes consistently if transformed.",
    },
    "anydigpayment": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Broad digital-payment participation indicator; usable for conditional usage profiling, with 28.55% overall missingness.",
        "overlap_note": "Related to merchantpay_dig and account measures; do not double-weight overlapping indicators.",
    },
    "merchantpay_dig": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Specific digital merchant-payment indicator for supplementary usage analysis.",
        "overlap_note": "Narrower behaviour than anydigpayment; avoid automatic double weighting.",
    },
    "fin3": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Use of a card or mobile phone for specified financial transactions.",
        "overlap_note": "Conditional question; inspect eligible population and missingness.",
    },
    "fin26a": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Use of a phone or computer for bill payments.",
        "overlap_note": "Conditional account-holder question; distinct from online purchasing.",
    },
    "fin26b": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Online purchase for delivery.",
        "overlap_note": "Conditional account-holder question; related to but distinct from bill payment.",
    },
    "con18": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Phone security protection indicator; supplementary safety analysis only because overall missingness is 55.12% and coverage is limited across economies.",
        "overlap_note": "Asked only of applicable phone owners; safety score has substantial country-level missingness.",
    },
    "con19": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Ability to change phone PIN or password without help.",
        "overlap_note": "Conditional on phone security setup; high missingness limits broad comparisons.",
    },
    "con21": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Experience of scam calls or messages requesting money; supplementary digital-safety risk indicator.",
        "overlap_note": "Risk indicator rather than a protective measure; substantial missingness and limited country coverage.",
    },
    "con22": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Experience of sending money to an unknown scammer.",
        "overlap_note": "Highly conditional question with very high missingness; use only where coverage is adequate.",
    },
    "con23": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Experience of unwanted or offensive messages/media.",
        "overlap_note": "Conditional digital-safety exposure indicator; assess coverage before comparison.",
    },
    "saved": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Whether the respondent saved or set aside money in the past year; supplementary financial-resilience indicator.",
        "overlap_note": "Financial behaviour indicator, not a direct measure of emergency resilience; 28.55% overall missingness.",
    },
    "borrowed": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Whether the respondent borrowed money in the past year.",
        "overlap_note": "Borrowing is not inherently positive or negative; interpret carefully.",
    },
    "fin24": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Main source of emergency money.",
        "overlap_note": "Multi-category response; do not treat response codes as a continuous scale.",
    },
    "fin24a": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Difficulty raising emergency funds within 30 days; supplementary resilience indicator.",
        "overlap_note": "Conditional response and substantial missingness; use only after reviewing the eligible population.",
    },
    "fin24b": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Length of time the household could cover expenses after loss of main income; supplementary resilience indicator.",
        "overlap_note": "Ordered response; verify direction and special codes. The combined resilience score requires all three components.",
    },
    "con9": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Smartphone versus basic-phone ownership.",
        "overlap_note": "Conditional on phone ownership; distinct from internet use.",
    },
    "con12": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Frequency of mobile-phone use.",
        "overlap_note": "Conditional on phone ownership; verify response-code direction and missingness.",
    },
    "con13": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Codebook definition and skip logic have been verified, but approximately 98.8% of responses are missing; unsuitable as a primary indicator.",
        "overlap_note": "Retain for documentation only unless a specific analysis justifies its limited coverage.",
    },
    "con24": {
        "priority": "REVIEW_REQUIRED",
        "reason": "Observed values 3 and 4 conflict with the currently documented expected codes; their meanings remain unresolved.",
        "overlap_note": "Resolve the 298 unexpected rows before considering model use; do not recode them by assumption.",
    },
    "con25": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Internet use in a different reference period/question path.",
        "overlap_note": "Conditional follow-up; high missingness may reflect survey routing.",
    },
    "con26": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Internet-use frequency indicator.",
        "overlap_note": "Conditional follow-up; verify codes, eligibility, and coverage.",
    },
    "domestic_remittances": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Method of receiving domestic remittances.",
        "overlap_note": "Applicable only to respondents in the relevant remittance pathway.",
    },
    "pay_utilities": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Method of paying utility bills.",
        "overlap_note": "Includes no-payment and special response categories; not a simple yes/no variable.",
    },
    "receive_wages": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Method of receiving wage payments.",
        "overlap_note": "Not applicable to all respondents; distinguish no wage payment from missing.",
    },
    "receive_transfers": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Method of receiving government transfers.",
        "overlap_note": "Not applicable to all respondents; distinguish no transfer from missing.",
    },
    "receive_pensions": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Method of receiving government pension payments.",
        "overlap_note": "Not applicable to all respondents; distinguish no pension from missing.",
    },
    "receive_agriculture": {
        "priority": "SECONDARY_CANDIDATE",
        "reason": "Method of receiving agricultural payments.",
        "overlap_note": "Not applicable to all respondents; distinguish no payment from missing.",
    },
}



def run_selection_register() -> dict:
    codebook = pd.read_csv(CODEBOOK_PATH)
    overlap = pd.read_csv(OVERLAP_PATH)

    if codebook["variable"].duplicated().any():
        raise ValueError("Duplicate variable names in codebook register.")

    if set(SELECTION_PLAN) != set(codebook["variable"]):
        missing_plan = sorted(set(codebook["variable"]) - set(SELECTION_PLAN))
        extra_plan = sorted(set(SELECTION_PLAN) - set(codebook["variable"]))
        raise ValueError(
            f"Plan/register mismatch. Missing plan: {missing_plan}; "
            f"Unexpected plan entries: {extra_plan}"
        )

    result = codebook.copy()
    result["provisional_priority"] = result["variable"].map(
        {key: value["priority"] for key, value in SELECTION_PLAN.items()}
    )
    result["selection_rationale"] = result["variable"].map(
        {key: value["reason"] for key, value in SELECTION_PLAN.items()}
    )
    result["overlap_review_note"] = result["variable"].map(
        {key: value["overlap_note"] for key, value in SELECTION_PLAN.items()}
    )

    result["approved_for_primary_model"] = False
    result["final_decision"] = "PENDING"
    result["raw_data_modified"] = False

    # Surface overlap partners for each variable for review.
    overlap_map = {}
    for variable in result["variable"]:
        partners = set(
            overlap.loc[
                overlap["variable_a"].eq(variable), "variable_b"
            ].dropna().astype(str)
        )
        partners.update(
            overlap.loc[
                overlap["variable_b"].eq(variable), "variable_a"
            ].dropna().astype(str)
        )
        overlap_map[variable] = "; ".join(sorted(partners))

    result["audited_overlap_partners"] = result["variable"].map(overlap_map)
    result.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "candidate_variable_count": int(len(result)),
        "provisional_priority_counts": {
            str(key): int(value)
            for key, value in result["provisional_priority"].value_counts().items()
        },
        "approved_for_primary_model_count": 0,
        "open_codebook_issues": result.loc[
            result["verification_status"].ne("VERIFIED"), "variable"
        ].tolist(),
        "outputs": {
            "selection_register": str(OUTPUT_PATH.relative_to(PROJECT_ROOT))
        },
        "raw_data_modified": False,
        "note": (
            "This is a transparent draft for expert review. Priorities are "
            "not final model approvals; check codebook definitions, skip "
            "patterns, response coding, missingness, and dimension balance."
        ),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nDraft selection register:")
    print(
        result[
            [
                "variable", "dimension", "verification_status",
                "provisional_priority", "selection_rationale",
                "approved_for_primary_model",
            ]
        ].to_string(index=False)
    )
    return summary


if __name__ == "__main__":
    run_selection_register()
