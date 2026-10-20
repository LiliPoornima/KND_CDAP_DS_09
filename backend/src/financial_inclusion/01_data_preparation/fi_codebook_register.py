
"""Create the codebook verification register for Component 02."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"

# Only definitions reviewed so far are marked VERIFIED.
# Values are recorded as documented; no source data is recoded here.
REGISTER = [
    {
        "variable": "con1",
        "dimension": "Connectivity",
        "meaning": "Personal mobile phone ownership",
        "codes": "1=yes; 2=no; 3=don't know; 4=refused",
        "skip_condition": "None recorded in this register",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "con9",
        "dimension": "Connectivity",
        "meaning": "Smartphone or basic text phone",
        "codes": "1=smartphone; 2=basic text phone; 8=don't know; 9=refused",
        "skip_condition": "Asked only of mobile phone owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "con12",
        "dimension": "Connectivity",
        "meaning": "Frequency of mobile phone use",
        "codes": "1=daily; 2=weekly; 3=monthly; 4=less than monthly; 7=never; 8=don't know; 9=refused",
        "skip_condition": "Asked only of mobile phone owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "con13",
        "dimension": "Connectivity",
        "meaning": "Mobile phone use in the past three months",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": (
            "Asked of phone owners who did not report using their "
            "mobile phone daily, weekly, or monthly"
        ),
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": (
            "Codebook definition and observed skip pattern are consistent. "
            "High missingness (98.80%) reflects the conditional question "
            "universe; review suitability before model inclusion."
        ),
    },
    {
        "variable": "con24",
        "dimension": "Connectivity",
        "meaning": "Internet use in the past seven days",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "",
        "verification_status": "VERIFIED_WITH_ISSUE",
        "primary_model_status": "PENDING",
        "issue": (
            "Online World Bank data dictionary confirms 298 records with "
            "codes 3 and 4, but their category labels are blank. The PDF "
            "codebook lists only 1, 2, 8 and 9. The meanings of codes 3 "
            "and 4 remain unconfirmed; do not recode without authoritative "
            "documentation."
        ),
    },
    {
        "variable": "con25",
        "dimension": "Connectivity",
        "meaning": "Internet use in the past three months",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "Asked only of respondents who did not report internet use in the past seven days",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "con26",
        "dimension": "Connectivity",
        "meaning": "Frequency of internet use",
        "codes": "1=daily; 2=weekly; 3=monthly; 4=less than monthly; 7=never; 8=don't know; 9=refused",
        "skip_condition": "Asked only of respondents who used the internet in the past three months",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "internet_use",
        "dimension": "Connectivity",
        "meaning": (
            "Used the internet within the past three months, "
            "corresponding to the Global Findex 2025 codebook's "
            "constructed variable named internet."
        ),
        "codes": (
            "1 = Used the internet in the past three months; "
            "0 = Did not use the internet in the past three months"
        ),
        "skip_condition": (
            "Constructed variable; based on the internet-use definition "
            "in the Global Findex 2025 codebook."
        ),
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": (
            "Codebook variable name is 'internet', while the source CSV "
            "column is 'internet_use'. The names differ, but the definition "
            "and observed 0/1 coding correspond. Retain 'internet_use' as "
            "the source column name; do not rename or modify the raw data."
        ),
    },
    {
        "variable": "account_fin",
        "dimension": "Financial Access",
        "meaning": "Financial institution account or debit card",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "account_mob",
        "dimension": "Financial Access",
        "meaning": "Mobile money service use in the past year",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable; additional coverage noted in codebook",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "62,166 missing values; investigate availability before modelling",
    },
    {
        "variable": "account",
        "dimension": "Financial Access",
        "meaning": "Financial institution account or mobile money account",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "dig_account",
        "dimension": "Financial Access",
        "meaning": "Account used to make digital payments by card or phone",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "41,136 missing values; investigate before modelling",
    },
    {
        "variable": "fin2",
        "dimension": "Financial Access",
        "meaning": "ATM/debit card ownership",
        "codes": "1=yes; 2=no; 3=don't know; 4=refused",
        "skip_condition": "",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "anydigpayment",
        "dimension": "Digital Financial Usage",
        "meaning": "Made or received a digital payment in the past year",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "41,136 missing values; investigate before modelling",
    },
    {
        "variable": "merchantpay_dig",
        "dimension": "Digital Financial Usage",
        "meaning": "Made a digital merchant payment in the past year",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "41,136 missing values; investigate before modelling",
    },
    {
        "variable": "pay_utilities",
        "dimension": "Digital Financial Usage",
        "meaning": "Method of utility bill payment",
        "codes": "1=from account; 2=cash only; 3=other method; 4=no payment; 5=don't know/refused",
        "skip_condition": "Check question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "domestic_remittances",
        "dimension": "Digital Financial Usage",
        "meaning": "Method of domestic remittance",
        "codes": "1=via account; 2=other method; 3=no remittance; 4=don't know/refused",
        "skip_condition": "Check question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "receive_wages",
        "dimension": "Digital Financial Usage",
        "meaning": "Method of wage payment",
        "codes": "1=account; 2=cash only; 3=other method; 4=no wage payment; 5=don't know/refused",
        "skip_condition": "Check question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "receive_transfers",
        "dimension": "Digital Financial Usage",
        "meaning": "Method of government transfer payment",
        "codes": "1=account; 2=cash only; 3=other method; 4=no transfer; 5=don't know/refused",
        "skip_condition": "Check question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "receive_pensions",
        "dimension": "Digital Financial Usage",
        "meaning": "Method of government pension payment",
        "codes": "1=account; 2=cash only; 3=other method; 4=no pension; 5=don't know/refused",
        "skip_condition": "Check question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "receive_agriculture",
        "dimension": "Digital Financial Usage",
        "meaning": "Method of agricultural payment",
        "codes": "1=account; 2=cash only; 3=other method; 4=no payment; 5=don't know/refused",
        "skip_condition": "Check question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "fin3",
        "dimension": "Digital Financial Usage",
        "meaning": "Used card or phone to access a financial institution account",
        "codes": "1=yes; 2=no; 3=don't know; 4=refused",
        "skip_condition": "Asked only of financial institution account owners, excluding mobile money accounts",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "90,782 missing values; conditional availability expected",
    },
    {
        "variable": "fin26a",
        "dimension": "Digital Financial Usage",
        "meaning": "Used a phone or computer to pay a bill",
        "codes": "1=yes; 2=no; 3=don't know; 4=refused",
        "skip_condition": "Asked only of bank or mobile money account owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "76,175 missing values; conditional availability expected",
    },
    {
        "variable": "fin26b",
        "dimension": "Digital Financial Usage",
        "meaning": "Bought something online for delivery",
        "codes": "1=yes; 2=no; 3=don't know; 4=refused",
        "skip_condition": "Asked only of bank or mobile money account owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "76,175 missing values; conditional availability expected",
    },
    {
        "variable": "con18",
        "dimension": "Digital Safety",
        "meaning": "Has a phone lock such as PIN, password, or fingerprint",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "Asked only of mobile phone owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "con19",
        "dimension": "Digital Safety",
        "meaning": "Can change phone PIN/password without help",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "Asked only of respondents with a phone PIN/password",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "con21",
        "dimension": "Digital Safety",
        "meaning": "Received a scam call or text asking for money",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "Asked only of mobile phone owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "Yes indicates exposure to risk, not better safety",
    },
    {
        "variable": "con22",
        "dimension": "Digital Safety",
        "meaning": "Sent money to an unknown person asking for money",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "Asked only of respondents who received such a call or text",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "Structural missingness expected",
    },
    {
        "variable": "con23",
        "dimension": "Digital Safety",
        "meaning": "Received unwanted or offensive messages/media",
        "codes": "1=yes; 2=no; 8=don't know; 9=refused",
        "skip_condition": "Asked only of mobile phone owners",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "Yes indicates exposure to risk, not better safety",
    },
    {
        "variable": "saved",
        "dimension": "Financial Resilience",
        "meaning": "Saved or set aside money in the past year",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "borrowed",
        "dimension": "Financial Resilience",
        "meaning": "Borrowed money in the past year",
        "codes": "1=yes; 0=no",
        "skip_condition": "Constructed variable",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "Borrowing alone does not indicate positive or negative resilience",
    },
    {
        "variable": "fin24",
        "dimension": "Financial Resilience",
        "meaning": "Main source of emergency money within 30 days",
        "codes": "1=savings; 2=family/friends; 3=work; 4=borrowing; 5=sale of assets; 6=other; 7=could not raise money; 8=don't know; 9=refused",
        "skip_condition": "Review exact question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
    {
        "variable": "fin24a",
        "dimension": "Financial Resilience",
        "meaning": "Difficulty raising emergency funds within 30 days",
        "codes": "1=very difficult; 2=somewhat difficult; 3=no difficulty; 4=don't know; 5=refused",
        "skip_condition": "Asked only if respondent said emergency funds could be raised",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "Structural missingness expected",
    },
    {
        "variable": "fin24b",
        "dimension": "Financial Resilience",
        "meaning": "How long household could cover expenses if main income was lost",
        "codes": "1=<2 weeks; 2=about 1 month; 3=about 2 months; 4=>2 months; 5=don't know; 6=refused",
        "skip_condition": "Review exact question universe and missingness",
        "verification_status": "VERIFIED",
        "primary_model_status": "PENDING",
        "issue": "",
    },
]


def create_register() -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    register = pd.DataFrame(REGISTER)
    output_path = REPORT_DIR / "fi_codebook_verification_register.csv"
    register.to_csv(output_path, index=False)

    summary = {
        "candidate_variables_in_register": int(len(register)),
        "verification_status_counts": {
            str(key): int(value)
            for key, value in register["verification_status"]
            .value_counts().items()
        },
        "primary_model_status": "All selections remain pending",
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "raw_data_modified": False,
        "note": (
            "The register documents reviewed definitions and open issues. "
            "It does not approve variables for scoring or modelling."
        ),
    }

    summary_path = REPORT_DIR / "fi_codebook_verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nRegister saved to:", output_path)
    return register


if __name__ == "__main__":
    create_register()
