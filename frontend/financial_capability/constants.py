"""Frontend constants for Component 04 – Financial Well-being Assessment & Bank Loan Decision Support."""

SECTIONS = [
    "Overview",
    "Dataset & Data Quality",
    "Model Performance",
    "Individual Assessment",
    "SHAP Explanation",
    "Capability Gaps & Suggested Categories",
    "What-if Analysis",
    "Methodology & Limitations",
]

COMPONENT_TITLE = (
    "Component 04 – Explainable Financial Well-being Assessment System "
    "for Bank Loan Decision Support"
)

COMPONENT_SUBTITLE = (
    "Bank Officer Decision Support Dashboard | CFPB NFWBS 2016 | "
    "Random Forest & XGBoost | SHAP Explainability | Financial Capability Profile"
)

SESSION_RESPONSE_KEY = "fc_response"
WIDGET_PREFIX = "fc_in_"

DISCLAIMER = (
    "This report provides supplementary information about the applicant's financial well-being. "
    "It does not replace the bank's standard credit assessment, identity and income verification, "
    "affordability checks or lending policies. It does not predict loan repayment or automatically "
    "approve or reject a loan."
)

BANK_LOAN_NOTICE = (
    "Component 04 does not automatically approve or reject the loan. "
    "It provides additional financial well-being information to support the bank officer's assessment."
)

# Illustrative loan applicant example answers (NOT a real person and NOT used for any metric).
EXAMPLE_RESPONSE = {
    "SAVEHABIT": 3, "SAVINGSRANGES": 3, "AUTOMATED_1": 0, "AUTOMATED_2": 0, "FS1_7": 2,
    "MANAGE1_1": 4, "MANAGE1_2": 2, "MANAGE1_3": 3, "MANAGE1_4": 3, "PROPPLAN_1": 2,
    "PROPPLAN_2": 3, "FS1_6": 3, "ACT1_1": 4, "SUBKNOWL1": 4, "LMscore": 2, "FS1_3": 3,
    "FS2_3": 3, "FINGOALS": 1, "PROPPLAN_3": 4, "PROPPLAN_4": 2, "GOALCONF": 3, "ACT1_2": 3,
    "SCFHORIZON": 2, "FS1_1": 3, "agecat": 3, "PPINCIMP": 4, "PPEDUC": 3, "EMPLOY": 2,
    "HOUSING": 2, "PPHHSIZE": 3, "PPMARIT": 1,
}


