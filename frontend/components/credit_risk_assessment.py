"""User-facing Component 1 credit-risk assessment form."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from backend.src.credit_risk.home_credit.prediction import (
    cr_prepare_home_credit_application,
    cr_predict_home_credit_application,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "credit_risk"


@st.cache_resource(show_spinner="Preparing the credit-risk assessment model…")
def _load_application_only_model(
    prepared_dir: str,
    signature: tuple[int, int, int],
) -> dict[str, Any]:
    del signature
    from backend.src.credit_risk.home_credit.prediction import (
        cr_fit_home_credit_application_model,
    )

    return cr_fit_home_credit_application_model(prepared_dir)


def _show_missing(path: Path) -> None:
    st.error("The assessment model data is not available.")
    st.info(f"Expected prepared data at `{path.relative_to(PROJECT_ROOT)}`.")


def render_credit_risk_assessment() -> None:
    """Render the applicant questionnaire and its research-only result."""
    st.title("Applicant Credit Assessment")
    st.caption(
        "Component 1 · Enter applicant and requested-credit information to "
        "receive a research-only payment-difficulty estimate."
    )
    st.warning(
        "This assessment uses a Home Credit application-only model. It is not "
        "a lender approval, financial advice, or a guarantee of repayment. "
        "Use the amount units represented in the source dataset."
    )

    prepared_dir = OUTPUT_ROOT / "home_credit" / "completion_run" / "prepared"
    split_paths = tuple(
        prepared_dir / filename
        for filename in (
            "train_processed.csv",
            "validation_processed.csv",
            "test_processed.csv",
        )
    )
    if not all(path.is_file() for path in split_paths):
        _show_missing(split_paths[0])
        return

    with st.form("applicant_credit_assessment"):
        st.subheader("Requested credit")
        amount_cols = st.columns(2)
        income = amount_cols[0].number_input(
            "Annual income (source data units)",
            min_value=0.0,
            value=150000.0,
            step=10000.0,
        )
        requested_credit = amount_cols[1].number_input(
            "Requested loan amount (source data units)",
            min_value=0.0,
            value=500000.0,
            step=10000.0,
        )
        amount_cols = st.columns(2)
        annuity = amount_cols[0].number_input(
            "Estimated installment amount (source data units)",
            min_value=0.0,
            value=25000.0,
            step=1000.0,
            help="Enter the proposed installment amount.",
        )
        goods_price = amount_cols[1].number_input(
            "Purchase price / goods value (source data units)",
            min_value=0.0,
            value=450000.0,
            step=10000.0,
        )

        st.subheader("Personal, household, and employment details")
        employment_options = [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Unemployed",
            "Businessman",
            "Student",
            "Maternity leave",
        ]
        education_options = [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ]
        family_options = [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow",
            "Unknown",
        ]
        housing_options = [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ]
        occupation_options = [
            "Unknown / not reported",
            "Laborers",
            "Sales staff",
            "Core staff",
            "Managers",
            "Drivers",
            "High skill tech staff",
            "Accountants",
            "Medicine staff",
            "Security staff",
            "Cooking staff",
            "Cleaning staff",
            "Private service staff",
            "Low-skill Laborers",
            "Waiters/barmen staff",
            "Secretaries",
            "Realty agents",
            "HR staff",
            "IT staff",
        ]
        left, right = st.columns(2)
        contract = left.selectbox("Loan product type", ["Cash loans", "Revolving loans"])
        income_type = right.selectbox("Employment / income category", employment_options)
        education = left.selectbox("Education category", education_options)
        family = right.selectbox("Family status", family_options)
        housing = left.selectbox("Housing category", housing_options)
        occupation = right.selectbox("Occupation", occupation_options)
        car = left.selectbox("Own a car", ["No", "Yes"])
        property_owner = right.selectbox("Own property", ["No", "Yes"])

        detail_cols = st.columns(3)
        children = detail_cols[0].number_input(
            "Number of children", min_value=0, value=0, step=1
        )
        household = detail_cols[1].number_input(
            "Household size", min_value=1, value=2, step=1
        )
        years_employed = detail_cols[2].number_input(
            "Years employed", min_value=0.0, value=5.0, step=1.0
        )
        submitted = st.form_submit_button(
            "Assess applicant", type="primary", use_container_width=True
        )

    if not submitted:
        st.info("Complete the applicant information and select **Assess applicant**.")
        return

    signature = tuple(path.stat().st_mtime_ns for path in split_paths)
    raw = pd.DataFrame(
        [
            {
                "NAME_CONTRACT_TYPE": contract,
                "FLAG_OWN_CAR": "Y" if car == "Yes" else "N",
                "FLAG_OWN_REALTY": "Y" if property_owner == "Yes" else "N",
                "CNT_CHILDREN": children,
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": requested_credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "NAME_INCOME_TYPE": income_type,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family,
                "NAME_HOUSING_TYPE": housing,
                "DAYS_EMPLOYED": -years_employed * 365.25,
                "OCCUPATION_TYPE": (
                    None if occupation == "Unknown / not reported" else occupation
                ),
                "CNT_FAM_MEMBERS": household,
            }
        ]
    )
    try:
        bundle = _load_application_only_model(str(prepared_dir), signature)
        applicant = cr_prepare_home_credit_application(raw)
        probability = cr_predict_home_credit_application(bundle, applicant)
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Could not assess this application: {exc}")
        return

    decision = "Further review recommended" if probability >= bundle["threshold"] else "Lower estimated risk"
    st.subheader("Assessment result")
    result_cols = st.columns(2)
    result_cols[0].metric("Research-only assessment", decision)
    result_cols[1].metric("Estimated payment-difficulty risk", f"{probability:.1%}")
    st.caption(
        "This estimate reflects patterns in the Home Credit research dataset. "
        "It does not replace a bank's affordability checks, policy rules, or "
        "human decision-making."
    )
