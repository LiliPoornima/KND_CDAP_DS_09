"""User-facing Component 1 credit-risk assessment form."""

from pathlib import Path
from typing import Any
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.src.credit_risk.home_credit.prediction import (
    APPLICATION_FEATURES,
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


def _validate_application(raw: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return blocking validation errors and non-blocking affordability warnings."""
    row = raw.iloc[0]
    errors = []
    warnings = []
    if row["AMT_INCOME_TOTAL"] <= 0:
        errors.append("Annual income must be greater than zero.")
    if row["AMT_ANNUITY"] > row["AMT_INCOME_TOTAL"] / 12:
        warnings.append(
            "The installment is higher than the entered monthly income; "
            "the affordability estimate may indicate higher payment difficulty."
        )
    if row["AMT_CREDIT"] > row["AMT_GOODS_PRICE"] > 0:
        warnings.append(
            "The requested loan is greater than the purchase-price value. "
            "This may be valid for cash loans, refinancing, fees, or additional expenses."
        )
    if row["CNT_FAM_MEMBERS"] < row["CNT_CHILDREN"] + 1:
        errors.append("Household size must be at least the number of children plus one.")
    return errors, warnings


def _risk_band(
    probability: float,
    raw: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """Return a presentation band using model risk plus affordability stress."""
    affordability_high = False
    affordability_elevated = False
    if raw is not None:
        row = raw.iloc[0]
        income = float(row["AMT_INCOME_TOTAL"])
        monthly_income = income / 12 if income > 0 else 0.0
        installment = float(row["AMT_ANNUITY"])
        loan_to_income = (
            float(row["AMT_CREDIT"]) / income if income > 0 else np.inf
        )
        affordability_high = monthly_income == 0 or installment > monthly_income or loan_to_income >= 5
        affordability_elevated = installment >= monthly_income * 0.4 or loan_to_income >= 2

    if probability >= 0.50 or affordability_high:
        return "High", "red"
    if probability >= 0.20 or affordability_elevated:
        return "Medium", "orange"
    if probability < 0.20:
        return "Low", "green"
    return "High", "red"


def _recommended_action(risk_band: str, affordability_reasons: list[str]) -> str:
    if risk_band == "High":
        return "Manual review required"
    if risk_band == "Medium" or affordability_reasons:
        return "Request additional information"
    return "Proceed to standard assessment"


def _build_summary(
    raw: pd.DataFrame,
    applicant: pd.DataFrame,
    probability: float,
    decision: str,
    applicant_reference: str,
    additional_information: dict[str, Any],
) -> pd.DataFrame:
    row = applicant.iloc[0]
    return pd.DataFrame(
        [
            {"Item": "Applicant reference", "Value": applicant_reference},
            {"Item": "Research-only assessment", "Value": decision},
            {"Item": "Estimated payment-difficulty likelihood", "Value": f"{probability:.1%}"},
            {"Item": "Annual income", "Value": raw.loc[0, "AMT_INCOME_TOTAL"]},
            {"Item": "Requested loan amount", "Value": raw.loc[0, "AMT_CREDIT"]},
            {"Item": "Estimated installment", "Value": raw.loc[0, "AMT_ANNUITY"]},
            {"Item": "Purchase price / goods value", "Value": raw.loc[0, "AMT_GOODS_PRICE"]},
            {"Item": "Loan-to-income ratio", "Value": f"{row['CREDIT_TO_INCOME_RATIO']:.2f}x"},
            {"Item": "Installment-to-income ratio", "Value": f"{row['ANNUITY_TO_INCOME_RATIO']:.1%}"},
            {"Item": "Loan-to-purchase ratio", "Value": f"{row['CREDIT_TO_GOODS_RATIO']:.1%}"},
            {"Item": "Employment / income category", "Value": raw.loc[0, "NAME_INCOME_TYPE"]},
            {"Item": "Education category", "Value": raw.loc[0, "NAME_EDUCATION_TYPE"]},
            {"Item": "Family status", "Value": raw.loc[0, "NAME_FAMILY_STATUS"]},
            {"Item": "Housing category", "Value": raw.loc[0, "NAME_HOUSING_TYPE"]},
            {"Item": "Loan product type", "Value": raw.loc[0, "NAME_CONTRACT_TYPE"]},
            {"Item": "Occupation", "Value": raw.loc[0, "OCCUPATION_TYPE"] or "Not reported"},
            {"Item": "Own a car", "Value": "Yes" if raw.loc[0, "FLAG_OWN_CAR"] == "Y" else "No"},
            {"Item": "Own property", "Value": "Yes" if raw.loc[0, "FLAG_OWN_REALTY"] == "Y" else "No"},
            {"Item": "Household size", "Value": raw.loc[0, "CNT_FAM_MEMBERS"]},
            {"Item": "Number of children", "Value": raw.loc[0, "CNT_CHILDREN"]},
            {"Item": "Years employed", "Value": applicant.loc[0, "YEARS_EMPLOYED"]},
            {
                "Item": "Existing debt information",
                "Value": "Provided" if additional_information["provided"] else "Not provided",
            },
            {
                "Item": "Existing monthly debt repayments",
                "Value": additional_information["monthly_debt"] if additional_information["provided"] else "Not provided",
            },
            {
                "Item": "Outstanding loan balances",
                "Value": additional_information["outstanding_balance"] if additional_information["provided"] else "Not provided",
            },
            {
                "Item": "Previous missed payments",
                "Value": additional_information["missed_payments"] if additional_information["provided"] else "Not provided",
            },
            {
                "Item": "Previous credit history",
                "Value": additional_information["credit_history"] if additional_information["provided"] else "Not provided",
            },
            {
                "Item": "Credit utilisation",
                "Value": additional_information["credit_utilisation"] if additional_information["provided"] else "Not provided",
            },
        ]
    )


def _build_pdf(summary: pd.DataFrame) -> bytes | None:
    """Create a small, dependency-backed PDF assessment summary."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    output = BytesIO()
    document = canvas.Canvas(output, pagesize=letter)
    _, height = letter
    document.setFont("Helvetica-Bold", 15)
    document.drawString(48, height - 52, "Applicant Credit Assessment")
    document.setFont("Helvetica", 9)
    document.drawString(48, height - 70, "Research-only summary; not a lending decision.")
    y = height - 100
    for item, value in summary.itertuples(index=False):
        text = f"{item}: {value}"
        document.drawString(48, y, text[:115])
        y -= 15
        if y < 55:
            document.showPage()
            y = height - 55
    document.setFont("Helvetica-Oblique", 8)
    document.drawString(
        48,
        35,
        "Use this estimate with bank affordability, identity, and policy checks.",
    )
    document.save()
    return output.getvalue()


FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Requested loan amount",
    "AMT_ANNUITY": "Estimated installment",
    "AMT_GOODS_PRICE": "Purchase price / goods value",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Household size",
    "DAYS_EMPLOYED": "Employment duration",
    "YEARS_EMPLOYED": "Years employed",
    "CREDIT_TO_INCOME_RATIO": "Loan amount / income",
    "ANNUITY_TO_INCOME_RATIO": "Installment / income",
    "CREDIT_TO_GOODS_RATIO": "Loan amount / goods value",
    "ANNUITY_TO_CREDIT_RATIO": "Installment / loan amount",
    "NAME_CONTRACT_TYPE": "Loan product type",
    "NAME_INCOME_TYPE": "Employment / income category",
    "NAME_EDUCATION_TYPE": "Education category",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing category",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_OWN_CAR": "Car ownership",
    "FLAG_OWN_REALTY": "Property ownership",
}


def _render_affordability_summary(
    raw: pd.DataFrame,
    applicant: pd.DataFrame,
) -> list[str]:
    """Explain the main affordability indicators in applicant language."""
    row = applicant.iloc[0]
    income = float(raw.loc[0, "AMT_INCOME_TOTAL"])
    annuity = float(raw.loc[0, "AMT_ANNUITY"])
    goods_price = float(raw.loc[0, "AMT_GOODS_PRICE"])

    st.subheader("2. Financial Affordability Analysis")
    summary_cols = st.columns(3)
    summary_cols[0].metric(
        "Loan-to-income ratio", f"{row['CREDIT_TO_INCOME_RATIO']:.2f}x"
    )
    monthly_income = income / 12
    monthly_installment_ratio = annuity / monthly_income if monthly_income else np.nan
    summary_cols[1].metric(
        "Installment / annual income (source ratio)",
        f"{row['ANNUITY_TO_INCOME_RATIO']:.1%}",
        help=(
            "This is the ratio used by the trained model. Because the installment "
            "is monthly and income is annual in the source data, use the monthly "
            "income indicator below for affordability review."
        ),
    )
    summary_cols[2].metric(
        "Loan-to-purchase ratio", f"{row['CREDIT_TO_GOODS_RATIO']:.1%}"
    )

    reasons: list[str] = []
    if income <= 0:
        reasons.append("Annual income is zero or missing, so affordability cannot be assessed reliably.")
    elif monthly_installment_ratio >= 0.4:
        reasons.append(
            "The proposed installment is a relatively large share of the estimated monthly income."
        )
    elif monthly_installment_ratio <= 0.2:
        reasons.append(
            "The proposed installment is a relatively small share of the estimated monthly income."
        )
    if row["CREDIT_TO_INCOME_RATIO"] >= 5:
        reasons.append("The requested loan is large compared with the entered annual income.")
    elif row["CREDIT_TO_INCOME_RATIO"] <= 2:
        reasons.append("The requested loan is comparatively moderate relative to the entered annual income.")
    if goods_price > 0 and row["CREDIT_TO_GOODS_RATIO"] > 1:
        reasons.append("The requested loan exceeds the entered purchase-price value.")
    if reasons:
        st.warning(f"Affordability warning: {reasons[0]}")
    else:
        st.success("No extreme affordability indicator was identified from the entered values.")
    st.info(
        f"Estimated monthly income: {monthly_income:,.2f} source units · "
        f"Installment share of estimated monthly income: "
        f"{monthly_installment_ratio:.1%}" if monthly_income else
        "Monthly affordability cannot be calculated from zero income."
    )
    st.caption(
        "These are screening indicators calculated from the values entered in this "
        "form. They are not bank affordability rules and do not account for every "
        "household expense or existing obligation."
    )
    return reasons


def _render_next_steps(
    raw: pd.DataFrame,
    applicant: pd.DataFrame,
    probability: float,
) -> list[str]:
    """Provide non-binding checks that a user can consider."""
    row = applicant.iloc[0]
    st.subheader("5. Personalised Recommendations")
    checks: list[str] = [
        "Confirm income, installment, loan amount, and purchase-price details before relying on the estimate.",
        "Review existing monthly obligations and essential household expenses separately.",
        "Collect existing loan balances, monthly repayments, and previous missed-payment information; these are not available in this form.",
    ]
    monthly_income = float(raw.loc[0, "AMT_INCOME_TOTAL"]) / 12
    monthly_installment_ratio = (
        float(raw.loc[0, "AMT_ANNUITY"]) / monthly_income
        if monthly_income
        else np.nan
    )
    if monthly_installment_ratio >= 0.4:
        checks.append("Consider whether a smaller installment or a longer repayment plan is affordable.")
    if row["CREDIT_TO_INCOME_RATIO"] >= 5:
        checks.append("Consider whether the requested amount can be reduced or supported by additional resources.")
    if raw.loc[0, "OCCUPATION_TYPE"] is None:
        checks.append("Provide an occupation where available; the current entry is treated as not reported.")
    if row["YEARS_EMPLOYED"] >= 5:
        checks.append("Include the entered employment duration as one part of an overall employment-stability review.")
    if probability >= 0.5:
        checks.append("A bank or adviser should complete additional affordability and identity checks before any decision.")
    for check in checks:
        st.write(f"- {check}")
    return checks


def _render_bank_decision_support(
    probability: float,
    threshold: float,
    affordability_reasons: list[str],
    recommendations: list[str],
    shap_factors: pd.DataFrame | None,
    occupation_reported: bool,
    additional_information: dict[str, Any],
) -> None:
    """Summarise how a bank officer can use the result without automating a decision."""
    st.subheader("6. Bank Decision Support")
    if probability >= threshold or affordability_reasons:
        action = "Manual review / request additional information"
    else:
        action = "Proceed to further assessment with standard checks"
    st.metric("Recommended action", action)

    factors = (
        shap_factors.sort_values("Contribution", key=abs, ascending=False)
        .head(3)["Factor"].tolist()
        if shap_factors is not None
        else []
    )
    support = pd.DataFrame(
        [
            {"Assessment area": "Predicted risk", "Information": f"{probability:.1%} estimated payment-difficulty likelihood"},
            {"Assessment area": "Prediction rule", "Information": f"{'Higher' if probability >= threshold else 'Lower'} likelihood relative to the model's validation-selected threshold"},
            {"Assessment area": "Main SHAP factors", "Information": ", ".join(factors) if factors else "Not available"},
            {"Assessment area": "Financial affordability", "Information": "Concerns require verification" if affordability_reasons else "No extreme input ratio identified"},
            {
                "Assessment area": "Existing debt",
                "Information": (
                    "Provided as contextual information only"
                    if additional_information["provided"]
                    else "Not provided; request balances and monthly commitments"
                ),
            },
            {"Assessment area": "Fairness information", "Information": "Review group-level fairness results on the Credit Risk research page; no individual fairness score is produced"},
            {"Assessment area": "Model confidence", "Information": "Not provided by this trained model; do not treat probability as certainty"},
            {"Assessment area": "Model reliability", "Information": "Review model-level calibration and performance on the Credit Risk research page"},
            {"Assessment area": "Employment information", "Information": "Occupation reported" if occupation_reported else "Occupation not reported; request additional information"},
            {"Assessment area": "Recommended action", "Information": action},
        ]
    )
    st.dataframe(support, hide_index=True, use_container_width=True)
    if recommendations:
        st.caption(
            "The recommended action supports human review. It does not approve, "
            "reject, or price the loan automatically."
        )


def _render_applicant_profile(raw: pd.DataFrame, applicant: pd.DataFrame) -> None:
    """Show the submitted values without adding unmodelled inputs."""
    row = raw.iloc[0]
    profile = pd.DataFrame(
        [
            {"Category": "Financial", "Information": f"Income: {row['AMT_INCOME_TOTAL']:,.2f}; loan: {row['AMT_CREDIT']:,.2f}; installment: {row['AMT_ANNUITY']:,.2f}"},
            {"Category": "Employment", "Information": f"{row['NAME_INCOME_TYPE']}; {row['OCCUPATION_TYPE'] or 'Occupation not reported'}; {applicant.iloc[0]['YEARS_EMPLOYED']:.1f} years employed"},
            {"Category": "Household", "Information": f"{row['NAME_FAMILY_STATUS']}; {row['CNT_CHILDREN']} children; household size {row['CNT_FAM_MEMBERS']}"},
            {"Category": "Assets", "Information": f"Car: {'Yes' if row['FLAG_OWN_CAR'] == 'Y' else 'No'}; property: {'Yes' if row['FLAG_OWN_REALTY'] == 'Y' else 'No'}"},
            {"Category": "Education", "Information": str(row["NAME_EDUCATION_TYPE"])},
            {"Category": "Housing", "Information": str(row["NAME_HOUSING_TYPE"])},
            {"Category": "Loan", "Information": str(row["NAME_CONTRACT_TYPE"])},
        ]
    )
    st.subheader("4. Applicant Profile Analysis")
    st.dataframe(profile, hide_index=True, use_container_width=True)


def _render_shap_explanation(
    prepared_dir: Path,
    bundle: dict[str, Any],
    applicant: pd.DataFrame,
) -> pd.DataFrame | None:
    """Show optional local SHAP evidence behind the plain-language summary."""
    with st.expander("Why the estimate changed: supporting factors", expanded=True):
        try:
            import shap

            model = bundle["model"]
            background = pd.read_csv(
                prepared_dir / "train_processed.csv",
                usecols=APPLICATION_FEATURES,
            ).sample(n=40, random_state=42)
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]
            transformed_background = preprocessor.transform(background)
            transformed_applicant = preprocessor.transform(applicant)
            if hasattr(transformed_background, "toarray"):
                transformed_background = transformed_background.toarray()
                transformed_applicant = transformed_applicant.toarray()
            transformed_names = list(preprocessor.get_feature_names_out())

            def predict_probability(values: np.ndarray) -> np.ndarray:
                raw_probability = classifier.predict_proba(values)[
                    :, list(model.classes_).index(1)
                ]
                clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
                logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
                return bundle["calibrator"].predict_proba(logits)[:, 1]

            explainer = shap.Explainer(
                predict_probability,
                transformed_background,
                algorithm="permutation",
                feature_names=transformed_names,
            )
            shap_values = explainer(
                transformed_applicant,
                max_evals=2 * transformed_applicant.shape[1] + 1,
            )
            contributions = dict.fromkeys(APPLICATION_FEATURES, 0.0)
            for transformed_name, contribution in zip(
                transformed_names, shap_values.values[0]
            ):
                candidate = transformed_name.split("__", 1)[-1]
                source_feature = next(
                    (
                        feature
                        for feature in sorted(APPLICATION_FEATURES, key=len, reverse=True)
                        if candidate == feature or candidate.startswith(feature + "_")
                    ),
                    None,
                )
                if source_feature is not None:
                    contributions[source_feature] += float(contribution)

            local = pd.DataFrame(
                {
                    "Factor": [
                        FEATURE_LABELS.get(feature, feature)
                        for feature in APPLICATION_FEATURES
                    ],
                    "Contribution": [
                        contributions[feature] for feature in APPLICATION_FEATURES
                    ],
                }
            )
            local = local.reindex(local["Contribution"].abs().sort_values().index).tail(10)
            top_factors = local.tail(5).sort_values("Contribution", key=abs, ascending=False)
            st.subheader("3. SHAP-Based Explainability")
            st.markdown("**Top factors influencing this prediction**")
            st.dataframe(
                top_factors.rename(columns={"Factor": "Feature", "Contribution": "Actual SHAP contribution"}),
                hide_index=True,
                use_container_width=True,
            )
            figure = px.bar(
                top_factors.sort_values("Contribution"),
                x="Contribution",
                y="Factor",
                orientation="h",
                color="Contribution",
                color_continuous_scale="RdBu_r",
                title="Factors influencing this estimate",
            )
            figure.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Approximate contribution to estimated risk",
            )
            st.plotly_chart(figure, use_container_width=True)
            waterfall = top_factors.sort_values("Contribution")
            waterfall_figure = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative"] * len(waterfall) + ["total"],
                    x=waterfall["Factor"].tolist() + ["Model estimate"],
                    y=waterfall["Contribution"].tolist() + [0],
                    connector={"line": {"color": "#7B8794"}},
                    increasing={"marker": {"color": "#D95F59"}},
                    decreasing={"marker": {"color": "#2C7FB8"}},
                    totals={"marker": {"color": "#4C78A8"}},
                )
            )
            waterfall_figure.update_layout(
                title="Individual prediction explanation",
                yaxis_title="Actual SHAP contribution to estimated risk",
                showlegend=False,
            )
            st.plotly_chart(waterfall_figure, use_container_width=True)
            st.caption(
                "Bars pointing right push the estimate toward higher payment-difficulty "
                "risk; bars pointing left push it lower. SHAP explains model behaviour "
                "relative to a training sample. It does not prove that a factor caused "
                "the result."
            )
            return local
        except (ImportError, KeyError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"The supporting SHAP explanation could not be generated: {exc}")
            return None


def render_credit_risk_assessment() -> None:
    """Render the applicant questionnaire and its research-only result."""
    st.title("Applicant Credit Assessment")
    st.caption(
        "Component 1 · Enter applicant and requested-credit information to "
        "receive a research-only payment-difficulty estimate."
    )
    st.markdown(
        "**Bank decision-support workspace**  \n"
        "Use the model output as one input to structured human review."
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
        applicant_reference = st.text_input(
            "Applicant reference",
            value="Applicant 001",
            help="A local reference for the summary only; it is not used by the model.",
        )
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
        compare_scenario = st.checkbox(
            "Compare with an alternative loan amount or installment",
            help="The alternative is scored using the same entered applicant details.",
        )
        alternative_credit = requested_credit
        alternative_annuity = annuity
        if compare_scenario:
            compare_cols = st.columns(2)
            alternative_credit = compare_cols[0].number_input(
                "Alternative loan amount",
                min_value=0.0,
                value=float(requested_credit),
                step=10000.0,
            )
            alternative_annuity = compare_cols[1].number_input(
                "Alternative installment",
                min_value=0.0,
                value=float(annuity),
                step=1000.0,
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
        st.markdown("**Optional bank-review information (not used by the current model)**")
        additional_information_provided = st.checkbox(
            "I want to provide existing debt and credit-history information"
        )
        additional_information = {
            "provided": additional_information_provided,
            "monthly_debt": "Not provided",
            "outstanding_balance": "Not provided",
            "missed_payments": "Not provided",
            "credit_history": "Not provided",
            "credit_utilisation": "Not provided",
        }
        if additional_information_provided:
            additional_cols = st.columns(2)
            additional_information["monthly_debt"] = additional_cols[0].number_input(
                "Existing monthly debt repayments",
                min_value=0.0,
                value=0.0,
                step=1000.0,
            )
            additional_information["outstanding_balance"] = additional_cols[1].number_input(
                "Outstanding loan balances",
                min_value=0.0,
                value=0.0,
                step=10000.0,
            )
            additional_cols = st.columns(3)
            additional_information["missed_payments"] = additional_cols[0].number_input(
                "Previous missed payments",
                min_value=0,
                value=0,
                step=1,
            )
            additional_information["credit_history"] = additional_cols[1].selectbox(
                "Previous credit history",
                ["No known adverse history", "Adverse history", "Unknown / not reported"],
            )
            additional_information["credit_utilisation"] = additional_cols[2].number_input(
                "Credit utilisation (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
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
    validation_errors, validation_warnings = _validate_application(raw)
    if validation_errors:
        st.error("Please correct the following before assessment:")
        for error in validation_errors:
            st.write(f"- {error}")
        return
    try:
        bundle = _load_application_only_model(str(prepared_dir), signature)
        applicant = cr_prepare_home_credit_application(raw)
        probability = cr_predict_home_credit_application(bundle, applicant)
        alternative_probability = None
        if compare_scenario:
            alternative_raw = raw.copy()
            alternative_raw.loc[0, "AMT_CREDIT"] = alternative_credit
            alternative_raw.loc[0, "AMT_ANNUITY"] = alternative_annuity
            alternative_errors, _ = _validate_application(alternative_raw)
            if alternative_errors:
                st.error("The alternative scenario cannot be assessed:")
                for error in alternative_errors:
                    st.write(f"- {error}")
                return
            alternative = cr_prepare_home_credit_application(alternative_raw)
            alternative_probability = cr_predict_home_credit_application(
                bundle, alternative
            )
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Could not assess this application: {exc}")
        return

    band, band_color = _risk_band(probability, raw)
    decision = (
        "Higher payment-difficulty likelihood"
        if probability >= bundle["threshold"]
        else "Lower payment-difficulty likelihood"
    )
    st.subheader("1. Credit Risk Assessment – Main Output")
    result_cols = st.columns(4)
    result_cols[0].metric("Applicant", applicant_reference)
    result_cols[1].metric("Predicted payment-difficulty probability", f"{probability:.1%}")
    result_cols[2].metric("Risk category", band)
    result_cols[3].metric("Assessment status", "Completed")
    st.markdown(f"**Model threshold comparison:** {decision}")
    st.caption(
        "The probability and threshold comparison come from the trained model. "
        "The risk category is a screening band that combines the model probability "
        "with affordability stress from the values entered in this form. It is not "
        "a lender approval or rejection and is not a confidence score."
    )
    band_messages = {
        "green": st.success,
        "orange": st.warning,
        "red": st.error,
    }
    band_messages[band_color](
        f"Risk category: **{band}**. This category is for research decision support only."
    )
    if alternative_probability is not None:
        alternative_band, _ = _risk_band(alternative_probability, alternative_raw)
        st.subheader("Alternative scenario comparison")
        comparison = pd.DataFrame(
            [
                {
                    "Scenario": "Current application",
                    "Loan amount": requested_credit,
                    "Installment": annuity,
                    "Payment-difficulty probability": f"{probability:.1%}",
                    "Risk category": band,
                },
                {
                    "Scenario": "Alternative",
                    "Loan amount": alternative_credit,
                    "Installment": alternative_annuity,
                    "Payment-difficulty probability": f"{alternative_probability:.1%}",
                    "Risk category": alternative_band,
                },
            ]
        )
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    if validation_warnings:
        st.warning("Please review these affordability notes:")
        for warning in validation_warnings:
            st.write(f"- {warning}")
    affordability_reasons = _render_affordability_summary(raw, applicant)
    recommended_action = _recommended_action(band, affordability_reasons)
    st.subheader("Recommended bank action")
    st.info(recommended_action)
    with st.expander("Detailed review", expanded=False):
        shap_factors = _render_shap_explanation(prepared_dir, bundle, applicant)
        _render_applicant_profile(raw, applicant)
        recommendations = _render_next_steps(raw, applicant, probability)
        _render_bank_decision_support(
            probability,
            bundle["threshold"],
            affordability_reasons,
            recommendations,
            shap_factors,
            raw.loc[0, "OCCUPATION_TYPE"] is not None,
            additional_information,
        )
        summary = _build_summary(
            raw,
            applicant,
            probability,
            decision,
            applicant_reference,
            additional_information,
        )
        st.subheader("Review and download")
        st.caption("Review the submitted application before downloading.")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.download_button(
            "Download CSV",
            summary.to_csv(index=False).encode("utf-8-sig"),
            "applicant_credit_assessment.csv",
            "text/csv",
            use_container_width=True,
        )
        pdf_summary = _build_pdf(summary)
        if pdf_summary is None:
            st.warning(
                "PDF download is unavailable because ReportLab is not installed. "
                "The CSV download is still available."
            )
        else:
            st.download_button(
                "Download PDF",
                pdf_summary,
                "applicant_credit_assessment.pdf",
                "application/pdf",
                use_container_width=True,
            )
    if "shap_factors" not in locals():
        shap_factors = _render_shap_explanation(prepared_dir, bundle, applicant)
    if shap_factors is not None:
        st.subheader("Main risk factors")
        top_factors = shap_factors.sort_values(
            "Contribution", key=abs, ascending=False
        ).head(3).copy()
        top_factors["Direction"] = np.where(
            top_factors["Contribution"] >= 0,
            "Increases estimated risk",
            "Decreases estimated risk",
        )
        st.dataframe(
            top_factors[["Factor", "Direction", "Contribution"]].rename(
                columns={"Factor": "Factor", "Contribution": "Actual SHAP contribution"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    if compare_scenario and (
        alternative_credit != requested_credit or alternative_annuity != annuity
    ):
        alternative_raw = raw.copy()
        alternative_raw.loc[0, "AMT_CREDIT"] = alternative_credit
        alternative_raw.loc[0, "AMT_ANNUITY"] = alternative_annuity
        alternative_errors, alternative_warnings = _validate_application(alternative_raw)
        if alternative_errors:
            st.warning("The alternative scenario cannot be scored:")
            for error in alternative_errors:
                st.write(f"- {error}")
        else:
            if alternative_warnings:
                st.warning("The alternative scenario includes these affordability notes:")
                for warning in alternative_warnings:
                    st.write(f"- {warning}")
            alternative_applicant = cr_prepare_home_credit_application(alternative_raw)
            alternative_probability = cr_predict_home_credit_application(
                bundle, alternative_applicant
            )
            comparison = pd.DataFrame(
                {
                    "Scenario": ["Current application", "Alternative scenario"],
                    "Loan amount": [requested_credit, alternative_credit],
                    "Installment": [annuity, alternative_annuity],
                    "Estimated likelihood": [probability, alternative_probability],
                }
            )
            st.subheader("Current vs alternative scenario")
            st.dataframe(
                comparison.style.format(
                    {
                        "Loan amount": "{:,.2f}",
                        "Installment": "{:,.2f}",
                        "Estimated likelihood": "{:.1%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            change = alternative_probability - probability
            st.info(
                f"The alternative changes the estimated likelihood by {change:+.1%}. "
                "This is a model comparison, not a guarantee that changing loan "
                "terms will produce the same result in practice."
            )
    st.caption(
        "This estimate reflects patterns in the Home Credit research dataset. "
        "It does not replace a bank's affordability checks, policy rules, or "
        "human decision-making."
    )
    st.info(
        "Responsible use: do not use this estimate as the sole basis to approve, "
        "reject, or price a loan. Verify applicant information and complete "
        "human, affordability, identity, and policy checks."
    )
