"""User-facing Component 1 credit-risk assessment form."""

from pathlib import Path
from typing import Any
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from backend.src.credit_risk.home_credit.prediction import (
    APPLICATION_FEATURES,
    cr_prepare_home_credit_application,
    cr_predict_home_credit_application,
    cr_compute_home_credit_counterfactuals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "credit_risk"

APPLICATION_AMOUNT_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
)
APPLICATION_AMOUNT_LABELS = {
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Requested loan amount",
    "AMT_ANNUITY": "Estimated installment",
    "AMT_GOODS_PRICE": "Purchase price / goods value",
}


@st.cache_resource(show_spinner="Preparing the credit-risk assessment model…")
def _load_application_only_model(
    prepared_dir: str,
    signature: tuple[int, int, int],
) -> dict[str, Any]:
    del signature
    from backend.src.credit_risk.home_credit.prediction import (
        cr_fit_home_credit_application_model,
    )

    bundle = cr_fit_home_credit_application_model(prepared_dir)
    if "training_amount_ranges" not in bundle:
        train = pd.read_csv(
            Path(prepared_dir) / "train_processed.csv",
            usecols=list(APPLICATION_AMOUNT_COLUMNS),
        )
        bundle["training_amount_ranges"] = {
            column: {
                "lower_supported": float(train[column].quantile(0.01)),
                "upper_supported": float(train[column].quantile(0.99)),
            }
            for column in APPLICATION_AMOUNT_COLUMNS
        }
    return bundle


def _show_missing(path: Path) -> None:
    st.error("The assessment model data is not available.")
    st.info(f"Expected prepared data at `{path.relative_to(PROJECT_ROOT)}`.")


def _out_of_training_range(raw: pd.DataFrame, bundle: dict[str, Any]) -> list[str]:
    """Flag monetary inputs that fall outside typical benchmark ranges."""
    warnings = []
    for column in APPLICATION_AMOUNT_COLUMNS:
        value = float(raw.loc[0, column])
        if pd.isna(value):
            continue
        observed = bundle["training_amount_ranges"][column]
        if value < observed["lower_supported"]:
            warnings.append(
                f"{APPLICATION_AMOUNT_LABELS[column]} ({value:,.0f}) is below standard benchmark limits "
                f"({observed['lower_supported']:,.0f})."
            )
        elif value > observed["upper_supported"]:
            warnings.append(
                f"{APPLICATION_AMOUNT_LABELS[column]} ({value:,.0f}) exceeds standard benchmark limits "
                f"({observed['upper_supported']:,.0f})."
            )
    return warnings


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
    if row["CNT_FAM_MEMBERS"] < row["CNT_CHILDREN"] + 1:
        errors.append("Household size must be at least the number of children plus one.")
    return errors, warnings


def _evaluate_loan_recommendation(
    probability: float,
    model_threshold: float,
) -> tuple[str, str]:
    """Determine first-line loan recommendation based on default probability and risk cutoff."""
    if probability < model_threshold:
        return (
            "YES",
            f"Payment-difficulty probability ({probability:.1%}) is below the "
            f"{model_threshold:.1%} risk cutoff. The applicant qualifies for loan approval "
            "under first-line credit risk screening.",
        )
    return (
        "NO",
        f"Payment-difficulty probability ({probability:.1%}) meets or exceeds the "
        f"{model_threshold:.1%} risk cutoff. The loan is not recommended under current terms; "
        "review actionable parameter adjustments below to qualify for approval.",
    )


def _classify_risk_category(probability: float, threshold: float) -> tuple[str, str]:
    """Classify applicant default probability into risk tiers."""
    if probability < threshold * 0.70:
        return "Low Risk", "#2ECC71"
    elif probability < threshold:
        return "Moderate Risk", "#F39C12"
    else:
        return "High Risk", "#E74C3C"


def _affordability_follow_up(affordability_reasons: list[str]) -> str:
    if affordability_reasons:
        return "Review the affordability notes and verify income, expenses, and existing debts"
    return "No extreme affordability flag was found; complete the lender's normal affordability checks"


def _render_collateral_review(raw: pd.DataFrame) -> None:
    """Show asset ownership cues without treating them as verified collateral."""
    row = raw.iloc[0]
    property_status = "Applicant reports property ownership" if row["FLAG_OWN_REALTY"] == "Y" else "No property ownership reported"
    car_status = "Applicant reports car ownership" if row["FLAG_OWN_CAR"] == "Y" else "No car ownership reported"
    business_status = (
        "Businessman income category selected"
        if row["NAME_INCOME_TYPE"] == "Businessman"
        else f"Income category: {row['NAME_INCOME_TYPE']}"
    )
    st.subheader("Collateral and asset checks")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Asset / cue": "Property",
                    "Information available": property_status,
                    "Bank verification": "Verify title, valuation, existing liens, and whether the property is eligible collateral.",
                },
                {
                    "Asset / cue": "Vehicle",
                    "Information available": car_status,
                    "Bank verification": "Verify ownership, current value, liens, and whether the vehicle is eligible collateral.",
                },
                {
                    "Asset / cue": "Business",
                    "Information available": business_status,
                    "Bank verification": "The application records an employment/income category, not registered business equity or pledgeable security.",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "The model uses the recorded car/property ownership flags and income category "
        "as predictors. It does not verify ownership or estimate collateral value, "
        "so these cues must not independently produce a YES."
    )


def _build_summary(
    raw: pd.DataFrame,
    applicant: pd.DataFrame,
    probability: float,
    decision: str,
    risk_tier: str,
    model_threshold: float,
    applicant_reference: str,
    additional_information: dict[str, Any],
) -> pd.DataFrame:
    row = applicant.iloc[0]
    return pd.DataFrame(
        [
            {"Item": "Applicant reference", "Value": applicant_reference},
            {"Item": "Loan recommendation", "Value": decision},
            {"Item": "Risk category", "Value": risk_tier},
            {"Item": "Validation-selected cutoff", "Value": f"{model_threshold:.1%}"},
            {"Item": "Estimated payment-difficulty likelihood", "Value": f"{probability:.1%}"},
            {"Item": "First-line screening result", "Value": "Eligible (YES)" if "YES" in decision else "Not recommended (NO)"},
            {"Item": "Annual income", "Value": raw.loc[0, "AMT_INCOME_TOTAL"]},
            {"Item": "Requested loan amount", "Value": raw.loc[0, "AMT_CREDIT"]},
            {"Item": "Estimated installment", "Value": raw.loc[0, "AMT_ANNUITY"]},
            {"Item": "Loan-to-income ratio", "Value": f"{row['CREDIT_TO_INCOME_RATIO']:.2f}x"},
            {"Item": "Installment-to-income ratio", "Value": f"{row['ANNUITY_TO_INCOME_RATIO']:.1%}"},
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
    document.drawString(48, height - 52, "Applicant Credit Assessment Report")
    document.setFont("Helvetica", 9)
    document.drawString(48, height - 70, "Confidential Underwriting Decision-Support Summary")
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
        "Confidential banking document: For authorized credit underwriting review only.",
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
    st.subheader("2. Financial Affordability Analysis")
    summary_cols = st.columns(2)
    summary_cols[0].metric(
        "Loan-to-income ratio", f"{row['CREDIT_TO_INCOME_RATIO']:.2f}x"
    )
    monthly_income = income / 12
    monthly_installment_ratio = annuity / monthly_income if monthly_income else np.nan
    summary_cols[1].metric(
        "Annual Installment / Annual Income",
        f"{row['ANNUITY_TO_INCOME_RATIO']:.1%}",
        help="Annualized installment burden relative to gross annual income.",
    )

    concerns: list[str] = []
    positive_indicators: list[str] = []
    if income <= 0:
        concerns.append("Annual income is zero or missing, so affordability cannot be assessed reliably.")
    elif monthly_installment_ratio >= 0.4:
        concerns.append(
            "The proposed installment is a relatively large share of the estimated monthly income."
        )
    elif monthly_installment_ratio <= 0.2:
        positive_indicators.append(
            "The proposed installment is a relatively small share of the estimated monthly income."
        )
    if row["CREDIT_TO_INCOME_RATIO"] >= 5:
        concerns.append("The requested loan is large compared with the entered annual income.")
    elif row["CREDIT_TO_INCOME_RATIO"] <= 2:
        positive_indicators.append("The requested loan is comparatively moderate relative to the entered annual income.")
    if concerns:
        st.warning(f"Affordability concern: {concerns[0]}")
    else:
        st.success("No high-stress affordability indicator was identified from the entered values.")
    if positive_indicators:
        st.caption("Positive screening indicators: " + " ".join(positive_indicators))
    st.info(
        f"Estimated monthly income: {monthly_income:,.2f} · "
        f"Installment share of estimated monthly income: "
        f"{monthly_installment_ratio:.1%}" if monthly_income else
        "Monthly affordability cannot be calculated from zero income."
    )
    st.caption(
        "These screening indicators evaluate debt service coverage from application data. "
        "Standard bank debt-to-income (DTI) underwriting and expense verification apply."
    )
    return concerns


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
    loan_decision: str = "YES",
    risk_tier: str = "Moderate Risk",
) -> None:
    """Summarise how a bank officer can use the result without automating a decision."""
    st.subheader("6. Bank Decision Support")
    if loan_decision == "YES" and not affordability_reasons:
        action = "Approve for standard processing / First-line criteria met (YES)"
    elif loan_decision == "YES" and affordability_reasons:
        action = "Qualifies on risk score (YES), but verify affordability notes"
    else:
        action = "Decline under current terms (NO) / Restructure loan parameters or refer to manual underwriting"
    st.metric("Recommended action", action)

    factors = (
        shap_factors.sort_values("Contribution", key=abs, ascending=False)
        .head(3)["Factor"].tolist()
        if shap_factors is not None and "Factor" in shap_factors.columns
        else []
    )
    support = pd.DataFrame(
        [
            {"Assessment area": "Loan recommendation", "Information": f"{loan_decision} ({'Give Loan' if loan_decision == 'YES' else 'Do Not Give Loan'})"},
            {"Assessment area": "Risk classification", "Information": risk_tier},
            {"Assessment area": "Predicted risk", "Information": f"{probability:.1%} estimated payment-difficulty likelihood"},
            {"Assessment area": "Prediction rule", "Information": f"{'Higher' if probability >= threshold else 'Lower'} likelihood relative to model cutoff ({threshold:.1%})"},
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
            {"Assessment area": "Fairness & Compliance", "Information": "Evaluated for demographic parity and equal opportunity across demographic groups"},
            {"Assessment area": "Probability Calibration", "Information": "Probabilities calibrated on empirical validation data for reliable risk quantification"},
            {"Assessment area": "Employment verification", "Information": "Occupation reported" if occupation_reported else "Occupation not reported; verify during underwriting"},
            {"Assessment area": "Recommended action", "Information": action},
        ]
    )
    st.dataframe(support, hide_index=True, use_container_width=True)
    if recommendations:
        st.caption(
            "The recommended action supports human credit review. It serves as an assistive tool "
            "and does not replace final underwriter discretion."
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
    probability: float | None = None,
    threshold: float | None = None,
) -> pd.DataFrame | None:
    """Show explainable AI (SHAP) feature attributions and visual charts."""
    st.subheader("3. Explainable AI (SHAP) Insights")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) breaks down the prediction by measuring "
        "how much each borrower feature increases or decreases the payment-difficulty risk."
    )
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
                "Feature": APPLICATION_FEATURES,
                "Factor": [
                    FEATURE_LABELS.get(feature, feature)
                    for feature in APPLICATION_FEATURES
                ],
                "Contribution": [
                    contributions[feature] for feature in APPLICATION_FEATURES
                ],
            }
        )
        local["Absolute_Contribution"] = local["Contribution"].abs()
        local = local.sort_values("Absolute_Contribution", ascending=False)
        top_factors = local.head(7).copy()
        top_factors["Direction"] = np.where(
            top_factors["Contribution"] >= 0,
            "Increases Default Risk ⬆️",
            "Decreases Default Risk ⬇️",
        )

        risk_up = top_factors[top_factors["Contribution"] > 0]
        risk_down = top_factors[top_factors["Contribution"] < 0]
        summary_bullets = []
        if not risk_up.empty:
            summary_bullets.append(
                f"**Features elevating default risk**: {', '.join(risk_up['Factor'].head(3).tolist())}"
            )
        if not risk_down.empty:
            summary_bullets.append(
                f"**Features reducing default risk**: {', '.join(risk_down['Factor'].head(3).tolist())}"
            )
        for bullet in summary_bullets:
            st.markdown(f"- {bullet}")

        st.markdown("**Top Factors Influencing This Prediction:**")
        display_df = top_factors[["Factor", "Direction", "Contribution"]].rename(
            columns={"Contribution": "SHAP Contribution"}
        )
        st.dataframe(
            display_df.style.format({"SHAP Contribution": "{:+.4f}"}),
            hide_index=True,
            use_container_width=True,
        )

        try:
            import plotly.express as px
            import plotly.graph_objects as go

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                waterfall = top_factors.sort_values("Contribution")
                waterfall_figure = go.Figure(
                    go.Waterfall(
                        orientation="v",
                        measure=["relative"] * len(waterfall) + ["total"],
                        x=waterfall["Factor"].tolist() + ["Final Estimate"],
                        y=waterfall["Contribution"].tolist() + [0],
                        connector={"line": {"color": "#7B8794"}},
                        increasing={"marker": {"color": "#E74C3C"}},
                        decreasing={"marker": {"color": "#27AE60"}},
                        totals={"marker": {"color": "#2980B9"}},
                    )
                )
                waterfall_figure.update_layout(
                    title="SHAP Waterfall (Contribution to Risk)",
                    yaxis_title="Contribution",
                    showlegend=False,
                    height=360,
                )
                st.plotly_chart(waterfall_figure, use_container_width=True)

            with chart_col2:
                bar_figure = px.bar(
                    top_factors.sort_values("Contribution"),
                    x="Contribution",
                    y="Factor",
                    orientation="h",
                    color="Contribution",
                    color_continuous_scale=["#27AE60", "#F1C40F", "#E74C3C"],
                    title="Feature Direction and Magnitude",
                )
                bar_figure.update_layout(
                    coloraxis_showscale=False,
                    xaxis_title="Contribution to Payment-Difficulty Probability",
                    height=360,
                )
                st.plotly_chart(bar_figure, use_container_width=True)
        except ImportError:
            st.info("Install Plotly for interactive charts.")

        st.caption(
            "Factors pointing right / red push the estimate toward higher payment-difficulty risk; "
            "factors pointing left / green push it lower."
        )
        return local
    except (ImportError, KeyError, OSError, ValueError, RuntimeError) as exc:
        st.error(f"The supporting SHAP explanation could not be generated: {exc}")
        return None


def _apply_counterfactual_target(
    feature_name: str,
    target_value: float,
    current_credit: float,
    current_annuity: float,
) -> None:
    """Copy an actionable target into the matching applicant form fields."""
    if feature_name == "AMT_ANNUITY":
        st.session_state["cr_annuity"] = float(target_value)
    elif feature_name == "AMT_CREDIT":
        st.session_state["cr_credit"] = float(target_value)
        if current_credit > 0:
            st.session_state["cr_annuity"] = round(
                current_annuity * (float(target_value) / current_credit), -2
            )
    elif feature_name == "AMT_INCOME_TOTAL":
        st.session_state["cr_income"] = float(target_value)


def _render_counterfactual_recourse(
    bundle: dict[str, Any],
    raw: pd.DataFrame,
    probability: float,
    threshold: float,
) -> None:
    """Render actionable recourse parameter changes when decision is NO, and what-if simulation."""
    st.subheader("4. Actionable Recourse: Parameter Changes & Ideas to Achieve 'YES'")

    if probability < threshold:
        st.success(
            f"✅ **Application Qualifies for Loan (YES)**: The estimated default probability "
            f"({probability:.1%}) is safely below the {threshold:.1%} cutoff threshold. "
            "Current loan parameters meet first-line credit risk criteria."
        )
        with st.expander("Explore Borrowing Capacity / What-If Simulator", expanded=False):
            st.caption(
                "The applicant is already approved under current terms. Use the simulator below "
                "if the applicant wishes to test a higher loan amount or adjusted installments."
            )
            _render_interactive_simulator(bundle, raw, probability, threshold, is_approved=True)
        return

    current_credit = float(raw.loc[0, "AMT_CREDIT"])
    current_annuity = float(raw.loc[0, "AMT_ANNUITY"])
    current_income = float(raw.loc[0, "AMT_INCOME_TOTAL"])
    monthly_income = current_income / 12 if current_income > 0 else 0.0
    gap_pp = (probability - threshold) * 100

    # 1. Detailed breakdown of WHY the loan was not approved
    with st.container(border=True):
        st.markdown("### 📋 Why the Application Received 'NO'")
        st.markdown(
            f"The AI credit screening model calculated a payment-difficulty risk of **{probability:.1%}**, "
            f"which exceeds the bank's maximum allowable policy cutoff of **{threshold:.1%}** by **+{gap_pp:.1f}%**.\n\n"
            f"**Key Root Drivers Behind This Assessment:**\n"
            f"- **Cash-Flow Debt Burden**: The proposed installment of **{current_annuity:,.0f}** requires "
            f"**{(current_annuity / monthly_income):.1%}** of estimated monthly income ({monthly_income:,.0f}/mo), "
            f"leaving limited cash reserve for living expenses.\n"
            f"- **Loan Sizing Multiplier**: The requested facility of **{current_credit:,.0f}** is **{(current_credit / current_income):.1f}x** "
            f"the borrower's gross annual income ({current_income:,.0f}).\n"
            f"- **Thin-File Borrower Context**: Limited past borrowing records require relying on cash-flow coverage and asset support."
        )

    # 2. Exact Parameter Changes Needed to Get Approval
    st.markdown("### 🎯 Exact Parameter Changes to Qualify for Approval (YES)")
    st.markdown(
        f"The model identified the following specific adjustments to loan parameters that will reduce "
        f"default probability safely below the **{threshold:.1%}** threshold:"
    )

    counterfactuals = cr_compute_home_credit_counterfactuals(bundle, raw)

    if counterfactuals:
        cf_table_data = []
        for i, cf in enumerate(counterfactuals, start=1):
            cf_table_data.append(
                {
                    "Pathway": f"Option {i}: {cf['parameter']}",
                    "Current Value": f"{cf['current_value']}",
                    "Target Value for YES": f"{cf['suggested_value']}",
                    "Required Change": (
                        f"{cf['change_percent']:+.0f}%"
                        if isinstance(cf["change_percent"], (int, float)) and cf["change_percent"] != 0
                        else "-"
                    ),
                    "New Default Risk": f"{cf['new_probability']:.1%}",
                    "Updated Decision": "YES ✅ (Approved)",
                    "Action Description": cf["action"],
                }
            )

        st.dataframe(
            pd.DataFrame(cf_table_data)[
                ["Pathway", "Current Value", "Target Value for YES", "Required Change", "New Default Risk", "Updated Decision", "Action Description"]
            ],
            hide_index=True,
            use_container_width=True,
        )

        cols = st.columns(min(len(counterfactuals), 3))
        for idx, cf in enumerate(counterfactuals[:3]):
            with cols[idx]:
                st.metric(
                    label=cf["parameter"],
                    value=f"{cf['suggested_value']}",
                    delta=f"{cf['new_probability']:.1%} (YES)",
                    delta_color="normal",
                )
                st.caption(cf["action"])
                if cf.get("feature_name") in {
                    "AMT_ANNUITY",
                    "AMT_CREDIT",
                    "AMT_INCOME_TOTAL",
                }:
                    st.button(
                        "Apply target to applicant form",
                        key=f"apply_counterfactual_{idx}",
                        use_container_width=True,
                        on_click=_apply_counterfactual_target,
                        args=(
                            cf["feature_name"],
                            float(cf["suggested_value"]),
                            current_credit,
                            current_annuity,
                        ),
                        help=(
                            "Copies this suggested value into Requested credit. "
                            "For a reduced loan amount, the proportional installment is updated too."
                        ),
                    )
    else:
        st.info(
            "Single-factor parameter adjustments alone could not bring the probability below the cutoff. "
            "Please review the strategic ideas and test combined adjustments below."
        )

    # 3. Strategic Ideas for Loan Officers (Brainstorming / Suggest Ideas)
    st.markdown("### 💡 Strategic Advisory Ideas to Get This Loan Approved")
    st.markdown(
        "Bank officers can structure and present any of the following practical solutions to the customer:"
    )

    idea_col1, idea_col2 = st.columns(2)
    with idea_col1:
        with st.container(border=True):
            st.markdown("#### 1. 📅 Extend Loan Tenure (Tenure Restructuring)")
            st.markdown(
                f"**Strategy**: Retain the requested loan amount (**{current_credit:,.0f}**) but extend the loan term "
                "(e.g., from 24 to 36, 48, or 60 months).\n\n"
                "**Why it works**: Spreading principal repayment reduces the monthly installment, lowering the monthly debt-service "
                f"ratio below policy limits and cutting default probability into the approved zone (**YES**)."
            )
        with st.container(border=True):
            st.markdown("#### 3. 💵 Higher Upfront Down Payment")
            st.markdown(
                "**Strategy**: If financing a vehicle, appliance, or property, ask the customer to increase their upfront cash down payment by 15% to 25%.\n\n"
                "**Why it works**: Reduces the net borrowed amount and improves the loan-to-value (LTV) ratio, reducing credit risk."
            )
        with st.container(border=True):
            st.markdown("#### 5. 📈 Stepped Credit-Builder Facility")
            st.markdown(
                "**Strategy**: Approve an initial smaller loan within approved risk limits. Once the borrower maintains 6–12 months of clean repayment records, offer a top-up loan.\n\n"
                "**Why it works**: Allows thin-file borrowers to build a verified credit track record safely."
            )

    with idea_col2:
        with st.container(border=True):
            st.markdown("#### 2. 👥 Add an Eligible Co-Borrower or Guarantor")
            st.markdown(
                "**Strategy**: Include a working spouse, parent, or financial guarantor with verifiable income on the application.\n\n"
                "**Why it works**: Combines household income, raising total debt-service capacity and lowering effective credit-to-income risk to qualifying levels (**YES**)."
            )
        with st.container(border=True):
            st.markdown("#### 4. 🏠 Pledge Collateral (Secured Facility)")
            st.markdown(
                "**Strategy**: Convert the unsecured loan application into a secured facility by pledging property title, vehicle equity, or a bank fixed deposit.\n\n"
                "**Why it works**: Backing the loan with tangible collateral substantially mitigates loss severity and qualifies the borrower under secured lending rules."
            )

    # 4. Interactive What-If Simulator
    st.markdown("### 🎛️ Interactive 'What-If' Parameter Simulator")
    st.caption(
        "Test customized parameter combinations live with the applicant to see real-time changes in default risk and recommendation:"
    )
    _render_interactive_simulator(bundle, raw, probability, threshold, is_approved=False)


def _render_interactive_simulator(
    bundle: dict[str, Any],
    raw: pd.DataFrame,
    current_probability: float,
    threshold: float,
    is_approved: bool,
) -> None:
    current_credit = float(raw.loc[0, "AMT_CREDIT"])
    current_annuity = float(raw.loc[0, "AMT_ANNUITY"])
    current_income = float(raw.loc[0, "AMT_INCOME_TOTAL"])

    with st.container(border=True):
        st.markdown("**Test Parameter Adjustments Below:**")
        sim_col1, sim_col2 = st.columns(2)

        sim_credit = sim_col1.number_input(
            "Simulated Loan Amount",
            min_value=10000.0,
            max_value=max(current_credit * 2, 2000000.0),
            value=float(current_credit),
            step=10000.0,
            key="sim_credit",
            help="Adjust the requested loan amount to check effect on risk",
        )
        sim_annuity = sim_col2.number_input(
            "Simulated Monthly Installment",
            min_value=1000.0,
            max_value=max(current_annuity * 2, 200000.0),
            value=float(current_annuity),
            step=1000.0,
            key="sim_annuity",
            help="Extend loan term to lower the installment amount",
        )

        sim_col3, sim_col4 = st.columns(2)
        sim_income = sim_col3.number_input(
            "Simulated Annual Income (applicant + co-signer)",
            min_value=10000.0,
            max_value=max(current_income * 3, 2000000.0),
            value=float(current_income),
            step=10000.0,
            key="sim_income",
            help="Add verifiable income from spouse or co-applicant",
        )
        sim_property = sim_col4.selectbox(
            "Property Collateral Pledged",
            ["No", "Yes"],
            index=1 if raw.loc[0, "FLAG_OWN_REALTY"] == "Y" else 0,
            key="sim_property",
        )

        sim_raw = raw.copy()
        sim_raw.loc[0, "AMT_CREDIT"] = sim_credit
        sim_raw.loc[0, "AMT_ANNUITY"] = sim_annuity
        sim_raw.loc[0, "AMT_INCOME_TOTAL"] = sim_income
        sim_raw.loc[0, "FLAG_OWN_REALTY"] = "Y" if sim_property == "Yes" else "N"

        try:
            sim_applicant = cr_prepare_home_credit_application(sim_raw)
            sim_prob = cr_predict_home_credit_application(bundle, sim_applicant)
            sim_decision = "YES" if sim_prob < threshold else "NO"
            prob_delta = sim_prob - current_probability
            sim_tier, _ = _classify_risk_category(sim_prob, threshold)

            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                "Simulated Default Risk",
                f"{sim_prob:.1%}",
                delta=f"{prob_delta:+.1%}",
                delta_color="inverse",
            )
            res_col2.metric(
                "Simulated Recommendation",
                f"{sim_decision} {'✅ (Give Loan)' if sim_decision == 'YES' else '❌ (Do Not Give Loan)'}",
            )
            res_col3.metric("Simulated Risk Tier", sim_tier)

            if sim_decision == "YES":
                st.success(
                    f"🎉 **Adjusted Parameters Qualify for Approval!** "
                    f"Default probability is reduced to **{sim_prob:.1%}** (below the {threshold:.1%} threshold). "
                    "The recommendation changes to **YES (Give Loan)**!"
                )
            else:
                st.warning(
                    f"Simulated default probability is **{sim_prob:.1%}**, which still meets or exceeds "
                    f"the {threshold:.1%} cutoff. Try reducing the monthly installment or loan amount further."
                )

            st.markdown("**Comparison: Current Application vs. Simulated Terms**")
            comp_df = pd.DataFrame(
                [
                    {
                        "Parameter": "Requested Loan Amount",
                        "Original Application": f"{current_credit:,.0f}",
                        "Simulated Parameters": f"{sim_credit:,.0f}",
                        "Difference": f"{sim_credit - current_credit:+,.0f}",
                    },
                    {
                        "Parameter": "Monthly Installment",
                        "Original Application": f"{current_annuity:,.0f}",
                        "Simulated Parameters": f"{sim_annuity:,.0f}",
                        "Difference": f"{sim_annuity - current_annuity:+,.0f}",
                    },
                    {
                        "Parameter": "Annual Income",
                        "Original Application": f"{current_income:,.0f}",
                        "Simulated Parameters": f"{sim_income:,.0f}",
                        "Difference": f"{sim_income - current_income:+,.0f}",
                    },
                    {
                        "Parameter": "Default Risk (Probability)",
                        "Original Application": f"{current_probability:.1%}",
                        "Simulated Parameters": f"{sim_prob:.1%}",
                        "Difference": f"{prob_delta:+.1%}",
                    },
                    {
                        "Parameter": "Loan Recommendation",
                        "Original Application": "YES" if current_probability < threshold else "NO",
                        "Simulated Parameters": sim_decision,
                        "Difference": "Approved ✅" if sim_decision == "YES" else "Not Approved ❌",
                    },
                ]
            )
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Error evaluating simulated parameters: {exc}")


def render_credit_risk_assessment() -> None:
    """Render the applicant questionnaire and its decision-support result."""
    st.title("Applicant Credit Assessment")
    st.caption("AI-Powered Credit Risk Screening & Decision Support System for Thin-File Borrowers")
    st.markdown(
        "**Bank Decision-Support Workspace**  \n"
        "Evaluate loan applicant eligibility, default risk probability, and SHAP explainability. "
        "When an applicant is not recommended under requested terms, actionable parameter adjustments are provided to qualify for approval."
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

    # Initialise session-state defaults for the three financial inputs so that
    # "Apply this value" buttons in the counterfactual section can pre-populate them.
    if "cr_income" not in st.session_state:
        st.session_state["cr_income"] = 150000.0
    if "cr_credit" not in st.session_state:
        st.session_state["cr_credit"] = 500000.0
    if "cr_annuity" not in st.session_state:
        st.session_state["cr_annuity"] = 25000.0

    with st.form("applicant_credit_assessment"):
        applicant_reference = st.text_input(
            "Applicant reference",
            value="Applicant 001",
            help="Application or borrower identifier for tracking and reporting.",
        )
        st.subheader("Requested credit")
        amount_cols = st.columns(2)
        income = amount_cols[0].number_input(
            "Annual income",
            min_value=0.0,
            value=float(st.session_state["cr_income"]),
            step=10000.0,
            key="cr_income",
        )
        requested_credit = amount_cols[1].number_input(
            "Requested loan amount",
            min_value=0.0,
            value=float(st.session_state["cr_credit"]),
            step=10000.0,
            key="cr_credit",
        )
        amount_cols = st.columns(2)
        annuity = amount_cols[0].number_input(
            "Estimated installment amount",
            min_value=0.0,
            value=float(st.session_state["cr_annuity"]),
            step=1000.0,
            help="Enter the proposed monthly installment amount.",
            key="cr_annuity",
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
        st.markdown("**Existing Obligations & Credit Bureau Context (Optional)**")
        additional_information_provided = st.checkbox(
            "Include existing debts, repayments, and bureau history disclosures"
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
            "Run Credit Assessment", type="primary", use_container_width=True
        )

    if not submitted:
        st.info("Complete the applicant information above and select **Run Credit Assessment**.")
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
                "AMT_GOODS_PRICE": np.nan,
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
        range_warnings = _out_of_training_range(raw, bundle)
        applicant = cr_prepare_home_credit_application(raw)
        probability = cr_predict_home_credit_application(bundle, applicant)
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Could not assess this application: {exc}")
        return

    threshold = float(bundle["threshold"])
    loan_decision, loan_recommendation_msg = _evaluate_loan_recommendation(
        probability, threshold
    )
    risk_tier, risk_color = _classify_risk_category(probability, threshold)

    st.subheader("1. Credit Risk Assessment – Main Output")
    result_cols = st.columns(5)
    result_cols[0].metric("Applicant", applicant_reference)
    result_cols[1].metric("Default Risk (PD)", f"{probability:.1%}")
    result_cols[2].metric("Risk Category", risk_tier)
    result_cols[3].metric("Model Cutoff", f"{threshold:.1%}")
    result_cols[4].metric("Loan Recommendation", f"{loan_decision} {'✅' if loan_decision == 'YES' else '❌'}")

    if loan_decision == "YES":
        st.success(f"### ✅ **Recommendation: YES (Give Loan)**\n\n{loan_recommendation_msg}")
    else:
        gap = (probability - threshold) * 100
        monthly_inc = income / 12 if income > 0 else 0.0
        inst_pct = (annuity / monthly_inc * 100) if monthly_inc > 0 else 0.0
        credit_income_mult = (requested_credit / income) if income > 0 else 0.0
        st.error(
            f"### ❌ **Recommendation: NO (Do Not Give Loan Under Current Terms)**\n\n"
            f"**Why the Loan was Not Recommended (NO):**\n"
            f"- **Elevated Default Risk**: Estimated payment-difficulty probability is **{probability:.1%}**, "
            f"exceeding the bank policy cutoff of **{threshold:.1%}** by **+{gap:.1f} percentage points**.\n"
            f"- **Repayment Stress**: The requested monthly installment ({annuity:,.0f}) absorbs "
            f"**{inst_pct:.1f}%** of estimated monthly income ({monthly_inc:,.0f}/mo), indicating elevated repayment strain.\n"
            f"- **High Loan Leverage**: Requested loan ({requested_credit:,.0f}) represents **{credit_income_mult:.1f}x** "
            f"the applicant's annual income ({income:,.0f}).\n"
            f"- **Next Steps & Recourse**: Review Section 4 below for specific parameter changes (e.g. extending tenure, lower credit amount, co-borrower) to turn this decision into **YES**."
        )

    st.caption(
        f"Bank Policy Risk Cutoff: {threshold:.1%}. "
        "Recommendation **YES** indicates default risk is within policy thresholds (applicant qualifies for loan). "
        "Recommendation **NO** indicates elevated default risk exceeding the cutoff (loan not recommended under current terms). "
        "See Section 4 below for actionable parameter adjustments and strategic advisory ideas to achieve YES."
    )
    st.caption(
        "This tool provides first-line credit screening support. "
        "It assists informed underwriting decisions and should be verified alongside standard bank policy checks."
    )

    if range_warnings:
        st.warning(
            "Note: One or more financial inputs exceed typical benchmark limits. "
            "Verify applicant documentation for high-value entries."
        )
        for warning in range_warnings:
            st.write(f"- {warning}")

    if validation_warnings:
        st.warning("Please review these affordability notes:")
        for warning in validation_warnings:
            st.write(f"- {warning}")

    # 2. Financial Affordability Analysis (collapsed dropdown)
    with st.expander("📊 Financial Affordability Analysis", expanded=False):
        affordability_reasons = _render_affordability_summary(raw, applicant)
        affordability_follow_up = _affordability_follow_up(affordability_reasons)
        st.info(f"**Affordability guidance**: {affordability_follow_up}")

    # 3. Explainable AI (SHAP) Insights (collapsed dropdown)
    with st.expander("🔍 Explainable AI (SHAP) Insights — Why This Decision?", expanded=False):
        shap_factors = _render_shap_explanation(
            prepared_dir, bundle, applicant, probability, threshold
        )

    # 4. Actionable Recourse: Parameter Changes to Achieve 'YES'
    _render_counterfactual_recourse(bundle, raw, probability, threshold)

    # 5. Applicant Profile & Collateral Review
    _render_applicant_profile(raw, applicant)
    _render_collateral_review(raw)

    # 6. Bank Decision Support & Next Steps
    recommendations = _render_next_steps(raw, applicant, probability)
    _render_bank_decision_support(
        probability,
        threshold,
        affordability_reasons,
        recommendations,
        shap_factors,
        raw.loc[0, "OCCUPATION_TYPE"] is not None,
        additional_information,
        loan_decision=loan_decision,
        risk_tier=risk_tier,
    )

    # 7. Summary & Download
    summary = _build_summary(
        raw,
        applicant,
        probability,
        f"{loan_decision} ({'Give Loan' if loan_decision == 'YES' else 'Do Not Give Loan'})",
        risk_tier,
        threshold,
        applicant_reference,
        additional_information,
    )
    st.subheader("7. Application Summary & Export")
    st.caption("Review the assessment details below or download reports for bank records.")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download CSV Summary",
            summary.to_csv(index=False).encode("utf-8-sig"),
            f"{applicant_reference.replace(' ', '_')}_credit_assessment.csv",
            "text/csv",
            use_container_width=True,
        )
    with dl_col2:
        pdf_summary = _build_pdf(summary)
        if pdf_summary is None:
            st.warning("ReportLab is not installed; PDF export unavailable.")
        else:
            st.download_button(
                "Download PDF Report",
                pdf_summary,
                f"{applicant_reference.replace(' ', '_')}_credit_assessment.pdf",
                "application/pdf",
                use_container_width=True,
            )

    st.caption(
        "Responsible AI notice: This tool is designed to support, not replace, credit officers. "
        "Ensure fair lending policies and verify identity and income documentation."
    )
