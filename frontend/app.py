"""Global financial analytics application and explicit component navigation."""

from pathlib import Path
import sys

import streamlit as st


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STYLES_PATH = FRONTEND_DIR / "assets" / "styles" / "main.css"

st.set_page_config(
    page_title="FinSight AI | Intelligent Loan Decision Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if STYLES_PATH.exists():
    st.markdown(
        f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _return_to_home_overview() -> None:
    st.session_state["home_component_nav"] = None


def _select_home_component(component: str) -> None:
    st.session_state["home_component_nav"] = component


def _render_home_overview() -> None:
    st.subheader("Four complementary perspectives")
    components = [
        (
            "01",
            "🛡️",
            "Credit Risk Assessment",
            "Evaluates potential credit default risk, particularly for thin-file borrowers with limited traditional credit histories.",
            ("Alternative-data credit assessment",
             "Default-risk prediction", "Fairness and explainability"),
        ),
        (
            "02",
            "📱",
            "Digital Financial Inclusion",
            "Identifies an individual's financial inclusion profile and gaps in financial access, connectivity, digital usage, safety, and resilience.",
            ("Five-dimensional assessment", "Inclusion profile identification",
             "Explainable individual insights"),
        ),
        (
            "03",
            "🌐",
            "Market Risk Analysis",
            "Examines external market conditions that may influence financial circumstances and lending decisions.",
            ("Market condition analysis",
             "Potential external risks", "Lending-context insights"),
        ),
        (
            "04",
            "💼",
            "Financial Well-being Assessment",
            "Assesses financial well-being, capabilities, and resilience to add context about an applicant's financial circumstances.",
            ("Financial capability", "Saving and budgeting", "Financial resilience"),
        ),
    ]

    for row_index in range(2):
        columns = st.columns(2, gap="medium")
        for column, component in zip(columns, components[row_index * 2:row_index * 2 + 2]):
            number, icon, title, description, highlights = component
            with column:
                with st.container(key=f"home_component_card_{number}", border=True):
                    st.markdown(
                        f"<span class='home-component-icon'>{icon}</span> "
                        f"<span class='home-component-number'>COMPONENT {number}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### {title}")
                    st.write(description)
                    items = "".join(f"<li>{item}</li>" for item in highlights)
                    st.markdown(
                        f"<ul class='home-feature-list'>{items}</ul>", unsafe_allow_html=True)

    st.subheader("How FinSight AI supports lending decisions")
    workflow = (
        ("01", "Customer approaches the bank"),
        ("02", "Four components assess complementary factors"),
        ("03", "The platform presents explainable insights"),
        ("04", "A bank officer reviews the findings"),
        ("05", "The institution makes its lending decision"),
    )
    workflow_columns = st.columns(5, gap="small")
    for column, (step, description) in zip(workflow_columns, workflow):
        with column:
            with st.container(key=f"home_workflow_step_{step}", border=True):
                st.markdown(f"**{step}**")
                st.caption(description)

    st.subheader("Why FinSight AI")
    value_props = (
        ("Comprehensive assessment",
         "Multiple perspectives on the applicant and financial environment."),
        ("Explainable insights",
         "Understand the factors contributing to available model outputs."),
        ("Financial inclusion awareness",
         "Identify digital financial access and participation gaps."),
        ("Human-led decisions",
         "Support officers without replacing lending policies or professional judgment."),
    )
    value_columns = st.columns(4, gap="small")
    for index, (column, (title, detail)) in enumerate(zip(value_columns, value_props), start=1):
        with column:
            with st.container(key=f"home_value_card_{index}", border=True):
                st.markdown(f"**{title}**")
                st.caption(detail)

    st.markdown(
        "<div class='home-disclaimer'><strong>Decision-support notice</strong><br>"
        "FinSight AI provides additional decision-support insights. It does not "
        "automatically approve or reject loan applications. Final lending decisions "
        "remain with the financial institution and its applicable policies.</div>",
        unsafe_allow_html=True,
    )


def _render_selected_component(selected: str) -> None:
    if selected == "credit":
        from frontend.components.credit_risk_assessment import render_credit_risk_assessment

        render_credit_risk_assessment()
    elif selected == "inclusion":
        from frontend.financial_inclusion.new_dashboard import render_home_assessment

        render_home_assessment()
    elif selected == "market":
        st.title("Component 03 · Market Risk Analysis")
        st.info(
            "A Market Risk assessment frontend is not implemented in this project yet. "
            "The existing sidebar destination remains available, but its page currently contains no rendered interface."
        )
    else:
        from frontend.financial_capability.constants import (
            BANK_LOAN_NOTICE,
            COMPONENT_SUBTITLE,
            SECTIONS,
        )
        from frontend.financial_capability.sections import (
            capability_gaps,
            data_quality,
            individual_assessment,
            methodology,
            model_performance,
            overview,
            shap_explanation,
            what_if,
        )
        from frontend.financial_capability.styles import inject_fintech_styles

        inject_fintech_styles()
        st.title("Component 04 · Financial Well-being Assessment")
        st.caption(COMPONENT_SUBTITLE)

        renderers = {
            SECTIONS[0]: overview.render,
            SECTIONS[1]: data_quality.render,
            SECTIONS[2]: model_performance.render,
            SECTIONS[3]: individual_assessment.render,
            SECTIONS[4]: shap_explanation.render,
            SECTIONS[5]: capability_gaps.render,
            SECTIONS[6]: what_if.render,
            SECTIONS[7]: methodology.render,
        }
        selected_tab = st.radio(
            "Component 04 section",
            SECTIONS,
            index=3,
            horizontal=True,
            key="home_fc_section_radio",
        )
        st.divider()
        renderers[selected_tab]()
        st.divider()
        st.caption(
            f"Component 04 | {BANK_LOAN_NOTICE} | Trained on CFPB NFWBS 2016 public survey data."
        )



def _render_home() -> None:
    with st.container(key="home_hero"):
        hero_copy, hero_visual = st.columns(
            [1.25, 0.75], gap="large", vertical_alignment="center")
        with hero_copy:
            st.markdown(
                "<div class='home-hero-eyebrow'>FinSight AI · Intelligent Loan Decision Support</div>", unsafe_allow_html=True)
            st.markdown(
                "# Smarter Lending Through Integrated Financial Intelligence")
            st.markdown(
                "<div class='home-hero-copy'>Explore a comprehensive decision-support platform "
                "that combines four complementary AI-driven assessments to provide a broader "
                "understanding of loan applicants and the financial environment.</div>",
                unsafe_allow_html=True,
            )
        with hero_visual:
            with st.container(key="home_hero_visual"):
                st.image(
                    FRONTEND_DIR / "assets" / "images" / "financial-review.jpg",
                    caption="Financial professionals reviewing information together",
                    use_container_width=True,
                )

    st.markdown("### Integrated component navigation")
    component_options = (
        ("credit", "🛡️  01 · Credit Risk"),
        ("inclusion", "📱  02 · Financial Inclusion"),
        ("market", "🌐  03 · Market Risk"),
        ("wellbeing", "💼  04 · Financial Well-being"),
    )
    selected = st.session_state.get("home_component_nav")
    if selected is not None:
        st.button(
            "Return to Home Overview",
            key="home_return_to_overview",
            on_click=_return_to_home_overview,
        )
    nav_columns = st.columns(4, gap="small")
    for column, (component, label) in zip(nav_columns, component_options):
        with column:
            st.button(
                label,
                key=f"home_nav_{component}",
                type="primary" if selected == component else "secondary",
                use_container_width=True,
                on_click=_select_home_component,
                args=(component,),
            )

    if selected is None:
        _render_home_overview()
    else:
        st.divider()
        _render_selected_component(selected)


navigation = st.navigation(
    [
        st.Page(_render_home, title="Home", icon="🏠", default=True),
        st.Page("pages/1_Credit_Risk.py", title="Credit Risk", icon="💳"),
        st.Page(
            "pages/6_Applicant_Credit_Assessment.py",
            title="Applicant Credit Assessment",
            icon="📝",
        ),
        st.Page(
            "pages/6_Financial_Inclusion_New_Dashboard.py",
            title="Financial Inclusion",
            icon="💹",
        ),




        st.Page("pages/4_Fraud_Detection.py",
                title="Fraud Detection", icon="🛡️"),
        st.Page("pages/7_Financial_Capability.py",
                title="Financial Well-being (Component 04)", icon="🏦"),
        # The new comparison page is separate; the original Financial Inclusion
        # destination above keeps its existing page and behavior.

    ],
    position="sidebar",
    expanded=True,
)
navigation.run()
