"""Global financial analytics application and explicit component navigation."""

from pathlib import Path

import streamlit as st


FRONTEND_DIR = Path(__file__).resolve().parent
STYLES_PATH = FRONTEND_DIR / "assets" / "styles" / "main.css"

st.set_page_config(
    page_title="Global Financial Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if STYLES_PATH.exists():
    st.markdown(
        f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _render_home() -> None:
    if st.session_state.get("home_component_view") == "financial_inclusion":
        if st.button("Back to Home", key="home_back_from_financial_inclusion"):
            st.session_state.pop("home_component_view", None)
            st.rerun()
        from frontend.financial_inclusion.new_dashboard import render_home_assessment

        render_home_assessment()
        return

    st.title("Global Financial Analytics")
    st.subheader("Research and Decision Support Dashboard")
    st.markdown(
        """
        Welcome to the group's research dashboard.

        Select a research component from the navigation sidebar to explore
        its datasets, analytical findings, model evaluation, and methodology.
        """
    )
    st.divider()
    st.subheader("Research Components")
    component_rows = [
        (
            ("01 · Credit Risk", "Credit-risk assessment for thin-file borrowers.", "pages/1_Credit_Risk.py"),
            ("02 · Financial Inclusion", "Digital financial inclusion assessment and explainable inclusion profiles.", "pages/6_Financial_Inclusion_New_Dashboard.py"),
        ),
        (
            ("03 · Market Risk", "External market-risk analysis and lending insights.", "pages/3_Market_Risk.py"),
            ("04 · Financial Well-being", "Financial well-being, capabilities, and resilience assessment.", None),
        ),
    ]
    for row in component_rows:
        columns = st.columns(2)
        for column, (title, description, page) in zip(columns, row):
            with column:
                with st.container(border=True):
                    st.markdown(f"### {title}")
                    st.write(description)
                    if page:
                        if st.button("Open component", key=f"home_{title}", use_container_width=True):
                            if "02" in title:
                                st.session_state["home_component_view"] = "financial_inclusion"
                                st.rerun()
                            else:
                                st.switch_page(page)
                    elif st.button("Open component", key=f"home_{title}", use_container_width=True):
                        st.info("A Financial Well-being assessment page is not available in this project.")
    st.info(
        "The components are developed independently. Select a component in "
        "the navigation sidebar to open its dashboard."
    )


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

        st.Page("pages/3_Market_Risk.py", title="Market Risk", icon="📈"),

        st.Page("pages/4_Fraud_Detection.py",
                title="Fraud Detection", icon="🛡️"),
        # The new comparison page is separate; the original Financial Inclusion
        # destination above keeps its existing page and behavior.

    ],
    position="sidebar",
    expanded=True,
)
navigation.run()
