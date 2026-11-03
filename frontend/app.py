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
    left, right = st.columns(2)
    with left:
        st.markdown("### Credit Risk")
        st.write("Explore credit-risk research findings and evaluation.")
        st.markdown("### Market Risk")
        st.write("Explore market-risk research findings and evaluation.")
    with right:
        st.markdown("### Fraud Detection")
        st.write("Explore fraud-detection research findings and evaluation.")
        st.markdown("### Financial Inclusion")
        st.write(
            "Explore saved inclusion profiles, explainable model outputs, "
            "population gaps, and individual assessment."
        )
    st.info(
        "The components are developed independently. Select a component in "
        "the navigation sidebar to open its dashboard."
    )


navigation = st.navigation(
    [
        st.Page(_render_home, title="Home", icon="🏠", default=True),
        st.Page("pages/1_Credit_Risk.py", title="Credit Risk", icon="💳"),
        st.Page(
            "pages/6_Financial_Inclusion_New_Dashboard.py",
            title="Financial Inclusion – New Dashboard",
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
