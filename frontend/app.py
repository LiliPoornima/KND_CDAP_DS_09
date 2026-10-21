
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

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Credit Risk")
    st.write("Explore credit-risk research findings and evaluation.")

    st.markdown("### Financial Inclusion")
    st.write(
        "Explore Global Findex 2025 data, digital financial inclusion "
        "dimensions, cluster profiles, and supervised model evaluation."
    )

with col2:
    st.markdown("### Market Risk")
    st.write("Explore market-risk research findings and evaluation.")

    st.markdown("### Fraud Detection")
    st.write("Explore fraud-detection research findings and evaluation.")

st.info(
    "The component pages are developed independently by the research "
    "team. Findings will be displayed as their dashboard implementations "
    "become available."
)
