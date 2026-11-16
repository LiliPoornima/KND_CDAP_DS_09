"""Additive five-stage Component 02 decision-support dashboard."""

from __future__ import annotations

import streamlit as st

from frontend.financial_inclusion import data_loader
from frontend.financial_inclusion.decision_support import render


@st.cache_data(show_spinner="Loading the validated Component 02 analytical sample…")
def load_dashboard_dataset():
    return data_loader.load_final_dataset()


st.set_page_config(
    page_title="Financial Inclusion Decision Support",
    page_icon="🌐",
    layout="wide",
)

try:
    analytical_df = load_dashboard_dataset()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.title("Global Digital Financial Inclusion Gap Profiler")
    st.error("Could not load the validated Component 02 analytical dataset.")
    st.info(str(exc))
    st.stop()

st.title("🌐 Component 02: Financial Inclusion Decision Support")
st.subheader("Explainable Profiling of Digital Financial Inclusion Gaps")
st.markdown(
    "This research component identifies distinct financial inclusion profiles, "
    "explores observed differences across populations, and provides explainable "
    "individual assessments to support stakeholder investigation. Results describe "
    "patterns in the available analytical sample and do not establish causal relationships."
)


render(analytical_df)
