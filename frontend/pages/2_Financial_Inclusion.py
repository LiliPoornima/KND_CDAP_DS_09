"""Thin Streamlit entry point for Component 02."""

from __future__ import annotations

import streamlit as st

from frontend.financial_inclusion import data_loader
from frontend.financial_inclusion.constants import SECTIONS
from frontend.financial_inclusion.sections import comparison, distribution, explainability, gaps, overview, profiles, recommendations, subgroups


@st.cache_data(show_spinner="Loading the validated Component 02 analytical sample…")
def load_dashboard_dataset():
    return data_loader.load_final_dataset()


st.set_page_config(page_title="Global Digital Financial Inclusion Gap Profiler", page_icon="🌐", layout="wide")
st.title("Global Digital Financial Inclusion Gap Profiler")
st.caption("Explainable Digital Financial Inclusion Decision Support System (EDFI-DSS)")
try:
    analytical_df = load_dashboard_dataset()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error("Could not load the validated Component 02 analytical dataset.")
    st.info(str(exc))
    st.stop()

section = st.radio("Dashboard section", SECTIONS, horizontal=True, key="financial_inclusion_section")
st.divider()
renderers = {
    SECTIONS[0]: overview.render,
    SECTIONS[1]: profiles.render,
    SECTIONS[2]: distribution.render,
    SECTIONS[3]: comparison.render,
    SECTIONS[4]: explainability.render,
    SECTIONS[5]: subgroups.render_demographics,
    SECTIONS[6]: subgroups.render_income,
    SECTIONS[7]: subgroups.render_region,
    SECTIONS[8]: gaps.render,
    SECTIONS[9]: recommendations.render,
}
renderers[section](analytical_df)
st.divider()
st.caption("Component 02 · Global Findex 2025 individual microdata · survey year 2024 · ")
