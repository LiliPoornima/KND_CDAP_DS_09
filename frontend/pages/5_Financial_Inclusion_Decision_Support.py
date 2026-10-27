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

render(analytical_df)
