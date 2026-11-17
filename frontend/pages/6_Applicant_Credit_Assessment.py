"""Applicant Credit Assessment dashboard."""

import streamlit as st
from frontend.components.credit_risk_assessment import render_credit_risk_assessment

st.set_page_config(
    page_title="Applicant Credit Assessment",
    page_icon="🏦",
    layout="wide",
)

render_credit_risk_assessment()
