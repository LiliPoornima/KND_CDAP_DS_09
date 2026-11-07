"""Streamlit page for Component 05: Explainable Financial Capability Assessment.

Thin entry point: all logic lives in ``frontend/financial_capability`` and
``backend/src/financial_capability``. Can be run standalone with
``streamlit run frontend/pages/7_Financial_Capability.py`` or registered in the
main app's navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.financial_capability.constants import SECTIONS  # noqa: E402
from frontend.financial_capability.styles import inject_fintech_styles  # noqa: E402
from frontend.financial_capability.sections import (  # noqa: E402
    capability_gaps,
    data_quality,
    individual_assessment,
    methodology,
    model_performance,
    overview,
    shap_explanation,
    what_if,
)

try:  # the main app already sets the page config; standalone runs set it here
    st.set_page_config(page_title="Financial Capability Assessment", page_icon="🎓", layout="wide")
except st.errors.StreamlitAPIException:
    pass

inject_fintech_styles()

st.title("Explainable Financial Capability Assessment")
st.caption("Component 05 - Phase 1 | CFPB NFWBS 2016 | Random Forest / XGBoost | SHAP | "
           "transparent capability rules")

section = st.radio("Dashboard section", SECTIONS, horizontal=True, key="financial_capability_section")
st.divider()

RENDERERS = {
    SECTIONS[0]: overview.render,
    SECTIONS[1]: data_quality.render,
    SECTIONS[2]: model_performance.render,
    SECTIONS[3]: individual_assessment.render,
    SECTIONS[4]: shap_explanation.render,
    SECTIONS[5]: capability_gaps.render,
    SECTIONS[6]: what_if.render,
    SECTIONS[7]: methodology.render,
}
RENDERERS[section]()

st.divider()
st.caption("Component 05 | Educational support only, not financial advice | US 2016 survey data.")

