"""Capability Gaps & Suggested Categories section (transparent rules, labels only)."""

from __future__ import annotations

import streamlit as st

from backend.src.financial_capability.fc_capability_rules import (
    RULES,
    STATUS_GAP,
    STATUS_INSUFFICIENT,
    identify_capability_gaps,
    map_gaps_to_categories,
)
from backend.src.financial_capability.fc_predictor import validate_response
from frontend.financial_capability.data_loader import require_response


def render() -> None:
    st.subheader("Capability Gaps & Suggested Categories")
    st.info("These results come from **transparent rules on your actual answers** (using the "
            "official codebook meaning of each response) - not from SHAP. Wording is "
            "'possible improvement area', never a diagnosis. Phase 1 suggests category labels "
            "only; lessons and quizzes are planned for a later phase.")
    response = require_response()
    if response is None:
        return
    row = validate_response(response).iloc[0].to_dict()
    results = identify_capability_gaps(row)

    categories = map_gaps_to_categories(results)
    if categories:
        st.success("Suggested learning categories: **" + ", ".join(categories) + "**")
    elif all(r.status == STATUS_INSUFFICIENT for r in results):
        st.warning("Insufficient information to suggest a category. Answer more questions.")
    else:
        st.success("No improvement area was indicated by the rules for the answers given.")

    cols = st.columns(2)
    for i, r in enumerate(results):
        with cols[i % 2].container(border=True):
            st.markdown(f"#### {r.area}")
            if r.status == STATUS_GAP:
                st.warning(f"{r.status} - {r.strength}")
                for reason in r.triggered:
                    st.markdown(f"- {reason}")
                st.markdown(f"Suggested learning category: **{r.learning_category}**")
            elif r.status == STATUS_INSUFFICIENT:
                st.info(r.status)
            else:
                st.success(r.status)
            st.caption(f"{r.n_answered} of {r.n_indicators} indicator questions answered.")

    with st.expander("Rule definitions (proposed thresholds - to be agreed with the supervisor)"):
        for area, indicators in RULES.items():
            st.markdown(f"**{area}**")
            for ind in indicators:
                st.markdown(f"- `{ind.variable}`: {ind.reason}")

