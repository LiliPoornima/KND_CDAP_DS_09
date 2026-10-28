"""Five-stage stakeholder workflow for the additive Component 02 dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .sections import (
    comparison,
    distribution,
    explainability,
    gaps,
    individual_assessment,
    overview,
    profiles,
    subgroups,
)


WORKFLOW = (
    "Executive Overview",
    "Discover Inclusion Profiles",
    "Profiles Explanation",
    "Explore Population Gaps",
    "Decision Support & Action Planning",
)


def render(df: pd.DataFrame) -> None:
    """Render existing verified Component 02 analyses in a five-step flow."""
    section = st.radio(
        "Financial Inclusion sections",
        WORKFLOW,
        horizontal=True,
        label_visibility="collapsed",
        key="fi_dss_workflow",
    )
    st.divider()

    if section == WORKFLOW[0]:
        
        
        _workflow_status()
        overview.render(df)
    elif section == WORKFLOW[1]:
        st.title("Discover Inclusion Profiles")
        profiles.render(df)
        with st.expander("Compare profile dimension patterns"):
            comparison.render(df)
        with st.expander("View profile distribution by economy"):
            distribution.render(df)
    elif section == WORKFLOW[2]:
        st.title("Profiles Explanation") 
        st.info(
            "The supervised model predicts membership in previously discovered profiles. "
            "It does not directly predict financial inclusion success or establish causes of exclusion."
        )
        explainability.render(df)
    elif section == WORKFLOW[3]:
        st.title("Explore Population Gaps")
        st.caption(
            "Compare dimension scores across populations and review saved profile "
            "mixes by demographic."
        )
        gap_tab, demographic_tab = st.tabs(["Gap detector", "Demographics"])
        with gap_tab:
            gaps.render(df)
        with demographic_tab:
            subgroups.render_demographics(df)
    else:
        st.title("Decision Support & Action Planning")
        individual_assessment.render()

    st.divider()
    st.caption(
        "Component 02 · Saved profile and analysis outputs · Findings describe predictive/descriptive associations; they do not establish causation."
    )


def _workflow_status() -> None:
    stages = [
        ("Data preparation and indicator validation", "Completed · validated dataset pipeline"),
        ("Five-dimensional representation", "Completed · saved dimension scores"),
        ("Inclusion profile discovery", "Completed · saved profile assignments"),
        ("Model explanation and interpretation", "Completed outputs · saved SHAP artifacts"),
        ("Population-gap analysis", "Pending · descriptive dashboard analysis"),
        ("Country-held-out validation", "Pending"),
    ]
    st.subheader("Research workflow status")
    cols = st.columns(3)
    for index, (stage, state) in enumerate(stages):
        with cols[index % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{stage}**")
                st.caption(state)
   
