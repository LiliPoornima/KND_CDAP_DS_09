"""Five-stage stakeholder workflow for the additive Component 02 dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .sections import (
    comparison,
    distribution,
    explainability,
    gaps,
    overview,
    profiles,
    recommendations,
    subgroups,
)


WORKFLOW = (
    "Executive Overview",
    "Discover Inclusion Profiles",
    "Explain the Profiles",
    "Explore Population Gaps",
    "Decision Support & Action Planning",
)


def render(df: pd.DataFrame) -> None:
    """Render existing verified Component 02 analyses in a five-step flow."""
    section = st.radio(
        "Decision workflow",
        WORKFLOW,
        horizontal=True,
        key="fi_dss_workflow",
    )
    st.divider()

    if section == WORKFLOW[0]:
        st.title("Global Digital Financial Inclusion Gap Profiler")
        st.caption("Explainable Digital Financial Inclusion Decision Support System (EDFI-DSS)")
        st.write(
            "This system identifies multidimensional digital financial inclusion profiles, "
            "explains their observed characteristics, compares population groups, and "
            "supports evidence-informed decisions about where further investigation or "
            "targeted support may be needed."
        )
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
        st.title("Explain the Profiles")
        st.info(
            "The supervised model predicts membership in previously discovered profiles. "
            "It does not directly predict financial inclusion success or establish causes of exclusion."
        )
        explainability.render(df)
        st.warning("Country-held-out validation is pending; current evaluation does not establish generalisation to unseen countries.")
    elif section == WORKFLOW[3]:
        st.title("Explore Population Gaps")
        st.caption("Compare descriptive sample patterns. These summaries are unweighted and do not estimate national prevalence.")
        gap_tab, demographic_tab, income_tab, region_tab = st.tabs(
            ["Gap detector", "Demographics", "Income group", "Region"]
        )
        with gap_tab:
            gaps.render(df)
        with demographic_tab:
            subgroups.render_demographics(df)
        with income_tab:
            subgroups.render_income(df)
        with region_tab:
            subgroups.render_region(df)
    else:
        st.title("Decision Support & Action Planning")
        st.caption("Potential follow-up categories are hypotheses for stakeholder review, not proven interventions or automated policy decisions.")
        recommendations.render(df)
        st.info(
            "Individual profile assessment remains available in the existing Financial Inclusion dashboard. "
            "This new page preserves that destination and its saved-model inference workflow."
        )

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
   
