import pandas as pd
import streamlit as st
from .. import data_loader as loader
from ..constants import CLUSTER_IDS, CLUSTER_LABELS
from .common import profile_frame

def render(df: pd.DataFrame) -> None:
    st.header("Policy recommendations")
    st.caption("Possible follow-up actions for stakeholder investigation, derived from observed dimension contrasts. These are not proven interventions.")
    profiles = profile_frame(df)
    cluster = st.selectbox("Profile to review", CLUSTER_IDS, format_func=lambda x: CLUSTER_LABELS[x], key="policy_profile")
    selected = profiles.loc[profiles.cluster_id == cluster].iloc[0]
    comparison = pd.DataFrame([{"dimension": loader.DIMENSION_LABELS[c], "difference": float(selected[f"{c}_mean"] - pd.to_numeric(df[c],errors="coerce").mean())} for c in loader.DIMENSION_COLUMNS])
    suggestions = loader.build_community_recommendations(comparison, reference_label="pooled analytical sample")
    st.metric("Profile sample", f"{int(selected.respondents):,} ({selected.percent_of_candidate:.1f}%)")
    st.dataframe(comparison.rename(columns={"dimension":"Dimension","difference":"Difference from pooled mean (score points)"}).round(3), hide_index=True, width="stretch")
    if not suggestions:
        st.info("No dimension mean is below the pooled reference for this profile; no rule-based follow-up is generated.")
    for item in suggestions:
        with st.container(border=True):
            st.markdown(f"**Observed pattern â€” {item['dimension']}**")
            st.write(item["evidence"])
            st.write(f"**Potential follow-up:** {item['action']}")
            st.write(f"**Stakeholders:** {item['stakeholder']}")
            st.write(f"**Possible monitoring indicator:** {item['monitor']}")
            st.caption(item["limitation"])
    st.caption("The profile mean and pooled mean are descriptive, unweighted summaries. The reference includes the selected profile; conclusions depend on the available operational indicators.")
