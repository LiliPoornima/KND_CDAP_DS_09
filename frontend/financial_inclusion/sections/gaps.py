import pandas as pd
import plotly.express as px
import streamlit as st
from .common import dimension_means, show_caution, subgroup_filter
from .. import data_loader as loader

def render(df: pd.DataFrame) -> None:
    st.header("Digital inclusion gap detector")
    label, selected = subgroup_filter(df, "gap")
    if selected.empty:
        st.info("No respondents match this selection."); return
    show_caution(len(selected))
    means = dimension_means(selected); ref = dimension_means(df).set_index("Dimension")["Mean score"]
    means["Pooled mean score"] = means.Dimension.map(ref)
    means["Difference in mean score"] = means["Mean score"] - means["Pooled mean score"]
    means["Selected valid n"] = means["Valid respondents"]
    st.metric("Available sample", f"{len(selected):,}")
    plot_data = means[["Dimension", "Mean score", "Pooled mean score"]].melt(id_vars="Dimension", value_vars=["Mean score","Pooled mean score"], var_name="Population", value_name="Score")
    st.plotly_chart(px.bar(plot_data, x="Dimension", y="Score", color="Population", barmode="group", range_y=[0,1], title=f"{label} versus pooled sample"), config={"responsive": True})
    st.dataframe(means.round(3), hide_index=True, width="stretch")
    gap = means.sort_values("Difference in mean score").iloc[0]
    st.markdown(f"Largest observed negative contrast: **{gap.Dimension}**, {gap['Difference in mean score']:+.3f} mean-score points versus the pooled sample (selected n={len(selected):,}). This is a difference in dimension means, not a percentage-point prevalence difference.")
    recs = loader.build_community_recommendations(means.rename(columns={"Dimension":"dimension", "Difference in mean score":"difference"}), reference_label="pooled analytical sample")
    if recs:
        st.markdown("**Areas stakeholders could investigate:** " + "; ".join(r['action'] for r in recs[:3]))
    st.caption("Observed differences do not identify causes. The pooled reference includes the selected subgroup. Summaries are unweighted.")
