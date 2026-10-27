import pandas as pd
import plotly.express as px
import streamlit as st
from .common import MIN_DISPLAY_N, counts_by_cluster, show_caution
from ..constants import CLUSTER_LABELS

def render(df: pd.DataFrame) -> None:
    st.header("Global cluster distribution")
    st.caption("Economy-level summaries are unweighted sample counts and shares, not national prevalence estimates or rankings.")
    table = df.groupby("economy", dropna=False).agg(sample_size=("cluster_id", "size")).reset_index()
    parts = df.groupby(["economy", "cluster_id"]).size().rename("respondents").reset_index()
    parts["percent_within_economy"] = parts.respondents / parts.groupby("economy").respondents.transform("sum") * 100
    dominant = parts.sort_values(["economy", "respondents", "cluster_id"], ascending=[True, False, True]).drop_duplicates("economy")[['economy','cluster_id']].rename(columns={'cluster_id':'dominant_cluster_id'})
    table = table.merge(dominant, on="economy")
    table["dominant_cluster"] = table.dominant_cluster_id.map(CLUSTER_LABELS)
    wide_counts = parts.pivot(index="economy", columns="cluster_id", values="respondents").fillna(0).astype(int)
    wide_counts.columns = [f"Cluster {int(cluster)} respondents" for cluster in wide_counts.columns]
    wide_shares = parts.pivot(index="economy", columns="cluster_id", values="percent_within_economy").fillna(0)
    wide_shares.columns = [f"Cluster {int(cluster)} share (%)" for cluster in wide_shares.columns]
    table = table.merge(wide_counts.reset_index(), on="economy").merge(wide_shares.reset_index(), on="economy")
    economy = st.selectbox("Explore economy", sorted(table.economy.dropna().unique(), key=str), key="global_economy")
    sample = int(table.loc[table.economy == economy, "sample_size"].iloc[0]); show_caution(sample)
    selected = parts.loc[parts.economy == economy].copy()
    selected['Profile'] = selected.cluster_id.map(CLUSTER_LABELS)
    st.plotly_chart(px.bar(selected, x="Profile", y="percent_within_economy", hover_data=["respondents"], title=f"Observed sample composition: {economy}"), config={"responsive": True})
    st.caption(f"Selected economy sample n={sample:,}. Dominant cluster is calculated from sample counts; ties resolve to the lowest saved cluster ID.")
    table = table.rename(columns={"economy":"Economy", "sample_size":"Sample size", "dominant_cluster":"Dominant sample cluster"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.info("A choropleth is unavailable: no validated economy-to-geometry mapping is included in the Component 02 artifacts. No geographic join has been inferred.")

