import pandas as pd
import plotly.express as px
import streamlit as st
from .common import dimension_long, export_csv, profile_frame
from ..constants import CLUSTER_IDS, CLUSTER_LABELS

def render(df: pd.DataFrame) -> None:
    st.header("Cluster comparison")
    profiles = profile_frame(df)
    selected = st.multiselect("Highlight profiles", CLUSTER_IDS, default=list(CLUSTER_IDS), format_func=lambda x: CLUSTER_LABELS[x], key="comparison_clusters")
    current = profiles.loc[profiles.cluster_id.isin(selected)]
    long = dimension_long(current)
    st.plotly_chart(px.bar(long, x="Dimension", y="Mean score", color="Profile", barmode="group", range_y=[0,1], title="Dimension means by saved profile"), config={"responsive": True})
    summary = current[['cluster_id','Profile','respondents','percent_of_candidate'] + [f'{c}_mean' for c in __import__('frontend.financial_inclusion.data_loader', fromlist=['DIMENSION_COLUMNS']).DIMENSION_COLUMNS]].copy()
    summary = summary.rename(columns={'cluster_id':'Cluster ID','respondents':'Respondents','percent_of_candidate':'Sample share (%)'})
    st.dataframe(summary.round(3), hide_index=True, width="stretch")
    export_csv(summary, "financial_inclusion_cluster_comparison.csv", "download_cluster_comparison")
    st.caption("All dimension means use the saved 5D_k5 assignments. A score of 0.65 is not 65% of respondents.")
