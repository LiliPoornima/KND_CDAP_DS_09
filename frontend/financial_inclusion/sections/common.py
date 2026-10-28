from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .. import data_loader as loader
from ..constants import CLUSTER_IDS, CLUSTER_LABELS, CLUSTER_NAMES

MIN_DISPLAY_N = 30  # display caution only; not a validated inferential threshold


def profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = loader.calculate_cluster_profiles(df)
    result["Profile"] = result.cluster_id.astype(int).map(CLUSTER_NAMES)
    return result


def dimension_long(profiles: pd.DataFrame) -> pd.DataFrame:
    return profiles.melt(
        id_vars=["cluster_id", "Profile", "respondents"],
        value_vars=[f"{col}_mean" for col in loader.DIMENSION_COLUMNS],
        var_name="dimension", value_name="Mean score",
    ).assign(Dimension=lambda x: x.dimension.str.removesuffix("_mean").map(loader.DIMENSION_LABELS))


def counts_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.cluster_id.value_counts().reindex(CLUSTER_IDS, fill_value=0)
    return pd.DataFrame({"Cluster": [CLUSTER_LABELS[i] for i in CLUSTER_IDS], "Respondents": counts.values,
                         "Share of sample (%)": counts.values / max(len(df), 1) * 100})


def show_distribution_chart(df: pd.DataFrame, *, title: str = "Profile composition") -> None:
    counts = counts_by_cluster(df)
    fig = px.bar(counts, x="Cluster", y="Share of sample (%)", hover_data=["Respondents"], title=title)
    st.plotly_chart(fig, config={"responsive": True})


def show_caution(n: int) -> None:
    if n < MIN_DISPLAY_N:
        st.warning(f"This subgroup has {n:,} respondents. Results are shown as a display caution only (fewer than {MIN_DISPLAY_N}); interpret carefully. This is not a validated statistical threshold.")


def export_csv(frame: pd.DataFrame, name: str, key: str) -> None:
    st.download_button("Download displayed summary (CSV)", frame.to_csv(index=False).encode("utf-8-sig"), file_name=name, mime="text/csv", key=key)


def subgroup_filter(df: pd.DataFrame, key: str) -> tuple[str, pd.DataFrame]:
    choices = ["All respondents", "Economy", "Region"]
    mode = st.selectbox("Population", choices, key=f"{key}_mode")
    if mode == "Economy":
        column, label = "economy", "Economy"
    elif mode == "Region" and "regionwb" in df:
        column, label = "regionwb", "Region"
    else:
        return "Pooled analytical sample", df
    vals = sorted(df[column].dropna().unique(), key=str)
    if not vals:
        st.info(f"No verified {label.lower()} values are available.")
        return label, df.iloc[0:0]
    selected = st.selectbox(label, vals, key=f"{key}_{column}")
    return f"{label}: {selected}", df.loc[df[column] == selected]


def dimension_means(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"Dimension": loader.DIMENSION_LABELS[c], "Mean score": pd.to_numeric(df[c], errors="coerce").mean(), "Valid respondents": int(pd.to_numeric(df[c], errors="coerce").notna().sum())} for c in loader.DIMENSION_COLUMNS])
