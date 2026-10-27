import pandas as pd
import plotly.express as px
import streamlit as st
from .common import dimension_means, export_csv, show_caution, subgroup_filter
from ..constants import CLUSTER_LABELS

DEMOGRAPHICS = {"female":"Gender code", "age":"Age (numeric)" ,"educ":"Education code", "emp_in":"Employment code", "urbanicity":"Residence code"}

def render_demographics(df: pd.DataFrame) -> None:
    st.header("Demographic insights")
    available = [c for c in DEMOGRAPHICS if c in df.columns]
    if not available:
        st.info("No supported demographic fields are present in the analytical dataset."); return
    column = st.selectbox("Verified variable", available, format_func=lambda x: DEMOGRAPHICS[x], key="demographic_field")
    data = df.dropna(subset=[column]).copy(); data[column] = data[column].astype(str)
    tab = data.groupby([column,"cluster_id"]).size().rename("Respondents").reset_index()
    tab["Share within category (%)"] = tab.Respondents / tab.groupby(column).Respondents.transform("sum") * 100
    st.plotly_chart(px.bar(tab, x=column, y="Share within category (%)", color="cluster_id", barmode="stack", labels={"cluster_id":"Saved cluster"}, title=f"Saved profile composition by {DEMOGRAPHICS[column]}"), config={"responsive": True})
    tab["Cluster"] = tab.cluster_id.map(CLUSTER_LABELS)
    st.dataframe(tab.rename(columns={column:DEMOGRAPHICS[column]}), hide_index=True, width="stretch")
    export_csv(tab, "financial_inclusion_demographic_summary.csv", "download_demographic")
    st.caption("Categories are shown as source numeric codes because verified display labels are not available here. Missing responses are excluded, not recoded. Unweighted descriptive association only.")

def render_region(df: pd.DataFrame) -> None:
    st.header("Region analysis")
    if "regionwb" not in df:
        st.info("Verified World Bank region field is unavailable."); return
    data = df.dropna(subset=["regionwb"])
    region = st.selectbox("World Bank region code", sorted(data.regionwb.unique(), key=str), key="region_selector")
    chosen = data.loc[data.regionwb == region]; show_caution(len(chosen))
    means = dimension_means(chosen); overall = dimension_means(data).set_index("Dimension")["Mean score"]
    means["Pooled mean score"] = means.Dimension.map(overall)
    st.metric("Respondents", f"{len(chosen):,}")
    st.plotly_chart(px.bar(means.melt(id_vars="Dimension", value_vars=["Mean score","Pooled mean score"], var_name="Population", value_name="Mean"), x="Dimension", y="Mean", color="Population", barmode="group", range_y=[0,1]), config={"responsive": True})
    st.plotly_chart(px.imshow(data.groupby("regionwb")[["score_connectivity","score_financial_access","score_digital_financial_usage","score_digital_safety","score_financial_resilience"]].mean(), aspect="auto", zmin=0,zmax=1, labels={"x":"Dimension","y":"World Bank region code","color":"Mean score"}, title="Dimension means by source region code"), config={"responsive": True})
    counts = chosen.cluster_id.value_counts().reindex(range(5), fill_value=0)
    summary = pd.DataFrame({"Cluster ID":range(5),"Respondents":counts.values,"Share (%)":counts.values/max(len(chosen),1)*100})
    st.dataframe(summary, hide_index=True, width="stretch"); export_csv(means,"financial_inclusion_region_means.csv","download_region")
    st.caption("The available metadata is a coded regionwb field; labels have not been inferred. Unweighted descriptive summaries; no causal claims.")

def render_income(df: pd.DataFrame) -> None:
    st.header("Income group analysis")
    st.info("Unavailable: the analytical dataset contains respondent income-quintile codes, not a verified economy-level World Bank income-group mapping. Income quintile is not a substitute. A dated, versioned country/economy mapping is a prerequisite for this analysis.")

