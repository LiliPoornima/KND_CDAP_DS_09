import pandas as pd
import plotly.express as px
import streamlit as st
from .common import dimension_means, export_csv, show_caution, subgroup_filter
from ..constants import CLUSTER_IDS, CLUSTER_LABELS


AGE_BANDS = ("15–24", "25–34", "35–44", "45–54", "55–64", "65+")
AGE_BINS = (15, 25, 35, 45, 55, 65, float("inf"))
DEMOGRAPHICS = {"age": "Age band", "regionwb": "World Bank region"}

# Exact, human-readable source labels confirmed in the saved variable audit.
VERIFIED_REGION_CATEGORIES = (
    "East Asia & Pacific (excluding high income)",
    "Europe & Central Asia (excluding high income)",
    "High income",
    "Latin America & Caribbean (excluding high income)",
    "Middle East & North Africa (excluding high income)",
    "South Asia",
    "Sub-Saharan Africa (excluding high income)",
)


def demographic_profile_distribution(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, int]:
    """Build profile shares within verified demographic categories."""
    if column not in DEMOGRAPHICS:
        raise ValueError(f"Unsupported demographic field: {column}")
    if column not in df.columns or "cluster_id" not in df.columns:
        raise ValueError("The selected demographic field or saved profile labels are unavailable.")

    clusters = pd.to_numeric(df["cluster_id"], errors="coerce")
    if (
        clusters.isna().any()
        or not clusters.isin(CLUSTER_IDS).all()
        or not clusters.eq(clusters.round()).all()
    ):
        raise ValueError("Demographic analysis requires valid saved profile labels.")

    if column == "age":
        ages = pd.to_numeric(df[column], errors="coerce")
        valid = ages.notna() & ages.ge(AGE_BINS[0]) & ages.lt(float("inf"))
        categories = pd.cut(
            ages.loc[valid],
            bins=AGE_BINS,
            labels=AGE_BANDS,
            right=False,
        )
        category_values = pd.Series(
            categories.astype(str).to_numpy(),
            index=df.index[valid],
            name="Category",
        )
    else:
        valid = df[column].notna() & df[column].isin(VERIFIED_REGION_CATEGORIES)
        category_values = df.loc[valid, column].rename("Category")

    prepared = pd.DataFrame(
        {
            "Category": category_values,
            "cluster_id": clusters.loc[valid].astype(int),
        }
    )
    excluded = len(df) - len(prepared)
    if prepared.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Category",
                    "cluster_id",
                    "Profile",
                    "Respondents",
                    "Category respondents",
                    "Share within category (%)",
                ]
            ),
            excluded,
        )

    if column == "age":
        category_order = [band for band in AGE_BANDS if band in set(prepared["Category"])]
    else:
        category_order = sorted(prepared["Category"].unique(), key=str.casefold)

    counts = (
        prepared.groupby(["Category", "cluster_id"], observed=True)
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [category_order, CLUSTER_IDS],
                names=["Category", "cluster_id"],
            ),
            fill_value=0,
        )
        .rename("Respondents")
        .reset_index()
    )
    counts["Category respondents"] = counts.groupby("Category")[
        "Respondents"
    ].transform("sum")
    counts["Share within category (%)"] = (
        counts["Respondents"] / counts["Category respondents"] * 100
    )
    counts["Profile"] = counts["cluster_id"].map(CLUSTER_LABELS)
    return counts, excluded


def render_demographics(df: pd.DataFrame) -> None:
    st.subheader("Profile mix by demographic")
    available = [column for column in DEMOGRAPHICS if column in df.columns]
    if not available:
        st.info("No supported demographic fields are present in the analytical dataset.")
        return

    column = st.selectbox(
        "Demographic",
        available,
        format_func=lambda value: DEMOGRAPHICS[value],
        key="demographic_field",
    )
    try:
        summary, excluded = demographic_profile_distribution(df, column)
    except ValueError as exc:
        st.warning(f"Demographic profile summary is unavailable: {exc}")
        return

    if summary.empty:
        st.info("No respondents have a valid value for this demographic field.")
        return

    st.caption(
        "Bars show the saved inclusion-profile mix within each demographic category. "
        "Percentages use that category's valid respondent count."
    )
    if column == "age":
        st.caption(
            "Age bands are 15–24, 25–34, 35–44, 45–54, 55–64, and 65 or older. "
            "Missing, non-numeric, non-finite, and below-15 ages are excluded."
        )
    elif excluded:
        st.caption(
            f"{excluded:,} respondents with missing or unverified region categories are excluded."
        )

    category_totals = summary.drop_duplicates("Category")[
        ["Category", "Category respondents"]
    ]
    if (category_totals["Category respondents"] < 30).any():
        st.caption(
            "Categories with fewer than 30 respondents remain visible; interpret "
            "small groups cautiously. This is a display caution, not an inferential threshold."
        )

    category_order = category_totals["Category"].tolist()
    profile_order = [CLUSTER_LABELS[cluster_id] for cluster_id in CLUSTER_IDS]
    fig = px.bar(
        summary,
        x="Category",
        y="Share within category (%)",
        color="Profile",
        barmode="stack",
        custom_data=["Respondents", "Category respondents"],
        category_orders={"Category": category_order, "Profile": profile_order},
        labels={
            "Category": DEMOGRAPHICS[column],
            "Profile": "Saved inclusion profile",
            "Share within category (%)": "Respondents within category (%)",
        },
        title=f"Saved inclusion profiles by {DEMOGRAPHICS[column].lower()}",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{y:.1f}%<br>"
            "Respondents in profile: %{customdata[0]:,}<br>"
            "Respondents in category: %{customdata[1]:,}"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title=None,
        yaxis_title="Respondents within category (%)",
        legend_title_text="Saved inclusion profile",
    )
    st.plotly_chart(fig, config={"responsive": True})

    st.dataframe(
        summary[
            [
                "Category",
                "Profile",
                "Respondents",
                "Category respondents",
                "Share within category (%)",
            ]
        ].style.format(
            {
                "Respondents": "{:,}",
                "Category respondents": "{:,}",
                "Share within category (%)": "{:.1f}%",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    export_csv(
        summary[
            [
                "Category",
                "Profile",
                "Respondents",
                "Category respondents",
                "Share within category (%)",
            ]
        ],
        "financial_inclusion_demographic_summary.csv",
        "download_demographic",
    )
    st.caption(
        "Only age and the seven readable region labels verified in the saved audit "
        "are shown. Other categorical codes lack verified display labels here. "
        "These unweighted summaries describe the observed sample and do not establish causation."
    )

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
