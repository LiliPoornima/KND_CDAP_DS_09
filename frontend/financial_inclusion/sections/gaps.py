import pandas as pd
import plotly.express as px
import streamlit as st

from .common import dimension_means, show_caution, subgroup_filter
from .subgroups import VERIFIED_REGION_CATEGORIES
from .. import data_loader as loader


def build_gap_comparison(selected: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """Compare selected and pooled means with per-dimension valid counts."""
    selected_means = dimension_means(selected).rename(
        columns={
            "Mean score": "Selected mean score",
            "Valid respondents": "Selected valid respondents",
        }
    )
    pooled_means = dimension_means(pooled).rename(
        columns={
            "Mean score": "Pooled mean score",
            "Valid respondents": "Pooled valid respondents",
        }
    )
    result = selected_means.merge(pooled_means, on="Dimension", validate="one_to_one")
    result["Difference in mean score"] = (
        result["Selected mean score"] - result["Pooled mean score"]
    )
    return result


def gap_insights(comparison: pd.DataFrame) -> list[str]:
    """Describe the largest observed differences with their correct direction."""
    valid = comparison.dropna(subset=["Difference in mean score"])
    if valid.empty:
        return []

    positive = valid.loc[valid["Difference in mean score"] > 0]
    negative = valid.loc[valid["Difference in mean score"] < 0]
    insights = []
    if not positive.empty:
        row = positive.loc[positive["Difference in mean score"].idxmax()]
        insights.append(
            f"The selected subgroup has a higher observed mean than the pooled "
            f"sample for **{row['Dimension']}** "
            f"(+{row['Difference in mean score']:.3f} score points)."
        )
    if not negative.empty:
        row = negative.loc[negative["Difference in mean score"].idxmin()]
        insights.append(
            f"The selected subgroup has a lower observed mean than the pooled "
            f"sample for **{row['Dimension']}** "
            f"({row['Difference in mean score']:.3f} score points)."
        )
    if not insights:
        insights.append(
            "Selected subgroup and pooled-sample means are equal across all dimensions."
        )
    return insights


def region_dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize dimension means for the verified region labels only."""
    columns = ["World Bank region", "Respondents", *loader.DIMENSION_LABELS.values()]
    if "regionwb" not in df.columns:
        return pd.DataFrame(columns=columns)
    data = df.loc[df["regionwb"].isin(VERIFIED_REGION_CATEGORIES)]
    if data.empty:
        return pd.DataFrame(columns=columns)
    counts = data.groupby("regionwb").size().rename("Respondents")
    numeric_scores = data[loader.DIMENSION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric_scores["regionwb"] = data["regionwb"]
    means = numeric_scores.groupby("regionwb")[loader.DIMENSION_COLUMNS].mean()
    result = means.rename(columns=loader.DIMENSION_LABELS).join(counts).reset_index()
    return result.rename(columns={"regionwb": "World Bank region"})[columns]


def render(df: pd.DataFrame) -> None:
    st.subheader("Gap detector")
    label, selected = subgroup_filter(df, "gap")
    if selected.empty:
        st.info("No respondents match this selection.")
        return

    show_caution(len(selected))
    comparison = build_gap_comparison(selected, df)
    sample_columns = st.columns(2)
    sample_columns[0].metric("Selected population", f"{len(selected):,} respondents")
    sample_columns[1].metric("Pooled reference", f"{len(df):,} respondents")
    st.caption(
        f"{label} is compared with the pooled analytical sample, which includes "
        "the selected subgroup. Dimension means use available scores; valid counts "
        "are shown for each dimension."
    )

    if comparison["Difference in mean score"].notna().any():
        plot_data = comparison.melt(
            id_vars=[
                "Dimension",
                "Selected valid respondents",
                "Pooled valid respondents",
            ],
            value_vars=["Selected mean score", "Pooled mean score"],
            var_name="Population",
            value_name="Mean dimension score",
        )
        fig = px.bar(
            plot_data,
            x="Dimension",
            y="Mean dimension score",
            color="Population",
            barmode="group",
            range_y=[0, 1],
            custom_data=[
                "Selected valid respondents",
                "Pooled valid respondents",
            ],
            labels={
                "Dimension": "Financial inclusion dimension",
                "Population": "Population",
            },
            title=f"Observed dimension scores: {label} and pooled sample",
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{fullData.name}: %{y:.3f}<br>"
                "Selected valid respondents: %{customdata[0]:,}<br>"
                "Pooled valid respondents: %{customdata[1]:,}"
                "<extra></extra>"
            )
        )
        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=55, b=20),
            bargap=0.25,
            xaxis_title=None,
            yaxis_title="Mean dimension score",
        )
        st.plotly_chart(fig, config={"responsive": True})

        display = comparison.rename(
            columns={
                "Dimension": "Dimension",
                "Selected mean score": "Selected mean",
                "Pooled mean score": "Pooled mean",
                "Selected valid respondents": "Selected valid n",
                "Pooled valid respondents": "Pooled valid n",
                "Difference in mean score": "Difference in mean score",
            }
        )
        st.dataframe(
            display.style.format(
                {
                    "Selected mean": "{:.3f}",
                    "Pooled mean": "{:.3f}",
                    "Difference in mean score": "{:+.3f}",
                    "Selected valid n": "{:,}",
                    "Pooled valid n": "{:,}",
                },
                na_rep="Unavailable",
            ),
            hide_index=True,
            width="stretch",
        )
        for insight in gap_insights(comparison):
            st.markdown(f"- {insight}")

        recommendations = loader.build_community_recommendations(
            comparison.rename(
                columns={
                    "Dimension": "dimension",
                    "Difference in mean score": "difference",
                }
            ),
            reference_label="pooled analytical sample",
        )
        if recommendations:
            st.markdown(
                "**Areas stakeholders could investigate:** "
                + "; ".join(item["action"] for item in recommendations[:3])
            )
        if label.startswith("Region:"):
            with st.expander("View dimension means across verified regions"):
                region_summary = region_dimension_summary(df)
                if region_summary.empty:
                    st.info("No verified World Bank region categories are available.")
                else:
                    region_chart = px.imshow(
                        region_summary.set_index("World Bank region")[
                            list(loader.DIMENSION_LABELS.values())
                        ].T,
                        aspect="auto",
                        zmin=0,
                        zmax=1,
                        labels={
                            "x": "World Bank region",
                            "y": "Financial inclusion dimension",
                            "color": "Mean score",
                        },
                        title="Observed dimension means by region",
                    )
                    region_chart.update_layout(
                        height=360,
                        margin=dict(l=20, r=20, t=55, b=20),
                    )
                    st.plotly_chart(region_chart, config={"responsive": True})
                    st.caption(
                        "This overview shows region means; the comparison above "
                        "uses the selected region versus the pooled sample."
                    )
    else:
        st.info("No dimension has valid scores in both populations for comparison.")

    st.caption(
        "These unweighted sample summaries do not estimate national prevalence "
        "or establish causation. Differences are in dimension scores, not "
        "percentage-point prevalence."
    )
