import pandas as pd
import plotly.express as px
import streamlit as st

from .common import (
    dimension_long,
    profile_frame,
    show_distribution_chart,
)
from .. import data_loader as loader


def render(df: pd.DataFrame) -> None:
    st.header("Executive Overview")

    st.caption(
        "Descriptive, unweighted summaries of the saved 5D_k5 "
        "analytical sample. "
    )

    # --------------------------------------------------
    # 1. Prepare existing research results
    # --------------------------------------------------
    profiles = profile_frame(df)
    dimensions = loader.calculate_dimension_summary(df)

    largest = profiles.sort_values(
        "respondents",
        ascending=False
    ).iloc[0]

    # --------------------------------------------------
    # 2. KPI cards
    # --------------------------------------------------
    columns = st.columns(4)

    columns[0].metric(
        label="Respondents Analysed",
        value=f"{df['wpid_random'].nunique():,}"
    )

    columns[1].metric(
        label="Economies Represented",
        value=f"{df['economy'].nunique():,}"
    )

    columns[2].metric(
        label="Saved Profiles",
        value="5 (5D_k5)"
    )

    with columns[3]:
        with st.container(border=True):
            st.markdown("**Largest Observed Profile**")
            st.markdown(f"**{largest['Profile']}**")

    # --------------------------------------------------
    # Understanding the Five Dimensions
    # --------------------------------------------------
    st.subheader("Understanding the Five Dimensions")

    st.caption(
        "Five dimensions capture key aspects of digital financial inclusion."
    )

    dimension_cards = [
        {
            "icon": "📶",
            "title": "Connectivity",
            "description": "Mobile phone access and internet use.",
        },
        {
            "icon": "🏦",
            "title": "Financial Access",
            "description": "Access to a formal financial account.",
        },
        {
            "icon": "💳",
            "title": "Digital Financial Usage",
            "description": "Use of digital financial services and payments.",
        },
        {
            "icon": "🛡️",
            "title": "Digital Safety",
            "description": "Phone-protection practices captured by the study.",
        },
        {
            "icon": "💰",
            "title": "Financial Resilience",
            "description": "Savings and capacity to manage financial shocks.",
        },
    ]

    dimension_columns = st.columns(5, gap="small")

    for column, dimension in zip(dimension_columns, dimension_cards):
        with column:
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size: 28px; text-align: center;'>"
                    f"{dimension['icon']}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div style='text-align: center; font-weight: 600; "
                    f"min-height: 42px;'>"
                    f"{dimension['title']}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div style='font-size: 13px; line-height: 1.4; "
                    f"min-height: 55px;'>"
                    f"{dimension['description']}</div>",
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------
    # 3. Five dimension means
    # --------------------------------------------------
    st.subheader("Five Dimension Means")

    d = dimensions.assign(
        Dimension=dimensions["variable"].map(
            loader.DIMENSION_LABELS
        )
    )

    dimension_chart = px.bar(
        d,
        x="mean",
        y="Dimension",
        orientation="h",
        range_x=[0, 1],
        title="Mean operational score (0 to 1 scale)",
        hover_data=["missing_percent"],
    )

    dimension_chart.update_layout(
        xaxis_title="Mean score",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=20, t=60, b=20),
    )

    st.plotly_chart(
        dimension_chart,
        config={"responsive": True},
    )

    st.caption(
        "Scores are operational dimension summaries, not percentages "
        "of people. Each mean excludes missing scores for that dimension."
    )

    # --------------------------------------------------
    # 4. Saved profile distribution
    # --------------------------------------------------
    st.subheader("Saved Profile Distribution")

    show_distribution_chart(
        df,
        title="Respondent shares by saved cluster"
    )

    # --------------------------------------------------
    # 5. Profile-by-dimension pattern
    # --------------------------------------------------
    st.subheader("Profile-by-Dimension Pattern")

    # Retain the existing helper call for compatibility.
    long = dimension_long(profiles)

    heat = profiles.set_index("Profile")[
        [f"{column}_mean" for column in loader.DIMENSION_COLUMNS]
    ].copy()

    heat.columns = [
        loader.DIMENSION_LABELS[column]
        for column in loader.DIMENSION_COLUMNS
    ]

    heat_chart = px.imshow(
        heat,
        zmin=0,
        zmax=1,
        aspect="auto",
        labels={
            "x": "Dimension",
            "y": "Saved profile",
            "color": "Mean score",
        },
        title="Cluster mean scores (0 to 1 scale)",
        text_auto=".2f",
    )

    heat_chart.update_layout(
        margin=dict(l=10, r=10, t=60, b=20),
    )

    st.plotly_chart(
        heat_chart,
        config={"responsive": True},
    )

    # --------------------------------------------------
    # 6. Findings from this sample
    # --------------------------------------------------
    st.subheader("Findings from this sample")

    high = d.loc[d["mean"].idxmax()]
    low = d.loc[d["mean"].idxmin()]

    # Find the source column for the highest-mean dimension
    # using its variable name rather than its DataFrame row index.
    high_variable = high["variable"]

    valid_n = pd.to_numeric(
        df[high_variable],
        errors="coerce"
    ).notna().sum()

    st.markdown(
        f"- Largest profile: **{largest['Profile']}**, "
        f"{int(largest['respondents']):,} of {len(df):,} respondents "
        f"({largest['percent_of_candidate']:.1f}%).\n"
        f"- Highest observed dimension mean: **{high['Dimension']}**, "
        f"{high['mean']:.3f} (valid n={int(valid_n):,}).\n"
        f"- Lowest observed dimension mean: **{low['Dimension']}**, "
        f"{low['mean']:.3f}."
    )
