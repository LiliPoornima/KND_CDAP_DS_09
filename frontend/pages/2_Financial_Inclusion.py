"""Component 02: explainable financial-inclusion decision support."""

from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.financial_inclusion import data_loader as loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend" / "src" / "financial_inclusion"
MODULE_DIRS = [
    BACKEND_DIR / "02_dimension_creation",
    BACKEND_DIR / "03_clustering",
    BACKEND_DIR / "04_supervised_learning",
]
for module_dir in MODULE_DIRS:
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from fi_individual_profiling import profile_individual
from fi_supervised_inference import (
    CLUSTER_LABELS,
    load_model,
    predict_profiles,
)


st.set_page_config(
    page_title="Digital Financial Inclusion",
    page_icon="🌍",
    layout="wide",
)

CLUSTER_NAMES = {
    0: "Digitally Active, Safety Gap",
    1: "Connected and Digitally Included",
    2: "Connected, Financially Excluded",
    3: "Multi-Dimensional Inclusion Gap",
    4: "Access Without Digital Usage",
}
CLUSTER_IDS = list(CLUSTER_NAMES)
DIMENSIONS = loader.DIMENSION_LABELS
FILTER_LABELS = {
    "female": "Gender code",
    "educ": "Education level code",
    "inc_q": "Income quintile code",
    "emp_in": "Employment status code",
    "urbanicity": "Urbanicity code",
    "regionwb": "World Bank region",
}
PROFILE_ACTION_LABELS = {
    "score_connectivity": "Connectivity",
    "score_financial_access": "Financial access",
    "score_digital_financial_usage": "Digital financial usage",
    "score_digital_safety": "Digital safety",
    "score_financial_resilience": "Financial resilience",
}


@st.cache_data(show_spinner="Loading the analytical dataset...")
def load_dashboard_dataset() -> pd.DataFrame:
    """Cache the immutable analytical CSV during Streamlit reruns."""
    return loader.load_final_dataset()


@st.cache_resource(show_spinner="Loading the saved Logistic Regression model...")
def load_saved_classifier():
    """Load the existing fitted pipeline once; this never trains a model."""
    return load_model()


def navigate_to_profile_explorer() -> None:
    st.session_state["financial_inclusion_navigation"] = (
        "Explore Inclusion Profiles"
    )


def show_overall_dashboard(df: pd.DataFrame) -> None:
    st.header("Overview & Key Findings")
    n = df["wpid_random"].nunique()
    economies = df["economy"].nunique()
    dimensions = loader.calculate_dimension_summary(df)
    profiles = loader.calculate_cluster_profiles(df)
    profiles["profile"] = profiles["cluster_id"].map(CLUSTER_NAMES)

    a, b, c, d = st.columns(4)
    a.metric("Analytical respondents", f"{n:,}")
    b.metric("Economies represented", f"{economies:,}")
    c.metric("Inclusion profiles", len(profiles))
    d.metric("Operational dimensions", len(DIMENSIONS))
    st.caption(
        f"Survey year: 2024 | Analytical sample: {n:,} respondents | "
        f"Economies: {economies:,} | Global Findex 2025 release"
    )

    means = dimensions.set_index("variable")["mean"]
    high, low = means.idxmax(), means.idxmin()
    gap = means["score_financial_access"] - means["score_digital_financial_usage"]
    largest = profiles.sort_values("respondents", ascending=False).iloc[0]
    st.subheader("What stands out")
    cols = st.columns(3)
    cols[0].metric("Highest mean", f"{DIMENSIONS[high]}: {means[high]:.3f}")
    cols[0].caption("Operational score mean. Which underlying indicators merit follow-up?")
    cols[1].metric("Lowest mean", f"{DIMENSIONS[low]}: {means[low]:.3f}")
    cols[1].caption("Selected indicators; this does not explain why the pattern occurs.")
    cols[2].metric("Largest profile", f"{largest['profile']}: {largest['percent_of_candidate']:.1f}%")
    cols[2].caption(f"{int(largest['respondents']):,} respondents. Profiles are descriptive summaries.")
    st.info(
        f"Financial access minus digital financial usage: {gap:+.3f}. "
        "This describes two different indicators, not whether account holders use digital services."
    )

    st.subheader("Five operational dimension scores")
    display = dimensions.assign(Dimension=dimensions["variable"].map(DIMENSIONS)).sort_values("mean")
    chart = px.bar(
        display, x="mean", y="Dimension", orientation="h",
        text=display["mean"].map(lambda value: f"{value:.3f}"),
        range_x=[0, 1], labels={"mean": "Mean score"},
        title="Mean scores in the analytical sample",
    )
    st.plotly_chart(chart, config={"responsive": True})
    st.caption("A score of 0.65 does not mean 65% of respondents are financially included.")

    st.subheader("Profile snapshot")
    for offset in range(0, len(profiles), 3):
        cols = st.columns(min(3, len(profiles) - offset))
        for col, (_, row) in zip(cols, profiles.iloc[offset:offset + 3].iterrows()):
            vals = row[[f"{key}_mean" for key in DIMENSIONS]].astype(float)
            high_key = vals.idxmax().removesuffix("_mean")
            low_key = vals.idxmin().removesuffix("_mean")
            with col.container(border=True):
                col.markdown(f"**{int(row['cluster_id'])} - {row['profile']}**")
                col.write(f"{int(row['respondents']):,} respondents - {row['percent_of_candidate']:.1f}%")
                col.caption(f"Relatively higher: {DIMENSIONS[high_key]}; lower: {DIMENSIONS[low_key]}.")
    st.button(
        "Explore the five profiles",
        on_click=navigate_to_profile_explorer,
        type="primary",
    )

    with st.expander("How these scores work"):
        st.dataframe(pd.DataFrame([
            ["Connectivity", "Phone ownership and internet use", "Mean of available components."],
            ["Financial access", "Account ownership", "Validated account indicator."],
            ["Digital financial usage", "Any digital payment", "Validated usage indicator."],
            ["Digital safety", "Phone protection (con18)", "Scam exposure is not included."],
            ["Financial resilience", "Saved, fin24a, fin24b", "All three mapped components are required."],
        ], columns=["Dimension", "Selected indicators", "Scoring rule and limit"]), hide_index=True, width="stretch")
        st.caption("Missing and unasked answers remain unavailable, never No or zero. Scores do not cover every aspect of inclusion.")

    st.subheader("Questions to investigate")
    st.markdown(
        "- Which indicators contribute most to the observed resilience pattern?\n"
        "- Which profiles show the largest access and digital-usage differences?\n"
        "- How do these descriptive patterns vary across population groups?"
    )


def show_profile_explorer(df: pd.DataFrame) -> None:
    st.header("Explore Inclusion Profiles")
    profiles = loader.calculate_cluster_profiles(df)
    profiles["profile"] = profiles["cluster_id"].astype(int).map(CLUSTER_NAMES)
    dim_cols = [f"{key}_mean" for key in DIMENSIONS]
    st.subheader("Five saved profiles")
    for offset in range(0, len(profiles), 3):
        cols = st.columns(min(3, len(profiles) - offset))
        for col, (_, row) in zip(cols, profiles.iloc[offset:offset + 3].iterrows()):
            scores = row[dim_cols].astype(float)
            high = DIMENSIONS[scores.idxmax().removesuffix("_mean")]
            low = DIMENSIONS[scores.idxmin().removesuffix("_mean")]
            with col.container(border=True):
                col.markdown(f"**{int(row['cluster_id'])} - {row['profile']}**")
                col.write(f"{int(row['respondents']):,} respondents - {row['percent_of_candidate']:.1f}%")
                col.caption(f"Relatively higher: {high}; lower: {low}.")

    overall = loader.calculate_dimension_summary(df).set_index("variable")["mean"]
    selected = st.selectbox(
        "Explore a profile", profiles["cluster_id"].astype(int).tolist(),
        format_func=lambda value: f"{value} - {CLUSTER_NAMES[value]}",
        key="profile_explorer_cluster",
    )
    row = profiles.loc[profiles["cluster_id"] == selected].iloc[0]
    a, b = st.columns(2)
    a.metric("Profile respondents", f"{int(row['respondents']):,}")
    b.metric("Analytical sample share", f"{row['percent_of_candidate']:.1f}%")
    chart_data = pd.DataFrame({
        "Dimension": list(DIMENSIONS.values()),
        "Selected profile": [float(row[f"{key}_mean"]) for key in DIMENSIONS],
        "Overall sample": [float(overall[key]) for key in DIMENSIONS],
    }).melt(id_vars="Dimension", var_name="Population", value_name="Mean score")
    fig = px.bar(
        chart_data, x="Mean score", y="Dimension", color="Population",
        barmode="group", orientation="h", range_x=[0, 1],
        title="Selected profile compared with the analytical sample",
    )
    st.plotly_chart(fig, config={"responsive": True})
    if st.checkbox("Show exact profile dimension means", key="profile_explorer_show_table"):
        st.dataframe(chart_data.pivot(index="Dimension", columns="Population", values="Mean score").round(3), width="stretch")
    lowest_key = min(DIMENSIONS, key=lambda key: float(row[f"{key}_mean"]) - float(overall[key]))
    difference = float(row[f"{lowest_key}_mean"]) - float(overall[lowest_key])
    relation = "below" if difference < 0 else "above"
    st.markdown(
        f"**Observed difference:** {DIMENSIONS[lowest_key]} is {abs(difference):.3f} points {relation} "
        "the overall mean within this profile. **Question to investigate:** which indicators or "
        "service experiences should be checked with respondents? This is descriptive and does not identify causes."
    )
    with st.expander("Compare an economy or population group"):
        show_community_analysis(df)


def show_individual_profile_explorer(df: pd.DataFrame) -> None:
    st.header("Individual Profile Explorer")
    assessment, prediction = st.tabs(["Indicator-based assessment", "Research model prediction"])
    with assessment:
        show_individual_profiling()
    with prediction:
        show_research_model_prediction(df)


def show_community_analysis(df: pd.DataFrame) -> None:
    st.subheader("Economy and population comparison")
    st.write(
        "Compare an economy's observed dimension scores with the analytical-"
        "sample reference. The reference is descriptive, not an official "
        "benchmark or economy ranking."
    )
    st.caption(
        "Scores are operational 0–1 summaries of selected survey indicators; "
        "for example, 0.65 does not mean 65% financial inclusion."
    )

    economies = sorted(df["economy"].dropna().astype(str).unique().tolist())
    selected_economy = st.selectbox(
        "Select an economy",
        economies,
        key="community_economy",
    )
    filter_options = [
        column for column in FILTER_LABELS if column in df.columns
    ]
    filter_column = st.selectbox(
        "Optional demographic or socioeconomic filter",
        options=[None, *filter_options],
        format_func=lambda column: (
            "No additional filter"
            if column is None
            else FILTER_LABELS[column]
        ),
        key="community_filter_column",
    )

    cluster_filter = st.selectbox(
        "Optional saved profile filter",
        options=[None, *CLUSTER_IDS],
        format_func=lambda value: (
            "All profiles" if value is None else f"{value} — {CLUSTER_NAMES[value]}"
        ),
        key="community_cluster_filter",
    )
    exclude_economy = st.checkbox(
        "Exclude the selected economy from the reference population",
        value=False,
        key="community_exclude_reference_economy",
    )

    filter_value = None
    if filter_column is not None:
        observed_values = sorted(
            df[filter_column].dropna().unique().tolist(), key=str
        )
        filter_value = st.selectbox(
            f"Select {FILTER_LABELS[filter_column]}",
            options=[None, *observed_values],
            format_func=lambda value: (
                "All categories" if value is None else f"Code/value {value}"
            ),
            key=f"community_filter_value_{filter_column}",
        )

    try:
        result = loader.calculate_community_analysis(
            df,
            selected_economy,
            filter_column=filter_column,
            filter_value=filter_value,
            cluster_id=cluster_filter,
            exclude_selected_economy=exclude_economy,
        )
    except (TypeError, ValueError, KeyError):
        st.error(
            "Could not calculate this comparison. Check the selected economy "
            "and filters, then try again."
        )
        return

    comparison = result["comparison"]
    metric1, metric2 = st.columns(2)
    metric1.metric(
        f"{selected_economy} respondents",
        f"{result['respondents']:,}",
    )
    metric2.metric(
        "Reference respondents",
        f"{result['reference_respondents']:,}",
    )
    has_small_dimension_sample = (
        comparison[["economy_valid_respondents", "reference_valid_respondents"]]
        .lt(30)
        .any()
        .any()
    )
    if (
        min(result["respondents"], result["reference_respondents"]) < 30
        or has_small_dimension_sample
    ):
        st.warning(
            "The selected population, reference, or at least one dimension's "
            "valid subset has fewer than 30 respondents. Interpret affected "
            "averages cautiously; 30 is a display caution, not a validated "
            "statistical threshold."
        )
    reference_description = (
        "the filtered analytical sample excluding the selected economy"
        if exclude_economy
        else "the filtered pooled analytical sample, including the selected economy"
    )
    st.caption(
        f"Reference: {reference_description}. Both economy and reference use "
        "the same optional category and saved-profile filters. "
        "Dimension means exclude missing values separately for each dimension; "
        "their valid denominators are shown in the table."
    )

    comparison_chart_data = comparison.melt(
        id_vars=["dimension"],
        value_vars=["economy_mean", "reference_mean"],
        var_name="population",
        value_name="mean_score",
    )
    comparison_chart_data["population"] = comparison_chart_data[
        "population"
    ].map(
        {
            "economy_mean": "Selected economy",
            "reference_mean": (
                "Other economies" if exclude_economy else "Pooled sample"
            ),
        }
    )
    comparison_chart = px.bar(
        comparison_chart_data,
        x="mean_score",
        y="dimension",
        orientation="h",
        color="population",
        barmode="group",
        range_x=[0, 1],
        labels={
            "dimension": "Dimension",
            "mean_score": "Mean score",
            "population": "Comparison group",
        },
        title=f"{selected_economy}: dimension means and sample reference",
    )
    st.plotly_chart(comparison_chart, config={"responsive": True})

    comparison_display = comparison[
        [
            "dimension",
            "economy_mean",
            "reference_mean",
            "difference",
            "economy_valid_respondents",
            "reference_valid_respondents",
            "economy_missing_respondents",
            "reference_missing_respondents",
        ]
    ].rename(
        columns={
            "dimension": "Dimension",
            "economy_mean": "Economy mean",
            "reference_mean": "Reference mean",
            "difference": "Difference",
            "economy_valid_respondents": "Economy valid n",
            "reference_valid_respondents": "Reference valid n",
            "economy_missing_respondents": "Economy missing n",
            "reference_missing_respondents": "Reference missing n",
        }
    )
    st.dataframe(
        comparison_display.drop(
            columns=["Economy missing n", "Reference missing n"]
        ).round(3),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Missing responses by dimension"):
        st.dataframe(
            comparison_display[
                ["Dimension", "Economy missing n", "Reference missing n"]
            ],
            hide_index=True,
            width="stretch",
        )

    negative_gaps = comparison.loc[comparison["difference"] < 0].sort_values(
        "difference"
    )
    st.caption(
        "Difference is the selected economy mean minus the reference mean; "
        "negative values are observed lower means, not statistically tested gaps."
    )

    st.subheader("Cluster composition")
    cluster_distribution = result["cluster_distribution"].copy()
    cluster_distribution["Cluster profile"] = cluster_distribution[
        "cluster_id"
    ].map(CLUSTER_NAMES)
    cluster_display = cluster_distribution.rename(
        columns={
            "cluster_id": "Cluster ID",
            "respondents": "Respondents",
            "percent_of_economy": "Selected economy (%)",
        }
    )
    cluster_chart_data = cluster_display.copy()
    cluster_chart_data["Share label"] = cluster_chart_data[
        "Selected economy (%)"
    ].map(lambda value: f"{value:.1f}%")
    cluster_chart = px.bar(
        cluster_chart_data,
        x="Selected economy (%)",
        y="Cluster profile",
        orientation="h",
        text="Share label",
        custom_data=["Respondents"],
        range_x=[0, 100],
        title="Saved profile shares in the selected group",
        labels={"Selected economy (%)": "Share of selected group"},
    )
    cluster_chart.update_traces(
        hovertemplate=(
            "%{y}<br>Share: %{x:.1f}%<br>Respondents: "
            "%{customdata[0]}<extra></extra>"
        )
    )
    st.plotly_chart(cluster_chart, config={"responsive": True})
    with st.expander("Profile counts and shares"):
        st.dataframe(
            cluster_display[
                [
                    "Cluster ID",
                    "Cluster profile",
                    "Respondents",
                    "Selected economy (%)",
                ]
            ].round(2),
            hide_index=True,
            width="stretch",
        )
    available_clusters = cluster_distribution.loc[
        cluster_distribution["respondents"] > 0, "cluster_id"
    ].astype(int).tolist()
    selected_profile = st.selectbox(
        "Inspect dimension means for a saved profile",
        available_clusters,
        format_func=lambda value: f"{value} — {CLUSTER_NAMES[value]}",
        key="community_inspect_cluster",
    )
    filtered_population = df
    if filter_column is not None and filter_value is not None:
        filtered_population = filtered_population.loc[
            filtered_population[filter_column] == filter_value
        ]
    if cluster_filter is not None:
        filtered_population = filtered_population.loc[
            filtered_population["cluster_id"] == cluster_filter
        ]
    profile_group = filtered_population.loc[
        (filtered_population["economy"] == selected_economy)
        & (filtered_population["cluster_id"] == selected_profile)
    ]
    profile_means = profile_group[list(DIMENSIONS)].mean().rename(DIMENSIONS)
    st.caption(
        f"Dimension means for {CLUSTER_NAMES[selected_profile]} in "
        f"{selected_economy} (n={len(profile_group):,}); these describe the "
        "selected cluster subset, while the table above describes cluster shares."
    )
    st.dataframe(
        profile_means.rename("Mean score").rename_axis("Dimension")
        .reset_index().round(3),
        hide_index=True,
        width="stretch",
    )

    most_common = cluster_distribution.sort_values(
        "respondents", ascending=False
    ).iloc[0]
    st.markdown("### Observed pattern")
    st.write(
        f"In the selected sample, **{most_common['Cluster profile']}** is the "
        f"largest cluster ({most_common['percent_of_economy']:.1f}%). "
        "Dimension gaps and cluster shares describe this analytical sample; "
        "they do not establish causes or characterize every resident."
    )

    st.subheader("Question to investigate")
    recommendations = loader.build_community_recommendations(
        comparison,
        reference_label=(
            "filtered pooled analytical sample excluding this economy"
            if exclude_economy
            else "filtered pooled analytical-sample reference"
        ),
    )
    if not recommendations:
        st.write("No lower dimension mean was observed for this selection.")
    else:
        lead = recommendations[0]
        st.markdown(
            f"**Observed difference:** {lead['observed_gap']}\n\n"
            f"**What it suggests:** {lead['interpretation']}\n\n"
            f"**Question to investigate:** {lead['action']}"
        )
        with st.expander("Stakeholders, monitoring and other observed gaps"):
            for recommendation in recommendations:
                st.markdown(
                    f"- **{recommendation['dimension']}:** "
                    f"{recommendation['stakeholder']}. Monitor: "
                    f"{recommendation['monitor']}"
                )
            st.caption(lead["limitation"])

    export_selection = pd.DataFrame(
        [
            {"Section": "Selection", "Field": "Selected economy", "Value": selected_economy},
            {"Section": "Selection", "Field": "Reference definition", "Value": reference_description},
            {"Section": "Selection", "Field": "Category filter", "Value": f"{filter_column}={filter_value}" if filter_column and filter_value is not None else "All categories"},
            {"Section": "Selection", "Field": "Saved profile filter", "Value": cluster_filter if cluster_filter is not None else "All profiles"},
            {"Section": "Selection", "Field": "Selected respondents", "Value": result["respondents"]},
            {"Section": "Selection", "Field": "Reference respondents", "Value": result["reference_respondents"]},
        ]
    )
    export_recommendations = pd.DataFrame(recommendations).assign(
        Section="Stakeholder follow-up"
    )
    st.download_button(
        "Download current filtered analysis (CSV)",
        data=pd.concat(
            [
                export_selection,
                comparison_display.assign(Section="Dimension comparison"),
                cluster_display[
                    ["Cluster ID", "Cluster profile", "Respondents", "Selected economy (%)"]
                ].assign(Section="Cluster composition"),
                export_recommendations,
            ],
            ignore_index=True,
            sort=False,
        ).to_csv(index=False).encode("utf-8"),
        file_name="financial_inclusion_community_analysis.csv",
        mime="text/csv",
        help=(
            f"Selected economy={selected_economy}; filter={filter_column} "
            f"{filter_value}; reference={reference_description}; "
            f"profile={cluster_filter}; respondents={result['respondents']}; "
            f"reference respondents={result['reference_respondents']}. "
            "No respondent identifiers are included."
        ),
    )

    st.caption(
        "Results are descriptive and unweighted respondent summaries from the "
        "saved analytical dataset. The comparison does not account for survey "
        "design uncertainty, prove barriers as causes, or establish official "
        "economy rankings. Category values are shown as stored in the source "
        "data; no unverified category meanings are assigned. Consult the "
        "applicable codebook before interpreting numeric codes."
    )


def show_individual_profiling() -> None:
    st.subheader("Indicator-based assessment")
    st.write(
        "Explore an illustrative inclusion profile from the selected "
        "survey-indicator responses. This calculates operational scores; it "
        "is not a credit score, a complete measure of financial wellbeing, or "
        "a verified assessment of a person's financial circumstances."
    )
    st.info(
        "Leave unavailable responses as Not answered. Missing indicators are "
        "not converted to No. Connectivity averages available components, "
        "while financial resilience requires all three components."
    )
    st.caption(
        "Scores range from 0 to 1 under the selected operational indicator "
        "mappings. A score such as 0.65 is not 65% financial inclusion and "
        "does not summarize every aspect of a person's financial wellbeing."
    )

    with st.form("individual_financial_inclusion_form"):
        st.subheader("Connectivity")
        con1 = st.selectbox(
            "Mobile-phone ownership",
            ["Not answered", "Yes", "No"],
            key="individual_con1",
        )
        internet_use = st.selectbox(
            "Internet use",
            ["Not answered", "Yes", "No"],
            key="individual_internet_use",
        )

        st.subheader("Financial access")
        account = st.selectbox(
            "Financial-account ownership",
            ["Not answered", "Yes", "No"],
            key="individual_account",
        )
        st.subheader("Digital financial usage")
        anydigpayment = st.selectbox(
            "Digital-payment use",
            ["Not answered", "Yes", "No"],
            key="individual_anydigpayment",
        )

        st.subheader("Digital safety")
        con18 = st.selectbox(
            "Phone lock (PIN, password, or fingerprint)",
            ["Not answered", "Yes", "No"],
            key="individual_con18",
            help=(
                "The verified con18 indicator asks about a phone lock. It "
                "does not represent every aspect of digital safety."
            ),
        )

        st.subheader("Financial resilience")
        saved = st.selectbox(
            "Saved money",
            ["Not answered", "Yes", "No"],
            key="individual_saved",
        )
        fin24a = st.selectbox(
            "Emergency financial-capacity response (fin24a)",
            ["Not answered", "1", "2", "3"],
            key="individual_fin24a",
            help="The validated mapping is 1 -> 0, 2 -> 0.5, 3 -> 1.",
        )
        fin24b = st.selectbox(
            "Emergency financial-capacity response (fin24b)",
            ["Not answered", "1", "2", "3", "4"],
            key="individual_fin24b",
            help="The validated mapping is 1 -> 0, 2 -> 1/3, 3 -> 2/3, 4 -> 1.",
        )
        submitted = st.form_submit_button(
            "Calculate profile",
            type="primary",
        )

    if not submitted:
        return

    def yes_no_to_code(value: str) -> int | None:
        return {"Yes": 1, "No": 2}.get(value)

    def yes_no_to_binary(value: str) -> int | None:
        return {"Yes": 1, "No": 0}.get(value)

    indicators = {
        "con1": yes_no_to_code(con1),
        "internet_use": yes_no_to_binary(internet_use),
        "account": yes_no_to_binary(account),
        "anydigpayment": yes_no_to_binary(anydigpayment),
        "con18": yes_no_to_code(con18),
        "saved": yes_no_to_binary(saved),
        "fin24a": int(fin24a) if fin24a != "Not answered" else None,
        "fin24b": int(fin24b) if fin24b != "Not answered" else None,
    }

    try:
        result = profile_individual(indicators)
    except (FileNotFoundError, TypeError, ValueError, KeyError):
        st.error(
            "Could not calculate a profile from these responses. Check the "
            "available inputs and try again."
        )
        return

    if result["status"] != "success":
        available_scores = {
            key: value
            for key, value in result["dimension_scores"].items()
            if value is not None and pd.notna(value)
        }
        if available_scores:
            st.subheader("Available dimension scores")
            score_columns = st.columns(min(5, len(available_scores)))
            for column, (score_key, value) in zip(
                score_columns, available_scores.items()
            ):
                column.metric(DIMENSIONS[score_key], f"{value:.3f}")
            chart_data = pd.DataFrame(
                {
                    "Dimension": [DIMENSIONS[key] for key in available_scores],
                    "Score": list(available_scores.values()),
                }
            )
            chart = px.bar(
                chart_data,
                x="Score",
                y="Dimension",
                orientation="h",
                text=chart_data["Score"].map(lambda value: f"{value:.3f}"),
                range_x=[0, 1],
                title="Available indicator-based scores",
            )
            st.plotly_chart(chart, config={"responsive": True})
        st.warning(result["message"])
        st.write(
            "Unavailable dimensions: "
            + ", ".join(
                PROFILE_ACTION_LABELS.get(name, name)
                for name in result["unavailable_dimensions"]
            )
        )
        st.write("Available component counts")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Dimension": PROFILE_ACTION_LABELS[key],
                        "Components available": value,
                    }
                    for key, value in result["components_available"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(result["missing_data_policy"])
        return

    scores = result["dimension_scores"]
    st.subheader("Five dimension scores")
    score_columns = st.columns(5)
    for column, (score_key, label) in zip(score_columns, DIMENSIONS.items()):
        value = scores[score_key]
        column.metric(label, f"{value:.3f}")

    chart_data = pd.DataFrame(
        {
            "Dimension": [DIMENSIONS[key] for key in DIMENSIONS],
            "Score": [scores[key] for key in DIMENSIONS],
        }
    )
    score_chart = px.bar(
        chart_data,
        x="Score",
        y="Dimension",
        orientation="h",
        text=chart_data["Score"].map(lambda value: f"{value:.3f}"),
        range_x=[0, 1],
        title="Individual dimension scores",
    )
    st.plotly_chart(score_chart, config={"responsive": True})

    profile = result["profile"]
    st.subheader("Assigned research profile")
    st.metric("Cluster ID", int(profile["cluster_id"]))
    st.markdown(f"**{profile['profile_name']}**")
    st.write(profile["profile_description"])
    st.caption(result["interpretation_limit"])

    evidence = pd.DataFrame(result["explanation"]["dimension_evidence"])
    evidence["feature"] = evidence["feature"].map(PROFILE_ACTION_LABELS)
    evidence = evidence.rename(
        columns={
            "feature": "Dimension",
            "individual_score": "Individual score",
            "cluster_mean_score": "Assigned-cluster mean",
            "difference_from_cluster_mean": "Difference from cluster mean",
            "observation": "Observed evidence",
        }
    )
    st.subheader("Dimension-level evidence")
    st.dataframe(
        evidence[
            [
                "Dimension",
                "Individual score",
                "Assigned-cluster mean",
                "Difference from cluster mean",
                "Observed evidence",
            ]
        ].round(3),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Suggested actions for review")
    actions = loader.format_suggested_actions(
        result["explanation"]["suggested_actions"]
    ).rename(
        columns={
            "dimension": "Dimension",
            "reason": "Reason",
            "action": "Possible action",
            "stakeholder": "Relevant stakeholders",
            "monitor": "Monitoring indicator",
        }
    )
    st.dataframe(actions, hide_index=True, width="stretch")
    st.caption(
        "These are decision-support suggestions, not validated causal "
        "interventions. Scores and cluster assignments should not be used "
        "alone to determine creditworthiness, loan eligibility, or wellbeing."
    )


def show_model_performance() -> None:
    st.subheader("Model performance and classification evaluation")
    st.warning(
        "Experimental research feature: the saved Logistic Regression "
        "classifier predicts previously derived cluster labels. Its performance "
        "is modest. Outputs are model scores, not probabilities of financial "
        "exclusion, credit default, or an individual's future outcome."
    )
    st.write(
        "The prediction uses the existing fitted pipeline and its documented "
        "demographic predictors. No model fitting occurs in this dashboard."
    )

    st.subheader("Saved model comparison")
    try:
        metrics = loader.prepare_supervised_model_comparison(
            loader.load_supervised_metrics()
        )
        st.dataframe(
            metrics.round(3),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Saved stratified train/test metrics. The task predicts the "
            "research-derived 5D_k5 labels, not an independent financial "
            "inclusion outcome."
        )
    except (FileNotFoundError, ValueError, KeyError):
        st.warning(
            "Saved model metrics are unavailable or could not be validated."
        )

    st.subheader("Per-cluster performance and confusion matrices")
    try:
        reports = loader.load_supervised_classification_reports()
        matrices = loader.load_supervised_confusion_matrices()
        report_models = list(reports)
        if not report_models:
            st.warning("No saved per-cluster classification reports were found.")
        else:
            report_model = st.selectbox(
                "Select a model's per-cluster report",
                report_models,
                key="evaluation_report_model",
            )
            report_rows = re.findall(
                r"^\s*([0-4])\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$",
                reports[report_model],
                flags=re.MULTILINE,
            )
            if report_rows:
                report_display = pd.DataFrame(
                    [
                        {
                            "Cluster profile": CLUSTER_NAMES[int(row[0])],
                            "Precision": float(row[1]),
                            "Recall": float(row[2]),
                            "F1 score": float(row[3]),
                            "Support": int(row[4]),
                        }
                        for row in report_rows
                    ]
                )
                st.dataframe(
                    report_display.round(3), hide_index=True, width="stretch"
                )
            with st.expander("Full saved classification report"):
                st.code(reports[report_model], language="text")
            if report_model in matrices:
                st.caption(
                    "Confusion-matrix class IDs: "
                    + "; ".join(
                        f"{cluster}: {CLUSTER_NAMES[cluster]}"
                        for cluster in CLUSTER_IDS
                    )
                )
                confusion = pd.DataFrame(
                    matrices[report_model],
                    index=[f"Actual {cluster}" for cluster in CLUSTER_IDS],
                    columns=[f"Predicted {cluster}" for cluster in CLUSTER_IDS],
                )
                confusion_chart = px.imshow(
                    confusion.to_numpy(),
                    x=[f"Cluster {cluster}" for cluster in CLUSTER_IDS],
                    y=[f"Cluster {cluster}" for cluster in CLUSTER_IDS],
                    text_auto=True,
                    labels={
                        "x": "Predicted cluster",
                        "y": "Actual cluster",
                        "color": "Respondents",
                    },
                    title=f"Confusion matrix — {report_model}",
                    color_continuous_scale="Blues",
                )
                confusion_chart.update_layout(height=620)
                st.plotly_chart(confusion_chart, config={"responsive": True})
                with st.expander("Confusion-matrix counts"):
                    st.dataframe(confusion, width="stretch")
            else:
                st.info(f"No saved confusion matrix is available for {report_model}.")
    except (FileNotFoundError, ValueError, KeyError):
        st.warning(
            "Saved classification reports or confusion matrices are unavailable "
            "or could not be validated."
        )

    st.info(
        "No country-held-out validation has been completed. These metrics "
        "should not be interpreted as evidence of generalisation to unseen economies."
    )




def show_research_model_prediction(df: pd.DataFrame) -> None:
    st.subheader("Research model prediction")
    st.write(
        "Predicts a previously derived research profile from demographic inputs. "
        "It does not predict financial exclusion, credit default, future outcomes, or service eligibility."
    )
    predictors = [
        "female",
        "age",
        "educ",
        "inc_q",
        "emp_in",
        "urbanicity",
        "regionwb",
    ]
    with st.form("demographic_cluster_prediction_form"):
        st.markdown("#### Demographic input")
        categorical_inputs = {}
        input_columns = st.columns(3)
        for index, feature in enumerate(
            ["female", "educ", "inc_q", "emp_in", "urbanicity", "regionwb"]
        ):
            categories = sorted(
                df[feature].dropna().unique().tolist(), key=str
            )
            categorical_inputs[feature] = input_columns[index % 3].selectbox(
                FILTER_LABELS.get(feature, feature),
                options=["Not answered", *categories],
                format_func=lambda value: (
                    "Not answered" if value == "Not answered" else f"Code/value {value}"
                ),
                key=f"prediction_{feature}",
                help=(
                    "Use source-data category codes. The model's fitted "
                    "preprocessor handles missing and unseen categories."
                ),
            )

        ages = pd.to_numeric(df["age"], errors="coerce").dropna()
        age_min = max(0, int(ages.min()))
        age_max = max(age_min + 1, int(ages.max()))
        age_unavailable = st.checkbox(
            "Age unavailable",
            key="prediction_age_missing",
        )
        age_value = st.number_input(
            "Age in years",
            min_value=age_min,
            max_value=age_max,
            value=int(ages.median()),
            step=1,
            disabled=age_unavailable,
            key="prediction_age",
        )
        predict_submitted = st.form_submit_button(
            "Predict research cluster",
            type="primary",
        )
    st.caption(
        "The saved fitted pipeline uses source-data predictor codes passed "
        "without recoding. Human-readable labels "
        "are not assigned where a verified context-variable codebook mapping "
        "is unavailable. Missing values are handled by the saved pipeline's "
        "fitted imputers."
    )

    if predict_submitted:
        inputs = {
            feature: (
                np.nan
                if value == "Not answered"
                else value
            )
            for feature, value in categorical_inputs.items()
        }
        inputs["age"] = np.nan if age_unavailable else age_value
        inputs = {feature: inputs[feature] for feature in predictors}
        try:
            pipeline = load_saved_classifier()
            prediction = predict_profiles(inputs, pipeline).iloc[0]
        except (FileNotFoundError, TypeError, ValueError, KeyError):
            st.error(
                "The saved research model could not produce a prediction. "
                "Check that its validated model artifact is available."
            )
        else:
            predicted_id = int(prediction["predicted_cluster_id"])
            prediction_cols = st.columns(2)
            prediction_cols[0].metric("Predicted cluster ID", predicted_id)
            prediction_cols[1].metric(
                "Predicted profile",
                CLUSTER_LABELS[predicted_id],
            )
            prediction_cols = st.columns(2)
            prediction_cols[0].metric(
                "Model output for predicted class",
                f"{prediction['predicted_cluster_probability']:.1%}",
            )
            prediction_cols[1].caption(
                "This is the model's predicted-class probability output, "
                "not an individual's probability of exclusion or default."
            )
            probability_frame = pd.DataFrame(
                {
                    "Cluster profile": [
                        f"{cluster_id} — {CLUSTER_NAMES[cluster_id]}"
                        for cluster_id in CLUSTER_IDS
                    ],
                    "Model output": [
                        prediction[f"probability_cluster_{cluster_id}"]
                        for cluster_id in CLUSTER_IDS
                    ],
                }
            )
            probability_chart = px.bar(
                probability_frame,
                x="Model output",
                y="Cluster profile",
                orientation="h",
                range_x=[0, 1],
                title="Saved classifier outputs across cluster classes",
                labels={"Model output": "Model output"},
            )
            st.plotly_chart(probability_chart, config={"responsive": True})
            st.info(
                "Newly entered inputs do not receive a local SHAP explanation. "
                "Saved validation-sample explanations are on the evidence page."
            )




def show_shap_explanations() -> None:
    st.subheader("Validated SHAP explanations")
    st.write(
        "SHAP values explain the saved Logistic Regression classifier's "
        "decision scores for previously explained validation samples. They do "
        "not explain the dimension-scoring formula and do not establish causes "
        "or policy importance."
    )
    try:
        shap_global = loader.load_shap_global_importance()
        shap_local = loader.load_shap_local_explanations()
        shap_predictions = loader.load_shap_predictions()
        shap_summary = loader.load_shap_validation_summary()
    except (FileNotFoundError, ValueError, KeyError):
        st.warning(
            "Saved SHAP outputs are unavailable or could not be validated."
        )
        return

    cluster_options = (
        shap_global[["cluster_id", "cluster_label"]]
        .drop_duplicates()
        .sort_values("cluster_id")
    )
    selected_cluster = st.selectbox(
        "Cluster class for saved global explanation",
        cluster_options["cluster_id"].astype(int).tolist(),
        format_func=lambda cluster_id: (
            f"{cluster_id} — {CLUSTER_NAMES[cluster_id]}"
        ),
        key="predictive_shap_cluster",
    )
    selected_global = shap_global.loc[
        shap_global["cluster_id"] == selected_cluster
    ].sort_values(
        "mean_absolute_shap_decision_score",
        ascending=False,
    ).head(7)
    global_chart = px.bar(
        selected_global.sort_values(
            "mean_absolute_shap_decision_score",
            ascending=True,
        ),
        x="mean_absolute_shap_decision_score",
        y="feature",
        orientation="h",
        labels={
            "mean_absolute_shap_decision_score": (
                "Average absolute contribution to decision score"
            ),
            "feature": "Predictor",
        },
        title=(
            "Global average absolute predictor contributions to the "
            "selected class decision score"
        ),
    )
    st.plotly_chart(global_chart, config={"responsive": True})
    st.caption(
        "Predictor labels are kept as saved when a verified human-readable "
        "mapping is unavailable. Absolute contribution magnitude does not "
        "establish causality or policy priority."
    )
    st.caption(
        f"Global SHAP is the average absolute contribution to the selected "
        f"class decision score across {shap_summary['explained_rows']} saved "
        f"test respondents, using a background sample of "
        f"{shap_summary['background_rows']}. The full held-out test set had "
        "11,301 respondents; the saved SHAP sample is not the full test set. "
        "Magnitude is not evidence of causality or policy importance."
    )

    saved_rows = sorted(shap_predictions["row_index"].astype(int).unique())
    selected_row = st.selectbox(
        "Previously explained sample (sample index only; no respondent ID)",
        saved_rows,
        key="predictive_saved_shap_row",
    )
    saved_prediction = shap_predictions.loc[
        shap_predictions["row_index"] == selected_row
    ].iloc[0]
    explained_cluster = int(saved_prediction["predicted_cluster_id"])
    local = shap_local.loc[
        (shap_local["row_index"] == selected_row)
        & (shap_local["explained_class_id"] == explained_cluster)
    ].copy()
    st.caption(
        f"Saved sample prediction: cluster {explained_cluster} — "
        f"{CLUSTER_NAMES[explained_cluster]} | predicted-class model output "
        f"{saved_prediction['predicted_cluster_probability']:.1%}. The local "
        "contributions below refer to this saved sample and predicted class, "
        "not the newly entered inputs."
    )
    if not local.empty:
        local_chart = px.bar(
            local.sort_values("shap_decision_score_contribution"),
            x="shap_decision_score_contribution",
            y="feature",
            orientation="h",
            labels={
                "shap_decision_score_contribution": (
                    "Contribution to Logistic Regression decision score "
                    "(not probability)"
                ),
                "feature": "Predictor",
            },
            title="Saved local explanation for the selected validation sample",
        )
        local_chart.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(local_chart, config={"responsive": True})
        st.caption(
            "Positive values move the selected class's decision score upward; "
            "negative values move it downward. Neither direction shows a "
            "causal effect on financial inclusion."
        )
    else:
        st.info("No saved local SHAP contributions are available for this class.")
    st.caption(
        f"Validation: {shap_summary['explained_rows']} explained rows; "
        f"{shap_summary['background_rows']} background rows; maximum "
        f"additivity error {shap_summary['max_additivity_error']:.2e}; "
        f"maximum grouped contribution error "
        f"{shap_summary['max_grouped_sum_error']:.2e}; SHAP "
        f"{shap_summary['shap_version']}. The saved global values use the "
        "explained sample, not all 11,301 test respondents. SHAP values explain "
        "decision scores, not probabilities or causes of financial exclusion."
    )
    with st.expander("SHAP validation checks"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Validation check": name.replace("_", " ").title(), "Passed": passed}
                    for name, passed in shap_summary["checks_passed"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )



def show_model_evaluation_and_explainability(df: pd.DataFrame) -> None:
    st.header("Model, Explainability & Evidence")
    performance, explanations, evidence = st.tabs(
        ["Model performance", "SHAP explanations", "Clustering, scoring and data"]
    )
    with performance:
        show_model_performance()
    with explanations:
        show_shap_explanations()
    with evidence:
        show_methodology_data_evidence(df)

def show_methodology_data_evidence(df: pd.DataFrame) -> None:
    st.subheader("Clustering, scoring, data quality and limitations")
    st.write(
        "This section documents the analytical sample and how the saved "
        "results used by the dashboard were produced. All displayed findings "
        "are descriptive summaries or saved model outputs."
    )
    st.subheader("Analysis pathway")
    st.markdown(
        """
        1. **Data and variable validation:** the prepared Global Findex 2025
           individual survey data (survey year 2024) was checked against the
           Component 02 variable definitions and missing-response rules.
        2. **Dimension scoring:** validated indicator mappings produce five
           operational scores. Connectivity averages available components;
           financial resilience requires all three mapped inputs.
        3. **Profile discovery:** candidate clustering approaches were
           evaluated and the saved, validated `5D_k5` K-Means solution is
           loaded without retraining.
        4. **Profile interpretation:** cluster means and sizes summarize the
           saved assignments. Profiles are analytical patterns, not official
           categories.
        5. **Supervised evaluation:** saved classifiers predict the derived
           cluster labels from demographic and socioeconomic predictors.
        6. **Explainability:** saved SHAP values explain Logistic Regression
           decision scores for a limited sample.
        7. **Decision support:** descriptive differences inform questions for
           stakeholder validation; they do not demonstrate causes or proven
           interventions.
        """
    )
    with st.expander("Scoring rules and indicator limits"):
        scoring_details = pd.DataFrame(
            [
                ["Connectivity", "Phone ownership and internet use", "Mean available components; unavailable inputs stay missing."],
                ["Financial access", "Account ownership", "Existing binary account indicator."],
                ["Digital financial usage", "Any digital payment", "Existing validated anydigpayment indicator."],
                ["Digital safety", "Phone protection (con18)", "Scam exposure is not included in this score."],
                ["Financial resilience", "saved, fin24a, fin24b", "All three mapped responses are required."],
            ],
            columns=["Dimension", "Selected indicators", "Scoring rule and limit"],
        )
        st.dataframe(scoring_details, hide_index=True, width="stretch")
        st.caption(
            "The operational scores are not percentages of financial inclusion. "
            "Ordinal mappings and selected indicators do not capture every aspect "
            "of access, safety, or resilience."
        )
        with st.expander("Validated response mappings"):
            st.markdown(
                "- Phone ownership and phone protection: code 1 maps to 1; "
                "code 2 maps to 0.\n"
                "- Internet use, account ownership, digital-payment use, and "
                "saving use the existing 0/1 indicators.\n"
                "- Emergency-funds difficulty (`fin24a`): 1 to 0, 2 to 0.5, "
                "3 to 1.\n"
                "- Income-loss buffer (`fin24b`): 1 to 0, 2 to 1/3, "
                "3 to 2/3, 4 to 1.\n"
                "- Resilience is the mean of saving and both mapped responses "
                "only when all three are available."
            )
    st.subheader("Analytical population and completeness")
    st.write(
        f"The saved dataset contains **{df['wpid_random'].nunique():,}** "
        f"respondents across **{df['economy'].nunique():,}** economies. "
        "Economy summaries are unweighted respondent-level descriptions; "
        "they are not official estimates or rankings."
    )
    quality = loader.calculate_dimension_summary(df).copy()
    quality["Dimension"] = quality["variable"].map(DIMENSIONS)
    with st.expander("Dimension score summaries and availability"):
        st.dataframe(
            quality[["Dimension", "mean", "median", "missing_percent", "distinct_nonmissing_values"]]
            .rename(columns={
                "mean": "Mean score", "median": "Median score",
                "missing_percent": "Missing (%)",
                "distinct_nonmissing_values": "Distinct observed values",
            }).round(3),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "Missing or unasked responses remain unavailable. They are never "
        "treated as No or silently converted to zero. Scores from 0 to 1 are "
        "operational indicator summaries, not percentages of inclusion."
    )
    context_labels = {
        "female": "Gender code",
        "age": "Age",
        "educ": "Education code",
        "inc_q": "Income-quintile code",
        "emp_in": "Employment code",
        "urbanicity": "Urbanicity code",
        "regionwb": "World Bank region",
    }
    with st.expander("Context-field missingness"):
        context = pd.DataFrame(
            [
                {
                    "Field": label,
                    "Missing respondents": int(df[column].isna().sum()),
                    "Missing (%)": df[column].isna().mean() * 100,
                }
                for column, label in context_labels.items()
                if column in df.columns
            ]
        )
        st.dataframe(context.round(2), hide_index=True, width="stretch")

    st.subheader("Saved analytical evidence")
    try:
        clustering = loader.load_clustering_evidence()
    except (FileNotFoundError, ValueError, KeyError):
        st.warning(
            "Saved clustering evidence is unavailable or could not be validated."
        )
    else:
        st.write(
            f"Selected solution: **{clustering['candidate']}** "
            f"({clustering['algorithm']}, {clustering['clusters']} clusters, "
            f"{clustering['scaling']} scale)."
        )
        validation_metrics = st.columns(3)
        validation_metrics[0].metric(
            "Silhouette", f"{clustering['silhouette_score']:.3f}"
        )
        validation_metrics[1].metric(
            "Davies-Bouldin", f"{clustering['davies_bouldin_score']:.3f}"
        )
        validation_metrics[2].metric(
            "Calinski-Harabasz",
            f"{clustering['calinski_harabasz_score']:,.1f}",
        )
        st.caption(
            f"Seed stability: mean pairwise ARI "
            f"{clustering['mean_pairwise_ari']:.3f} across "
            f"{clustering['seed_runs']} runs; K-Means/GMM agreement ARI "
            f"{clustering['kmeans_gmm_ari']:.3f}. These describe internal "
            "separation and stability, not causal validity or effectiveness."
        )
        if clustering["legacy_summary_conflict"]:
            st.caption(
                "The exploratory summary still calls the selection provisional, "
                "but the final decision report, candidate matrix, stability report, "
                "and validated model metadata confirm 5D_k5. The final records are "
                "used here; the older summary remains unchanged."
            )

    codebook_path = (
        PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"
        / "01_data_preparation" / "fi_codebook_verification_register.csv"
    )
    st.subheader("Codebook review status")
    if codebook_path.is_file():
        codebook = pd.read_csv(codebook_path)
        unresolved = codebook.loc[
            codebook.get("issue", pd.Series(index=codebook.index, dtype=object))
            .fillna("").astype(str).str.strip().ne("")
        ]
        if unresolved.empty:
            st.info(
                "The saved register contains no entries with a recorded open issue. "
                "Variable inclusion and interpretation should still follow the "
                "validated scoring implementation."
            )
        else:
            columns = [
                column for column in ["variable", "meaning", "verification_status", "issue"]
                if column in unresolved.columns
            ]
            st.dataframe(unresolved[columns], hide_index=True, width="stretch")
        if "primary_model_status" in codebook.columns and codebook[
            "primary_model_status"
        ].eq("PENDING").any():
            st.caption(
                "The codebook register marks primary-model approval as pending "
                "for registered variables; this dashboard reports the existing "
                "validated analysis and does not upgrade that status."
            )
    else:
        st.info("The saved codebook verification register is unavailable.")

    st.subheader("Interpretation limits")
    st.markdown(
        """
        - Differences are descriptive and do not establish causes or intervention effects.
        - Respondent summaries do not account for survey design uncertainty.
        - Supervised models reproduce derived cluster labels; they do not predict
          creditworthiness, default, or independently measured exclusion.
        - SHAP values describe model contributions to decision scores, not causes.
        - Numeric demographic codes are retained where verified readable
          category mappings are not available; consult the source codebook.
        """
    )
st.title("Digital Financial Inclusion")
st.caption(
    "Explore financial access, digital participation, safety, and financial "
    "resilience across the observed survey sample."
)
with st.sidebar:
    st.subheader("Component 02")
    section = st.radio(
        "Explore",
        [
            "Overview & Key Findings",
            "Explore Inclusion Profiles",
            "Individual Profile Explorer",
            "Model, Explainability & Evidence",
        ],
        key="financial_inclusion_navigation",
    )
st.divider()

try:
    analytical_df = load_dashboard_dataset()
except (FileNotFoundError, ValueError, KeyError):
    st.error("Could not load the validated analytical dataset.")
    st.info(
        "Ensure the validated final dataset is available under "
        "data/financial_inclusion/processed/."
    )
    st.stop()

if section == "Overview & Key Findings":
    show_overall_dashboard(analytical_df)
elif section == "Explore Inclusion Profiles":
    show_profile_explorer(analytical_df)
elif section == "Individual Profile Explorer":
    show_individual_profile_explorer(analytical_df)
else:
    show_model_evaluation_and_explainability(analytical_df)

st.divider()
st.caption(
    "Component 02 | Global Findex 2025 individual microdata | "
    "Survey year: 2024"
)
