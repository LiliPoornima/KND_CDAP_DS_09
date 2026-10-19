
"""Component 02: Global Digital Financial Inclusion Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px

from frontend.financial_inclusion import data_loader as loader


st.set_page_config(
    page_title="Digital Financial Inclusion",
    page_icon="🌍",
    layout="wide",
)

# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("🌍 Global Digital Financial Inclusion")
st.caption(
    "Explainable Digital Financial Inclusion Gap Profiling "
    "Using Global Household Data"
)

st.markdown(
    """
    Explore digital connectivity, financial access, digital financial
    usage, phone protection and financial resilience using Global
    Findex 2025 individual-level survey data.
    """
)

# --------------------------------------------------
# Load existing research outputs
# --------------------------------------------------

try:
    coverage = loader.load_dimension_coverage()
    dimension_df = loader.load_dimension_distributions()
    cluster_df = loader.load_cluster_profiles()
    metrics_df = loader.load_model_metrics()
    clustering_summary = loader.load_clustering_summary()

except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Unable to load a research output: {exc}")
    st.info(
        "Check that the Component 02 research output files exist "
        "in outputs/financial_inclusion/reports and metrics."
    )
    st.stop()

# --------------------------------------------------
# Overview
# --------------------------------------------------

st.header("1. Research Overview")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Survey respondents",
    f"{coverage['total_respondents']:,}",
)

metric2.metric(
    "Economies in source data",
    f"{coverage['total_economies']:,}",
)

metric3.metric(
    "5D clustering respondents",
    f"{int(cluster_df.loc[cluster_df['candidate'] == '5D_k4', 'respondents'].sum()):,}",
)

st.info(
    "The complete five-dimension clustering sample is smaller than "
    "the full dataset because some respondents have missing dimension "
    "scores. Missing values were not imputed for dimension creation."
)

st.markdown("### Research dimensions")

st.dataframe(
    pd.DataFrame(
        {
            "Dimension": [
                "Connectivity",
                "Financial access",
                "Digital financial usage",
                "Digital safety",
                "Financial resilience",
            ],
            "Focus": [
                "Phone and internet connectivity",
                "Access to a financial account",
                "Use of digital payments",
                "Phone protection indicator",
                "Savings and emergency financial capacity",
            ],
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# --------------------------------------------------
# Dimensions
# --------------------------------------------------

st.header("2. Dimension Analysis")

st.markdown(
    "Mean scores describe the observed sample for each derived "
    "indicator. They should not be interpreted as country rankings."
)

dimension_labels = {
    "score_connectivity": "Connectivity",
    "score_financial_access": "Financial access",
    "score_digital_financial_usage": "Digital financial usage",
    "score_digital_safety": "Digital safety",
    "score_financial_resilience": "Financial resilience",
}

score_rows = dimension_df[
    dimension_df["variable"].isin(dimension_labels)
].copy()

if not score_rows.empty:
    score_rows["Dimension"] = score_rows["variable"].map(dimension_labels)

    score_rows["Missing (%)"] = score_rows["variable"].map(
        coverage["score_missing_percent"]
    )

    score_rows = score_rows.rename(
        columns={
            "mean": "Mean score",
            "median": "Median score",
            "distinct_nonmissing_values": "Distinct values",
        }
    )

    score_chart = px.bar(
        score_rows,
        x="Dimension",
        y="Mean score",
        range_y=[0, 1],
        text_auto=".3f",
        title="Mean score by dimension",
    )
    score_chart.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(score_chart, use_container_width=True)

    st.markdown("### Score summary")
    st.dataframe(
        score_rows[
            [
                "Dimension",
                "Mean score",
                "Median score",
                "Missing (%)",
                "Distinct values",
            ]
        ].round(3),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.warning("No matching dimension score rows were found.")

# --------------------------------------------------
# Clustering
# --------------------------------------------------

st.header("3. Respondent Cluster Profiles")

st.markdown(
    """
    The five-dimension K-Means solution uses four candidate clusters.
    Profiles summarize respondent groups; they are not country rankings
    or verified real-world outcome categories.
    """
)

five_d = cluster_df[
    cluster_df["candidate"] == "5D_k4"
].copy()

cluster_names = {
    0: "High usage, lower phone protection",
    1: "Broad digital-financial inclusion",
    2: "Connected but financially underserved",
    3: "Lower digital-financial inclusion",
}

if not five_d.empty:
    five_d["Profile"] = five_d["cluster_id"].map(cluster_names)

    c1, c2 = st.columns(2)

    with c1:
        size_chart = px.bar(
            five_d,
            x="Profile",
            y="respondents",
            text="respondents",
            title="Respondents by cluster",
        )
        size_chart.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(size_chart, use_container_width=True)

    with c2:
        score_columns = {
            "score_connectivity_mean": "Connectivity",
            "score_financial_access_mean": "Financial access",
            "score_digital_financial_usage_mean": "Digital usage",
            "score_digital_safety_mean": "Digital safety",
            "score_financial_resilience_mean": "Resilience",
        }

        heatmap_data = five_d.set_index("Profile")[
            list(score_columns)
        ].rename(columns=score_columns)

        heatmap_data.index.name = "Cluster profile"

        heatmap_chart = px.imshow(
            heatmap_data,
            labels={
                "x": "Dimension",
                "y": "Cluster profile",
                "color": "Mean score",
            },
            zmin=0,
            zmax=1,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Dimension scores across clusters",
        )
        st.plotly_chart(heatmap_chart, use_container_width=True)

    st.markdown("### Cluster summary")
    display_clusters = five_d[
        [
            "cluster_id",
            "Profile",
            "respondents",
            "percent_of_candidate",
            "economies_represented",
            "score_connectivity_mean",
            "score_financial_access_mean",
            "score_digital_financial_usage_mean",
            "score_digital_safety_mean",
            "score_financial_resilience_mean",
        ]
    ].rename(
        columns={
            "cluster_id": "Cluster",
            "respondents": "Respondents",
            "percent_of_candidate": "Sample (%)",
            "economies_represented": "Economies represented",
            "score_connectivity_mean": "Connectivity",
            "score_financial_access_mean": "Financial access",
            "score_digital_financial_usage_mean": "Digital usage",
            "score_digital_safety_mean": "Digital safety",
            "score_financial_resilience_mean": "Resilience",
        }
    )

    st.dataframe(
        display_clusters.round(3),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.warning("The 5D_k4 cluster profiles were not found.")

with st.expander("Clustering method details"):
    st.write(
        "Final model selected:",
        clustering_summary.get("final_model_selected", "Not specified"),
    )
    st.write(
        "Eligible respondents:",
        clustering_summary.get("eligible_respondents", "Not specified"),
    )
    st.write(
        "Economies:",
        clustering_summary.get("economies", "Not specified"),
    )
    st.write(
        "Stability analysis completed:",
        clustering_summary.get("stability_analysis_completed", "Not specified"),
    )
    st.write(clustering_summary.get("note", ""))

# --------------------------------------------------
# Supervised model evaluation
# --------------------------------------------------

st.header("4. Supervised Model Evaluation")

st.markdown(
    """
    The supervised models predict the cluster labels generated by
    K-Means. They do not directly predict financial inclusion outcomes.
    """
)

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

display_metrics = metrics_df.copy()

for column in metric_columns:
    if column in display_metrics.columns:
        display_metrics[column] = display_metrics[column].round(3)

st.dataframe(
    display_metrics,
    hide_index=True,
    use_container_width=True,
)

if not metrics_df.empty:
    metric_choice = st.selectbox(
        "Select evaluation metric",
        options=metric_columns,
        format_func=lambda value: value.replace("_", " ").title(),
    )

    model_chart = px.bar(
        metrics_df.sort_values(metric_choice, ascending=False),
        x="model",
        y=metric_choice,
        range_y=[0, 1],
        text_auto=".3f",
        title=f"Model comparison: {metric_choice.replace('_', ' ').title()}",
    )
    st.plotly_chart(model_chart, use_container_width=True)

    st.caption(
        "Logistic Regression has the strongest balanced accuracy and "
        "macro-F1 in the saved baseline results. XGBoost has the highest "
        "overall accuracy. Model selection depends on the evaluation objective."
    )

# --------------------------------------------------
# Methodology and limitations
# --------------------------------------------------

st.header("5. Methodology & Limitations")

with st.expander("Dimension scoring and missing data", expanded=True):
    st.markdown(
        """
        - Five derived dimensions are analyzed separately.
        - Missing dimension scores are not automatically imputed.
        - Digital safety currently represents the phone-protection indicator,
          not every aspect of digital safety.
        - Financial resilience includes ordinal mappings whose scores assume
          equal spacing between mapped response levels.
        - Different analyses use different eligible respondent samples.
        """
    )

with st.expander("Clustering interpretation"):
    st.markdown(
        """
        - Cluster labels are data-driven groups, not naturally occurring
          or officially defined categories.
        - The four-cluster five-dimension solution is used for profiling.
        - Profiles describe the selected sample and should not be treated as
          economy-level rankings.
        """
    )

with st.expander("Supervised model limitations"):
    st.markdown(
        """
        - The target is the K-Means cluster label.
        - Predictors are demographic and socioeconomic variables.
        - Performance measures describe held-out records from a stratified
          random split, not a country-held-out evaluation.
        - Classification performance does not establish causal relationships.
        """
    )

st.divider()
st.caption(
    "Component 02 | Global Findex 2025 individual microdata "
    "| Survey year: 2024"
)
