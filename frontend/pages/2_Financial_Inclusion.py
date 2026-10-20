
"""Component 02: Global Digital Financial Inclusion Dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.financial_inclusion import data_loader as loader


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Digital Financial Inclusion",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Global Digital Financial Inclusion")
st.caption(
    "Explainable Digital Financial Inclusion Gap Profiling "
    "Using Global Household Data"
)

st.markdown(
    """
    Explore digital connectivity, financial access, digital financial
    usage, digital safety and financial resilience using Global Findex
    2025 individual-level survey data.
    """
)


# --------------------------------------------------
# Load and validate the final analytical dataset
# --------------------------------------------------

try:
    df = loader.load_final_dataset()
    dimension_df = loader.calculate_dimension_summary(df)
    cluster_df = loader.calculate_cluster_profiles(df)

except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Unable to load the final analytical dataset: {exc}")
    st.info(
        "Ensure that data/financial_inclusion/processed/"
        "fi_final_analytical_dataset.csv exists and is valid."
    )
    st.stop()


# --------------------------------------------------
# Shared definitions
# --------------------------------------------------

DIMENSIONS = {
    "score_connectivity": "Connectivity",
    "score_financial_access": "Financial access",
    "score_digital_financial_usage": "Digital financial usage",
    "score_digital_safety": "Digital safety",
    "score_financial_resilience": "Financial resilience",
}

CLUSTER_NAMES = {
    0: "Digitally Active, Safety Gap",
    1: "Connected and Digitally Included",
    2: "Connected, Financially Excluded",
    3: "Multi-Dimensional Inclusion Gap",
    4: "Access Without Digital Usage",
}

CLUSTER_ORDER = list(CLUSTER_NAMES.keys())

df = df.copy()
df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
df["Cluster profile"] = df["cluster_id"].map(CLUSTER_NAMES)

if df["Cluster profile"].isna().any():
    st.error("Some cluster IDs do not have a configured profile label.")
    st.stop()

cluster_df["Profile"] = cluster_df["cluster_id"].map(CLUSTER_NAMES)
cluster_df["cluster_id"] = pd.to_numeric(
    cluster_df["cluster_id"], errors="raise"
).astype(int)
cluster_df["Profile"] = cluster_df["cluster_id"].map(CLUSTER_NAMES)

dimension_df["Dimension"] = dimension_df["variable"].map(DIMENSIONS)
dimension_df = dimension_df.dropna(subset=["Dimension"]).copy()


# --------------------------------------------------
# 1. Research overview
# --------------------------------------------------

st.header("1. Research Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Analytical respondents",
    f"{df['wpid_random'].nunique():,}",
)
m2.metric(
    "Economies represented",
    f"{df['economy'].nunique():,}",
)
m3.metric(
    "Final clusters",
    f"{df['cluster_id'].nunique():,}",
)
m4.metric(
    "Analytical variables",
    f"{len(df.columns):,}",
)

st.info(
    "This dashboard uses the selected 5D_k5 analytical sample. "
    "Each row represents one respondent assigned to one of five "
    "clusters. Findings describe this sample and are not causal claims "
    "or official economy rankings."
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
                "Mobile-phone ownership and internet use",
                "Financial-account ownership",
                "Use of digital payments",
                "Phone-protection indicator",
                "Savings and emergency financial capacity",
            ],
        }
    ),
    hide_index=True,
    use_container_width=True,
)


# --------------------------------------------------
# 2. Dimension analysis
# --------------------------------------------------

st.header("2. Dimension Analysis")

st.markdown(
    "The following statistics are calculated directly from the final "
    "analytical dataset. Scores range from 0 to 1 under the study's "
    "indicator-construction rules."
)

dimension_display = dimension_df.rename(
    columns={
        "mean": "Mean score",
        "median": "Median score",
        "missing_percent": "Missing (%)",
        "distinct_nonmissing_values": "Distinct values",
    }
)

dimension_chart = px.bar(
    dimension_display,
    x="Dimension",
    y="Mean score",
    text_auto=".3f",
    range_y=[0, 1],
    title="Mean dimension scores",
)
dimension_chart.update_layout(xaxis_tickangle=-20)
st.plotly_chart(dimension_chart, use_container_width=True)

st.markdown("### Dimension score summary")

st.dataframe(
    dimension_display[
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

with st.expander("How are these dimensions interpreted?"):
    st.markdown(
        """
        - **Connectivity:** mobile-phone and internet indicators.
        - **Financial access:** financial-account ownership.
        - **Digital financial usage:** digital-payment use.
        - **Digital safety:** phone protection, not every aspect of
          digital security.
        - **Financial resilience:** savings and mapped emergency
          financial-capacity indicators.
        - Missing values are not automatically converted to zero.
        """
    )


# --------------------------------------------------
# 3. Five-dimensional cluster profiles
# --------------------------------------------------

st.header("3. Respondent Cluster Profiles")

st.markdown(
    "Cluster sizes and dimension means are recalculated from the "
    "respondent-level dataset. Cluster IDs are identifiers, not rankings."
)

left, right = st.columns(2)

with left:
    size_chart = px.bar(
        cluster_df.sort_values("cluster_id"),
        x="Profile",
        y="respondents",
        text="respondents",
        category_orders={"Profile": [CLUSTER_NAMES[i] for i in CLUSTER_ORDER]},
        title="Respondents by cluster",
    )
    size_chart.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(size_chart, use_container_width=True)

with right:
    score_columns = {
        "score_connectivity_mean": "Connectivity",
        "score_financial_access_mean": "Financial access",
        "score_digital_financial_usage_mean": "Digital usage",
        "score_digital_safety_mean": "Digital safety",
        "score_financial_resilience_mean": "Resilience",
    }

    heatmap_data = (
        cluster_df.sort_values("cluster_id")
        .set_index("Profile")[list(score_columns.keys())]
        .rename(columns=score_columns)
    )
    heatmap_data.index.name = "Cluster profile"

    heatmap = px.imshow(
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
        title="Mean dimension scores by cluster",
    )
    st.plotly_chart(heatmap, use_container_width=True)

cluster_display = cluster_df[
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
        "cluster_id": "Cluster ID",
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

st.markdown("### Cluster summary")
st.dataframe(
    cluster_display.round(3),
    hide_index=True,
    use_container_width=True,
)

with st.expander("Clustering method and interpretation"):
    st.markdown(
        """
        - Selected solution: **5D_k5**.
        - Algorithm: K-Means with five dimension scores and five clusters.
        - The dashboard uses the cluster assignments already stored in
          the final analytical dataset; it does not refit K-Means.
        - Profiles describe the selected complete-case analytical sample.
        - These groups are research-derived profiles, not official
          Global Findex categories.
        """
    )


# --------------------------------------------------
# 4. Demographic and socioeconomic comparisons
# --------------------------------------------------

st.header("4. Demographic and Socioeconomic Analysis")

st.markdown(
    "Compare the composition of the five clusters using the demographic "
    "and socioeconomic variables available in the analytical dataset. "
    "Percentages below are calculated among non-missing values within "
    "each cluster."
)

demographic_options = {
    "female": "Gender code",
    "educ": "Education level",
    "inc_q": "Income quintile",
    "emp_in": "Employment status",
    "urbanicity": "Urbanicity",
    "regionwb": "World Bank region",
    "economy": "Economy",
}

available_demographics = {
    key: label
    for key, label in demographic_options.items()
    if key in df.columns
}

selected_variable = st.selectbox(
    "Select a demographic or geographic variable",
    options=list(available_demographics.keys()),
    format_func=lambda value: available_demographics[value],
)

demographic_df = df.dropna(subset=[selected_variable]).copy()

demographic_df[selected_variable] = (
    demographic_df[selected_variable].astype(str)
)

demographic_counts = (
    demographic_df.groupby(
        ["cluster_id", "Cluster profile", selected_variable],
        observed=True,
    )
    .size()
    .reset_index(name="Respondents")
)

demographic_counts["Percentage within cluster"] = (
    demographic_counts["Respondents"]
    / demographic_counts.groupby("cluster_id")["Respondents"].transform("sum")
    * 100
)

if selected_variable == "economy":
    st.caption(
        "Economy comparison: showing the 20 economies with the most "
        "respondents in the selected analytical sample."
    )

    top_economies = (
        demographic_counts.groupby(selected_variable)["Respondents"]
        .sum()
        .nlargest(20)
        .index
    )
    plot_data = demographic_counts[
        demographic_counts[selected_variable].isin(top_economies)
    ].copy()
else:
    plot_data = demographic_counts

demographic_chart = px.bar(
    plot_data,
    x=selected_variable,
    y="Percentage within cluster",
    color="Cluster profile",
    barmode="group",
    title=f"Cluster composition by {available_demographics[selected_variable]}",
)
demographic_chart.update_layout(xaxis_tickangle=-30)
st.plotly_chart(demographic_chart, use_container_width=True)

with st.expander("Demographic comparison data"):
    st.dataframe(
        demographic_counts.rename(
            columns={
                selected_variable: available_demographics[selected_variable]
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Gender, education, income, employment and urbanicity use the "
    "coded values available in the dataset. Consult the Global Findex "
    "codebook when interpreting category codes. Missing values are "
    "excluded from each variable's composition percentages."
)


# --------------------------------------------------
# 5. Data quality and missingness
# --------------------------------------------------

st.header("5. Data Quality")

st.markdown(
    "Missingness is calculated from the final analytical dataset. "
    "The selected clustering population already requires complete "
    "values for all five dimension scores."
)

missing_df = (
    df.isna()
    .sum()
    .rename_axis("Variable")
    .reset_index(name="Missing values")
)
missing_df["Missing (%)"] = (
    missing_df["Missing values"] / len(df) * 100
)
missing_df = missing_df[missing_df["Missing values"] > 0].copy()
missing_df = missing_df.sort_values("Missing (%)", ascending=False)

if missing_df.empty:
    st.success("No missing values were found in the final dataset.")
else:
    missing_chart = px.bar(
        missing_df,
        x="Variable",
        y="Missing (%)",
        hover_data=["Missing values"],
        title="Missing values by variable",
    )
    missing_chart.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(missing_chart, use_container_width=True)

    with st.expander("View missingness table"):
        st.dataframe(
            missing_df,
            hide_index=True,
            use_container_width=True,
        )


# --------------------------------------------------
# 6. Methodology and limitations
# --------------------------------------------------

st.header("6. Methodology & Limitations")

with st.expander("Dimension scoring and missing data", expanded=True):
    st.markdown(
        """
        - Five dimensions represent connectivity, financial access,
          digital-financial usage, digital safety and financial resilience.
        - Missing values are not automatically imputed in this dashboard.
        - Digital safety currently reflects a phone-protection indicator,
          not every aspect of digital safety.
        - Financial resilience uses mapped ordinal responses; its scores
          assume equal spacing between mapped response levels.
        - This dashboard uses the final five-dimensional clustering sample,
          not every respondent in the original microdata.
        """
    )

with st.expander("Clustering interpretation"):
    st.markdown(
        """
        - Cluster labels are data-driven profiles, not naturally occurring
          or officially defined categories.
        - The final solution contains five clusters.
        - Cluster assignments are read from the saved final analytical
          dataset; clustering is not refitted by the dashboard.
        - Profiles describe the selected sample and should not be treated
          as economy-level rankings.
        """
    )

with st.expander("Supervised modelling status"):
    st.markdown(
        """
        Supervised model metrics are intentionally not displayed in this
        version of the dashboard. The previous implementation read saved
        metrics from generated report files. A reproducible model-evaluation
        workflow should be implemented separately before showing those
        results here.

        Cluster classification predicts the cluster labels created by
        unsupervised learning. It does not directly predict a financial
        inclusion outcome, establish causality or demonstrate generalisation
        to unseen economies.
        """
    )


# ---------------------------------------------------------
# 7. SUPERVISED MODELLING & MODEL COMPARISON
# ---------------------------------------------------------

st.header("7. Supervised Modelling & Model Comparison")

st.markdown("""
The supervised classification task predicts the selected `5D_k5`
cluster label using demographic and socioeconomic predictors.

The cluster labels are generated by the unsupervised clustering stage.
Therefore, this task evaluates how well those cluster assignments can
be predicted; it does not establish causal relationships.
""")

st.subheader("7.1 Classification Setup")

setup_col1, setup_col2, setup_col3 = st.columns(3)

with setup_col1:
    st.metric("Target", "cluster_id")

with setup_col2:
    st.metric("Predictor variables", "7")

with setup_col3:
    st.metric("Target classes", "5")

st.write("**Predictors used in the previous evaluation:**")
st.code(
    "female, age, educ, inc_q, emp_in, urbanicity, regionwb",
    language="text",
)

st.info(
    "The previous audit identified 56,503 respondents and 991 rows "
    "excluded from supervised modelling because predictor values "
    "were incomplete. The resulting complete-case sample contained "
    "55,512 respondents."
)

st.subheader("7.2 Model Performance Comparison")

# Results from the previously completed evaluation run.
# Replace these values with metrics returned by the reproducible
# model-evaluation pipeline when it is integrated.

supervised_results = pd.DataFrame([
    {
        "Model": "Majority-class baseline",
        "Accuracy": 0.417220,
        "Balanced Accuracy": 0.200000,
        "Macro-F1": 0.117757,
        "Weighted-F1": 0.245653,
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.403858,
        "Balanced Accuracy": 0.378270,
        "Macro-F1": 0.362406,
        "Weighted-F1": 0.416086,
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.354393,
        "Balanced Accuracy": 0.308032,
        "Macro-F1": 0.303060,
        "Weighted-F1": 0.363997,
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.472879,
        "Balanced Accuracy": 0.333557,
        "Macro-F1": 0.335383,
        "Weighted-F1": 0.424340,
    },
])

metric_columns = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "Weighted-F1",
]

display_results = supervised_results.copy()
display_results[metric_columns] = (
    display_results[metric_columns].round(4)
)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Values are from the previous evaluation run. They are displayed "
    "here for reporting and are not recalculated by this UI block."
)

st.subheader("7.3 Compare Performance Metrics")

selected_metrics = st.multiselect(
    "Select metrics to compare",
    options=metric_columns,
    default=["Accuracy", "Balanced Accuracy", "Macro-F1"],
    key="supervised_metric_selection",
)

if selected_metrics:
    chart_data = supervised_results.melt(
        id_vars="Model",
        value_vars=selected_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Supervised Classification Performance",
        range_y=[0, 1],
    )
    fig.update_layout(
        yaxis_title="Score",
        xaxis_title="Classification model",
        legend_title="Evaluation metric",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("7.4 Best-performing Model by Metric")

best_balanced = supervised_results.loc[
    supervised_results["Balanced Accuracy"].idxmax()
]
best_macro = supervised_results.loc[
    supervised_results["Macro-F1"].idxmax()
]
best_accuracy = supervised_results.loc[
    supervised_results["Accuracy"].idxmax()
]
best_weighted = supervised_results.loc[
    supervised_results["Weighted-F1"].idxmax()
]

best_col1, best_col2 = st.columns(2)

with best_col1:
    st.metric(
        "Highest Balanced Accuracy",
        best_balanced["Model"],
        f'{best_balanced["Balanced Accuracy"]:.4f}',
    )
    st.metric(
        "Highest Macro-F1",
        best_macro["Model"],
        f'{best_macro["Macro-F1"]:.4f}',
    )

with best_col2:
    st.metric(
        "Highest Accuracy",
        best_accuracy["Model"],
        f'{best_accuracy["Accuracy"]:.4f}',
    )
    st.metric(
        "Highest Weighted-F1",
        best_weighted["Model"],
        f'{best_weighted["Weighted-F1"]:.4f}',
    )

st.markdown("""
**Interpretation**

- **Logistic Regression** achieved the highest Balanced Accuracy
  (0.3783) and Macro-F1 (0.3624) among the evaluated models.
- **XGBoost** achieved the highest Accuracy (0.4729) and Weighted-F1
  (0.4243).
- The majority-class baseline achieved 0.4172 Accuracy, showing why
  Accuracy alone is insufficient for comparing these imbalanced
  cluster classes.
- Logistic Regression is the preferred candidate when balanced
  class-level performance is prioritized. XGBoost is the candidate
  with the highest overall Accuracy in this evaluation.
""")

st.subheader("7.5 Model Evaluation Limitations")

st.markdown("""
- The target is the cluster assignment produced by the selected
  unsupervised clustering model, not an independently observed
  financial-inclusion outcome.
- Predictive performance does not demonstrate causation or validate
  the substantive meaning of the clusters.
- Incomplete predictor records were excluded in the previous
  evaluation; the results apply to the resulting complete-case sample.
- Aggregate metrics can hide poor prediction for individual clusters.
  A per-cluster classification report and confusion matrix should
  also be generated from the evaluation pipeline before final reporting.
- The metrics displayed above are from the previous evaluation run.
  For full reproducibility, connect this section to the model-evaluation
  script rather than treating these displayed values as live results.
""")


st.divider()
st.caption(
    "Component 02 | Global Findex 2025 individual microdata "
    "| Survey year: 2024"
)
