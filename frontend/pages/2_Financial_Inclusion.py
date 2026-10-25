
"""Component 02: Global Digital Financial Inclusion Dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.financial_inclusion import data_loader as loader


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIMENSION_MODULE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "financial_inclusion"
    / "02_dimension_creation"
)

CLUSTERING_MODULE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "financial_inclusion"
    / "03_clustering"
)

for module_dir in (DIMENSION_MODULE_DIR, CLUSTERING_MODULE_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from fi_individual_profiling import profile_individual


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
# Main dashboard navigation
# --------------------------------------------------

dashboard_section = st.radio(
    "Navigate to",
    options=[
        "Overall Dashboard",
        "Community Analysis",
        "Individual Profiling",
    ],
    horizontal=True,
    key="financial_inclusion_navigation",
)

st.divider()


# --------------------------------------------------
# Load and validate the final analytical dataset
# --------------------------------------------------

try:
    df = loader.load_final_dataset()
    dimension_df = loader.calculate_dimension_summary(df)
    cluster_df = loader.calculate_cluster_profiles(df)
    supervised_results = loader.load_supervised_metrics()
    classification_reports = loader.load_supervised_classification_reports()
    confusion_matrices = loader.load_supervised_confusion_matrices()

except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Unable to load the final analytical dataset or supervised outputs: {exc}")
    st.info(
        "Ensure that the final dataset and reproducible supervised-model outputs exist "
        "under data/financial_inclusion/processed/ and outputs/financial_inclusion/metrics/."
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
if dashboard_section == "Overall Dashboard":
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

if dashboard_section == "Community Analysis":
    
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

if dashboard_section == "Overall Dashboard":
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
            This dashboard displays saved outputs from the existing supervised
            evaluation workflow. The workflow evaluates the full 56,503-respondent
            cohort using a stratified 80/20 split with random_state=42 (45,202
            training and 11,301 test respondents). It retains respondents with
            missing predictors and fits its imputers on the training partition
            before applying them to the test partition.

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
        "The evaluation uses all 56,503 respondents in a stratified 80/20 split: "
        "45,202 training and 11,301 test respondents. A total of 991 respondents "
        "have one or more missing predictor values; they were retained and handled "
        "by the training pipeline's imputers. Imputation parameters are fitted on "
        "the training partition and applied to the test partition. The 55,512 "
        "complete-case count is not the evaluation sample size."
    )

    st.subheader("7.2 Model Performance Comparison")

    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro-F1",
        "Weighted-F1",
    ]

    # Reuse the saved metrics generated by the reproducible evaluation pipeline.
    display_results = supervised_results.copy()
    display_results[metric_columns] = display_results[metric_columns].round(4)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Results are loaded from the saved baseline-evaluation output generated by the component's "
        "supervised-model workflow and are not recomputed on every page rerun."
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
    - The majority-class baseline achieved 0.4172 Accuracy but only 0.2000
    balanced accuracy, showing why accuracy alone is insufficient for
    these imbalanced cluster classes.
    - Logistic Regression is the preferred candidate when balanced class-level
    performance is prioritized. XGBoost has the highest overall accuracy,
    but its balanced accuracy and macro-F1 are lower than Logistic Regression.
    """)

    st.subheader("7.5 Per-cluster Performance and Confusion Matrix")

    model_choice = st.selectbox(
        "Select a model to inspect",
        options=list(classification_reports.keys()),
        index=0,
    )

    if model_choice in classification_reports:
        st.text(classification_reports[model_choice])

        if model_choice in confusion_matrices:
            matrix = confusion_matrices[model_choice]
            labels = [0, 1, 2, 3, 4]
            fig = px.imshow(
                matrix,
                x=labels,
                y=labels,
                labels=dict(x="Predicted cluster", y="Actual cluster", color="Count"),
                title=f"Confusion matrix for {model_choice}",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("7.6 Model Evaluation Limitations")

    st.markdown("""
    - The target is the cluster assignment produced by the selected unsupervised
    clustering model rather than an independently observed financial-inclusion outcome.
    - Predictive performance does not establish causality or validate the substantive
    meaning of the clusters.
    - The evaluation retains the full cohort, including 991 respondents with one or
    more missing predictors. Imputers are fitted on the training partition and then
    applied to the held-out test partition. This avoids excluding those respondents,
    but the missingness pattern and imputation assumptions may still affect model
    performance and should be considered when interpreting results.
    - Aggregate metrics can mask poor performance for some clusters; the per-cluster report
    and confusion matrix should be interpreted alongside the summary metrics.
    - These values are loaded from the reproducible evaluation outputs rather than being
    recomputed in this page at runtime.
    """)


    st.subheader("7.7 Explainability with SHAP")

    st.markdown("""
    This section explains the saved Logistic Regression classifier's
    predictions of the previously derived financial-inclusion clusters.
    SHAP contributions describe changes in the model's decision scores;
    they do not establish causal explanations for financial exclusion.
    """)

    try:
        shap_global = loader.load_shap_global_importance()
        shap_local = loader.load_shap_local_explanations()
        shap_predictions = loader.load_shap_predictions()
        shap_summary = loader.load_shap_validation_summary()

        st.caption(
            f"Explained sample: {len(shap_predictions):,} respondents | "
            f"Background sample: {shap_summary['background_rows']:,} respondents"
        )

        cluster_options = (
            shap_global[["cluster_id", "cluster_label"]]
            .drop_duplicates()
            .sort_values("cluster_id")
        )

        selected_shap_cluster = st.selectbox(
            "Select a predicted cluster for global explanation",
            options=cluster_options["cluster_id"].tolist(),
            format_func=lambda cid: (
                f"{cid} — "
                f"{cluster_options.loc[
                    cluster_options['cluster_id'] == cid, 'cluster_label'
                ].iloc[0]}"
            ),
            key="shap_global_cluster",
        )

        selected_global = shap_global[
            shap_global["cluster_id"] == selected_shap_cluster
        ].sort_values("mean_absolute_shap_decision_score", ascending=True)

        fig_global = px.bar(
            selected_global,
            x="mean_absolute_shap_decision_score",
            y="feature",
            orientation="h",
            labels={
                "mean_absolute_shap_decision_score": "Mean absolute SHAP contribution",
                "feature": "Predictor",
            },
            title="Global feature importance for the selected cluster",
            hover_data=["rank_within_cluster"],
        )
        st.plotly_chart(fig_global, use_container_width=True)

        st.caption(
            "Larger mean absolute SHAP values indicate greater average influence "
            "on the selected class's decision score within this explained sample. "
            "They do not indicate whether a feature increases or decreases the score."
        )

        st.divider()
        st.markdown("#### Individual prediction explanation")

        prediction_options = shap_predictions["row_index"].tolist()

        selected_row = st.selectbox(
            "Select an explained respondent",
            options=prediction_options,
            format_func=lambda row_id: (
                f"Sample {row_id} — source row "
                f"{int(shap_predictions.loc[
                    shap_predictions['row_index'] == row_id,
                    'source_row_index'
                ].iloc[0])}"
            ),
            key="shap_local_row",
        )

        prediction = shap_predictions.loc[
            shap_predictions["row_index"] == selected_row
        ].iloc[0]

        predicted_cluster_id = int(prediction["predicted_cluster_id"])
        predicted_cluster_label = prediction["predicted_cluster_label"]
        predicted_probability = float(
            prediction["predicted_cluster_probability"]
        )

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Predicted cluster",
                f"{predicted_cluster_id} — {predicted_cluster_label}",
            )

        with metric2:
            st.metric(
                "Probability assigned to predicted cluster",
                f"{predicted_probability:.1%}",
            )

        selected_local = shap_local[
            (shap_local["row_index"] == selected_row)
            & (shap_local["explained_class_id"] == predicted_cluster_id)
        ].copy()

        if selected_local.empty:
            st.warning(
                "No local SHAP contributions were found for this prediction."
            )
        else:
            selected_local["contribution_direction"] = (
                selected_local["shap_decision_score_contribution"]
                .apply(lambda value: "Positive" if value >= 0 else "Negative")
            )

            selected_local = selected_local.sort_values(
                "shap_decision_score_contribution"
            )

            fig_local = px.bar(
                selected_local,
                x="shap_decision_score_contribution",
                y="feature",
                orientation="h",
                color="contribution_direction",
                color_discrete_map={
                    "Positive": "#2E8B57",
                    "Negative": "#D95F59",
                },
                labels={
                    "shap_decision_score_contribution": (
                        "SHAP contribution to decision score"
                    ),
                    "feature": "Predictor",
                    "contribution_direction": "Direction",
                },
                title="Contributions to the predicted class decision score",
            )

            fig_local.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_local, use_container_width=True)

            st.markdown("""
            **How to read this chart**

            - Positive contributions push the selected class's decision score upward.
            - Negative contributions push that class's decision score downward.
            - The values are contributions to the Logistic Regression decision score,
            not direct changes in predicted probability.
            - These contributions explain the model prediction, not the respondent's
            actual financial-inclusion status or its causes.
            """)

        with st.expander("SHAP validation and limitations"):
            st.write(
                f"SHAP version: {shap_summary['shap_version']}"
            )
            st.write(
                f"Background rows: {shap_summary['background_rows']}"
            )
            st.write(
                f"Explained rows: {shap_summary['explained_rows']}"
            )
            st.write(
                f"Maximum additivity error: "
                f"{shap_summary['max_additivity_error']:.3e}"
            )
            st.write(
                f"Maximum grouped contribution error: "
                f"{shap_summary['max_grouped_sum_error']:.3e}"
            )
            st.json(shap_summary["checks_passed"])

            st.warning(
                "Global importance is currently based on a sample of 100 test "
                "respondents. It may change when SHAP is computed for the full "
                "11,301-row test partition. The classifier predicts cluster labels, "
                "not an independently observed outcome."
            )

    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.warning(f"SHAP explainability results are unavailable: {exc}")



# --------------------------------------------------
# 8. Individual financial inclusion profiling
# --------------------------------------------------

if dashboard_section == "Individual Profiling":
    st.header("Individual Financial Inclusion Profiling")

    st.markdown(
        """
        Enter the available financial inclusion indicators to calculate
        five dimension scores and assign a research-derived inclusion profile.

        This is a profile of financial inclusion indicators, not a credit
        score, loan approval, or default-risk prediction.
        """
    )

    st.info(
        "Use the indicator coding documented for this research component. "
        "Unknown or unavailable responses should be left unanswered; they "
        "must not be interpreted automatically as No."
    )

    with st.form("individual_financial_inclusion_form"):
        st.subheader("1. Connectivity")

        con1 = st.selectbox(
            "Does the respondent have a mobile phone?",
            options=["Not answered", "Yes", "No"],
            key="individual_con1",
        )

        internet_use = st.selectbox(
            "Has the respondent used the internet?",
            options=["Not answered", "Yes", "No"],
            key="individual_internet_use",
        )

        st.subheader("2. Financial access")

        account = st.selectbox(
            "Does the respondent have a financial account?",
            options=["Not answered", "Yes", "No"],
            key="individual_account",
        )

        st.subheader("3. Digital financial usage")

        anydigpayment = st.selectbox(
            "Has the respondent made a digital payment?",
            options=["Not answered", "Yes", "No"],
            key="individual_anydigpayment",
        )

        st.subheader("4. Digital safety")

        con18 = st.selectbox(
            "Does the respondent use the phone-protection indicator?",
            options=["Not answered", "Yes", "No"],
            key="individual_con18",
        )

        st.subheader("5. Financial resilience")

        saved = st.selectbox(
            "Has the respondent saved money?",
            options=["Not answered", "Yes", "No"],
            key="individual_saved",
        )

        fin24a = st.selectbox(
            "Emergency financial-capacity indicator (fin24a)",
            options=[
                "Not answered",
                "1",
                "2",
                "3",
            ],
            key="individual_fin24a",
            help=(
                "Mapped codes: 1 → 0, 2 → 0.5, 3 → 1. "
                "Confirm the applicable question wording before deployment."
            ),
        )

        fin24b = st.selectbox(
            "Emergency financial-capacity indicator (fin24b)",
            options=[
                "Not answered",
                "1",
                "2",
                "3",
                "4",
            ],
            key="individual_fin24b",
            help=(
                "Mapped codes: 1 → 0, 2 → 1/3, "
                "3 → 2/3, 4 → 1."
            ),
        )

        submitted = st.form_submit_button(
            "Calculate profile",
            type="primary",
        )

    if submitted:
        def yes_no_to_code(value):
            if value == "Yes":
                return 1
            if value == "No":
                return 2
            return None

        def binary_to_value(value):
            if value == "Yes":
                return 1
            if value == "No":
                return 0
            return None

        indicators = {
            "con1": yes_no_to_code(con1),
            "internet_use": binary_to_value(internet_use),
            "account": binary_to_value(account),
            "anydigpayment": binary_to_value(anydigpayment),
            "con18": yes_no_to_code(con18),
            "saved": binary_to_value(saved),
            "fin24a": (
                int(fin24a) if fin24a != "Not answered" else None
            ),
            "fin24b": (
                int(fin24b) if fin24b != "Not answered" else None
            ),
        }

        try:
            result = profile_individual(indicators)

            if result["status"] != "success":
                st.warning(
                    "A complete profile cannot be assigned because one "
                    "or more required dimension scores are unavailable."
                )

                st.write("Unavailable dimensions:")
                st.write(result["unavailable_dimensions"])

                st.write("Available component counts:")
                st.json(result["components_available"])

                st.caption(result["missing_data_policy"])

            else:
                st.subheader("Individual profile result")

                score_labels = {
                    "score_connectivity": "Connectivity",
                    "score_financial_access": "Financial access",
                    "score_digital_financial_usage": "Digital financial usage",
                    "score_digital_safety": "Digital safety",
                    "score_financial_resilience": "Financial resilience",
                }

                score_columns = st.columns(5)

                for column, (key, label) in zip(
                    score_columns, score_labels.items()
                ):
                    column.metric(
                        label,
                        f"{result['dimension_scores'][key]:.3f}",
                    )

                profile = result["profile"]
                explanation = result["explanation"]

                st.success(
                    f"Assigned profile: {profile['profile_name']}"
                )

                st.markdown("### Dimension-level evidence")
                evidence_df = pd.DataFrame(
                    explanation["dimension_evidence"]
                )
                st.dataframe(
                    evidence_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("### Suggested actions")
                for action in explanation["suggested_actions"]:
                    st.markdown(f"- {action}")

                st.caption(
                    "The assigned profile is based on the five calculated "
                    "dimension scores and the saved K-Means model. Suggested "
                    "actions are decision-support guidance, not validated "
                    "causal interventions or guaranteed outcomes."
                )

        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            st.error(f"Unable to calculate the individual profile: {exc}")


    st.divider()
    st.caption(
        "Component 02 | Global Findex 2025 individual microdata "
        "| Survey year: 2024"
    )
