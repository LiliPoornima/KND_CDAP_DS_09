"""Component 1: saved credit-risk model and fairness research results."""

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "credit_risk"

DATASETS = {
    "Give Me Some Credit": {
        "key": "give_me_some_credit",
        "label": "Serious delinquency within two years",
        "geography": "Geography is not identified in the supplied competition files.",
        "comparison": "completion_run/model_comparison.csv",
        "metrics_dir": "completion_run",
        "explainability_dir": "completion_run",
        "limitations": (
            "The available fields support age and dependents subgroup reviews. "
            "The dataset does not include sex, race, or ethnicity fields."
        ),
    },
    "Taiwan credit-card default": {
        "key": "credit_card_default",
        "label": "Next-month credit-card payment default",
        "geography": "Taiwan; the supplied dataset describes Taiwan cardholders.",
        "comparison": "completion_run/metrics/model_comparison.csv",
        "metrics_dir": "completion_run/metrics",
        "explainability_dir": "completion_run/explainability",
        "limitations": (
            "The audit uses the dataset's sex, age, education, and marital-status "
            "fields. These historical sample results do not represent all borrowers."
        ),
    },
    "Home Credit": {
        "key": "home_credit",
        "label": "Payment difficulty on early installments of a sampled loan",
        "geography": "Geography is not established by the supplied competition files.",
        "comparison": "completion_run/metrics/model_comparison.csv",
        "metrics_dir": "completion_run/metrics",
        "explainability_dir": "completion_run/explainability",
        "limitations": (
            "The audit uses gender and age groups. The four XNA gender records "
            "are too few for a reliable standalone subgroup conclusion."
        ),
    },
}


@st.cache_data(show_spinner=False)
def _load_csv(path_string: str) -> pd.DataFrame | None:
    """Load small, saved research summaries without loading row-level splits."""

    path = Path(path_string)
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_json(
    path_string: str,
    modified_ns: int = 0,
    size_bytes: int = 0,
) -> dict | None:
    """Load a small dataset run manifest."""
    del modified_ns, size_bytes  # Included in the cache key for file refreshes.
    path = Path(path_string)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_text(
    path_string: str,
    modified_ns: int = 0,
    size_bytes: int = 0,
) -> str | None:
    """Load a small research report and refresh it when the file changes."""
    del modified_ns, size_bytes  # Included in the cache key for file refreshes.
    path = Path(path_string)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _dataset_path(dataset: dict[str, str], relative_path: str) -> Path:
    return OUTPUT_ROOT / dataset["key"] / relative_path


def _show_missing(path: Path) -> None:
    st.info(f"No saved result is available at `{path.relative_to(PROJECT_ROOT)}` yet.")


def _fairness_specs(dataset: dict[str, str]) -> list[dict[str, str]]:
    metrics_dir = _dataset_path(dataset, dataset["metrics_dir"])
    specs = []
    if not metrics_dir.is_dir():
        return specs

    for path in sorted(metrics_dir.glob("*_fairness.csv")):
        stem = path.stem.removesuffix("_fairness")
        if stem.startswith("logistic_regression_"):
            model_key = "logistic_regression"
            model_name = "Logistic Regression"
            attribute = stem.removeprefix("logistic_regression_")
        elif stem.startswith("random_forest_"):
            model_key = "random_forest"
            model_name = "Random Forest"
            attribute = stem.removeprefix("random_forest_")
        else:
            continue
        specs.append(
            {
                "path": str(path),
                "model_key": model_key,
                "model_name": model_name,
                "attribute": attribute,
            }
        )
    return specs


def _render_performance(dataset: dict[str, str]) -> None:
    comparison_path = _dataset_path(dataset, dataset["comparison"])
    comparison = _load_csv(str(comparison_path))
    if comparison is None or comparison.empty:
        _show_missing(comparison_path)
        return

    metric_columns = [
        "Test F1-score",
        "Test ROC-AUC",
        "Test PR-AUC",
        "Test Brier score",
    ]
    available_columns = ["Model", "Validation-selected threshold"] + [
        column for column in metric_columns if column in comparison.columns
    ]
    if not set(available_columns).issubset(comparison.columns):
        st.error("The saved model comparison file does not have the expected columns.")
        return

    st.dataframe(
        comparison[available_columns].style.format(
            {
                column: "{:.3f}"
                for column in available_columns
                if column != "Model"
            }
        ),
        hide_index=True,
        width="stretch",
    )

    interval_tables = []
    for model_name in comparison["Model"]:
        model_key = "random_forest" if model_name == "Random Forest" else "logistic_regression"
        interval_path = _dataset_path(
            dataset,
            f"{dataset['metrics_dir']}/{model_key}_performance_confidence_intervals.csv",
        )
        interval_table = _load_csv(str(interval_path))
        if interval_table is not None and not interval_table.empty:
            interval_table.insert(0, "Model", model_name)
            interval_tables.append(interval_table)
    if interval_tables:
        intervals = pd.concat(interval_tables, ignore_index=True)
        st.markdown("**95% stratified bootstrap intervals for held-out metrics**")
        st.dataframe(
            intervals[["Model", "metric", "estimate", "lower", "upper", "n"]].style.format(
                {"estimate": "{:.3f}", "lower": "{:.3f}", "upper": "{:.3f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Intervals resample test rows within each observed outcome class and "
            "condition on the fitted model and frozen validation threshold."
        )
    else:
        st.info("Held-out performance confidence intervals have not been generated yet.")

    paired_path = _dataset_path(
        dataset,
        f"{dataset['metrics_dir']}/paired_model_comparison_confidence_intervals.csv",
    )
    paired_intervals = _load_csv(str(paired_path))
    if paired_intervals is not None and not paired_intervals.empty:
        st.markdown("**Paired 95% intervals for model differences**")
        st.dataframe(
            paired_intervals[
                [
                    "metric",
                    "model_a",
                    "model_b",
                    "difference_b_minus_a",
                    "lower",
                    "upper",
                ]
            ].style.format(
                {
                    "difference_b_minus_a": "{:+.3f}",
                    "lower": "{:+.3f}",
                    "upper": "{:+.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Differences are Random Forest minus Logistic Regression. Each paired "
            "bootstrap replicate uses the same resampled test rows for both models, "
            "stratified by outcome. Thresholds remain frozen at their validation "
            "choices; intervals are conditional on the fitted models and have no "
            "multiple-comparison adjustment."
        )
    else:
        st.info("Paired model-comparison intervals have not been generated yet.")

    if "Test F1-score" in comparison.columns:
        chart_data = comparison.melt(
            id_vars="Model",
            value_vars=[
                column
                for column in ("Test F1-score", "Test ROC-AUC", "Test PR-AUC")
                if column in comparison.columns
            ],
            var_name="Metric",
            value_name="Score",
        )
        figure = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Held-out test performance",
            range_y=[0, 1],
            text_auto=".3f",
        )
        figure.update_layout(legend_title_text="Model", yaxis_title="Score")
        st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "Thresholds were selected on validation data and then applied to the "
        "held-out test split. Brier score is a probability-calibration measure; "
        "lower is better. Dataset results are not pooled because their targets "
        "and populations differ."
    )


def _render_calibration(dataset: dict[str, str], model_name: str) -> None:
    model_key = "random_forest" if model_name == "Random Forest" else "logistic_regression"
    path = _dataset_path(
        dataset,
        f"{dataset['metrics_dir']}/{model_key}_calibration_bins.csv",
    )
    calibration = _load_csv(str(path))
    if calibration is None or calibration.empty:
        _show_missing(path)
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash", "color": "#7B8794"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=calibration["mean_predicted_probability"],
            y=calibration["observed_rate"],
            mode="lines+markers",
            name=model_name,
            text=calibration["count"].map(lambda value: f"n={value:,}"),
            hovertemplate=(
                "Predicted: %{x:.3f}<br>Observed: %{y:.3f}<br>%{text}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{model_name} reliability by probability bin",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed event rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend_title_text="",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(calibration, hide_index=True, width="stretch")
    _render_group_calibration(dataset, model_name)


def _render_group_calibration(
    dataset: dict[str, str],
    model_name: str,
    *,
    ablation: bool = False,
) -> None:
    metrics_name = (
        "feature_ablation_subgroup_calibration.csv"
        if ablation
        else "subgroup_calibration_metrics.csv"
    )
    bins_name = (
        "feature_ablation_subgroup_calibration_bins.csv"
        if ablation
        else "subgroup_calibration_bins.csv"
    )
    metrics_path = _dataset_path(
        dataset, f"{dataset['metrics_dir']}/{metrics_name}"
    )
    bins_path = _dataset_path(dataset, f"{dataset['metrics_dir']}/{bins_name}")
    metrics = _load_csv(str(metrics_path))
    bins = _load_csv(str(bins_path))
    if metrics is None or metrics.empty or bins is None or bins.empty:
        _show_missing(metrics_path)
        return

    st.subheader(
        "Subgroup calibration across feature scenarios"
        if ablation
        else "Calibration by observed subgroup"
    )
    selected = metrics.copy()
    if "Model" in selected:
        selected = selected[selected["Model"] == model_name]
    if ablation and "Scenario" in selected:
        scenarios = selected["Scenario"].dropna().unique().tolist()
        scenario = st.selectbox(
            "Feature scenario",
            scenarios,
            key=f"calibration_scenario_{dataset['key']}",
        )
        selected = selected[selected["Scenario"] == scenario]
    attribute = st.selectbox(
        "Audit attribute",
        sorted(selected["attribute"].dropna().unique()),
        key=f"calibration_attribute_{dataset['key']}_{'ablation' if ablation else 'baseline'}",
    )
    selected = selected[selected["attribute"] == attribute]
    group = st.selectbox(
        "Group",
        selected["group"].astype(str).tolist(),
        key=f"calibration_group_{dataset['key']}_{'ablation' if ablation else 'baseline'}",
    )
    row = selected[selected["group"].astype(str) == group].iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Brier score", f"{row['brier_score']:.3f}")
    metric_columns[1].metric("Log loss", f"{row['log_loss']:.3f}")
    metric_columns[2].metric(
        "ECE (10 equal-width bins)",
        f"{row['expected_calibration_error']:.3f}",
    )
    metric_columns[3].metric(
        "Calibration in the large",
        f"{row['calibration_in_the_large']:+.3f}",
        help="Mean predicted probability minus observed outcome rate.",
    )
    st.caption(
        f"n={int(row['n']):,}; events={int(row['positive_support']):,}; "
        f"non-events={int(row['negative_support']):,}. "
        f"Meets the project calibration reporting support rule: "
        f"{bool(row['calibration_reportable'])}."
    )
    if not ablation:
        interval_path = _dataset_path(
            dataset,
            f"{dataset['metrics_dir']}/subgroup_calibration_confidence_intervals.csv",
        )
        intervals = _load_csv(str(interval_path))
        if intervals is not None and not intervals.empty:
            selected_intervals = intervals[
                (intervals["Model"] == model_name)
                & (intervals["attribute"] == attribute)
                & (intervals["group"].astype(str) == group)
            ]
            if not selected_intervals.empty and selected_intervals[
                "calibration_reportable"
            ].all():
                st.markdown("**95% within-group calibration intervals**")
                st.dataframe(
                    selected_intervals[
                        ["metric", "estimate", "lower", "upper"]
                    ].style.format(
                        {
                            "estimate": "{:.3f}",
                            "lower": "{:.3f}",
                            "upper": "{:.3f}",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Intervals use 1,000 bootstrap replicates sampled within this "
                    "group and outcome class. They condition on the fitted model, "
                    "predicted probabilities, and observed within-group outcome "
                    "counts; they do not include model-refit or population-shift "
                    "uncertainty."
                )
            else:
                st.info(
                    "This group does not meet the project support rule for a "
                    "subgroup calibration interval."
                )
        else:
            st.info("Subgroup calibration intervals have not been generated yet.")

    selected_bins = bins[
        (bins["attribute"] == attribute)
        & (bins["group"].astype(str) == group)
    ]
    if "Model" in selected_bins:
        selected_bins = selected_bins[selected_bins["Model"] == model_name]
    if ablation:
        selected_bins = selected_bins[selected_bins["Scenario"] == scenario]
    selected_bins = selected_bins[selected_bins["count"] > 0]
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash", "color": "#7B8794"},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=selected_bins["mean_predicted_probability"],
            y=selected_bins["observed_rate"],
            mode="lines+markers",
            name=f"{attribute}: {group}",
            text=selected_bins["count"].map(lambda value: f"n={value:,}"),
            hovertemplate=(
                "Predicted: %{x:.3f}<br>Observed: %{y:.3f}<br>"
                "%{text}<extra></extra>"
            ),
        )
    )
    chart.update_layout(
        title="Subgroup reliability curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed event rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        showlegend=False,
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(selected_bins, hide_index=True, width="stretch")


def _render_fairness(dataset: dict[str, str]) -> None:
    specs = _fairness_specs(dataset)
    if not specs:
        st.info("No subgroup fairness tables are available for this dataset.")
        return

    model_options = sorted({spec["model_name"] for spec in specs})
    selected_model = st.selectbox("Model", model_options, key="credit_fairness_model")
    model_specs = [spec for spec in specs if spec["model_name"] == selected_model]
    attributes = sorted({spec["attribute"] for spec in model_specs})
    attribute = st.selectbox(
        "Audit attribute",
        attributes,
        format_func=lambda value: value.replace("_", " ").title(),
        key="credit_fairness_attribute",
    )
    selected = next(spec for spec in model_specs if spec["attribute"] == attribute)
    fairness_path = Path(selected["path"])
    fairness = _load_csv(str(fairness_path))
    if fairness is None:
        _show_missing(fairness_path)
        return

    st.caption(dataset["limitations"])
    display_columns = [
        column
        for column in (
            "group",
            "n",
            "positive_support",
            "negative_support",
            "outcome_rate",
            "selection_rate",
            "true_positive_rate",
            "false_positive_rate",
            "precision",
            "sample_sufficient",
            "positive_support_sufficient",
            "negative_support_sufficient",
        )
        if column in fairness.columns
    ]
    st.dataframe(
        fairness[display_columns].style.format(
            {
                column: "{:.3f}"
                for column in display_columns
                if column.endswith("_rate") or column == "precision"
            }
        ),
        hide_index=True,
        width="stretch",
    )
    if "sample_sufficient" in fairness and not fairness["sample_sufficient"].all():
        st.warning(
            "At least one subgroup is below the project's support threshold; "
            "interpret its rates cautiously."
        )

    model_key = selected["model_key"]
    group_ci_path = fairness_path.with_name(
        f"{model_key}_{attribute}_group_confidence_intervals.csv"
    )
    gap_ci_path = fairness_path.with_name(
        f"{model_key}_{attribute}_gap_confidence_intervals.csv"
    )
    group_ci = _load_csv(str(group_ci_path))
    gap_ci = _load_csv(str(gap_ci_path))
    if group_ci is not None:
        st.markdown("**95% bootstrap intervals for group rates**")
        st.dataframe(group_ci, hide_index=True, width="stretch")
    if gap_ci is not None:
        st.markdown("**95% bootstrap intervals for between-group gaps**")
        st.dataframe(gap_ci, hide_index=True, width="stretch")
    st.caption(
        "Group-level rates are eligible when the group has at least 30 records; "
        "TPR and FPR additionally require at least 5 positive or negative cases. "
        "Fairness intervals condition on the fitted model and observed group sizes."
    )


def _render_explanations(dataset: dict[str, str]) -> None:
    global_path = _dataset_path(
        dataset,
        f"{dataset['explainability_dir']}/random_forest_global_shap_importance.csv",
    )
    if dataset["key"] == "give_me_some_credit":
        global_path = _dataset_path(
            dataset,
            "completion_run/global_shap_importance.csv",
        )
    importance = _load_csv(str(global_path))
    if importance is None or importance.empty:
        _show_missing(global_path)
    else:
        feature_column = "feature"
        value_column = "mean_absolute_shap"
        if not {feature_column, value_column}.issubset(importance.columns):
            st.warning("The saved global explanation table has an unexpected format.")
        else:
            top_features = importance.nlargest(20, value_column).sort_values(value_column)
            figure = px.bar(
                top_features,
                x=value_column,
                y=feature_column,
                orientation="h",
                title="Top global Random Forest features",
                labels={
                    value_column: "Mean absolute SHAP value",
                    feature_column: "Feature",
                },
            )
            st.plotly_chart(figure, use_container_width=True)

    stability_path = _dataset_path(
        dataset,
        f"{dataset['explainability_dir']}/explanation_stability_pairwise.csv",
    )
    if dataset["key"] == "give_me_some_credit":
        stability_path = _dataset_path(
            dataset,
            "completion_run/explanation_stability_pairwise.csv",
        )
    stability = _load_csv(str(stability_path))
    if stability is None or stability.empty:
        _show_missing(stability_path)
    else:
        st.markdown("**Repeated-fit explanation stability**")
        stability_metrics = {
            "spearman_rank_correlation": "Mean feature-rank correlation",
            "top_k_jaccard_overlap": "Mean top-10 overlap",
            "attribution_cosine_similarity": "Mean attribution cosine similarity",
        }
        metric_columns = st.columns(len(stability_metrics))
        for column, (metric, label) in zip(metric_columns, stability_metrics.items()):
            if metric in stability.columns:
                column.metric(label, f"{stability[metric].mean():.3f}")
        st.dataframe(stability, hide_index=True, width="stretch")
    resampling_dir = (
        "completion_run"
        if dataset["key"] == "give_me_some_credit"
        else dataset["explainability_dir"]
    )
    resampling_path = _dataset_path(
        dataset,
        f"{resampling_dir}/training_subsample_stability_summary.csv",
    )
    resampling = _load_csv(str(resampling_path))
    if resampling is not None and not resampling.empty:
        st.markdown("**Stability across training-row resamples**")
        row = resampling.iloc[0]
        stability_metrics = {
            "mean_spearman_rank_correlation": "Mean feature-rank correlation",
            "mean_top_k_jaccard_overlap": "Mean top-10 overlap",
            "mean_attribution_cosine_similarity": "Mean attribution cosine similarity",
        }
        metric_columns = st.columns(len(stability_metrics))
        for column, (metric, label) in zip(metric_columns, stability_metrics.items()):
            if metric in resampling.columns:
                column.metric(label, f"{row[metric]:.3f}")
        st.caption(
            f"Compared {int(row['runs_compared'])} fits trained on stratified "
            f"{row['training_fraction']:.0%} samples of the training partition. "
            f"All runs used the same {int(row['evaluation_rows'])} held-out rows "
            f"and {int(row['background_rows'])} training rows as the SHAP background."
        )
        pairwise_path = _dataset_path(
            dataset,
            f"{resampling_dir}/training_subsample_stability_pairwise.csv",
        )
        pairwise = _load_csv(str(pairwise_path))
        if pairwise is not None:
            st.dataframe(pairwise, hide_index=True, width="stretch")
    st.caption(
        "SHAP values summarize model associations for these fitted models; they "
        "do not establish causal effects."
    )

    if dataset["key"] == "give_me_some_credit":
        local_path = _dataset_path(dataset, "completion_run/shap_attributions_sample.csv")
    else:
        local_path = _dataset_path(
            dataset,
            f"{dataset['explainability_dir']}/random_forest_shap_attributions_sample.csv",
        )
    local = _load_csv(str(local_path))
    st.markdown("**Local Random Forest example**")
    if local is None or local.empty:
        _show_missing(local_path)
        return

    # The first column is the saved test-row index; omit it from the UI.
    feature_columns = [
        column for column in local.columns if column != "Unnamed: 0"
    ]
    sample_number = st.selectbox(
        "Example",
        range(len(local)),
        format_func=lambda value: f"Example {value + 1}",
        key=f"credit_local_shap_{dataset['key']}",
    )
    contributions = pd.to_numeric(local.iloc[sample_number][feature_columns], errors="coerce")
    contributions = contributions.dropna()
    contributions = contributions.reindex(
        contributions.abs().nlargest(min(12, len(contributions))).index
    ).sort_values()
    local_figure = px.bar(
        x=contributions.values,
        y=contributions.index,
        orientation="h",
        title="Largest signed SHAP contributions for this example",
        labels={"x": "SHAP contribution", "y": "Feature"},
    )
    st.plotly_chart(local_figure, use_container_width=True)
    st.caption(
        "Positive contributions move the model output above its SHAP baseline; "
        "negative contributions move it below. This is a model explanation, not "
        "a causal effect or a credit recommendation."
    )


def _render_ablation(dataset: dict[str, str]) -> None:
    path = _dataset_path(
        dataset,
        f"{dataset['metrics_dir']}/feature_ablation.csv",
    )
    if dataset["key"] == "give_me_some_credit":
        path = _dataset_path(dataset, "completion_run/feature_ablation.csv")
    results = _load_csv(str(path))
    if results is None or results.empty:
        _show_missing(path)
        return

    st.dataframe(results, hide_index=True, width="stretch")
    paired_path = path.with_name(
        "feature_ablation_paired_comparison_confidence_intervals.csv"
    )
    paired_results = _load_csv(str(paired_path))
    if paired_results is not None and not paired_results.empty:
        st.subheader("Paired uncertainty for feature ablations")
        visible_paired = paired_results[
            [
                "comparison",
                "metric",
                "difference_b_minus_a",
                "lower",
                "upper",
            ]
        ]
        st.dataframe(
            visible_paired.style.format(
                {
                    "difference_b_minus_a": "{:+.3f}",
                    "lower": "{:+.3f}",
                    "upper": "{:+.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Differences are each reduced-feature scenario minus the full-feature "
            "model. Paired outcome-stratified bootstrap replicates use the same "
            "test rows for both scenarios and keep validation-selected thresholds "
            "fixed. Intervals are conditional on the fitted models and have no "
            "multiple-comparison adjustment."
        )
    else:
        st.info(
            "Paired feature-ablation intervals have not been generated yet. "
            "Run the saved-split ablation uncertainty review to create them."
        )

    score_column = "Test F1-score" if "Test F1-score" in results else "F1-score"
    if score_column in results and "Scenario" in results:
        figure = px.bar(
            results,
            x="Scenario",
            y=score_column,
            color="Model" if "Model" in results else None,
            title="Held-out test F1 by feature scenario",
            range_y=[0, 1],
            text_auto=".3f",
        )
        figure.update_layout(xaxis_title="Feature scenario", yaxis_title="Test F1")
        st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Thresholds are selected using validation data. These controlled feature "
        "removals are descriptive comparisons, not causal estimates."
    )

    subgroup_path = path.with_name("feature_ablation_subgroup_metrics.csv")
    subgroup_results = _load_csv(str(subgroup_path))
    if subgroup_results is None or subgroup_results.empty:
        _show_missing(subgroup_path)
        return
    st.subheader("Subgroup outcomes across feature scenarios")
    attributes = sorted(subgroup_results["attribute"].dropna().unique())
    selected_attribute = st.selectbox(
        "Audit attribute",
        attributes,
        key=f"ablation_subgroup_attribute_{dataset['key']}",
    )
    visible = subgroup_results[
        subgroup_results["attribute"] == selected_attribute
    ].copy()
    st.dataframe(
        visible[
            [
                "Scenario",
                "group",
                "n",
                "positive_support",
                "negative_support",
                "selection_rate",
                "true_positive_rate",
                "false_positive_rate",
                "precision",
                "sample_sufficient",
                "positive_support_sufficient",
                "negative_support_sufficient",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Rates use held-out test rows and each scenario's validation-selected "
        "threshold. Small groups remain visible; interpret TPR/FPR only when "
        "their class-support flags are true. Gap columns are saved with the CSV."
    )
    fairness_interval_path = path.with_name(
        "feature_ablation_paired_fairness_gap_confidence_intervals.csv"
    )
    fairness_intervals = _load_csv(str(fairness_interval_path))
    if fairness_intervals is not None and not fairness_intervals.empty:
        selected_fairness_intervals = fairness_intervals[
            fairness_intervals["attribute"] == selected_attribute
        ]
        if not selected_fairness_intervals.empty:
            st.subheader("Paired uncertainty in fairness-gap changes")
            fairness_columns = [
                "comparison",
                "metric",
                "estimate_a",
                "estimate_b",
                "difference_b_minus_a",
                "lower",
                "upper",
            ]
            st.dataframe(
                selected_fairness_intervals[fairness_columns].style.format(
                    {
                        "estimate_a": "{:.3f}",
                        "estimate_b": "{:.3f}",
                        "difference_b_minus_a": "{:+.3f}",
                        "lower": "{:+.3f}",
                        "upper": "{:+.3f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Each difference is the ablation fairness gap minus the full-model "
                "gap. The same test rows are resampled within group and outcome "
                "class for both models. Positive values indicate a wider gap; "
                "negative values indicate a narrower gap. Intervals condition on "
                "the fitted models, fixed validation thresholds, and observed "
                "group-by-outcome counts; no multiple-comparison adjustment is applied."
            )
    else:
        st.info(
            "Paired fairness-gap intervals have not been generated yet. Run the "
            "saved-split ablation uncertainty review to create them."
        )
    _render_group_calibration(dataset, "Random Forest", ablation=True)


st.title("Credit Risk")
st.caption(
    "Component 1 research results: independent public datasets, validation-selected "
    "thresholds, held-out evaluation, subgroup fairness, calibration, and explanations."
)

dataset_name = st.selectbox("Dataset", list(DATASETS), key="credit_risk_dataset")
dataset = DATASETS[dataset_name]
comparison_path = _dataset_path(dataset, dataset["comparison"])
comparison = _load_csv(str(comparison_path))

st.markdown(f"**Target:** {dataset['label']}  \n**Coverage:** {dataset['geography']}")
st.info(
    "The datasets have different populations and target definitions. Their scores "
    "are reported separately and should not be read as a global ranking of borrower risk."
)

manifest_path = _dataset_path(dataset, "completion_run/run_manifest.json")
manifest_signature = (
    manifest_path.stat() if manifest_path.is_file() else None
)
manifest = _load_json(
    str(manifest_path),
    manifest_signature.st_mtime_ns if manifest_signature else 0,
    manifest_signature.st_size if manifest_signature else 0,
)
with st.expander("Run reproducibility details"):
    if manifest is None:
        st.info(f"No run manifest is available at `{manifest_path.relative_to(PROJECT_ROOT)}` yet.")
    else:
        source = manifest["dataset"]["source_archive"]
        implementation = manifest["implementation"]
        st.caption(
            f"Generated {manifest['generated_at_utc']} · revision "
            f"{implementation.get('commit') or 'unknown'} · working tree "
            f"{'dirty' if implementation.get('working_tree_dirty') else 'clean'}"
        )
        st.markdown("**Source archive**")
        st.write(f"{source['filename']} · SHA-256 `{source['sha256']}`")
        st.dataframe(
            pd.DataFrame(source["members_used"]),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Saved split artifacts**")
        split_rows = []
        for name, artifact in manifest["preparation"]["saved_splits"].items():
            split_rows.append(
                {
                    "partition": name,
                    "rows": artifact["rows"],
                    "target_counts": artifact.get("target_counts", "audit groups"),
                    "sha256": artifact["sha256"],
                }
            )
        st.dataframe(pd.DataFrame(split_rows), hide_index=True, width="stretch")
        with st.popover("Model and software settings"):
            st.json(
                {
                    "models": manifest["experiments"]["models"],
                    "threshold_selection": manifest["experiments"]["threshold_selection"],
                    "performance_uncertainty": manifest["experiments"]["performance_uncertainty"],
                    "fairness_uncertainty": manifest["experiments"]["fairness_uncertainty"],
                    "software": manifest["software"],
                },
                expanded=False,
            )

if comparison is not None and not comparison.empty:
    best_row = comparison.sort_values("Test F1-score", ascending=False).iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Top test F1", f"{best_row['Test F1-score']:.3f}")
    metric_columns[1].metric("Test ROC-AUC", f"{best_row['Test ROC-AUC']:.3f}")
    metric_columns[2].metric("Test PR-AUC", f"{best_row['Test PR-AUC']:.3f}")
    metric_columns[3].metric(
        "Validation threshold",
        f"{best_row['Validation-selected threshold']:.2f}",
        help=f"Selected for {best_row['Model']} on the validation split.",
    )

performance_tab, fairness_tab, explanation_tab, ablation_tab, report_tab = st.tabs(
    [
        "Performance & calibration",
        "Fairness",
        "Explainability",
        "Feature ablation",
        "Research report",
    ]
)

with performance_tab:
    st.subheader("Model comparison")
    _render_performance(dataset)
    model_names = (
        comparison["Model"].tolist()
        if comparison is not None and "Model" in comparison
        else ["Random Forest"]
    )
    calibration_model = st.selectbox(
        "Calibration model",
        model_names,
        key="credit_calibration_model",
    )
    _render_calibration(dataset, calibration_model)

with fairness_tab:
    st.subheader("Subgroup performance")
    _render_fairness(dataset)

with explanation_tab:
    st.subheader("Global feature attribution and stability")
    _render_explanations(dataset)

with ablation_tab:
    st.subheader("Controlled feature-group ablation")
    _render_ablation(dataset)

with report_tab:
    st.subheader("Component 1 final research report")
    report_path = PROJECT_ROOT / "docs" / "research" / "credit_risk_final_report.md"
    report_signature = report_path.stat() if report_path.is_file() else None
    report = _load_text(
        str(report_path),
        report_signature.st_mtime_ns if report_signature else 0,
        report_signature.st_size if report_signature else 0,
    )
    if report is None:
        _show_missing(report_path)
    else:
        st.markdown(report)

st.divider()
st.caption(
    "Research outputs only. These historical public datasets and baseline models "
    "are not lending recommendations or validated production decision systems."
)
