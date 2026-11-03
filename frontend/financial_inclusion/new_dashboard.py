"""Independent professional dashboard for Component 02 comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import data_loader as loader
from .constants import CLUSTER_IDS, CLUSTER_LABELS
from .sections import (
    distribution,
    gaps,
    individual_assessment,
    overview,
    subgroups,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "financial_inclusion"
DATA_REPORTS = OUTPUTS / "reports" / "01_data_preparation"
DIMENSION_REPORTS = OUTPUTS / "reports" / "02_dimension_creation"
CLUSTER_REPORTS = OUTPUTS / "reports" / "03_clustering"
SUPERVISED_REPORTS = OUTPUTS / "reports" / "04_supervised_learning"
SHAP_REPORTS = OUTPUTS / "reports" / "05_shap_explainability"
TUNED_EXPERIMENT = OUTPUTS / "experiments" / "supervised_tuning"
HELDOUT_EXPERIMENT = OUTPUTS / "experiments" / "kmeans_heldout_validation"

DIMENSION_ICONS = {
    "score_connectivity": "◉",
    "score_financial_access": "▣",
    "score_digital_financial_usage": "↗",
    "score_digital_safety": "◇",
    "score_financial_resilience": "⌂",
}
FEATURE_LABELS = {
    "female": "Gender",
    "age": "Age",
    "educ": "Education",
    "inc_q": "Income group",
    "emp_in": "Employment",
    "urbanicity": "Urbanicity",
    "regionwb": "World Bank region",
}


@st.cache_data(show_spinner="Loading validated financial inclusion data…")
def _load_dataset() -> pd.DataFrame:
    return loader.load_final_dataset()


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _missing_artifact(path: Path) -> None:
    st.info(f"Saved result is not available: {path.relative_to(ROOT)}")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .st-key-new_fi_dark_dashboard {
          --fi-bg: #0e1117; --fi-card: #171b24; --fi-card-alt: #202631;
          --fi-text: #ffffff; --fi-muted: #b8c0cc; --fi-border: #343b46;
          --fi-accent: #14b8a6;
          background: var(--fi-bg); color: var(--fi-text); padding: 1rem;
          border: 1px solid #252b35; border-radius: 18px;
        }
        .st-key-new_fi_dark_dashboard,
        .st-key-new_fi_dark_dashboard [data-testid="stMarkdownContainer"],
        .st-key-new_fi_dark_dashboard [data-testid="stWidgetLabel"] p,
        .st-key-new_fi_dark_dashboard h1,
        .st-key-new_fi_dark_dashboard h2,
        .st-key-new_fi_dark_dashboard h3,
        .st-key-new_fi_dark_dashboard h4,
        .st-key-new_fi_dark_dashboard p,
        .st-key-new_fi_dark_dashboard li,
        .st-key-new_fi_dark_dashboard label { color: var(--fi-text); }
        .st-key-new_fi_dark_dashboard [data-testid="stCaptionContainer"],
        .st-key-new_fi_dark_dashboard [data-testid="stCaptionContainer"] p { color: var(--fi-muted); }
        .st-key-new_fi_dark_dashboard .fi-hero {
          padding: 1.45rem 1.65rem; border-radius: 16px; margin: .15rem 0 1.15rem;
          background: linear-gradient(115deg, #102a43 0%, #164e63 58%, #0f766e 100%);
          color: #f8fafc; box-shadow: 0 14px 36px rgba(0, 0, 0, .24);
        }
        .st-key-new_fi_dark_dashboard .fi-eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem;
          color: #a7f3d0; font-weight: 700; margin-bottom: .45rem; }
        .st-key-new_fi_dark_dashboard .fi-hero h1 { color: white; margin: 0; font-size: 2rem; letter-spacing: -.025em; }
        .st-key-new_fi_dark_dashboard .fi-hero p { color: #dbeafe; max-width: 850px; margin: .45rem 0 0; }
        .st-key-new_fi_dark_dashboard [data-testid="stMetric"],
        .st-key-new_fi_dark_dashboard [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-new_fi_dark_dashboard [data-testid="stExpander"] details {
          background: var(--fi-card) !important; color: var(--fi-text) !important;
          border: 1px solid var(--fi-border) !important; border-radius: 14px;
        }
        .st-key-new_fi_dark_dashboard [data-testid="stMetric"] { padding: .8rem 1rem; }
        .st-key-new_fi_dark_dashboard [data-testid="stMetricLabel"],
        .st-key-new_fi_dark_dashboard [data-testid="stMetricLabel"] p { color: var(--fi-muted) !important; }
        .st-key-new_fi_dark_dashboard [data-testid="stMetricValue"],
        .st-key-new_fi_dark_dashboard [data-testid="stMetricValue"] div,
        .st-key-new_fi_dark_dashboard [data-testid="stMetricDelta"] { color: var(--fi-text) !important; }
        .st-key-new_fi_dark_dashboard [data-testid="stAlert"] {
          background: var(--fi-card-alt) !important; border-color: var(--fi-border) !important;
          color: var(--fi-text) !important;
        }
        .st-key-new_fi_dark_dashboard [data-testid="stDataFrame"],
        .st-key-new_fi_dark_dashboard [data-testid="stTable"] {
          background: var(--fi-card); border: 1px solid var(--fi-border); border-radius: 10px;
        }
        .st-key-new_fi_dark_dashboard [data-baseweb="select"] > div {
          background: var(--fi-card-alt); border-color: var(--fi-border); color: var(--fi-text);
        }
        .st-key-new_fi_dark_dashboard input[type="radio"] { accent-color: var(--fi-accent); }
        .st-key-new_fi_dark_dashboard div[data-testid="stRadio"] > div { gap: .45rem; }
        .st-key-new_fi_dark_dashboard div[data-testid="stRadio"] label {
          background: var(--fi-card-alt); padding: .45rem .8rem; border-radius: 999px;
          border: 1px solid var(--fi-border);
        }
        .st-key-new_fi_dark_dashboard div[data-testid="stRadio"] label:has(input:checked) {
          background: #123b3b; border-color: var(--fi-accent); color: var(--fi-text);
        }
        .st-key-new_fi_dark_dashboard [data-testid="stPlotlyChart"] { background: var(--fi-card); border-radius: 12px; }
        .st-key-new_fi_dark_dashboard .js-plotly-plot .bg { fill: var(--fi-card) !important; stroke: var(--fi-border) !important; }
        .st-key-new_fi_dark_dashboard .js-plotly-plot .main-svg { background: transparent !important; }
        .st-key-new_fi_dark_dashboard .js-plotly-plot text { fill: var(--fi-muted) !important; }
        .st-key-new_fi_dark_dashboard .js-plotly-plot .gtitle { fill: var(--fi-text) !important; }
        @media (max-width: 760px) {
          .st-key-new_fi_dark_dashboard { padding: .55rem; border-radius: 12px; }
          .st-key-new_fi_dark_dashboard .fi-hero { padding: 1rem; border-radius: 12px; }
          .st-key-new_fi_dark_dashboard .fi-hero h1 { font-size: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(view: str) -> None:
    tagline = (
        "A clear view of inclusion patterns, profiles, and practical areas for follow-up."
        if view == "Stakeholder Dashboard"
        else "A research workspace for data quality, profile validation, model evaluation, and explanation."
    )
    st.markdown(
        f"<div class='fi-hero'><div class='fi-eyebrow'>Component 02 · Global Findex 2025</div>"
        f"<h1>{view}</h1><p>{tagline}</p></div>",
        unsafe_allow_html=True,
    )


def _stakeholder_dimensions(df: pd.DataFrame) -> None:
    st.subheader("Five dimensions at a glance")
    dimension_means = loader.calculate_dimension_summary(df)
    cards = st.columns(5, gap="small")
    for card, row in zip(cards, dimension_means.itertuples(index=False)):
        label = loader.DIMENSION_LABELS[row.variable]
        with card:
            st.metric(label, f"{row.mean:.2f}" if pd.notna(row.mean) else "—")
            st.caption(f"Mean score · {row.missing_percent:.1f}% missing")

    profiles = loader.calculate_cluster_profiles(df)
    selected_id = st.selectbox(
        "Explore a saved profile", CLUSTER_IDS,
        format_func=lambda value: CLUSTER_LABELS[value], key="new_fi_profile",
    )
    selected = profiles.loc[profiles.cluster_id == selected_id].iloc[0]
    left, right = st.columns([1.05, 1.5], gap="large")
    with left:
        st.markdown(f"### {CLUSTER_LABELS[selected_id]}")
        st.caption(
            f"{int(selected.respondents):,} respondents · "
            f"{selected.percent_of_candidate:.1f}% of the analytical sample"
        )
        radar = go.Figure(go.Scatterpolar(
            r=[float(selected[f"{column}_mean"]) for column in loader.DIMENSION_COLUMNS],
            theta=[loader.DIMENSION_LABELS[column] for column in loader.DIMENSION_COLUMNS],
            fill="toself", line={"color": "#0f766e", "width": 3},
            fillcolor="rgba(15,118,110,.16)", name="Mean dimension score",
        ))
        radar.update_layout(
            paper_bgcolor="#171b24",
            plot_bgcolor="#171b24",
            polar={
                "bgcolor": "#171b24",
                "radialaxis": {
                    "visible": True, "range": [0, 1],
                    "tickfont": {"size": 10, "color": "#b8c0cc"},
                    "gridcolor": "#343b46", "linecolor": "#343b46",
                },
                "angularaxis": {
                    "tickfont": {"color": "#b8c0cc"},
                    "gridcolor": "#343b46", "linecolor": "#343b46",
                },
            },
            showlegend=False, height=360, margin={"l": 35, "r": 35, "t": 25, "b": 25},
        )
        st.plotly_chart(radar, use_container_width=True, config={"responsive": True})
    with right:
        values = pd.DataFrame({
            "Dimension": [loader.DIMENSION_LABELS[c] for c in loader.DIMENSION_COLUMNS],
            "Mean score": [float(selected[f"{c}_mean"]) for c in loader.DIMENSION_COLUMNS],
        }).sort_values("Mean score")
        fig = px.bar(values, x="Mean score", y="Dimension", orientation="h", range_x=[0, 1],
                     color="Mean score", color_continuous_scale=["#f59e8b", "#f6c85f", "#0f766e"])
        fig.update_layout(height=320, margin={"l": 8, "r": 18, "t": 12, "b": 16}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
        st.caption("Dimension scores range from 0 to 1 and describe the available indicators.")


def _stakeholder_view(df: pd.DataFrame) -> None:
    areas = (
        "Overview",
        "Individual assessment",
        "Dimensions & profiles",
        "Gap analysis",
        "Stakeholder insights",
    )
    area = st.radio("Stakeholder area", areas, horizontal=True,
                    label_visibility="collapsed", key="new_fi_stakeholder_area")
    st.divider()
    if area == areas[0]:
        overview.render(df)
    elif area == areas[1]:
        st.subheader("Individual financial inclusion assessment")
        st.caption("Use the existing questionnaire and scoring workflow. Choose ‘Not available’ when an answer was not observed.")
        individual_assessment.render()
    elif area == areas[2]:
        _stakeholder_dimensions(df)
    elif area == areas[3]:
        gaps.render(df)
    else:
        st.subheader("Population and economy insights")
        st.caption("These are unweighted summaries of the available analytical sample, not national prevalence estimates.")
        distribution.render(df)
        with st.expander("Explore patterns by World Bank region"):
            subgroups.render_region(df)


def _dataset_quality() -> None:
    st.subheader("Dataset overview")
    raw = _read_json(DATA_REPORTS / "fi_preprocessing_report.json")
    analytical = _load_dataset()
    kpis = st.columns(4)
    kpis[0].metric("Source respondents", f"{int(raw.get('row_count', 0)):,}" if raw else "Unavailable")
    kpis[1].metric("Source variables", str(raw.get("column_count", "Unavailable")))
    kpis[2].metric("Analytical respondents", f"{len(analytical):,}")
    kpis[3].metric("Economies in analytical cohort", f"{analytical.economy.nunique():,}")
    st.caption("The analytical cohort requires complete profile dimensions; its coverage differs from the full source microdata.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### Variable availability and missingness")
        missing = _read_csv(DATA_REPORTS / "fi_missingness_by_variable.csv")
        if missing is None:
            _missing_artifact(DATA_REPORTS / "fi_missingness_by_variable.csv")
        else:
            missing = missing.sort_values("missing_percent", ascending=False).head(15)
            st.dataframe(missing[["variable", "dimension", "verification_status", "missing_percent"]],
                         hide_index=True, width="stretch")
    with col_b:
        st.markdown("#### Codebook and response review")
        codebook = _read_csv(DATA_REPORTS / "fi_codebook_verification_register.csv")
        if codebook is None:
            _missing_artifact(DATA_REPORTS / "fi_codebook_verification_register.csv")
        else:
            counts = codebook.verification_status.value_counts().rename_axis("Review status").reset_index(name="Variables")
            st.dataframe(counts, hide_index=True, width="stretch")
            st.caption("Question universes and skip patterns affect availability; missing responses are not treated as ‘No’.")
    with st.expander("Skip patterns and indicator verification details"):
        if codebook is not None:
            st.dataframe(codebook[["variable", "dimension", "skip_condition", "verification_status", "issue"]],
                         hide_index=True, width="stretch")
        audit = _read_csv(DATA_REPORTS / "fi_categorical_response_audit.csv")
        if audit is not None:
            st.markdown("**Categorical response-code audit**")
            st.dataframe(audit, hide_index=True, width="stretch")
        st.caption("Saved audit outputs are shown as recorded; this page does not rerun preprocessing or alter codes.")


def _dimension_research(df: pd.DataFrame) -> None:
    st.subheader("Five-dimensional measurement")
    policy_path = ROOT / "docs" / "research" / "fi_dimension_scoring_policy.md"
    st.caption("Dimension definitions and score construction follow the project's documented scoring policy.")
    mapping = pd.DataFrame([
        ("Connectivity", "con1 · internet_use", "Mean of phone access and internet use"),
        ("Financial Access", "account", "Account indicator"),
        ("Digital Financial Usage", "anydigpayment", "Digital payment indicator"),
        ("Digital Safety", "con18", "Phone protection indicator"),
        ("Financial Resilience", "saved · fin24a · fin24b", "Mean of three available components; all required"),
    ], columns=["Dimension", "Selected indicators", "Score rule"])
    st.dataframe(mapping, hide_index=True, width="stretch")
    summary = loader.calculate_dimension_summary(df).copy()
    summary["Dimension"] = summary.variable.map(loader.DIMENSION_LABELS)
    st.dataframe(summary[["Dimension", "mean", "median", "missing_percent", "distinct_nonmissing_values"]]
                 .rename(columns={"mean": "Mean", "median": "Median", "missing_percent": "Missing (%)", "distinct_nonmissing_values": "Distinct values"})
                 .style.format({"Mean": "{:.3f}", "Median": "{:.3f}", "Missing (%)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    long = df[loader.DIMENSION_COLUMNS].rename(columns=loader.DIMENSION_LABELS).melt(var_name="Dimension", value_name="Score").dropna()
    fig = px.violin(long, x="Dimension", y="Score", box=True, points=False, color="Dimension",
                    color_discrete_sequence=["#0f766e", "#1d6a8a", "#f2a65a", "#b86b77", "#64748b"])
    fig.update_layout(showlegend=False, height=400, margin={"l": 10, "r": 10, "t": 20, "b": 30})
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
    availability = _read_csv(DIMENSION_REPORTS / "fi_dimension_country_coverage.csv")
    if availability is not None:
        with st.expander("Coverage by dimension and economy"):
            st.dataframe(availability, hide_index=True, width="stretch")
    with st.expander("Full scoring policy"):
        if policy_path.is_file():
            st.markdown(policy_path.read_text(encoding="utf-8"))
        else:
            _missing_artifact(policy_path)


def _clustering_research(df: pd.DataFrame) -> None:
    from .sections import profiles

    st.subheader("Selected K-Means solution and five profiles")
    st.caption("The existing validated 5D_k5 model and profile definitions are displayed without retraining.")
    profiles.render(df)


def _heldout_validation() -> None:
    st.subheader("Repeated held-out validation")
    summary_path = HELDOUT_EXPERIMENT / "heldout_metric_summary.csv"
    metrics_path = HELDOUT_EXPERIMENT / "heldout_metrics_by_seed.csv"
    summary, results = _read_csv(summary_path), _read_csv(metrics_path)
    if summary is None or results is None:
        _missing_artifact(summary_path)
        return
    values = {row.metric: row for row in summary.itertuples(index=False)}
    cards = st.columns(2)
    silhouette = values.get("test_silhouette")
    ari = values.get("reference_ari")
    cards[0].metric("Held-out silhouette · mean", f"{silhouette.mean:.4f}" if silhouette else "Unavailable",
                    delta=f"SD {silhouette.std:.4f}" if silhouette else None)
    cards[1].metric("ARI vs full-data reference", f"{ari.mean:.3f}" if ari else "Unavailable",
                    delta="Reference agreement, not ground truth" if ari and ari.mean == 1 else None)
    st.warning("ARI compares held-out assignments with the existing full-data reference partition. It is not accuracy or independent ground-truth validation.")
    st.dataframe(results[["seed", "train_size", "test_size", "test_silhouette", "reference_ari"]]
                 .style.format({"test_silhouette": "{:.4f}", "reference_ari": "{:.3f}", "train_size": "{:,}", "test_size": "{:,}"}),
                 hide_index=True, width="stretch")
    chart = px.line(results, x="seed", y="test_silhouette", markers=True, title="Test silhouette by split seed")
    chart.update_traces(line_color="#0f766e")
    chart.update_layout(height=330, yaxis_title="Silhouette", xaxis_title="Split seed")
    st.plotly_chart(chart, use_container_width=True)
    profile_path = HELDOUT_EXPERIMENT / "dimension_profile_comparison_summary.csv"
    profile = _read_csv(profile_path)
    if profile is not None:
        st.markdown("#### Profile dimension means: reference and held-out")
        profile["Profile"] = profile.cluster_id.map(CLUSTER_LABELS)
        st.dataframe(profile[["Profile", "dimension", "full_reference_mean", "heldout_mean_across_seeds", "heldout_mean_sd_across_seeds"]]
                     .rename(columns={"dimension": "Dimension", "full_reference_mean": "Full reference mean", "heldout_mean_across_seeds": "Held-out mean", "heldout_mean_sd_across_seeds": "Held-out SD"})
                     .style.format({"Full reference mean": "{:.3f}", "Held-out mean": "{:.3f}", "Held-out SD": "{:.3f}"}),
                     hide_index=True, width="stretch")
    for image_name, caption in [("cluster_distribution_comparison.png", "Cluster shares across reference and hold-out samples"),
                                ("dimension_profile_comparison.png", "Dimension patterns across aligned profiles")]:
        image_path = HELDOUT_EXPERIMENT / image_name
        if image_path.is_file():
            st.image(str(image_path), caption=caption, width="stretch")


def _supervised_research() -> None:
    st.subheader("Supervised profile classification")
    try:
        baseline = loader.load_supervised_metrics()
        tuned = _read_csv(TUNED_EXPERIMENT / "metrics.csv")
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved supervised evaluation is unavailable: {exc}")
        return
    common = baseline.rename(columns={"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro-F1": "macro_f1", "Weighted-F1": "weighted_f1"})
    common["experiment"] = "Original baseline"
    if tuned is not None:
        revised = tuned.rename(columns={"model": "Model", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "macro_f1", "weighted_f1": "weighted_f1"})
        revised["Model"] = revised.Model.map({"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"})
        revised = revised[["Model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
        revised["experiment"] = "Tuned experiment"
        compare = pd.concat([common[["Model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "experiment"]], revised], ignore_index=True)
    else:
        compare = common[["Model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "experiment"]]
    chart_df = compare.melt(id_vars=["Model", "experiment"], var_name="Metric", value_name="Score")
    chart = px.bar(chart_df, x="Model", y="Score", color="experiment", facet_col="Metric", barmode="group",
                   color_discrete_sequence=["#164e63", "#f59e0b"], range_y=[0, 1])
    chart.update_layout(height=380, margin={"l": 10, "r": 10, "t": 45, "b": 15}, legend_title_text="")
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(compare.style.format({"accuracy": "{:.3f}", "balanced_accuracy": "{:.3f}", "macro_f1": "{:.3f}", "weighted_f1": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("The original and tuned runs use the same fixed held-out split. Tuned artifacts are exploratory; the dashboard continues to use its original baseline model.")
    try:
        matrices = loader.load_supervised_confusion_matrices()
        reports = loader.load_supervised_classification_reports()
    except (OSError, ValueError, KeyError) as exc:
        st.info(f"Class-level evaluation artifacts are unavailable: {exc}")
        return
    model_name = st.selectbox("Baseline confusion matrix", list(matrices), key="new_fi_confusion_model")
    matrix = pd.DataFrame(matrices[model_name], index=[CLUSTER_LABELS[i] for i in CLUSTER_IDS], columns=[CLUSTER_LABELS[i] for i in CLUSTER_IDS])
    heatmap = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                        labels={"x": "Predicted profile", "y": "Actual profile", "color": "Respondents"})
    heatmap.update_layout(height=440, margin={"l": 8, "r": 8, "t": 20, "b": 20})
    st.plotly_chart(heatmap, use_container_width=True)
    if model_name in reports:
        with st.expander(f"{model_name} per-class report"):
            st.code(reports[model_name], language="text")


def _shap_research() -> None:
    st.subheader("SHAP explanations for Logistic Regression")
    global_path = SHAP_REPORTS / "fi_shap_global_importance.csv"
    local_path = SHAP_REPORTS / "fi_shap_local_explanations.csv"
    predictions_path = SHAP_REPORTS / "fi_shap_predictions.csv"
    validation_path = SHAP_REPORTS / "fi_shap_validation_summary.json"
    global_imp, local, predictions = map(_read_csv, [global_path, local_path, predictions_path])
    validation = _read_json(validation_path)
    if global_imp is None or local is None or predictions is None:
        _missing_artifact(global_path)
        return
    cluster_id = st.selectbox("Profile class to explain", CLUSTER_IDS,
                              format_func=lambda value: CLUSTER_LABELS[value], key="new_fi_shap_class")
    selected = global_imp[global_imp.cluster_id == cluster_id].copy()
    selected["Predictor"] = selected.feature.map(FEATURE_LABELS).fillna(selected.feature)
    selected = selected.sort_values("mean_absolute_shap_decision_score", ascending=True)
    chart = px.bar(selected, x="mean_absolute_shap_decision_score", y="Predictor", orientation="h",
                   color="mean_absolute_shap_decision_score", color_continuous_scale=["#cbd5e1", "#0f766e"])
    chart.update_layout(height=350, coloraxis_showscale=False, xaxis_title="Mean absolute contribution to decision score", yaxis_title="")
    st.plotly_chart(chart, use_container_width=True)
    options = predictions["row_index"].astype(int).tolist()
    row_index = st.selectbox("Saved validation example", options, format_func=lambda row: f"Example {row + 1}", key="new_fi_shap_example")
    row = predictions.loc[predictions.row_index == row_index].iloc[0]
    st.caption(f"Predicted profile: {row.predicted_cluster_label} · saved predicted probability {row.predicted_cluster_probability:.3f}")
    detail = local[(local.row_index == row_index) & (local.explained_class_id == row.predicted_cluster_id)].copy()
    detail["Predictor"] = detail.feature.map(FEATURE_LABELS).fillna(detail.feature)
    detail = detail.sort_values("shap_decision_score_contribution")
    local_fig = px.bar(detail, x="shap_decision_score_contribution", y="Predictor", orientation="h",
                       color="shap_decision_score_contribution", color_continuous_scale=["#b4535b", "#e2e8f0", "#0f766e"])
    local_fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
    local_fig.update_layout(height=350, coloraxis_showscale=False, xaxis_title="Contribution to predicted class decision score", yaxis_title="")
    st.plotly_chart(local_fig, use_container_width=True)
    st.caption("SHAP values describe the saved classifier's decision scores. They are not causal explanations or direct effects on financial inclusion.")
    if validation:
        st.metric("SHAP validation", "Passed" if validation.get("checks_passed", {}).get("additivity") else "Review saved validation",
                  delta=f"Max additivity error {validation.get('max_additivity_error', float('nan')):.2e}" if validation.get("max_additivity_error") is not None else None)


def _findings() -> None:
    st.subheader("Research findings, limitations, and implementation status")
    st.markdown("""
    - The selected five-profile solution is stable across the saved repeated-seed checks and the separate respondent hold-out experiment.
    - Profile names summarize observed dimension-score patterns; they are not causal categories or individual diagnoses.
    - The supervised classifier predicts K-Means-generated profile labels from seven demographic and socioeconomic predictors. Performance is modest and differs by class.
    - The held-out ARI is agreement with a reference clustering fitted on the complete analytical cohort, not independent ground truth.
    - Digital Safety and Financial Resilience have lower coverage in the full source data. Cross-economy comparisons should account for coverage and should not be treated as national rankings.
    - Community recommendations are follow-up questions for local validation, not proven interventions.
    """)
    st.info("The page displays saved findings and model outputs. It does not retrain models, rerun preprocessing, or change profile assignments.")


def _research_view(df: pd.DataFrame) -> None:
    sections = (
        "Dataset & quality", "Dimension construction", "K-Means profiles",
        "Held-out validation", "Supervised models", "SHAP explainability", "Findings & limits",
    )
    selected = st.radio("Research section", sections, horizontal=True,
                        label_visibility="collapsed", key="new_fi_research_area")
    st.divider()
    if selected == sections[0]:
        _dataset_quality()
    elif selected == sections[1]:
        _dimension_research(df)
    elif selected == sections[2]:
        _clustering_research(df)
    elif selected == sections[3]:
        _heldout_validation()
    elif selected == sections[4]:
        _supervised_research()
    elif selected == sections[5]:
        _shap_research()
    else:
        _findings()


def render() -> None:
    with st.container(key="new_fi_dark_dashboard"):
        _inject_styles()
        try:
            df = _load_dataset()
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.error("Could not load the validated Component 02 analytical dataset.")
            st.info(str(exc))
            st.stop()
        view = st.radio(
            "Dashboard audience",
            ("Stakeholder Dashboard", "Research & Model Insights"),
            horizontal=True, label_visibility="collapsed", key="new_fi_dashboard_view",
        )
        _hero(view)
        if view == "Stakeholder Dashboard":
            _stakeholder_view(df)
        else:
            _research_view(df)
        st.divider()
        st.caption("Component 02 · Existing validated data and saved outputs · Descriptive results do not establish causation")
