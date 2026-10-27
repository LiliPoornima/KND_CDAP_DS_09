import pandas as pd
import plotly.express as px
import streamlit as st
from .. import data_loader as loader
from ..constants import CLUSTER_IDS, CLUSTER_LABELS

def render(df: pd.DataFrame) -> None:
    st.header("Explainable AI")
    st.caption("Saved classifiers predict derived cluster labels. Performance is modest; SHAP values describe model decision behavior, not causes of exclusion.")
    try:
        metrics = loader.prepare_supervised_model_comparison(loader.load_supervised_metrics())
        reports = loader.load_supervised_classification_reports()
        matrices = loader.load_supervised_confusion_matrices()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved model evaluation could not be validated: {exc}")
    else:
        st.subheader("Saved model evaluation")
        st.dataframe(metrics.round(4), hide_index=True, width="stretch")
        model = st.selectbox("Saved model report", list(reports), key="fi_classification_report")
        st.code(reports[model])
        if model in matrices:
            st.dataframe(pd.DataFrame(matrices[model], index=[CLUSTER_LABELS[i] for i in CLUSTER_IDS], columns=[CLUSTER_LABELS[i] for i in CLUSTER_IDS]), width="stretch")
        st.info("Logistic Regression remains selected for SHAP in the saved analysis because its class-balanced metrics and interpretability were the stated selection rationale; XGBoost has the highest saved accuracy. No model is selected automatically here.")
    try:
        global_imp = loader.load_shap_global_importance()
        local = loader.load_shap_local_explanations()
        predictions = loader.load_shap_predictions()
        validation = loader.load_shap_validation_summary()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved SHAP artifacts could not be validated: {exc}")
        return
    st.subheader("Global SHAP importance")
    cls = st.selectbox("Class explained", CLUSTER_IDS, format_func=lambda x: CLUSTER_LABELS[x], key="shap_global_class")
    shown = global_imp.loc[global_imp.cluster_id == cls].sort_values("mean_absolute_shap_decision_score").tail(12)
    st.plotly_chart(px.bar(shown, x="mean_absolute_shap_decision_score", y="feature", orientation="h", title=f"Mean absolute SHAP decision-score contribution Â· {CLUSTER_LABELS[cls]}"), config={"responsive": True})
    st.caption(f"Saved sampled evaluation: {validation.get('explained_rows','Unavailable')} explained rows. These are grouped feature importance values, not a beeswarm plot.")
    st.subheader("Saved local explanation")
    row_ids = sorted(predictions.row_index.astype(int).unique())
    row_id = st.selectbox("Previously explained sample index", row_ids, key="shap_local_sample")
    pred = predictions.loc[predictions.row_index == row_id].iloc[0]
    c = int(pred.predicted_cluster_id)
    detail = local.loc[(local.row_index == row_id) & (local.explained_class_id == c)].sort_values("shap_decision_score_contribution")
    st.caption(f"Saved prediction: {CLUSTER_LABELS[c]} Â· model output {pred.predicted_cluster_probability:.1%}; this is not an individual probability of exclusion.")
    st.plotly_chart(px.bar(detail, x="shap_decision_score_contribution", y="feature", orientation="h", title="Contributions to predicted class decision score (logits)").add_vline(x=0, line_dash="dash"), config={"responsive": True})
    st.caption("Positive/negative values move the saved class decision score up/down. They do not show causal effects. Only saved validation-sample explanations are shown; personal IDs are not exposed.")
    st.subheader("Saved SHAP validation")
    st.json({k: validation.get(k) for k in ["explained_output", "background_rows", "explained_rows", "max_additivity_error", "max_grouped_sum_error", "checks_passed", "split_provenance_note"]})
