"""Model Performance section: Random Forest vs XGBoost on the same held-out test set."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.financial_capability.data_loader import require_bundle

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "macro_precision": "Macro precision", "macro_recall": "Macro recall", "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1", "roc_auc_ovr_macro": "ROC-AUC (OvR, macro)", "log_loss": "Log loss",
}


def render() -> None:
    st.subheader("Model Performance")
    bundle = require_bundle()
    if bundle is None:
        return
    meta = bundle.metadata
    results = meta["test_metrics"]
    selected = meta["selected_model"]
    if meta.get("quick_run"):
        st.warning("These artifacts come from a QUICK smoke run (tiny grids) - not final results.")

    st.success(f"**Selected model: {selected}.** Rule: {meta['selection_rule']}")
    st.caption(f"Same stratified split for both models: {meta['split']['n_train']:,} train / "
               f"{meta['split']['n_test']:,} test, seed {meta['random_seed']}, "
               f"{meta['split']['cv_folds']}-fold CV on the training set only.")

    table = pd.DataFrame([{"Model": name, **{METRIC_LABELS[k]: m[k] for k in METRIC_LABELS}}
                          for name, m in results.items()])
    st.markdown("### Test-set metrics")
    st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns if c != "Model"}),
                 hide_index=True, use_container_width=True)
    base = meta["majority_baseline"]
    st.caption(f"Reference: always predicting the biggest class gives accuracy {base['accuracy']:.4f} "
               f"and macro-F1 {base['macro_f1']:.4f}. Accuracy alone is not enough because the "
               "classes are uneven, so macro metrics and per-class results are shown.")

    long = table.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1", "Balanced accuracy",
                                                   "ROC-AUC (OvR, macro)"])
    fig = px.bar(long, x="variable", y="value", color="Model", barmode="group", range_y=[0, 1])
    fig.update_layout(xaxis_title="", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Per-class metrics")
    rows = [{"Model": n, "Class": c, **vals} for n, m in results.items()
            for c, vals in m["per_class"].items()]
    st.dataframe(pd.DataFrame(rows).style.format(
        {"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}),
        hide_index=True, use_container_width=True)

    st.markdown("### Confusion matrices (rows = actual, columns = predicted)")
    cols = st.columns(len(results))
    for col, (name, m) in zip(cols, results.items()):
        labels = m["confusion_matrix_labels"]
        fig_cm = px.imshow(m["confusion_matrix"], x=labels, y=labels, text_auto=True,
                           color_continuous_scale="Blues", title=name)
        fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                             coloraxis_showscale=False)
        col.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("Cross-validation results used for model selection (training set only)"):
        tun = meta["tuning"]
        st.write(f"Candidates tried: Random Forest {tun['n_rf_candidates']}, "
                 f"XGBoost {tun['n_xgb_candidates']}.")
        st.write("Random Forest - top 5")
        st.dataframe(pd.DataFrame(tun["rf_top5"]), hide_index=True, use_container_width=True)
        st.write("XGBoost - top 5")
        st.dataframe(pd.DataFrame(tun["xgb_top5"]), hide_index=True, use_container_width=True)

    with st.expander("Reliability checks (selected model)"):
        st.markdown("**Performance by age and income group** - differences are reported, "
                    "this is not a fairness guarantee.")
        st.dataframe(pd.DataFrame(meta["subgroup_metrics"]), hide_index=True,
                     use_container_width=True)
        stab = meta["explanation_stability"]
        st.markdown(f"**Explanation stability:** mean pairwise Jaccard overlap of the top-"
                    f"{stab['top_k']} SHAP features across {stab['n_splits']} re-splits = "
                    f"**{stab['mean_pairwise_jaccard']:.3f}** (1.0 = identical). Macro-F1 across the "
                    f"re-splits: {stab['macro_f1_mean']:.3f} +/- {stab['macro_f1_std']:.3f}.")
        st.markdown("**Feature-group ablation** (diagnostic, single split)")
        st.dataframe(pd.DataFrame(meta["feature_group_ablation"]), hide_index=True,
                     use_container_width=True)

