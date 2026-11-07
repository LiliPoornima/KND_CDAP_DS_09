"""SHAP Explanation section: Global model importance and Individual local attributions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.src.financial_capability.fc_config import CLASS_NAMES
from backend.src.financial_capability.fc_explainability import describe_local_explanation
from backend.src.financial_capability.fc_predictor import validate_response
from frontend.financial_capability.data_loader import (
    get_explainer,
    require_bundle,
    require_response,
)

SHAP_DISCLAIMER = (
    "SHAP explains how individual answers influence the model's prediction. "
    "These explanations do not prove that one factor causes a person's financial well-being."
)


def render() -> None:
    """Render the enhanced SHAP Explainability section."""
    st.subheader("Explainable AI: SHAP Feature Attribution")
    bundle = require_bundle()
    if bundle is None:
        return

    st.info(f"ℹ️ {SHAP_DISCLAIMER}")

    # Tabs for Local (Personal) vs Global (Model-Wide) Explanations
    tab_local, tab_global = st.tabs(["🔍 Individual Explanation (Your Assessment)", "🌐 Global Feature Importance (Model-Wide)"])

    with tab_local:
        st.markdown("### Why this prediction? (your current assessment)")
        response = require_response()
        if response is None:
            st.info("Complete an assessment in the **Individual Assessment** section to view your personalized feature attributions.")
            return

        explainer = get_explainer()
        row = validate_response(response)
        res = explainer.explain_individual(row)
        pred_class = res["class_name"]
        pred_prob = res["probability"]

        # Prediction Summary Banner
        st.markdown(
            f"""
            <div class="fc-card" style="border-left: 4px solid #1E3A8A;">
                <div style="font-size: 0.9rem; color: #64748B;">Target Outcome Analyzed:</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #0F172A;">
                    Predicted Band: <span style="color: #2563EB;">{pred_class}</span> (Model Probability: {pred_prob:.1%})
                </div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                    Bars to the right (<span style="color: #059669; font-weight: 600;">green</span>) pushed the model towards <b>{pred_class}</b>. 
                    Bars to the left (<span style="color: #EF4444; font-weight: 600;">red</span>) pushed the model away.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Primary Influential Factors")
        sentences = describe_local_explanation(res, top_n=5)
        for s in sentences:
            st.markdown(f"• {s}")

        # Horizontal Bar Chart for Top 10 Local Features
        table = res["table"].head(10).iloc[::-1]
        colors = ["#10B981" if v > 0 else "#EF4444" for v in table["shap_value"]]

        fig = go.Figure(
            go.Bar(
                x=table["shap_value"],
                y=table["label"],
                orientation="h",
                marker_color=colors,
                customdata=table["answer"],
                hovertemplate="<b>%{y}</b><br>Answer: %{customdata}<br>SHAP Impact: %{x:+.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=f"Top Factor Contributions for Predicted Class '{pred_class}'", font=dict(size=14, color="#0F172A")),
            xaxis=dict(title=f"SHAP Contribution (relative to class base value)", zeroline=True, zerolinewidth=1.5, zerolinecolor="#94A3B8"),
            yaxis=dict(title=""),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Feature Attribution Table"):
            st.dataframe(
                res["table"][["label", "answer", "shap_value", "effect"]]
                .rename(columns={"label": "Survey Question", "answer": "Your Answer", "shap_value": "SHAP Impact", "effect": "Effect on Prediction"})
                .style.format({"SHAP Impact": "{:+.4f}"}),
                hide_index=True,
                use_container_width=True,
            )

    with tab_global:
        st.markdown("### Which Questions Matter Most Across All Respondents?")
        st.caption("Aggregated importance across the entire held-out evaluation dataset.")

        imp = pd.DataFrame(bundle.metadata["global_importance"])
        top_n = st.slider("Number of questions to show", 5, len(imp), 12)
        top = imp.head(top_n).iloc[::-1]

        fig_global = px.bar(
            top,
            x="mean_abs_shap",
            y="label",
            color="group",
            orientation="h",
            color_discrete_map={
                "Saving": "#D97706",
                "Budgeting": "#2563EB",
                "Basic Financial Knowledge": "#059669",
                "Financial Goal Planning": "#0F172A",
                "Background (context only)": "#94A3B8",
            },
            labels={"mean_abs_shap": "Mean |SHAP| Value", "label": "", "group": "Category"},
        )
        fig_global.update_layout(
            title=dict(text=f"Top {top_n} Most Influential Questions Across All Classes", font=dict(size=14, color="#0F172A")),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=40, b=20),
            height=420,
        )
        st.plotly_chart(fig_global, use_container_width=True)

        with st.expander("Global Feature Importance Breakdown by Class"):
            class_cols = [f"mean_abs_shap_{c}" for c in CLASS_NAMES]
            st.dataframe(
                imp[["label", "group", "mean_abs_shap"] + class_cols]
                .rename(columns={"label": "Question", "group": "Category", "mean_abs_shap": "Overall Importance"})
                .style.format({c: "{:.4f}" for c in ["Overall Importance"] + class_cols}),
                hide_index=True,
                use_container_width=True,
            )
