"""Overview section: Component mission, Phase 1 scope, and suggested final user journey."""

from __future__ import annotations

import streamlit as st

from frontend.financial_capability.constants import DISCLAIMER
from frontend.financial_capability.data_loader import get_metadata


def render() -> None:
    """Render the Component Overview with metrics, Phase 1 features checklist, and user journey."""
    st.subheader("Component Overview")
    st.markdown(
        """
        Having a bank account, a digital payment service or credit access does not automatically mean a
        person can manage money well. This component links behavioral indicators into an explainable, educational workflow:

        **Input Survey Responses ➔ Model Prediction ➔ SHAP Explanation ➔ Capability Gaps & Strengths ➔ Suggested Actions & Report**
        """
    )
    st.caption(DISCLAIMER)

    # KPI Metrics
    meta = get_metadata()
    c1, c2, c3, c4 = st.columns(4)
    if meta is None:
        c1.metric("Model status", "Not trained yet")
        c2.metric("Dataset", "CFPB NFWBS 2016")
        c3.metric("Models", "Random Forest / XGBoost")
        c4.metric("Explainability", "SHAP")
        st.warning(
            "The model has not been trained, so no performance numbers are shown. "
            "Open any model-dependent section for the training command."
        )
    else:
        v = meta["dataset"]["validation"]
        sel = meta["selected_model"]
        m = meta["test_metrics"][sel]
        c1.metric("Selected Model", sel)
        c2.metric("Dataset Respondents", f"{meta['split']['n_train'] + meta['split']['n_test']:,}")
        c3.metric("Test Macro-F1", f"{m['macro_f1']:.3f}")
        c4.metric("Test Accuracy", f"{m['accuracy']:.3f}")
        st.caption(
            f"Model trained {meta['training_date_utc']} (UTC) on held-out evaluation test set."
        )

    st.divider()

    # Phase 1 Features Checklist & Suggested User Journey (Matching User Images 1 & 2)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        checklist_html = """
        <div class="fc-card" style="border-radius: 12px; border: 1px solid #E2E8F0; padding: 22px;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                Component 05 – Phase 1 Features
            </div>
            <div style="margin-bottom: 14px;">
                <span class="fc-badge badge-good">Recommended scope</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <span style="font-weight: 600; font-size: 0.95rem; color: #1E293B;">Implementation checklist</span>
                <span style="font-size: 0.82rem; color: #64748B;">8/8 selected</span>
            </div>
            <ul style="list-style-type: none; padding-left: 0; margin: 0; color: #334155; font-size: 0.9rem; line-height: 1.9;">
                <li>✔ <b>Financial well-being prediction</b> (Low / Medium / High)</li>
                <li>✔ <b>Class probability and prediction reliability</b></li>
                <li>✔ <b>Financial capability overview</b> (four core areas)</li>
                <li>✔ <b>Financial strengths identification</b></li>
                <li>✔ <b>Improvement priorities and explanations</b></li>
                <li>✔ <b>Individual SHAP explanation</b></li>
                <li>✔ <b>Personalised action summary and learning categories</b></li>
                <li>✔ <b>What-if simulation and downloadable assessment report</b></li>
            </ul>
        </div>
        """
        st.markdown(checklist_html, unsafe_allow_html=True)

    with right_col:
        journey_html = """
        <div class="fc-card" style="border-radius: 12px; border: 1px solid #E2E8F0; padding: 22px;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-bottom: 14px;">
                Suggested Final User Journey
            </div>
            <ol style="padding-left: 18px; margin: 0; color: #334155; font-size: 0.88rem; line-height: 1.6;">
                <li style="margin-bottom: 8px;">
                    <b>Answer questions</b><br/>
                    <span style="color: #64748B;">User completes the financial capability assessment.</span>
                </li>
                <li style="margin-bottom: 8px;">
                    <b>View prediction</b><br/>
                    <span style="color: #64748B;">Low / Medium / High and class probabilities.</span>
                </li>
                <li style="margin-bottom: 8px;">
                    <b>Understand strengths</b><br/>
                    <span style="color: #64748B;">Identify positive financial habits and knowledge.</span>
                </li>
                <li style="margin-bottom: 8px;">
                    <b>Understand weaknesses</b><br/>
                    <span style="color: #64748B;">Identify possible improvement areas and their triggers.</span>
                </li>
                <li style="margin-bottom: 8px;">
                    <b>Understand the prediction</b><br/>
                    <span style="color: #64748B;">View individual SHAP explanations.</span>
                </li>
                <li style="margin-bottom: 8px;">
                    <b>Get suggested actions</b><br/>
                    <span style="color: #64748B;">See relevant improvement suggestions and learning categories.</span>
                </li>
                <li style="margin-bottom: 8px;">
                    <b>Try what-if simulation</b><br/>
                    <span style="color: #64748B;">Explore how changing answers affects the prediction.</span>
                </li>
                <li>
                    <b>Download report</b><br/>
                    <span style="color: #64748B;">Keep a PDF copy of the assessment results.</span>
                </li>
            </ol>
        </div>
        """
        st.markdown(journey_html, unsafe_allow_html=True)

    st.divider()

    # Scope Boundaries Table
    st.markdown("### Research Scope Boundaries")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Included in Phase 1**\n"
            "- CFPB NFWBS 2016 public-use dataset (6,389 valid rows)\n"
            "- Leakage-safe feature selection (31 approved features)\n"
            "- Random Forest vs XGBoost stratified tuning\n"
            "- Global and individual SHAP local attribution\n"
            "- Transparent, codebook-based capability rules\n"
            "- Downloadable assessment report (PDF/TXT)\n"
            "- Interactive What-if scenario simulator"
        )
    with right:
        st.markdown(
            "**Not Included (Planned for Phase 2)**\n"
            "- User authentication, passwords, or persistent account databases\n"
            "- Video streaming, lesson courseware delivery, or quizzes\n"
            "- User learning progress tracking over time\n"
            "- Any modification to teammates' Components 01–04"
        )
