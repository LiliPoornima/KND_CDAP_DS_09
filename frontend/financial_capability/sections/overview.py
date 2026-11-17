"""Overview section: Component 04 independent system workflow and Bank Loan Decision Support scenario."""

from __future__ import annotations

import streamlit as st

from frontend.financial_capability.constants import BANK_LOAN_NOTICE, DISCLAIMER
from frontend.financial_capability.data_loader import get_metadata


def _render_diagram_a_workflow(n_raw: str, n_valid: str, sel_model: str) -> None:
    """Render Diagram A: Component 04 – Financial Well-Being Assessment System (Independent)."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1E3A8A 0%, #1D4ED8 100%); color: #FFFFFF;
                    padding: 12px 20px; border-radius: 10px; font-weight: 700; font-size: 1.12rem;
                    text-align: center; margin-bottom: 14px; letter-spacing: 0.2px;">
            Part A — Component 04: Financial Well-Being Assessment System (Independent Workflow)
        </div>
        """,
        unsafe_allow_html=True,
    )

    stages = [
        (
            "1. CFPB Survey (NFWBS 2016)",
            "📋",
            "#1E3A8A",
            [
                f"<b>{n_raw}</b> raw / <b>{n_valid}</b> valid records",
                "Survey variables: Saving, Budgeting, Knowledge, Goals & Context",
                "Official CFPB public-use survey dataset",
            ],
        ),
        (
            "2. Data Preparation",
            "🗄️",
            "#0284C7",
            [
                "Clean data & special codes (-1, 98, 99)",
                "Select <b>31 approved</b> input features",
                "Target: <b>FWBscore</b> (Low ≤49, Med 50–57, High ≥58)",
                "Strict leakage prevention (exclude 10 FWB items)",
            ],
        ),
        (
            "3. Model Training & Eval",
            "🤖",
            "#2563EB",
            [
                "<b>Random Forest</b> & <b>XGBoost</b>",
                f"Selected model: <b>{sel_model}</b> (5-fold CV)",
                "Compare Accuracy, Precision, Recall, F1 & ROC-AUC",
            ],
        ),
        (
            "4. Explainability (SHAP)",
            "📊",
            "#7C3AED",
            [
                "Global feature importance",
                "Individual applicant explanation",
                "Key drivers: saving habit, budget control, knowledge, goal planning",
                "Describes model behavior (non-causal)",
            ],
        ),
        (
            "5. Financial Profile & Guidance",
            "🎯",
            "#D97706",
            [
                "4 Dimensions: <b>Saving, Budgeting, Knowledge, Goal Planning</b>",
                "👍 <b>Strengths</b> & ⚠️ <b>Gap Areas</b>",
                "🏦 <b>Bank Officer Guidance</b> & follow-up questions",
            ],
        ),
        (
            "6. Assessment Report",
            "📄",
            "#059669",
            [
                "Predicted level (High / Medium / Low)",
                "Key SHAP explanation factors",
                "Capability gaps & resilience indicators",
                "Downloadable PDF & TXT report for Bank Officer",
            ],
        ),
    ]

    cols = st.columns(6)
    for idx, (title, icon, color, bullets) in enumerate(stages):
        bullet_html = "".join(
            f"<li style='margin-bottom: 5px;'>✔ {b}</li>" for b in bullets
        )
        card_html = f"""
        <div class="fc-card" style="border-top: 4px solid {color}; border-radius: 10px;
                                    padding: 14px 12px; min-height: 255px; background: #FFFFFF;">
            <div style="font-size: 1.35rem; margin-bottom: 4px;">{icon}</div>
            <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A; margin-bottom: 10px;">
                {title}
            </div>
            <ul style="list-style-type: none; padding-left: 0; margin: 0;
                       font-size: 0.80rem; color: #334155; line-height: 1.45;">
                {bullet_html}
            </ul>
        </div>
        """
        cols[idx].markdown(card_html, unsafe_allow_html=True)


def _render_diagram_b_bank_scenario() -> None:
    """Render Diagram B: How Component 04 Supports the Bank Loan Scenario."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #0F172A 0%, #1E3A8A 100%); color: #FFFFFF;
                    padding: 12px 20px; border-radius: 10px; font-weight: 700; font-size: 1.12rem;
                    text-align: center; margin-top: 10px; margin-bottom: 14px; letter-spacing: 0.2px;">
            Part B — How Component 04 Supports the Bank Loan Scenario
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns([1.15, 1.75, 1.85, 1.25, 1.35])

    with c1:
        st.markdown(
            """
            <div class="fc-card" style="border: 2px solid #CBD5E1; border-radius: 10px; padding: 14px; min-height: 235px; text-align: center;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🧑‍💼📝</div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #0F172A; margin-bottom: 6px;">
                    1. Customer<br/>(Loan Applicant)
                </div>
                <div style="font-size: 0.82rem; color: #475569; line-height: 1.45;">
                    Submits loan application and completes the financial well-being assessment questionnaire.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="fc-card" style="border: 2px solid #94A3B8; border-radius: 10px; padding: 14px; min-height: 235px; background: #F8FAFC;">
                <div style="font-weight: 700; font-size: 0.92rem; color: #1E293B; margin-bottom: 6px;">
                    2A. Bank's Normal Loan Check<br/>
                    <span style="font-size: 0.76rem; font-weight: 500; color: #64748B;">(Existing Bank Process — Contextual Only)</span>
                </div>
                <ul style="list-style-type: none; padding-left: 0; margin: 0; font-size: 0.81rem; color: #334155; line-height: 1.55;">
                    <li>🪪 <b>Identity & Income Verification</b></li>
                    <li>📈 <b>Credit History / Score</b></li>
                    <li>🪙 <b>Repayment Capacity</b> (Income/expenses, DTI)</li>
                </ul>
                <div style="font-size: 0.74rem; color: #64748B; margin-top: 8px;">
                    <i>Performed outside Component 04 under normal bank systems.</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="fc-card" style="border: 2px solid #7C3AED; border-radius: 10px; padding: 14px; min-height: 235px; background: #FAF5FF;">
                <div style="font-weight: 700; font-size: 0.92rem; color: #4C1D95; margin-bottom: 6px;">
                    ➕ 2B. Component 04<br/>
                    <span style="font-size: 0.76rem; font-weight: 600; color: #6D28D9;">(Independent Financial Well-Being Assessment)</span>
                </div>
                <ul style="list-style-type: none; padding-left: 0; margin: 0; font-size: 0.81rem; color: #334155; line-height: 1.5;">
                    <li>✅ Assesses applicant's financial well-being (<b>High / Medium / Low</b>)</li>
                    <li>✅ Explains key factors via <b>SHAP</b> (saving, budgeting, knowledge, goals)</li>
                    <li>✅ Identifies capability strengths, gaps & resilience indicators</li>
                    <li>✅ Generates supplementary report & officer follow-up guidance</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="fc-card" style="border: 2px solid #2563EB; border-radius: 10px; padding: 14px; min-height: 235px; text-align: center;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🏦💻</div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #1E3A8A; margin-bottom: 6px;">
                    3. Bank Officer<br/>(Decision Support)
                </div>
                <div style="font-size: 0.81rem; color: #334155; line-height: 1.45;">
                    Reviews the bank's normal loan-check results <b>and</b> the Component 04 Assessment Report as separate inputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            """
            <div class="fc-card" style="border: 2px solid #10B981; border-radius: 10px; padding: 14px; min-height: 235px; background: #ECFDF5;">
                <div style="font-size: 1.5rem; margin-bottom: 4px;">📋✅</div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #065F46; margin-bottom: 8px;">
                    4. Loan Decision (Final)
                </div>
                <ul style="list-style-type: none; padding-left: 0; margin: 0; font-size: 0.83rem; color: #064E3B; line-height: 1.65;">
                    <li>✔ <b>Approve</b></li>
                    <li>✔ <b>Approve with Conditions</b></li>
                    <li>✔ or <b>Reject</b></li>
                </ul>
                <div style="font-size: 0.74rem; color: #047857; margin-top: 8px;">
                    <i>Made by the Bank Officer under bank lending policies.</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-left: 5px solid #DC2626;
                    border-radius: 8px; padding: 12px 16px; margin-top: 10px; color: #7F1D1D; font-size: 0.9rem;">
            <b>ⓘ Important Decision-Support Notice:</b> {BANK_LOAN_NOTICE}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render the Component 04 Overview with KPI metrics, Diagram A, Diagram B, and Scope Boundaries."""
    st.subheader("Bank Officer Decision Support — System Overview")
    st.markdown(
        """
        **Component 04** is an **independent Explainable Financial Well-being Assessment System** designed to
        assist a **Bank Officer** during a loan application review. Alongside the bank's standard verification
        and credit checks, Component 04 evaluates the loan applicant's day-to-day financial capability across
        four core dimensions (**Saving, Budgeting, Basic Financial Knowledge, and Financial Goal Planning**),
        explains the machine-learning prediction with **SHAP**, and generates a supplementary assessment report.
        """
    )
    st.caption(DISCLAIMER)

    # KPI Metrics from actual saved metadata
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
        n_raw, n_valid, sel_model = "6,394", "6,389", "XGBoost"
    else:
        v = meta["dataset"]["validation"]
        sel_model = meta["selected_model"]
        m = meta["test_metrics"][sel_model]
        n_raw = f"{v.get('n_rows', 6394):,}"
        n_valid_int = meta["split"]["n_train"] + meta["split"]["n_test"]
        n_valid = f"{n_valid_int:,}"
        c1.metric("Selected Model", sel_model)
        c2.metric("Valid Survey Records", n_valid)
        c3.metric("Test Macro-F1", f"{m['macro_f1']:.3f}")
        c4.metric("Test Accuracy", f"{m['accuracy']:.3f}")
        st.caption(
            f"Model trained {meta['training_date_utc']} (UTC) on held-out evaluation test set "
            f"(ROC-AUC: {m.get('roc_auc_ovr_macro', 0.0):.3f})."
        )

    st.divider()

    # Diagram A: Independent Component 04 Workflow
    _render_diagram_a_workflow(n_raw=n_raw, n_valid=n_valid, sel_model=sel_model)

    st.divider()

    # Diagram B: How Component 04 Supports the Bank Loan Scenario
    _render_diagram_b_bank_scenario()

    st.divider()

    # Component Independence & Scope Boundaries
    st.markdown("### Component 04 Independence & Governance Boundaries")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**What Component 04 Provides (Included)**\n"
            "- Independent Financial Well-being Assessment (`Low`, `Medium`, `High`) and model prediction probabilities\n"
            "- CFPB NFWBS 2016 dataset (`6,394` raw / `6,389` valid rows) and `31` leakage-safe predictor features\n"
            "- Random Forest and XGBoost evaluation with 5-fold stratified cross-validation\n"
            "- Global and individual applicant **SHAP** explanations in plain language\n"
            "- Four-dimension Financial Capability Profile (**Saving, Budgeting, Knowledge, Goal Planning**)\n"
            "- Supported Financial Resilience indicators, Bank Officer follow-up questions, What-if simulation, and PDF/TXT reports"
        )
    with right:
        st.markdown(
            "**Strict Governance & Independence Boundaries (Excluded)**\n"
            "- **Independent of Components 01, 02, and 03**: Uses its own dataset, features, models, and rules\n"
            "- **Not an Automatic Loan Decision**: Never approves or rejects a loan or predicts loan repayment/default probability\n"
            "- **Does Not Replace Standard Bank Checks**: Identity/income verification, credit score, and DTI remain separate bank processes\n"
            "- **For Bank Officer Use Only**: Excludes customer learning platforms, educational videos, lessons, or quizzes"
        )

