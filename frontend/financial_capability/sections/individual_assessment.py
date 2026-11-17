"""Bank Officer Dashboard (Individual Assessment): Complete 8-section loan applicant financial well-being review."""

from __future__ import annotations

from datetime import date
import html
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.src.financial_capability.fc_capability_rules import (
    STATUS_GAP,
    STATUS_INSUFFICIENT,
    STATUS_NONE,
    extract_resilience_indicators,
    generate_officer_guidance,
    generate_officer_questions,
    identify_capability_gaps,
    identify_capability_strengths,
)
from backend.src.financial_capability.fc_config import (
    CLASS_NAMES,
    FEATURE_SPECS,
    GROUP_BUDGETING,
    GROUP_CONTEXT,
    GROUP_GOALS,
    GROUP_KNOWLEDGE,
    GROUP_SAVING,
)
from backend.src.financial_capability.fc_explainability import describe_local_explanation
from backend.src.financial_capability.fc_predictor import (
    InvalidResponseError,
    predict_assessment,
    validate_response,
)
from backend.src.financial_capability.fc_what_if import run_what_if
from frontend.financial_capability.constants import (
    BANK_LOAN_NOTICE,
    COMPONENT_TITLE,
    DISCLAIMER,
    EXAMPLE_RESPONSE,
    SESSION_RESPONSE_KEY,
    WIDGET_PREFIX,
)
from frontend.financial_capability.data_loader import get_explainer, require_bundle
from frontend.financial_capability.pdf_report import (
    REPORT_TITLE,
    build_capability_assessment_pdf,
    build_capability_assessment_txt,
)


def _fill_example() -> None:
    """Callback: fill illustrative loan applicant answers into form widgets."""
    for name, value in EXAMPLE_RESPONSE.items():
        st.session_state[WIDGET_PREFIX + name] = value


def _clear_answers() -> None:
    """Callback: reset all form widgets and session results."""
    for spec in FEATURE_SPECS:
        st.session_state[WIDGET_PREFIX + spec.name] = None
    st.session_state.pop(SESSION_RESPONSE_KEY, None)


def _question(spec) -> None:
    """Render a clean, accessible question dropdown."""
    key = WIDGET_PREFIX + spec.name
    if key not in st.session_state:
        st.session_state[key] = None
    st.selectbox(
        spec.label,
        options=[None] + list(spec.options),
        format_func=lambda v, s=spec: "Not answered" if v is None else s.options[v],
        help=spec.question + (f" ({spec.note})" if spec.note else ""),
        key=key,
    )


def probability_chart(
    probabilities: dict,
    title: str = "Model Prediction Probability Distribution (Financial Well-being)",
) -> go.Figure:
    """Create a banking-grade probability distribution bar chart."""
    colors = [
        "#EF4444" if c == "Low" else "#F59E0B" if c == "Medium" else "#10B981"
        for c in probabilities
    ]
    fig = go.Figure(
        go.Bar(
            x=list(probabilities),
            y=list(probabilities.values()),
            marker_color=colors,
            marker_line_width=1,
            marker_line_color="#E2E8F0",
            text=[f"{p:.1%}" for p in probabilities.values()],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0F172A", family="sans-serif")),
        yaxis=dict(range=[0, 1], title="Model Probability", gridcolor="#F1F5F9"),
        xaxis=dict(title="", showgrid=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=40, b=20),
        height=230,
    )
    return fig


def _render_section_2_wellbeing_result(result: dict, response: dict, total_count: int) -> None:
    """Render Section 2 – Applicant Financial Well-being Result."""
    st.markdown("### Section 2 – Applicant Financial Well-being Result")
    pred = result["predicted_class"]
    conf = result["confidence"]

    badge_color = "#DC2626" if pred == "Low" else "#D97706" if pred == "Medium" else "#059669"
    badge_bg = "#FEF2F2" if pred == "Low" else "#FFFBEB" if pred == "Medium" else "#ECFDF5"
    level_explanation = {
        "Low": (
            "The applicant's reported indicators align with the Low financial well-being band "
            "(CFPB score ≤ 49). Closer review of emergency buffers, cash-flow margin, and debt load is advised."
        ),
        "Medium": (
            "The applicant's reported indicators align with the Medium financial well-being band "
            "(CFPB score 50–57). Moderate day-to-day financial stability is indicated, with specific capability areas to review."
        ),
        "High": (
            "The applicant's reported indicators align with the High financial well-being band "
            "(CFPB score ≥ 58). Strong saving, budgeting, and planning behaviors are indicated."
        ),
    }.get(pred, "")

    res_col1, res_col2 = st.columns([1.15, 1.85])
    with res_col1:
        st.metric("Predicted financial well-being category", pred)
        st.metric("Model prediction probability", f"{conf:.1%}")
        st.markdown(
            f"""
            <div style="background: {badge_bg}; border-left: 4px solid {badge_color};
                        padding: 10px 12px; border-radius: 6px; margin-top: 6px; font-size: 0.85rem; color: #1E293B;">
                <b>Visual Indicator: <span style="color: {badge_color};">{pred} Financial Well-being</span></b><br/>
                {html.escape(level_explanation)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"📌 **Label Notice**: The percentage above is the **model prediction probability** for the "
            f"applicant's financial well-being class ({len(response)} of {total_count} questions answered). "
            f"It is **not** a loan repayment probability or default probability."
        )
        if result["low_confidence"]:
            st.warning("Model prediction probability is below 50%. Treat this classification with extra caution.")
    with res_col2:
        st.plotly_chart(probability_chart(result["probabilities"]), use_container_width=True)

    missing_count = total_count - len(response)
    if missing_count == 0:
        st.success("✅ **Complete Questionnaire**: All 31 approved indicator questions were answered directly.")
    else:
        st.warning(
            f"⚠️ **Completeness Notice**: {missing_count} of {total_count} questions were left unanswered. "
            "Standard training-set median baseline values were imputed by the pipeline."
        )


def _render_section_3_capability_profile(
    gaps: list,
    strengths: dict,
    officer_guidance: dict,
) -> None:
    """Render Section 3 – Financial Capability Profile (Four Attractive Cards)."""
    st.markdown("### Section 3 – Financial Capability Profile (Four Core Dimensions)")
    st.caption(
        "Evaluated independently across Saving, Budgeting, Basic Financial Knowledge, and Financial Goal Planning "
        "using transparent CFPB codebook rules."
    )

    meta = {
        GROUP_SAVING: {"icon": "🐷", "color": "#D97706", "label": "Saving"},
        GROUP_BUDGETING: {"icon": "👛", "color": "#2563EB", "label": "Budgeting"},
        GROUP_KNOWLEDGE: {"icon": "📖", "color": "#059669", "label": "Basic Financial Knowledge"},
        GROUP_GOALS: {"icon": "🎯", "color": "#0F172A", "label": "Financial Goal Planning"},
    }

    gap_dict = {g.area: g for g in gaps}
    ordered_areas = [GROUP_SAVING, GROUP_BUDGETING, GROUP_KNOWLEDGE, GROUP_GOALS]

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    grid_cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for i, area in enumerate(ordered_areas):
        col = grid_cols[i]
        g = gap_dict.get(area)
        m = meta.get(area, {"icon": "📊", "color": "#0F172A", "label": area})
        area_strengths = strengths.get(area, [])
        guidance_txt = officer_guidance.get(area, "")

        if g is None or g.status == STATUS_INSUFFICIENT:
            badge_html = '<span class="fc-badge badge-insufficient">Insufficient Data</span>'
            assess_txt = "Insufficient answers provided"
        elif g.status == STATUS_GAP:
            if "Stronger" in g.strength:
                badge_html = '<span class="fc-badge badge-high-priority">Attention Needed (Stronger Signal)</span>'
            else:
                badge_html = '<span class="fc-badge badge-lower-priority">Attention Needed (1 Indicator)</span>'
            assess_txt = f"{len(g.triggered)} capability gap indicator(s) flagged"
        else:
            badge_html = '<span class="fc-badge badge-good">Healthy Indicators</span>'
            assess_txt = "No capability gap indicators flagged"

        strengths_html = (
            "".join(f"<li>{html.escape(s)}</li>" for s in area_strengths)
            if area_strengths
            else "<li><i>No positive strength triggers flagged from current responses.</i></li>"
        )
        gaps_html = (
            "".join(f"<li>{html.escape(r)}</li>" for r in g.triggered)
            if (g and g.triggered)
            else "<li><i>No gap indicators triggered under current codebook rules.</i></li>"
        )

        card_html = f"""
        <div class="fc-insight-card" style="min-height: 285px;">
            <div class="fc-insight-header">
                <span class="fc-insight-title">{m['icon']} {html.escape(m['label'])}</span>
                {badge_html}
            </div>
            <div style="font-size: 0.86rem; font-weight: 600; color: {m['color']}; margin-bottom: 8px;">
                Current Assessment: {html.escape(assess_txt)}
            </div>
            <div style="font-size: 0.82rem; color: #15803D; font-weight: 600; margin-top: 6px;">👍 Strengths:</div>
            <ul style="font-size: 0.80rem; color: #334155; margin: 2px 0 8px 0; padding-left: 18px;">
                {strengths_html}
            </ul>
            <div style="font-size: 0.82rem; color: #B45309; font-weight: 600;">⚠️ Gap Areas:</div>
            <ul style="font-size: 0.80rem; color: #334155; margin: 2px 0 8px 0; padding-left: 18px;">
                {gaps_html}
            </ul>
            <div style="background: #F8FAFC; border-left: 3px solid {m['color']}; padding: 7px 10px;
                        border-radius: 4px; font-size: 0.80rem; color: #1E293B; margin-top: 6px;">
                <b>🏦 Practical Guidance for Bank Officer:</b> {html.escape(guidance_txt)}
            </div>
        </div>
        """
        col.markdown(card_html, unsafe_allow_html=True)


def _render_section_4_shap_explanation(bundle, response: dict, result: dict) -> list[str]:
    """Render Section 4 – SHAP Explanation (Individual + Global summary)."""
    st.markdown("### Section 4 – SHAP Explanation (Key Prediction Factors)")
    st.caption(
        "Explains which applicant answers pushed the machine-learning model toward or away from the predicted "
        "financial well-being level."
    )

    shap_reasons: list[str] = []
    col_local, col_global = st.columns([1.25, 1])

    with col_local:
        try:
            explainer = get_explainer()
            row_df = validate_response(response)
            local_res = explainer.explain_individual(row_df)
            shap_reasons = describe_local_explanation(local_res, top_n=5)

            with st.container(border=True):
                st.markdown(
                    f"**Individual Applicant Explanation (Predicted Class: `{result['predicted_class']}`)**"
                )
                for r in shap_reasons:
                    st.markdown(f"• {r}")
                st.caption(
                    "⚠️ *SHAP describes how each answer influenced the model's prediction. "
                    "It does not prove causation.*"
                )
        except Exception as exc:
            st.caption(f"SHAP explanation details unavailable: {exc}")

    with col_global:
        with st.container(border=True):
            st.markdown("**Top Global Model Drivers (Across All Survey Respondents)**")
            imp = pd.DataFrame(bundle.metadata.get("global_importance", [])).head(5)
            if not imp.empty:
                for _, row_imp in imp.iterrows():
                    st.markdown(
                        f"• **{row_imp['label']}** (*{row_imp['group']}*) — Mean |SHAP|: `{row_imp['mean_abs_shap']:.3f}`"
                    )
            st.caption("Open the **SHAP Explanation** tab above for full interactive local and global charts.")

    return shap_reasons


def _render_section_5_resilience_info(resilience_items: list) -> None:
    """Render Section 5 – Financial Resilience Information (Supported by Actual Survey Variables)."""
    st.markdown("### Section 5 – Financial Resilience Information")
    st.caption(
        "Derived strictly from supported CFPB survey variables in the 31 approved features "
        "(liquid savings buffer, saving habit, bill timeliness, budget control, and planning horizon)."
    )

    cols = st.columns(3)
    for idx, item in enumerate(resilience_items):
        badge_cls = (
            "badge-good"
            if item.status == "Supportive"
            else "badge-high-priority"
            if item.status == "Watch"
            else "badge-insufficient"
        )
        card_html = f"""
        <div class="fc-secondary-card" style="min-height: 145px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; font-size: 0.90rem; color: #0F172A;">
                    {html.escape(item.indicator_label)}
                </span>
                <span class="fc-badge {badge_cls}">{html.escape(item.status)}</span>
            </div>
            <div style="font-size: 0.85rem; color: #1E3A8A; font-weight: 600; margin-bottom: 5px;">
                Response: {html.escape(item.applicant_response)}
            </div>
            <div style="font-size: 0.79rem; color: #475569;">
                {html.escape(item.officer_note)}
            </div>
        </div>
        """
        cols[idx % 3].markdown(card_html, unsafe_allow_html=True)


def _render_section_6_whatif_preview(bundle, response: dict, result: dict) -> None:
    """Render Section 6 – What-if Simulation preview."""
    st.markdown("### Section 6 – What-if Simulation (Hypothetical Scenario Check)")
    st.caption(
        "Simulate how hypothetical changes in the applicant's financial habits or liquid savings buffer would "
        "affect the model's financial well-being prediction (original applicant answers remain unchanged)."
    )

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            test_saving = st.selectbox(
                "Hypothetical change: Regular saving habit (SAVEHABIT)",
                options=[None, 1, 3, 5],
                format_func=lambda x: (
                    "Keep Applicant's Original Answer"
                    if x is None
                    else {1: "Strongly disagree (1)", 3: "Disagree slightly (3)", 5: "Agree strongly (5)"}[x]
                ),
                key="ui_quick_saving",
            )
        with c2:
            test_balance = st.selectbox(
                "Hypothetical change: Liquid savings balance (SAVINGSRANGES)",
                options=[None, 2, 4, 6],
                format_func=lambda x: (
                    "Keep Applicant's Original Answer"
                    if x is None
                    else {2: "$1 – $99", 4: "$1,000 – $4,999", 6: "$20,000 – $74,999"}[x]
                ),
                key="ui_quick_balance",
            )

        sim_changes = {}
        if test_saving is not None:
            sim_changes["SAVEHABIT"] = test_saving
        if test_balance is not None:
            sim_changes["SAVINGSRANGES"] = test_balance

        if sim_changes:
            sim_res = run_what_if(bundle, response, sim_changes)
            st.divider()
            r1, r2 = st.columns(2)
            with r1:
                st.metric(
                    "Original Applicant Prediction",
                    result["predicted_class"],
                    f"Model Probability: {result['confidence']:.1%}",
                )
            with r2:
                st.metric(
                    "Simulated Scenario Prediction",
                    sim_res["what_if"]["predicted_class"],
                    f"Model Probability: {sim_res['what_if']['confidence']:.1%}",
                )
            diff_text = ", ".join([f"{k}: {v:+.1%}" for k, v in sim_res["probability_change"].items()])
            st.caption(f"Simulated class probability shifts: {diff_text}")
        else:
            st.caption(
                "Select a hypothetical answer above (or open the **What-if Analysis** tab) to compare original vs. simulated outcomes."
            )

        st.caption(
            "⚠️ *Simulation Disclaimer: What-if results show how the model responds to hypothetical inputs. "
            "They do not guarantee real-world outcomes or loan repayment performance.*"
        )


def _render_section_7_loan_support_summary(
    result: dict,
    gaps: list,
    strengths: dict,
    resilience_items: list,
    officer_questions: list[str],
) -> None:
    """Render Section 7 – Loan Assessment Support Summary."""
    st.markdown("### Section 7 – Loan Assessment Support Summary")

    all_strengths = [s for area_list in strengths.values() for s in area_list]
    gap_areas = [g for g in gaps if g.status == STATUS_GAP]
    watch_resilience = [r for r in resilience_items if r.status == "Watch"]
    supportive_resilience = [r for r in resilience_items if r.status == "Supportive"]

    with st.container(border=True):
        st.markdown("#### 🏦 Component 04 – Additional Financial Well-being Information for Loan Assessment")
        st.caption(
            "Supplementary summary prepared for the Bank Officer to review alongside the bank's normal "
            "identity/income verification, credit history/score, and repayment capacity (DTI) checks."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**1. Financial Strengths**")
            if all_strengths:
                for s in all_strengths[:6]:
                    st.markdown(f"- ✅ {s}")
            else:
                st.markdown("- *No distinct capability strengths triggered from the current responses.*")

            st.markdown("**2. Financial Resilience Summary**")
            st.markdown(
                f"- **Supportive Indicators**: `{len(supportive_resilience)}` of `{len(resilience_items)}` "
                f"| **Watch Indicators**: `{len(watch_resilience)}` of `{len(resilience_items)}`"
            )
            for item in resilience_items[:3]:
                st.markdown(f"- **{item.indicator_label}**: {item.applicant_response} (*{item.status}*)")

        with col_b:
            st.markdown("**3. Areas for Further Discussion**")
            if gap_areas:
                for g in gap_areas:
                    reasons_joined = "; ".join(g.triggered)
                    st.markdown(f"- ⚠️ **{g.area}** ({g.strength}): {reasons_joined}")
            else:
                st.markdown("- ✅ *No capability gap areas flagged under the codebook rules.*")

            st.markdown("**4. Suggested Questions for the Bank Officer**")
            for q in officer_questions:
                st.markdown(f"- 💬 *{q}*")

        st.info(f"ℹ️ **Bank Officer Decision Governance**: {BANK_LOAN_NOTICE}")


def _render_section_8_report_downloads(
    bundle,
    response: dict,
    result: dict,
    gaps: list,
    shap_reasons: list[str],
    strengths: dict,
    resilience_items: list,
    officer_questions: list[str],
    officer_guidance: dict,
) -> None:
    """Render Section 8 – Assessment Report (Preview + PDF & TXT Downloads)."""
    st.markdown("### Section 8 – Financial Well-being Assessment Report (PDF & TXT)")
    st.write(
        "Generate and download the formal **Component 04 Financial Well-being Assessment Report** to attach "
        "as supplementary supporting documentation for the loan file."
    )

    selected_model_name = bundle.metadata.get("selected_model", "XGBoost")
    today = date.today()

    pdf_data = build_capability_assessment_pdf(
        response=response,
        result=result,
        gaps=gaps,
        shap_reasons=shap_reasons,
        assessment_date=today,
        strengths=strengths,
        resilience_items=resilience_items,
        officer_questions=officer_questions,
        officer_guidance=officer_guidance,
        selected_model_name=selected_model_name,
    )

    txt_data = build_capability_assessment_txt(
        response=response,
        result=result,
        gaps=gaps,
        shap_reasons=shap_reasons,
        assessment_date=today,
        strengths=strengths,
        resilience_items=resilience_items,
        officer_questions=officer_questions,
        officer_guidance=officer_guidance,
        selected_model_name=selected_model_name,
    )

    with st.expander("📋 Preview Assessment Report Content (Component 04)", expanded=False):
        st.markdown(f"**Report Title**: `{REPORT_TITLE}`")
        st.code(txt_data, language="markdown")

    d1, d2, _ = st.columns([1.6, 1.6, 2.8])
    with d1:
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_data,
            file_name=f"component04_wellbeing_report_{today.isoformat()}.pdf",
            mime="application/pdf",
            type="primary",
            key="btn_download_assessment_pdf",
        )
    with d2:
        st.download_button(
            label="📝 Download Report (TXT)",
            data=txt_data,
            file_name=f"component04_wellbeing_report_{today.isoformat()}.txt",
            mime="text/plain",
            key="btn_download_assessment_txt",
        )


def render() -> None:
    """Render the main Bank Officer Dashboard with all 8 structured decision-support sections."""
    st.subheader("Section 1 – Bank Officer Dashboard: Applicant Financial Well-being Assessment")
    bundle = require_bundle()
    if bundle is None:
        return

    st.markdown(
        f"**{COMPONENT_TITLE}**  \n"
        "Use this interface to assess a loan applicant's financial well-being and financial capability profile "
        "as **supplementary supporting information** alongside the bank's standard credit, income, and affordability checks."
    )
    st.caption(DISCLAIMER)

    # Count answered questions
    answered_count = sum(
        1 for s in FEATURE_SPECS
        if st.session_state.get(WIDGET_PREFIX + s.name) is not None
    )
    total_count = len(FEATURE_SPECS)
    progress_val = min(1.0, max(0.0, answered_count / total_count))

    # Completion Progress Bar
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        st.progress(progress_val)
    with p_col2:
        st.markdown(f"**Questionnaire Progress: {answered_count} of {total_count}** ({progress_val:.0%})")

    st.markdown(
        "Enter the loan applicant's questionnaire responses below. Unanswered items default to training-set "
        "median values during inference. Responses remain in the active browser session only."
    )

    b1, b2, _ = st.columns([1.2, 1.2, 2.6])
    b1.button("Use example answers", on_click=_fill_example)
    b2.button("Clear answers", on_click=_clear_answers)

    # Form with 5 structured groups
    with st.form("fc_assessment_form"):
        groups = [
            (GROUP_SAVING, "🐷 Saving Indicators", True),
            (GROUP_BUDGETING, "👛 Budgeting & Bill Management Indicators", True),
            (GROUP_KNOWLEDGE, "📖 Basic Financial Knowledge Indicators", True),
            (GROUP_GOALS, "🎯 Financial Goal Planning Indicators", True),
            (GROUP_CONTEXT, "👤 Applicant Background Context (Demographics & Household)", False),
        ]

        for grp_id, title, expanded in groups:
            specs = [s for s in FEATURE_SPECS if s.group == grp_id]
            with st.expander(f"{title} ({len(specs)} questions)", expanded=expanded):
                cols = st.columns(2)
                for i, spec in enumerate(specs):
                    with cols[i % 2]:
                        _question(spec)

        submitted = st.form_submit_button("Run assessment", type="primary")

    if submitted:
        response = {
            s.name: st.session_state.get(WIDGET_PREFIX + s.name)
            for s in FEATURE_SPECS
            if st.session_state.get(WIDGET_PREFIX + s.name) is not None
        }
        try:
            predict_assessment(bundle, response)
        except InvalidResponseError as exc:
            st.error(f"Invalid answer: {exc}")
        else:
            st.session_state[SESSION_RESPONSE_KEY] = response

    response = st.session_state.get(SESSION_RESPONSE_KEY)
    if not response:
        st.info("Click **Run assessment** above to generate the Applicant Financial Well-being Assessment and Bank Officer Decision Support summary.")
        return

    result = predict_assessment(bundle, response)
    row_dict = validate_response(response).iloc[0].to_dict()
    gaps = identify_capability_gaps(row_dict)
    strengths = identify_capability_strengths(row_dict)
    resilience_items = extract_resilience_indicators(row_dict)
    officer_guidance = generate_officer_guidance(gaps)
    officer_questions = generate_officer_questions(gaps, row_dict)

    st.divider()

    # Section 2 – Applicant Financial Well-being Result
    _render_section_2_wellbeing_result(result, response, total_count)

    st.divider()

    # Section 3 – Financial Capability Profile (4 Cards)
    _render_section_3_capability_profile(gaps, strengths, officer_guidance)

    st.divider()

    # Section 4 – SHAP Explanation
    shap_reasons = _render_section_4_shap_explanation(bundle, response, result)

    st.divider()

    # Section 5 – Financial Resilience Information
    _render_section_5_resilience_info(resilience_items)

    st.divider()

    # Section 6 – What-if Simulation
    _render_section_6_whatif_preview(bundle, response, result)

    st.divider()

    # Section 7 – Loan Assessment Support Summary
    _render_section_7_loan_support_summary(
        result=result,
        gaps=gaps,
        strengths=strengths,
        resilience_items=resilience_items,
        officer_questions=officer_questions,
    )

    st.divider()

    # Section 8 – Assessment Report (Preview + PDF & TXT Downloads)
    _render_section_8_report_downloads(
        bundle=bundle,
        response=response,
        result=result,
        gaps=gaps,
        shap_reasons=shap_reasons,
        strengths=strengths,
        resilience_items=resilience_items,
        officer_questions=officer_questions,
        officer_guidance=officer_guidance,
    )

