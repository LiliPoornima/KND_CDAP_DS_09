"""Individual Assessment section: Redesigned modern fintech assessment & capability insights."""

from __future__ import annotations

from datetime import date
import html
import plotly.graph_objects as go
import streamlit as st

from backend.src.financial_capability.fc_capability_rules import (
    STATUS_GAP,
    STATUS_INSUFFICIENT,
    STATUS_NONE,
    identify_capability_gaps,
    map_gaps_to_categories,
)
from backend.src.financial_capability.fc_config import (
    CAPABILITY_GROUPS,
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
    DISCLAIMER,
    EXAMPLE_RESPONSE,
    SESSION_RESPONSE_KEY,
    WIDGET_PREFIX,
)
from frontend.financial_capability.data_loader import get_explainer, require_bundle
from frontend.financial_capability.pdf_report import build_capability_assessment_pdf


def _fill_example() -> None:
    """Callback: fill illustrative answers into form widgets."""
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


def probability_chart(probabilities: dict, title: str = "Class Probabilities") -> go.Figure:
    """Create a modern fintech probability distribution bar chart."""
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
        yaxis=dict(range=[0, 1], title="Probability", gridcolor="#F1F5F9"),
        xaxis=dict(title="", showgrid=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
    )
    return fig


def _render_insights_grid(gaps: list) -> None:
    """Render the 2x2 Financial Capability Insights card grid (matching user mockup Image 1)."""
    st.markdown("### Financial Capability Insights")
    st.caption("Your personalized assessment results • Phase 1")

    # Map areas to specific metadata
    meta = {
        GROUP_SAVING: {"icon": "🐷", "color": "#D97706", "label": "Saving"},
        GROUP_BUDGETING: {"icon": "👛", "color": "#2563EB", "label": "Budgeting"},
        GROUP_GOALS: {"icon": "🎯", "color": "#0F172A", "label": "Goal Planning"},
        GROUP_KNOWLEDGE: {"icon": "📖", "color": "#059669", "label": "Financial Knowledge"},
    }

    gap_dict = {g.area: g for g in gaps}
    ordered_areas = [GROUP_SAVING, GROUP_BUDGETING, GROUP_GOALS, GROUP_KNOWLEDGE]

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    grid_cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for i, area in enumerate(ordered_areas):
        col = grid_cols[i]
        g = gap_dict.get(area)
        m = meta.get(area, {"icon": "📊", "color": "#0F172A", "label": area})

        if g is None or g.status == STATUS_INSUFFICIENT:
            issue_text = "N/A"
            subtext = "Insufficient information"
            badge_html = '<span class="fc-badge badge-insufficient">Incomplete</span>'
            count_color = "#64748B"
        elif g.status == STATUS_GAP:
            n_issues = len(g.triggered)
            issue_text = f"{n_issues} issue" if n_issues == 1 else f"{n_issues} issues"
            subtext = "Possible improvement indicator" if n_issues == 1 else "Possible improvement indicators"
            count_color = m["color"]
            if "Stronger" in g.strength:
                badge_html = '<span class="fc-badge badge-high-priority">High priority</span>'
            else:
                badge_html = '<span class="fc-badge badge-lower-priority">Lower priority</span>'
        else:
            issue_text = "0 issues"
            subtext = "No improvement area indicated"
            count_color = "#059669"
            badge_html = '<span class="fc-badge badge-good">Good indicators</span>'

        card_html = f"""
        <div class="fc-insight-card">
            <div class="fc-insight-header">
                <span class="fc-insight-title">{m['icon']} {m['label']}</span>
                {badge_html}
            </div>
            <div class="fc-insight-count" style="color: {count_color};">{issue_text}</div>
            <div class="fc-insight-label">{subtext}</div>
        </div>
        """
        col.markdown(card_html, unsafe_allow_html=True)


def _render_priorities_section(gaps: list) -> None:
    """Render Your Top Improvement Priority hero card and secondary cards (matching user mockup Images 1, 2, 3)."""
    # Filter gap areas and healthy areas
    gap_areas = [g for g in gaps if g.status == STATUS_GAP]
    healthy_areas = [g for g in gaps if g.status == STATUS_NONE]

    # Sort gaps: Stronger signal first, then Weaker signal
    gap_areas.sort(key=lambda x: 0 if "Stronger" in x.strength else 1)

    if not gap_areas and not healthy_areas:
        st.info("Insufficient information across all capability areas. Please answer more questions.")
        return

    # Render Top Priority (Hero Card)
    if gap_areas:
        top = gap_areas[0]
        st.markdown("### Your Top Improvement Priority")
        
        icon = "🐷" if top.area == GROUP_SAVING else "👛" if top.area == GROUP_BUDGETING else "🎯" if top.area == GROUP_GOALS else "📖"
        badge_text = "Priority 01 • Stronger signal" if "Stronger" in top.strength else "Priority 01 • Weaker signal"

        reasons_list = "".join([f"<li>{html.escape(r)}</li>" for r in top.triggered])

        hero_html = f"""
        <div class="fc-hero-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div style="display: flex; gap: 12px; align-items: center;">
                    <div style="background: #FEF3C7; padding: 10px 14px; border-radius: 10px; font-size: 1.4rem;">
                        {icon}
                    </div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">{html.escape(top.area)} habits</div>
                        <div style="font-size: 0.85rem; color: #64748B;">{badge_text}</div>
                    </div>
                </div>
                <span class="fc-badge fc-hero-badge">Focus first</span>
            </div>
            <div style="font-weight: 600; color: #0F172A; margin: 12px 0 6px 0;">Why was this identified?</div>
            <ul style="color: #334155; margin-bottom: 14px; padding-left: 20px;">
                {reasons_list}
            </ul>
            <div class="fc-hero-box">
                <div style="font-weight: 600; color: #92400E; margin-bottom: 4px;">Your suggested focus</div>
                <div style="color: #78350F; font-size: 0.9rem;">
                    {html.escape(top.area)}: building regular habits and understanding effective practical methods.
                </div>
            </div>
        </div>
        """
        st.markdown(hero_html, unsafe_allow_html=True)

    # Render Other Improvement Areas
    other_gaps = gap_areas[1:]
    if other_gaps:
        st.markdown("### Your Other Improvement Areas")
        for g in other_gaps:
            icon = "👛" if g.area == GROUP_BUDGETING else "🎯" if g.area == GROUP_GOALS else "🐷" if g.area == GROUP_SAVING else "📖"
            badge = '<span class="fc-badge badge-high-priority">High priority</span>' if "Stronger" in g.strength else '<span class="fc-badge badge-lower-priority">Lower priority</span>'
            reasons_summary = "; ".join(g.triggered)

            card_html = f"""
            <div class="fc-secondary-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 1.05rem; color: #0F172A;">{icon} {html.escape(g.area)}</div>
                    {badge}
                </div>
                <div style="color: #475569; font-size: 0.9rem; margin-bottom: 8px;">
                    {html.escape(reasons_summary)}
                </div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #2563EB;">
                    Suggested focus: {html.escape(g.learning_category)}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # Render Healthy Areas (Strengths)
    if healthy_areas:
        st.markdown("### Healthy Areas & Strengths")
        for h in healthy_areas:
            icon = "✅"
            healthy_html = f"""
            <div class="fc-healthy-card">
                <div style="display: flex; gap: 8px; align-items: center; font-weight: 600; font-size: 1.05rem; color: #15803D; margin-bottom: 6px;">
                    <span>{icon}</span> <span>{html.escape(h.area)}</span>
                </div>
                <div style="color: #166534; font-size: 0.9rem;">
                    Your current answers do not indicate an improvement area under the selected rules.
                </div>
            </div>
            """
            st.markdown(healthy_html, unsafe_allow_html=True)


def _render_shap_local_summary(bundle, response: dict, result: dict) -> None:
    """Render plain-language SHAP drivers and attribution summary."""
    st.markdown("### Why Did the Model Predict This?")
    st.caption("SHAP attributions reveal which answers had the greatest influence on this prediction.")

    try:
        explainer = get_explainer()
        row_df = validate_response(response)
        local_res = explainer.explain_individual(row_df)
        reasons = describe_local_explanation(local_res, top_n=4)

        with st.container(border=True):
            st.markdown(f"**Key factors influencing the {result['predicted_class']} prediction:**")
            for r in reasons:
                st.markdown(f"• {r}")
            st.caption(
                "⚠️ *SHAP explains how individual answers influence the model's prediction. "
                "These explanations do not prove that one factor causes a person's financial well-being.*"
            )
    except Exception as exc:
        st.caption(f"SHAP explanation details unavailable: {exc}")


def _render_learning_categories(gaps: list) -> None:
    """Render the recommended learning categories with Phase 1 disclaimer."""
    st.markdown("### Recommended Learning Categories")
    categories = map_gaps_to_categories(gaps)

    if categories:
        cols = st.columns(len(categories))
        for i, cat in enumerate(categories):
            with cols[i].container(border=True):
                st.markdown(f"#### 📚 {cat}")
                st.caption("Identified based on your reported behavioral indicators.")
    else:
        st.success("No critical learning categories identified under current rules.")

    st.info(
        "ℹ️ **Phase 1 Notice**: Phase 1 identifies suitable learning categories only. "
        "Actual lessons, videos, quizzes, and learning progress tracking are planned for Phase 2."
    )


def _render_whatif_preview(bundle, response: dict, result: dict) -> None:
    """Render interactive quick what-if simulation preview."""
    st.markdown("### What-if Simulation: Test Potential Improvements")
    st.write("See how modifying specific capability behaviors impacts your well-being prediction:")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            test_saving = st.selectbox(
                "What if you adopted a regular saving habit?",
                options=[None, 1, 3, 5],
                format_func=lambda x: "Current Answer" if x is None else {1: "Disagree strongly", 3: "Somewhat agree", 5: "Agree strongly"}[x],
                key="ui_quick_saving",
            )
        with c2:
            test_balance = st.selectbox(
                "What if your savings balance increased?",
                options=[None, 2, 4, 6],
                format_func=lambda x: "Current Answer" if x is None else {2: "$100 - $999", 4: "$5,000 - $19,999", 6: "$75,000 or more"}[x],
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
                st.metric("Original Prediction", result["predicted_class"], f"{result['confidence']:.1%}")
            with r2:
                st.metric(
                    "Simulated Prediction",
                    sim_res["what_if"]["predicted_class"],
                    f"{sim_res['what_if']['confidence']:.1%}",
                )
            diff_text = ", ".join([f"{k}: {v:+.1%}" for k, v in sim_res["probability_change"].items()])
            st.caption(f"Probability shifts: {diff_text}")
        else:
            st.caption("Select an alternative answer above to preview simulated well-being shifts.")

        st.caption("⚠️ *Model-based scenario: What-if results do not prove that changing a behavior will cause real-world financial improvement.*")


def _render_download_section(bundle, response: dict, result: dict, gaps: list) -> None:
    """Render Step 8: Downloadable Assessment Report (PDF)."""
    st.markdown("### 📥 Download Assessment Report")
    st.write(
        "Keep a personal copy of your financial capability assessment results, "
        "including your predicted category, capability gaps, identified strengths, and SHAP drivers."
    )

    shap_reasons = []
    try:
        explainer = get_explainer()
        row_df = validate_response(response)
        local_res = explainer.explain_individual(row_df)
        shap_reasons = describe_local_explanation(local_res, top_n=4)
    except Exception:
        pass

    pdf_data = build_capability_assessment_pdf(
        response=response,
        result=result,
        gaps=gaps,
        shap_reasons=shap_reasons,
        assessment_date=date.today(),
    )

    d1, d2, _ = st.columns([1.5, 1.5, 3])
    with d1:
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_data,
            file_name=f"financial_capability_report_{date.today().isoformat()}.pdf",
            mime="application/pdf",
            type="primary",
            key="btn_download_assessment_pdf",
        )
    with d2:
        # Also provide a clean Markdown text summary for download
        summary_md = f"""# Financial Capability Assessment Report
Date: {date.today().isoformat()}
Model: {bundle.metadata.get('selected_model', 'XGBoost')}
Predicted Category: {result['predicted_class']} (Confidence: {result['confidence']:.1%})
Probabilities: Low {result['probabilities']['Low']:.1%}, Medium {result['probabilities']['Medium']:.1%}, High {result['probabilities']['High']:.1%}

## Identified Capability Areas:
""" + "\n".join([f"- {g.area}: {g.status} ({len(g.triggered)} issues)" for g in gaps])
        st.download_button(
            label="📝 Download Summary (TXT)",
            data=summary_md,
            file_name=f"financial_capability_summary_{date.today().isoformat()}.txt",
            mime="text/plain",
            key="btn_download_assessment_txt",
        )


def render() -> None:
    """Render the main Individual Assessment page with interactive questionnaire and complete capability report."""
    st.subheader("Individual Assessment")
    bundle = require_bundle()
    if bundle is None:
        return

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
        st.markdown(f"**{answered_count} of {total_count}** completed ({progress_val:.0%})")

    st.markdown(
        "Answer the questions below to evaluate your financial capability and well-being. "
        "Optional questions may be left as *Not answered* — standard median baseline values will be used. "
        "Answers stay in this browser session only; no personal data is stored."
    )

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Use example answers", on_click=_fill_example)
    b2.button("Clear answers", on_click=_clear_answers)

    # Form with 5 structured groups
    with st.form("fc_assessment_form"):
        groups = [
            (GROUP_SAVING, "🐷 Saving", True),
            (GROUP_BUDGETING, "👛 Budgeting", True),
            (GROUP_KNOWLEDGE, "📖 Basic Financial Knowledge", True),
            (GROUP_GOALS, "🎯 Financial Goal Planning", True),
            (GROUP_CONTEXT, "👤 Background Information (Context Only)", False),
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
        st.info("Press **Run assessment** to view your financial capability insights and report.")
        return

    result = predict_assessment(bundle, response)
    st.divider()

    # 1. Executive Prediction & Distribution
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Predicted financial well-being category", result["predicted_class"])
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        st.caption(f"{len(response)} of {total_count} questions answered.")
        if result["low_confidence"]:
            st.warning("The model confidence is below 50%. Treat this prediction with caution.")
    with res_col2:
        st.plotly_chart(probability_chart(result["probabilities"]), use_container_width=True)

    # 2. Data Completeness Notice
    missing_count = total_count - len(response)
    if missing_count == 0:
        st.success("✅ **Complete Assessment**: All 31 questions answered directly without imputation.")
    else:
        st.warning(
            f"⚠️ **Assessment Notice**: {missing_count} of {total_count} questions were left unanswered. "
            "Typical baseline values were imputed by the model. For maximum reliability, answer all questions."
        )

    # 3. Capability Gaps Evaluation
    row_dict = validate_response(response).iloc[0].to_dict()
    gaps = identify_capability_gaps(row_dict)

    st.divider()

    # 4. Render 2x2 Financial Capability Insights Grid (Mockup Image 1)
    _render_insights_grid(gaps)

    st.divider()

    # 5. Render Priorities (Mockup Images 1, 2, 3)
    _render_priorities_section(gaps)

    st.divider()

    # 6. Render SHAP Local Drivers
    _render_shap_local_summary(bundle, response, result)

    st.divider()

    # 7. Render Recommended Learning Categories
    _render_learning_categories(gaps)

    st.divider()

    # 8. Render Quick What-If Simulation
    _render_whatif_preview(bundle, response, result)

    st.divider()

    # 9. Download Assessment Report (PDF / TXT)
    _render_download_section(bundle, response, result, gaps)
