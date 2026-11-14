"""Stage 5 – Financial Capability Profile, Gaps, Strengths & Bank Officer Guidance (transparent rules)."""

from __future__ import annotations

import streamlit as st

from backend.src.financial_capability.fc_capability_rules import (
    RULES,
    STATUS_GAP,
    STATUS_INSUFFICIENT,
    extract_resilience_indicators,
    generate_officer_guidance,
    generate_officer_questions,
    identify_capability_gaps,
    identify_capability_strengths,
)
from backend.src.financial_capability.fc_predictor import validate_response
from frontend.financial_capability.data_loader import require_response


def render() -> None:
    st.subheader("Stage 5 – Financial Capability Profile & Bank Officer Guidance")
    st.info(
        "These findings come from **transparent codebook rules applied to the loan applicant's actual answers** "
        "(independent of SHAP). They highlight capability strengths, areas requiring further attention, "
        "supported financial resilience indicators, and neutral follow-up questions for the Bank Officer."
    )
    response = require_response()
    if response is None:
        return
    row = validate_response(response).iloc[0].to_dict()
    results = identify_capability_gaps(row)
    strengths = identify_capability_strengths(row)
    resilience_items = extract_resilience_indicators(row)
    officer_guidance = generate_officer_guidance(results)
    officer_questions = generate_officer_questions(results, row)

    flagged_areas = [r.area for r in results if r.is_gap]
    if flagged_areas:
        st.warning(
            "Areas requiring further attention during officer consultation: **"
            + ", ".join(flagged_areas)
            + "**"
        )
    elif all(r.status == STATUS_INSUFFICIENT for r in results):
        st.warning("Insufficient information to evaluate capability dimensions. Complete more questionnaire items.")
    else:
        st.success("No capability gap areas were flagged by the codebook rules for the answers provided.")

    st.markdown("### Four Financial Capability Dimensions")
    cols = st.columns(2)
    for i, r in enumerate(results):
        area_strengths = strengths.get(r.area, [])
        guidance_txt = officer_guidance.get(r.area, "")
        with cols[i % 2].container(border=True):
            st.markdown(f"#### {r.area}")
            if r.status == STATUS_GAP:
                st.warning(f"{r.status} — {r.strength}")
            elif r.status == STATUS_INSUFFICIENT:
                st.info(r.status)
            else:
                st.success(r.status)

            st.markdown("**Identified Strengths:**")
            if area_strengths:
                for s in area_strengths:
                    st.markdown(f"- ✅ {s}")
            else:
                st.caption("No distinct strength triggers flagged from current responses.")

            st.markdown("**Identified Gap Areas:**")
            if r.triggered:
                for reason in r.triggered:
                    st.markdown(f"- ⚠️ {reason}")
            else:
                st.caption("No capability gap triggers flagged.")

            st.markdown(f"**Bank Officer Practical Guidance:** {guidance_txt}")
            st.caption(f"{r.n_answered} of {r.n_indicators} indicator questions answered.")

    st.divider()
    col_res, col_q = st.columns(2)
    with col_res:
        st.markdown("### Financial Resilience Indicators")
        for item in resilience_items:
            st.markdown(
                f"- **{item.indicator_label}**: `{item.applicant_response}` "
                f"(*{item.status}* — {item.officer_note})"
            )
    with col_q:
        st.markdown("### Suggested Follow-up Questions for the Bank Officer")
        for q in officer_questions:
            st.markdown(f"- 💬 *{q}*")

    with st.expander("Codebook Rule Definitions (Transparent Capability Thresholds)"):
        for area, indicators in RULES.items():
            st.markdown(f"**{area}**")
            for ind in indicators:
                st.markdown(f"- `{ind.variable}`: {ind.reason}")


