"""Respondent-level assessment UI backed by the existing scoring pipeline."""

from __future__ import annotations

from datetime import date
from html import escape
from io import BytesIO
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .. import data_loader as loader
from ..constants import CLUSTER_LABELS


INPUT_LABELS = {
    "con1": "Personal mobile phone ownership",
    "internet_use": "Recent internet use",
    "account": "Financial account ownership",
    "anydigpayment": "Digital payments",
    "con18": "Phone security protection",
    "saved": "Saving money",
    "fin24a": "Ability to find emergency funds",
    "fin24b": "Household income-loss buffer",
}


def _profile_individual():
    root = Path(__file__).resolve().parents[3]
    for folder in ("02_dimension_creation", "03_clustering"):
        path = root / "backend" / "src" / "financial_inclusion" / folder
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return import_module("fi_individual_profiling").profile_individual


def _assessment_form() -> tuple[bool, dict[str, Any]]:
    yes_no_options = {
        "Not available": None,
        "Yes": 1,
        "No": 2,
    }
    binary_options = {
        "Not available": None,
        "Yes": 1,
        "No": 0,
    }
    emergency_options = {
        "Not available": None,
        "Very difficult": 1,
        "Somewhat difficult": 2,
        "Not difficult at all": 3,
        "Don't know": 4,
        "Prefer not to answer": 5,
    }
    income_loss_options = {
        "Not available": None,
        "Less than 2 weeks": 1,
        "About 1 month": 2,
        "About 2 months": 3,
        "More than 2 months": 4,
        "Don't know": 5,
        "Prefer not to answer": 6,
    }

    with st.form("fi_individual_profile_form", border=True):
        st.markdown("### 1. Access and connectivity")
        st.caption(
            "Tell us about the respondent's access to a phone and the internet."
        )
        col1, col2 = st.columns(2)
        with col1:
            mobile_answer = st.selectbox(
                "Does the respondent personally own a mobile phone?",
                options=list(yes_no_options),
                index=0,
                help=(
                    "Choose Yes if the respondent owns a mobile phone, No if "
                    "they do not, or Not available if this information is unknown."
                ),
            )
        with col2:
            internet_answer = st.selectbox(
                "Has the respondent used the internet recently?",
                options=list(binary_options),
                index=0,
                help=(
                    "Use the validated internet-use indicator from your prepared "
                    "dataset. Yes means the indicator is 1; No means it is 0."
                ),
            )

        st.divider()
        st.markdown("### 2. Financial access and digital activity")
        st.caption("Describe account access, digital payments, and phone protection.")
        col3, col4 = st.columns(2)
        with col3:
            account_answer = st.selectbox(
                "Does the respondent have a financial account?",
                options=list(binary_options),
                index=0,
                help="Use the validated account indicator. Yes = 1; No = 0.",
            )
            payment_answer = st.selectbox(
                "Has the respondent made or received digital payments?",
                options=list(binary_options),
                index=0,
                help=(
                    "Use the validated digital-payment indicator. Yes = 1; No = 0."
                ),
            )
        with col4:
            protection_answer = st.selectbox(
                "Does the respondent use phone security protection?",
                options=list(yes_no_options),
                index=0,
                help="Use the project's validated phone-protection coding: Yes = 1 and No = 2.",
            )
            saving_answer = st.selectbox(
                "Has the respondent saved money?",
                options=list(binary_options),
                index=0,
                help="Use the validated saving indicator. Yes = 1; No = 0.",
            )

        st.divider()
        st.markdown("### 3. Financial resilience")
        st.caption(
            "These questions help describe how the respondent could manage "
            "an unexpected expense or loss of income."
        )
        resilience_col1, resilience_col2 = st.columns(2)
        with resilience_col1:
            st.markdown("**Emergency funds within 30 days**")
            st.caption(
                "How difficult would it be to find emergency money within "
                "the next 30 days?"
            )
            emergency_answer = st.selectbox(
                "Ability to find emergency funds",
                options=list(emergency_options),
                index=0,
                help=(
                    "Very difficult, Somewhat difficult, or Not difficult at all "
                    "are the three substantive responses. The question may only "
                    "have been asked of respondents who reported a possible "
                    "source of emergency funds."
                ),
            )
        with resilience_col2:
            st.markdown("**Coping with loss of income**")
            st.caption(
                "If the household lost its main income source, how long could "
                "it cover expenses?"
            )
            income_loss_answer = st.selectbox(
                "How long could expenses be covered?",
                options=list(income_loss_options),
                index=0,
                help=(
                    "Choose the closest response: less than 2 weeks, about "
                    "1 month, about 2 months, or more than 2 months."
                ),
            )

        st.info(
            "Choose 'Not available' if the respondent was not asked a question "
            "or the answer is unknown. Do not select 'No' as a substitute for "
            "missing information. 'Don't know' and 'Prefer not to answer' are "
            "separate responses where provided."
        )
        submitted = st.form_submit_button(
            "Calculate inclusion profile",
            type="primary",
            use_container_width=True,
        )

    entries = {
        "con1": yes_no_options[mobile_answer],
        "internet_use": binary_options[internet_answer],
        "account": binary_options[account_answer],
        "anydigpayment": binary_options[payment_answer],
        "con18": yes_no_options[protection_answer],
        "saved": binary_options[saving_answer],
        "fin24a": emergency_options[emergency_answer],
        "fin24b": income_loss_options[income_loss_answer],
    }
    return submitted, entries


def _dimension_rows(scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for column in loader.DIMENSION_COLUMNS:
        value = scores.get(column)
        try:
            score = float(value) if value is not None else None
        except (TypeError, ValueError):
            score = None
        if score is not None and (not pd.notna(score) or not 0 <= score <= 1):
            score = None
        rows.append(
            {"Dimension": loader.DIMENSION_LABELS[column], "Score": score}
        )
    return rows


def _dimension_extrema(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return all tied highest and lowest available dimension scores."""
    available = [row for row in rows if row["Score"] is not None]
    if not available:
        return [], []
    highest = max(row["Score"] for row in available)
    lowest = min(row["Score"] for row in available)
    tolerance = 1e-9
    strongest = [row for row in available if abs(row["Score"] - highest) <= tolerance]
    weakest = [row for row in available if abs(row["Score"] - lowest) <= tolerance]
    return strongest, weakest


def _overall_dimension_score(rows: list[dict[str, Any]]) -> float | None:
    """Return the existing five-score mean on a display-only 0–100 scale."""
    scores = [row["Score"] for row in rows if row["Score"] is not None]
    if len(scores) != len(loader.DIMENSION_COLUMNS):
        return None
    return sum(scores) / len(scores) * 100


def _actions_for_available_scores(
    result: dict[str, Any], rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], float]]:
    scores = {row["Dimension"].casefold(): row["Score"] for row in rows}
    actions = result.get("explanation", {}).get("suggested_actions", [])
    selected = []
    for action in actions:
        dimension = str(action.get("dimension", "")).casefold()
        score = scores.get(dimension)
        if score is not None:
            selected.append((action, score))
    return selected


def _information_notes(entries: dict[str, Any]) -> list[str]:
    notes = []
    for key, value in entries.items():
        if value is None:
            note = "Not available"
        elif key == "fin24a" and value == 4:
            note = "Don't know"
        elif key == "fin24a" and value == 5:
            note = "Prefer not to answer"
        elif key == "fin24b" and value == 5:
            note = "Don't know"
        elif key == "fin24b" and value == 6:
            note = "Prefer not to answer"
        else:
            continue
        notes.append(f"{INPUT_LABELS[key]}: {note}")
    return notes


def build_assessment_pdf(
    result: dict[str, Any],
    entries: dict[str, Any],
    assessment_date: date,
) -> bytes:
    """Create a concise, findings-led PDF from this submitted assessment."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        KeepTogether,
        LongTable,
        Paragraph,
        SimpleDocTemplate,
        TableStyle,
    )
    import reportlab

    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    if "AssessmentVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("AssessmentVera", str(font_dir / "Vera.ttf")))
        pdfmetrics.registerFont(
            TTFont("AssessmentVera-Bold", str(font_dir / "VeraBd.ttf"))
        )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Individual Financial Inclusion Assessment",
        author="Financial Inclusion Decision Support",
    )
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "AssessmentBody",
        parent=base["BodyText"],
        fontName="AssessmentVera",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "AssessmentSmall",
        parent=body,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#465360"),
    )
    heading = ParagraphStyle(
        "AssessmentHeading",
        parent=base["Heading2"],
        fontName="AssessmentVera-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=9,
        spaceAfter=4,
    )
    title = ParagraphStyle(
        "AssessmentTitle",
        parent=base["Title"],
        fontName="AssessmentVera-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    rows = _dimension_rows(result.get("dimension_scores", {}))
    available = [row for row in rows if row["Score"] is not None]
    strongest, weakest = _dimension_extrema(rows)
    overall = _overall_dimension_score(rows)
    profile = result.get("profile") if result.get("status") == "success" else None

    content = [
        Paragraph("Individual Financial Inclusion Assessment", title),
        Paragraph(f"Assessment date: {assessment_date.isoformat()}", body),
        Paragraph(
            "This report summarizes the information submitted for this assessment.",
            body,
        ),
    ]

    content.append(Paragraph("Assessment overview", heading))
    if profile:
        cluster_id = profile.get("cluster_id")
        profile_name = profile.get("profile_name") or CLUSTER_LABELS.get(
            cluster_id, "Saved inclusion profile"
        )
        content.append(
            Paragraph(
                f"<b>{escape(str(profile_name))}</b> "
                f"(Cluster {escape(str(cluster_id))})",
                body,
            )
        )
        description = profile.get("profile_description")
        if description:
            content.append(Paragraph(escape(str(description)), body))
        if overall is not None:
            content.append(
                Paragraph(f"Overall dimension score: <b>{overall:.1f} / 100</b>", body)
            )
    else:
        message = result.get(
            "message",
            "A profile was not assigned because required dimension information is unavailable.",
        )
        content.append(Paragraph(escape(str(message)), body))

    content.append(Paragraph("Dimension score summary", heading))
    table_data = [["Dimension", "Score (0 to 1)"]]
    table_data.extend(
        [
            row["Dimension"],
            f"{row['Score']:.3f}" if row["Score"] is not None else "Not available",
        ]
        for row in rows
    )
    score_table = LongTable(
        table_data,
        colWidths=[4.5 * inch, 1.6 * inch],
        repeatRows=1,
    )
    score_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "AssessmentVera-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "AssessmentVera"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#17212B")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    content.append(score_table)

    if available:
        content.append(Paragraph("Key findings", heading))
        content.append(
            Paragraph(
                "Strongest observed dimension(s): "
                + escape(", ".join(row["Dimension"] for row in strongest))
                + f" ({strongest[0]['Score']:.3f}).",
                body,
            )
        )
        content.append(
            Paragraph(
                "Lowest available dimension score(s): "
                + escape(", ".join(row["Dimension"] for row in weakest))
                + f" ({weakest[0]['Score']:.3f}). These are descriptive contrasts, not causes.",
                body,
            )
        )

    actions = _actions_for_available_scores(result, rows) if profile else []
    if actions:
        content.append(Paragraph("Recommended follow-up actions", heading))
        for action, score in actions:
            dimension = escape(str(action.get("dimension", "Dimension")))
            parts = [
                Paragraph(
                    f"<b>Priority area: {dimension}</b> - Observed score: {score:.3f} / 1",
                    body,
                ),
                Paragraph(
                    "<b>Observed Finding:</b> "
                    + escape(str(action.get("reason", "Selected from the observed dimension scores."))),
                    small,
                ),
                Paragraph(
                    "<b>Suggested follow-up:</b> "
                    + escape(str(action.get("action", "Further review is needed."))),
                    body,
                ),
            ]
            if action.get("stakeholder"):
                parts.append(
                    Paragraph(
                        "<b>Potential Stakeholders:</b> "
                        + escape(str(action["stakeholder"])),
                        small,
                    )
                )
            if action.get("monitor"):
                parts.append(
                    Paragraph(
                        "<b>Possible Monitoring Indicator:</b> "
                        + escape(str(action["monitor"])),
                        small,
                    )
                )
            content.append(KeepTogether(parts))
    elif profile:
        content.extend(
            [
                Paragraph("Recommended follow-up actions", heading),
                Paragraph("No dimension-specific action was available for this assessment.", body),
            ]
        )

    completeness = _information_notes(entries)
    completeness.extend(
        f"{row['Dimension']}: score not calculated"
        for row in rows
        if row["Score"] is None
    )
    if completeness:
        content.append(Paragraph("Information completeness", heading))
        content.extend(
            Paragraph("- " + escape(note), small) for note in dict.fromkeys(completeness)
        )

    content.append(
        Paragraph(
            "This assessment describes observed financial inclusion characteristics using the project's existing scoring and profile-assignment methodology. Findings are descriptive, do not establish causality, and should support further investigation rather than automated financial or policy decisions.",
            small,
        )
    )
    document.build(content)
    return buffer.getvalue()

def _show_assessment_results(
    result: dict[str, Any],
    entries: dict[str, Any],
    assessment_date: date,
) -> None:
    
    rows = _dimension_rows(result.get("dimension_scores", {}))
    available = [row for row in rows if row["Score"] is not None]
    strongest, weakest = _dimension_extrema(rows)
    overall_score = _overall_dimension_score(rows)
    notes = _information_notes(entries)

    profile = result.get("profile") if result.get("status") == "success" else None
    st.markdown("### Assessment Result Overview")
    with st.container(border=True):
        if profile:
            cluster_id = profile.get("cluster_id")
            profile_name = profile.get("profile_name") or CLUSTER_LABELS.get(
                cluster_id, "Saved inclusion profile"
            )
            header, cluster = st.columns([4, 1])
            with header:
                st.markdown(f"#### {profile_name}")
                if profile.get("profile_description"):
                    st.write(profile["profile_description"])
            with cluster:
                st.metric("Cluster identifier", str(cluster_id))
        else:
            st.markdown("#### Assessment incomplete")
            st.write(
                result.get(
                    "message",
                    "A saved profile was not assigned because required dimension information is unavailable.",
                )
            )
        st.caption(
            "Descriptive financial inclusion assessment; not a creditworthiness, "
            "loan eligibility, or automated policy decision."
        )

    if available:
        chart_rows = []
        strongest_names = {row["Dimension"] for row in strongest}
        weakest_names = {row["Dimension"] for row in weakest}
        for row in available:
            is_high = row["Dimension"] in strongest_names
            is_low = row["Dimension"] in weakest_names
            if is_high and is_low:
                level = "Highest and lowest"
            elif is_high:
                level = "Highest"
            elif is_low:
                level = "Lowest"
            else:
                level = "Other"
            chart_rows.append({**row, "Position in this assessment": level})

        chart = px.bar(
            pd.DataFrame(chart_rows).sort_values("Score"),
            x="Score",
            y="Dimension",
            color="Position in this assessment",
            orientation="h",
            range_x=[0, 1],
            text="Score",
            color_discrete_map={
                "Other": "#477DA8",
                "Highest": "#2A9D8F",
                "Lowest": "#E9A23B",
                "Highest and lowest": "#8E6CBB",
            },
            category_orders={
                "Position in this assessment": [
                    "Other", "Highest", "Lowest", "Highest and lowest"
                ]
            },
            labels={
                "Score": "Dimension score (0 to 1)",
                "Dimension": "",
                "Position in this assessment": "Observed rank",
            },
            title="Five-dimension score summary",
        )
        chart.update_traces(
            texttemplate="%{x:.2f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<br>%{fullData.name}<extra></extra>",
        )
        chart.update_layout(
            height=330,
            margin=dict(l=10, r=40, t=50, b=20),
            legend_title_text="Observed rank",
        )
        if overall_score is not None:
            gauge_column, dimensions_column = st.columns([1, 2])
            with gauge_column:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=overall_score,
                        number={"suffix": "/100", "valueformat": ".1f"},
                        title={"text": "Overall dimension score"},
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickmode": "array",
                                "tickvals": [0, 25, 50, 75, 100],
                            },
                            "bar": {"color": "#2A9D8F"},
                            "steps": [{"range": [0, 100], "color": "#E9EEF3"}],
                        },
                    )
                )
                gauge.update_layout(
                    height=280,
                    margin=dict(t=45, b=20, l=20, r=20),
                )
                st.plotly_chart(gauge, config={"responsive": True})
                st.caption(
                    "Arithmetic mean of the five available 0-1 dimension scores, "
                    "displayed on a 0-100 scale. It is not a validated index or probability."
                )
            with dimensions_column:
                st.plotly_chart(chart, config={"responsive": True})
        else:
            st.plotly_chart(chart, config={"responsive": True})
            st.caption(
                f"{len(available)} of 5 dimension scores are available. "
                "Unavailable dimensions are omitted, not treated as zero; an overall score is withheld."
            )

        insight_columns = st.columns(3 if notes else 2)
        with insight_columns[0]:
            with st.container(border=True):
                st.markdown("**Strongest dimensions**")
                st.write(
                    ", ".join(row["Dimension"] for row in strongest)
                    + f" - {strongest[0]['Score']:.2f}"
                )
                meanings = {
                    "Connectivity": "phone ownership and recent internet use",
                    "Financial access": "financial-account ownership",
                    "Digital financial usage": "digital-payment activity",
                    "Digital safety": "the available phone-protection measure",
                    "Financial resilience": "saving, emergency-fund access, and income-loss buffering",
                }
                st.caption(
                    "; ".join(
                        f"{row['Dimension']} reflects {meanings.get(row['Dimension'], 'its mapped operational indicators')}"
                        for row in strongest
                    )
                    + "."
                )
        with insight_columns[1]:
            with st.container(border=True):
                st.markdown("**Priority areas for follow-up**")
                st.write(", ".join(row["Dimension"] for row in weakest))
                st.caption(
                    "These are the lowest available dimension scores in this assessment. "
                    "Use the action cards below to review possible next questions."
                )
        if notes:
            with insight_columns[2]:
                with st.container(border=True):
                    st.markdown("**Information completeness**")
                    st.write(f"{len(notes)} input(s) marked unavailable or non-substantive")
                    with st.expander("Review information notes"):
                        for note in notes:
                            st.markdown(f"- {note}")
                    st.caption(
                        "Unavailable, not-asked, and non-substantive responses are not interpreted as No."
                    )
    elif notes:
        st.warning("No dimension scores could be calculated from the available substantive inputs.")
        st.markdown("**Information completeness**")
        for note in notes:
            st.markdown(f"- {note}")

    if profile:
        st.subheader("Recommended areas for follow-up")
        actions = _actions_for_available_scores(result, rows)
        if actions:
            action_columns = st.columns(min(3, len(actions)))
            for index, (action, score) in enumerate(actions):
                with action_columns[index % len(action_columns)]:
                    with st.container(border=True):
                        st.markdown(f"#### {action['dimension']}")
                        st.metric("Observed score", f"{score:.2f} / 1")
                        st.markdown("**Why this appears**")
                        st.write(action.get("reason", "Selected from the observed dimension scores."))
                        st.markdown("**Suggested follow-up**")
                        st.write(action.get("action", "Further review is needed."))
                        if action.get("stakeholder"):
                            st.caption(f"Potential stakeholders: {action['stakeholder']}")
                        if action.get("monitor"):
                            st.caption(f"Possible monitoring indicator: {action['monitor']}")
            st.caption(
                "These are evidence-linked prompts for stakeholder investigation, not proven or guaranteed interventions."
            )
        else:
            st.info("No dimension-specific follow-up action is available for the calculated scores.")

    st.download_button(
        "Download assessment summary (PDF)",
        data=build_assessment_pdf(result, entries, assessment_date),
        file_name="individual_financial_inclusion_assessment.pdf",
        mime="application/pdf",
        key="download_individual_assessment_pdf",
    )

def render() -> None:
    st.subheader("Individual profile scoring")
    st.caption(
        "Assess an individual's financial inclusion characteristics using the "
        "available information. Review the resulting inclusion profile, "
        "understand its dimension scores, identify missing information, and "
        "explore potential areas for further investigation."
    )
    st.info(
        "This is a descriptive financial inclusion assessment, not a credit "
        "score, loan eligibility assessment, or automated policy decision."
    )
    with st.expander("How to use this assessment"):
        st.markdown(
            "1. Enter the respondent's known information.\n"
            "2. Select **Not available** when an answer is unknown or was not collected.\n"
            "3. Review the assigned profile and five dimension scores.\n"
            "4. Use the interpretation to identify areas for further investigation.\n"
            "5. Download the assessment summary if needed."
        )

    submitted, entries = _assessment_form()
    if submitted:
        assessment_date = date.today()
        try:
            result = _profile_individual()(entries)
            st.session_state["individual_assessment"] = {
                "result": result,
                "entries": entries,
                "assessment_date": assessment_date,
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.session_state.pop("individual_assessment", None)
            st.error(f"The assessment could not be completed: {exc}")

    saved_assessment = st.session_state.get("individual_assessment")
    if saved_assessment:
        _show_assessment_results(
            saved_assessment["result"],
            saved_assessment["entries"],
            saved_assessment["assessment_date"],
        )
