"""PDF & TXT Assessment Report generator for Component 04 using ReportLab.

Produces a bank-officer decision-support report summarizing a loan applicant's
financial well-being prediction, model prediction probabilities, SHAP explanation
factors, four financial capability dimensions (strengths, gap areas, officer guidance),
supported financial resilience indicators, suggested follow-up questions, limitations,
and the mandatory supplementary-information disclaimer.
"""

from __future__ import annotations

import html
import io
import re
from datetime import date
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.src.financial_capability.fc_capability_rules import (
    GapResult,
    ResilienceItem,
    STATUS_GAP,
    STATUS_INSUFFICIENT,
    STATUS_NONE,
    extract_resilience_indicators,
    generate_officer_guidance,
    generate_officer_questions,
    identify_capability_strengths,
)
from backend.src.financial_capability.fc_config import CLASS_NAMES
from backend.src.financial_capability.fc_predictor import validate_response

REPORT_TITLE = "Financial Well-being Assessment Report – Component 04"

MANDATORY_REPORT_DISCLAIMER = (
    "This report provides supplementary information about the applicant's financial well-being. "
    "It does not replace the bank's standard credit assessment, identity and income verification, "
    "affordability checks or lending policies. It does not predict loan repayment or automatically "
    "approve or reject a loan."
)


def _prepare_report_details(
    response: Dict[str, Any],
    gaps: List[GapResult],
    strengths: Optional[Dict[str, List[str]]] = None,
    resilience_items: Optional[List[ResilienceItem]] = None,
    officer_questions: Optional[List[str]] = None,
    officer_guidance: Optional[Dict[str, str]] = None,
):
    """Ensure strengths, resilience, officer guidance, and questions are populated."""
    try:
        row_dict = validate_response(response).iloc[0].to_dict()
    except Exception:
        row_dict = dict(response)

    if strengths is None:
        strengths = identify_capability_strengths(row_dict)
    if resilience_items is None:
        resilience_items = extract_resilience_indicators(row_dict)
    if officer_guidance is None:
        officer_guidance = generate_officer_guidance(gaps)
    if officer_questions is None:
        officer_questions = generate_officer_questions(gaps, row_dict)

    return strengths, resilience_items, officer_guidance, officer_questions


def build_capability_assessment_pdf(
    response: Dict[str, Any],
    result: Dict[str, Any],
    gaps: List[GapResult],
    shap_reasons: Optional[List[str]] = None,
    assessment_date: Optional[date] = None,
    strengths: Optional[Dict[str, List[str]]] = None,
    resilience_items: Optional[List[ResilienceItem]] = None,
    officer_questions: Optional[List[str]] = None,
    officer_guidance: Optional[Dict[str, str]] = None,
    selected_model_name: str = "XGBoost",
) -> bytes:
    """Generate a structured PDF decision-support report for the bank officer."""
    if assessment_date is None:
        assessment_date = date.today()

    strengths, resilience_items, officer_guidance, officer_questions = _prepare_report_details(
        response, gaps, strengths, resilience_items, officer_questions, officer_guidance
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=REPORT_TITLE,
        author="Global Financial Analytics - Component 04",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=9,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=colors.HexColor("#1E293B"),
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=12,
        spaceAfter=2.5,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("Bank Loan Decision Support — Independent Supplementary Assessment", meta_style))
    story.append(Paragraph(REPORT_TITLE, title_style))
    story.append(
        Paragraph(
            f"Prepared for: <b>Bank Loan Officer</b> | Assessment Date: <b>{assessment_date.strftime('%B %d, %Y')}</b> | "
            f"Model: <b>{selected_model_name}</b> (CFPB NFWBS 2016)",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=8))

    # 2. Applicant Assessment Overview & Predicted Financial Well-being Level
    story.append(Paragraph("1. Applicant Financial Well-being Assessment Overview", section_heading))
    pred_class = result.get("predicted_class", "Unknown")
    conf = result.get("confidence", 0.0)
    probs = result.get("probabilities", {})
    probs_str = " | ".join([f"{c}: {probs.get(c, 0.0):.1%}" for c in CLASS_NAMES])

    n_gaps = sum(1 for g in gaps if g.status == STATUS_GAP)
    n_supportive = sum(1 for r in resilience_items if r.status == "Supportive")
    n_watch = sum(1 for r in resilience_items if r.status == "Watch")

    pred_color = "#EF4444" if pred_class == "Low" else "#F59E0B" if pred_class == "Medium" else "#10B981"
    summary_text = (
        f"<b>Predicted Financial Well-being Level:</b> <font color='{pred_color}'><b>{pred_class}</b></font><br/>"
        f"<b>Model Prediction Probability (Top Class):</b> <b>{conf:.1%}</b> "
        f"<i>(Model prediction probability for financial well-being — NOT loan repayment or default probability)</i><br/>"
        f"<b>Class Probability Distribution:</b> {probs_str}<br/>"
        f"<b>Executive Capability Scorecard:</b> <b>{n_gaps} of {len(gaps)}</b> capability dimensions flagged for attention | "
        f"Resilience Indicators: <b>{n_supportive} Supportive</b>, <b>{n_watch} Watch</b><br/>"
        f"<b>Questionnaire Completeness:</b> {len(response)} of 31 approved indicator questions answered directly."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 6))

    # 3. SHAP Explanation Factors
    story.append(Paragraph("2. Key Model Explanation Factors (SHAP)", section_heading))
    if shap_reasons:
        story.append(
            Paragraph(
                f"Top applicant responses influencing the <b>{pred_class}</b> financial well-being prediction:",
                body_style,
            )
        )
        for s in shap_reasons[:5]:
            # Convert Markdown **bold** to ReportLab <b>bold</b> safely
            clean_s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s))
            story.append(Paragraph(f"• {clean_s}", bullet_style))
    else:
        story.append(
            Paragraph(
                "Individual SHAP factor summary was not attached; refer to the interactive SHAP tab.",
                body_style,
            )
        )
    story.append(
        Paragraph(
            "<i>Note: SHAP describes how applicant answers influence the machine-learning model's prediction. "
            "It does not prove causation.</i>",
            meta_style,
        )
    )
    story.append(Spacer(1, 6))

    # 4. Financial Capability Profile (Four Dimensions)
    story.append(
        Paragraph(
            "3. Financial Capability Profile (Saving, Budgeting, Knowledge, Goal Planning)",
            section_heading,
        )
    )
    table_data = [
        [
            Paragraph("<b>Dimension</b>", body_style),
            Paragraph("<b>Assessment Result</b>", body_style),
            Paragraph("<b>Strengths & Gap Areas</b>", body_style),
            Paragraph("<b>Bank Officer Guidance</b>", body_style),
        ]
    ]

    for g in gaps:
        status_txt = (
            f"Needs attention ({g.strength})"
            if g.status == STATUS_GAP
            else "Healthy indicators"
            if g.status == STATUS_NONE
            else "Incomplete data"
        )
        area_strengths = strengths.get(g.area, [])
        details_parts = []
        if area_strengths:
            details_parts.append(f"<b>Strengths ({len(area_strengths)}):</b> " + "; ".join(area_strengths[:2]))
        if g.triggered:
            details_parts.append(f"<b>Gaps ({len(g.triggered)}):</b> " + "; ".join(g.triggered[:2]))
        if not details_parts:
            details_parts.append("No specific triggers flagged.")
        details_txt = "<br/>".join(details_parts)
        guidance_txt = officer_guidance.get(g.area, "Review alongside standard bank checks.")

        table_data.append(
            [
                Paragraph(f"<b>{g.area}</b>", body_style),
                Paragraph(status_txt, body_style),
                Paragraph(details_txt, body_style),
                Paragraph(guidance_txt, body_style),
            ]
        )

    overview_table = Table(table_data, colWidths=[1.35 * inch, 1.35 * inch, 2.45 * inch, 2.05 * inch])
    overview_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(overview_table)
    story.append(Spacer(1, 6))

    # 5. Financial Resilience Indicators
    story.append(Paragraph("4. Financial Resilience Information (Supported Survey Indicators)", section_heading))
    for item in resilience_items:
        story.append(
            Paragraph(
                f"• <b>{item.indicator_label}:</b> {item.applicant_response} "
                f"(<i>{item.status}</i> — {item.officer_note})",
                bullet_style,
            )
        )
    story.append(Spacer(1, 6))

    # 6. Suggested Follow-up Questions for the Bank Officer
    story.append(Paragraph("5. Suggested Follow-up Questions for the Bank Officer", section_heading))
    for q in officer_questions:
        story.append(Paragraph(f"• {q}", bullet_style))
    story.append(Spacer(1, 6))

    # 7. Bank Officer Decision-Support Takeaway
    story.append(
        Paragraph(
            "6. Bank Officer Decision-Support Takeaway (How to Use alongside Normal Bank Checks)",
            section_heading,
        )
    )
    takeaway_by_class = {
        "Low": (
            "• <b>Interpretation of 'Low' Well-being Signal:</b> Indicates higher sensitivity to unexpected financial shocks "
            "and thinner day-to-day cash-flow buffers.<br/>"
            "• <b>How to Combine with Normal Loan Checks:</b> If the bank's standard credit history, income verification, and "
            "DTI checks are satisfactory, the officer may consider <b>Approve with Conditions</b> (e.g., conservative "
            "installment-to-income margin, aligning repayment date with salary credit date, or verifying emergency reserves). "
            "If standard credit or DTI checks are already borderline, this profile highlights elevated household cash-flow vulnerability."
        ),
        "Medium": (
            "• <b>Interpretation of 'Medium' Well-being Signal:</b> Indicates moderate day-to-day financial stability with "
            "specific capability or saving-buffer areas that merit discussion.<br/>"
            "• <b>How to Combine with Normal Loan Checks:</b> Review the flagged capability areas and resilience watch items "
            "during the applicant interview before finalizing loan terms under standard bank policy."
        ),
        "High": (
            "• <b>Interpretation of 'High' Well-being Signal:</b> Indicates disciplined saving, budgeting, and financial "
            "planning habits.<br/>"
            "• <b>How to Combine with Normal Loan Checks:</b> Serves as positive supplementary context alongside the bank's "
            "mandatory identity, income, credit-score, and repayment-capacity (DTI) checks."
        ),
    }
    story.append(
        Paragraph(
            takeaway_by_class.get(pred_class, takeaway_by_class["Medium"]),
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    # 8. Assessment Limitations
    story.append(Paragraph("7. Assessment Scope & Limitations", section_heading))
    story.append(
        Paragraph(
            "• <b>Dataset Context:</b> Trained on the CFPB 2016 National Financial Well-Being Survey (U.S. adults, "
            "6,389 valid respondents). It does not represent Sri Lankan borrowers directly.<br/>"
            "• <b>Self-Reported Indicators:</b> Inputs reflect applicant-reported financial behaviors and perceptions; "
            "verify material figures against official bank documents.<br/>"
            "• <b>Independent Component:</b> Component 04 operates independently of Components 01–03 and provides "
            "supplementary context only.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # 9. Mandatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#CBD5E1"), spaceAfter=5))
    story.append(
        Paragraph(
            f"<b>Mandatory Bank Decision Support Disclaimer:</b> {MANDATORY_REPORT_DISCLAIMER}",
            meta_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()


def build_capability_assessment_txt(
    response: Dict[str, Any],
    result: Dict[str, Any],
    gaps: List[GapResult],
    shap_reasons: Optional[List[str]] = None,
    assessment_date: Optional[date] = None,
    strengths: Optional[Dict[str, List[str]]] = None,
    resilience_items: Optional[List[ResilienceItem]] = None,
    officer_questions: Optional[List[str]] = None,
    officer_guidance: Optional[Dict[str, str]] = None,
    selected_model_name: str = "XGBoost",
) -> str:
    """Generate a comprehensive plain-text assessment report for the bank officer."""
    if assessment_date is None:
        assessment_date = date.today()

    strengths, resilience_items, officer_guidance, officer_questions = _prepare_report_details(
        response, gaps, strengths, resilience_items, officer_questions, officer_guidance
    )

    pred_class = result.get("predicted_class", "Unknown")
    conf = result.get("confidence", 0.0)
    probs = result.get("probabilities", {})
    probs_str = ", ".join([f"{c}: {probs.get(c, 0.0):.1%}" for c in CLASS_NAMES])

    n_gaps = sum(1 for g in gaps if g.status == STATUS_GAP)
    n_supportive = sum(1 for r in resilience_items if r.status == "Supportive")
    n_watch = sum(1 for r in resilience_items if r.status == "Watch")

    lines = [
        f"# {REPORT_TITLE}",
        f"Date: {assessment_date.isoformat()}",
        f"Prepared For: Bank Loan Officer (Loan Application Decision Support)",
        f"Selected Model: {selected_model_name} (Trained on CFPB NFWBS 2016)",
        "",
        "## 1. Applicant Financial Well-being Prediction",
        f"- Predicted Financial Well-being Level: {pred_class}",
        f"- Model Prediction Probability (Top Class): {conf:.1%} (Model prediction probability for financial well-being — NOT loan repayment or default probability)",
        f"- Class Probability Distribution: {probs_str}",
        f"- Executive Capability Scorecard: {n_gaps} of {len(gaps)} capability dimensions flagged for attention | Resilience Indicators: {n_supportive} Supportive, {n_watch} Watch",
        f"- Questionnaire Completeness: {len(response)} of 31 questions answered directly",
        "",
        "## 2. Key SHAP Explanation Factors",
    ]
    if shap_reasons:
        for s in shap_reasons[:5]:
            lines.append(f"- {s.replace('**', '')}")
    else:
        lines.append("- Refer to the interactive SHAP tab for individual factor breakdown.")
    lines.append("Note: SHAP describes model behavior and does not prove causation.")
    lines.append("")

    lines.append("## 3. Financial Capability Profile (Four Dimensions)")
    for g in gaps:
        lines.append(f"### {g.area}")
        lines.append(f"- Assessment Result: {g.status}" + (f" ({g.strength})" if g.strength else ""))
        area_str = strengths.get(g.area, [])
        if area_str:
            lines.append("- Identified Strengths:")
            for st_item in area_str:
                lines.append(f"  * {st_item}")
        else:
            lines.append("- Identified Strengths: None specifically flagged")
        if g.triggered:
            lines.append("- Capability Gap Areas:")
            for tr in g.triggered:
                lines.append(f"  * {tr}")
        else:
            lines.append("- Capability Gap Areas: None flagged")
        lines.append(f"- Bank Officer Guidance: {officer_guidance.get(g.area, '')}")
        lines.append("")

    lines.append("## 4. Financial Resilience Indicators (Supported Survey Variables)")
    for item in resilience_items:
        lines.append(
            f"- {item.indicator_label}: {item.applicant_response} [{item.status}] — {item.officer_note}"
        )
    lines.append("")

    lines.append("## 5. Suggested Follow-up Questions for the Bank Officer")
    for q in officer_questions:
        lines.append(f"- {q}")
    lines.append("")

    lines.append("## 6. Bank Officer Decision-Support Takeaway (How to Use alongside Normal Bank Checks)")
    if pred_class == "Low":
        lines.append(
            "- Interpretation of 'Low' Well-being Signal: Indicates higher sensitivity to unexpected financial shocks and thinner day-to-day cash-flow buffers."
        )
        lines.append(
            "- How to Combine with Normal Loan Checks: If standard credit history, income verification, and DTI checks pass, consider Approve with Conditions (e.g., conservative installment-to-income margin, aligning repayment date with salary credit date, or verifying emergency reserves). If standard checks are borderline, this profile highlights elevated cash-flow vulnerability."
        )
    elif pred_class == "High":
        lines.append(
            "- Interpretation of 'High' Well-being Signal: Indicates disciplined saving, budgeting, and financial planning habits supporting standard bank verification."
        )
    else:
        lines.append(
            "- Interpretation of 'Medium' Well-being Signal: Indicates moderate day-to-day financial stability with specific capability or saving-buffer areas to clarify during the applicant interview."
        )
    lines.append("")

    lines.append("## 7. Assessment Limitations")
    lines.append(
        "- Dataset represents U.S. adults surveyed in the 2016 CFPB National Financial Well-Being Survey (6,389 valid respondents), not Sri Lankan loan applicants."
    )
    lines.append(
        "- Inputs are self-reported behavioral and financial capability indicators; verify material figures via standard bank checks."
    )
    lines.append("")

    lines.append("## 8. Mandatory Disclaimer")
    lines.append(MANDATORY_REPORT_DISCLAIMER)

    return "\n".join(lines)


