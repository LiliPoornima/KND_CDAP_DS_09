"""PDF Assessment Report generator for Component 05 using ReportLab.

Produces a clean, publication-ready summary of an individual's
financial capability assessment results, capability gaps, strengths,
SHAP drivers, and Phase 1 learning recommendations.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.src.financial_capability.fc_capability_rules import GapResult, STATUS_GAP, STATUS_INSUFFICIENT, STATUS_NONE
from backend.src.financial_capability.fc_config import CLASS_NAMES


def build_capability_assessment_pdf(
    response: Dict[str, Any],
    result: Dict[str, Any],
    gaps: List[GapResult],
    shap_reasons: Optional[List[str]] = None,
    assessment_date: Optional[date] = None,
) -> bytes:
    """Generate a clean, structured PDF report of the assessment results."""
    if assessment_date is None:
        assessment_date = date.today()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="Financial Capability Assessment Report",
        author="Global Financial Analytics - Component 05",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=12,
        spaceAfter=3,
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
    story.append(Paragraph("Global Financial Analytics", meta_style))
    story.append(Paragraph("Personalized Financial Capability Assessment Report", title_style))
    story.append(
        Paragraph(
            f"Component 05 — Phase 1 Prototype | Assessment Date: <b>{assessment_date.strftime('%B %d, %Y')}</b> | Dataset: CFPB NFWBS 2016",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10))

    # 2. Executive Prediction Summary
    story.append(Paragraph("1. Financial Well-being Prediction", section_heading))
    pred_class = result.get("predicted_class", "Unknown")
    conf = result.get("confidence", 0.0)
    probs = result.get("probabilities", {})
    probs_str = " | ".join([f"{c}: {probs.get(c, 0.0):.1%}" for c in CLASS_NAMES])

    pred_color = "#EF4444" if pred_class == "Low" else "#F59E0B" if pred_class == "Medium" else "#10B981"
    summary_text = (
        f"<b>Predicted Financial Well-being Category:</b> <font color='{pred_color}'><b>{pred_class}</b></font> "
        f"(Model Confidence: <b>{conf:.1%}</b>)<br/>"
        f"<b>Class Probability Distribution:</b> {probs_str}<br/>"
        f"<b>Assessment Completeness:</b> {len(response)} of 31 questions answered directly."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # 3. Financial Capability Overview (4 Core Areas Table)
    story.append(Paragraph("2. Financial Capability Overview (Four Areas)", section_heading))
    table_data = [
        [
            Paragraph("<b>Capability Area</b>", body_style),
            Paragraph("<b>Observed Status</b>", body_style),
            Paragraph("<b>Indicators / Issues</b>", body_style),
            Paragraph("<b>Suggested Learning Focus</b>", body_style),
        ]
    ]

    for g in gaps:
        status_txt = "Possible improvement area" if g.status == STATUS_GAP else "No improvement area indicated" if g.status == STATUS_NONE else "Insufficient information"
        issues_txt = f"{len(g.triggered)} issue(s) ({g.strength})" if g.status == STATUS_GAP else "0 issues (Healthy)" if g.status == STATUS_NONE else "Incomplete"
        focus_txt = g.learning_category if g.status == STATUS_GAP else "—"
        table_data.append([
            Paragraph(f"<b>{g.area}</b>", body_style),
            Paragraph(status_txt, body_style),
            Paragraph(issues_txt, body_style),
            Paragraph(focus_txt, body_style),
        ])

    overview_table = Table(table_data, colWidths=[1.8 * inch, 2.0 * inch, 1.8 * inch, 1.6 * inch])
    overview_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(overview_table)
    story.append(Spacer(1, 10))

    # 4. Improvement Priorities & Triggers
    gap_areas = [g for g in gaps if g.status == STATUS_GAP]
    gap_areas.sort(key=lambda x: 0 if "Stronger" in x.strength else 1)

    if gap_areas:
        story.append(Paragraph("3. Improvement Priorities & Identified Factors", section_heading))
        for idx, g in enumerate(gap_areas, 1):
            rank_label = f"Priority #{idx} — {g.area} ({g.strength})"
            story.append(Paragraph(f"<b>{rank_label}</b>", body_style))
            story.append(Paragraph("<i>Identified behavioral factors:</i>", meta_style))
            for r in g.triggered:
                story.append(Paragraph(f"• {r}", bullet_style))
            story.append(Paragraph(f"<b>Suggested Learning Focus:</b> {g.learning_category}", body_style))
            story.append(Spacer(1, 4))

    # 5. Financial Strengths (Healthy Areas)
    healthy_areas = [g for g in gaps if g.status == STATUS_NONE]
    if healthy_areas:
        story.append(Paragraph("4. Financial Strengths & Healthy Habits", section_heading))
        for h in healthy_areas:
            story.append(
                Paragraph(
                    f"• <b>{h.area}:</b> Current answers reflect healthy practices with no improvement gaps detected under selected rules.",
                    bullet_style,
                )
            )
        story.append(Spacer(1, 6))

    # 6. SHAP Key Drivers
    if shap_reasons:
        story.append(Paragraph("5. Model Attribution & Key Influential Factors (SHAP)", section_heading))
        story.append(
            Paragraph(
                f"The following behavioral responses had the greatest impact on predicting the <b>{pred_class}</b> category:",
                body_style,
            )
        )
        for s in shap_reasons[:4]:
            story.append(Paragraph(f"• {s}", bullet_style))
        story.append(
            Paragraph(
                "<i>Note: SHAP explains how individual answers influence the model's prediction. "
                "These explanations do not prove that one factor causes a person's financial well-being.</i>",
                meta_style,
            )
        )
        story.append(Spacer(1, 6))

    # 7. Recommended Learning Categories (Phase 1)
    story.append(Paragraph("6. Recommended Learning Focus (Phase 1 Scope)", section_heading))
    suggested_cats = [g.learning_category for g in gap_areas]
    if suggested_cats:
        cats_text = ", ".join(dict.fromkeys(suggested_cats))
        story.append(Paragraph(f"Recommended educational categories: <b>{cats_text}</b>", body_style))
    else:
        story.append(Paragraph("No critical learning categories indicated. Maintain positive financial habits.", body_style))

    story.append(
        Paragraph(
            "<b>Phase 1 Scope Notice:</b> Phase 1 identifies suitable learning categories only. "
            "Actual interactive lessons, video tutorials, quizzes, and learning progress tracking are planned for Phase 2.",
            meta_style,
        )
    )
    story.append(Spacer(1, 10))

    # 8. Formal Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> Educational support only — not financial advice, a credit decision, or a diagnosis "
            "of a person's abilities. The machine learning model was trained on US adults surveyed in the CFPB 2016 "
            "National Financial Well-Being Survey. No personal data was stored in any database.",
            meta_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()

