"""Capability-gap rule tests on controlled cases (rules are response-based, not SHAP-based)."""

import numpy as np
import pytest

from backend.src.financial_capability.fc_capability_rules import (
    RULES,
    STATUS_GAP,
    STATUS_INSUFFICIENT,
    STATUS_NONE,
    identify_capability_gaps,
    map_gaps_to_categories,
)
from backend.src.financial_capability.fc_config import (
    GROUP_BUDGETING,
    GROUP_GOALS,
    GROUP_KNOWLEDGE,
    GROUP_SAVING,
)

STRONG = {
    "SAVEHABIT": 6, "SAVINGSRANGES": 6, "FS1_7": 5,
    "MANAGE1_1": 5, "MANAGE1_2": 5, "PROPPLAN_1": 5, "PROPPLAN_2": 5, "FS1_6": 5,
    "SUBKNOWL1": 6, "LMscore": 3, "FS1_3": 5, "FS2_3": 1,
    "FINGOALS": 1, "PROPPLAN_3": 5, "PROPPLAN_4": 5, "GOALCONF": 4, "ACT1_2": 5,
}


def _by_area(results):
    return {r.area: r for r in results}


def test_every_area_has_rules_using_codebook_variables():
    assert set(RULES) == {GROUP_SAVING, GROUP_BUDGETING, GROUP_KNOWLEDGE, GROUP_GOALS}
    assert all(len(v) >= 3 for v in RULES.values())


def test_strong_answers_give_no_gap():
    res = _by_area(identify_capability_gaps(STRONG))
    assert all(r.status == STATUS_NONE for r in res.values())
    assert map_gaps_to_categories(list(res.values())) == []


@pytest.mark.parametrize("area,changes,category", [
    (GROUP_SAVING, {"SAVEHABIT": 1, "SAVINGSRANGES": 1}, "Saving"),
    (GROUP_BUDGETING, {"MANAGE1_2": 1, "PROPPLAN_1": 1}, "Budgeting"),
    (GROUP_KNOWLEDGE, {"SUBKNOWL1": 2, "LMscore": 0}, "Basic Financial Knowledge"),
    (GROUP_GOALS, {"FINGOALS": 0, "GOALCONF": 1}, "Financial Goal Planning"),
])
def test_each_rule_triggers_with_reason_and_category(area, changes, category):
    res = _by_area(identify_capability_gaps({**STRONG, **changes}))
    assert res[area].status == STATUS_GAP
    assert len(res[area].triggered) == 2 and "Stronger" in res[area].strength
    assert all(": " in reason for reason in res[area].triggered)  # reason text included
    others = [r for a, r in res.items() if a != area]
    assert all(r.status == STATUS_NONE for r in others)
    assert map_gaps_to_categories(list(res.values())) == [category]


def test_single_indicator_is_a_weaker_signal():
    res = _by_area(identify_capability_gaps({**STRONG, "SAVEHABIT": 2}))
    assert res[GROUP_SAVING].status == STATUS_GAP and "Weaker" in res[GROUP_SAVING].strength


def test_missing_answers_are_insufficient_information():
    res = _by_area(identify_capability_gaps({}))
    assert all(r.status == STATUS_INSUFFICIENT for r in res.values())
    nan_row = {k: np.nan for k in STRONG}
    assert all(r.status == STATUS_INSUFFICIENT for r in identify_capability_gaps(nan_row))


def test_trigger_fires_even_when_other_answers_missing():
    res = _by_area(identify_capability_gaps({"SAVEHABIT": 1}))
    assert res[GROUP_SAVING].status == STATUS_GAP


def test_knowledge_rule_uses_high_value_for_struggle_question():
    res = _by_area(identify_capability_gaps({**STRONG, "FS2_3": 5}))  # always struggles
    assert res[GROUP_KNOWLEDGE].status == STATUS_GAP


def test_categories_are_labels_only_and_ordered_by_signal_strength():
    row = {**STRONG, "SAVEHABIT": 1, "SAVINGSRANGES": 1, "FINGOALS": 0}
    cats = map_gaps_to_categories(identify_capability_gaps(row))
    assert cats == ["Saving", "Financial Goal Planning"]
    assert all(isinstance(c, str) for c in cats)


def test_rules_do_not_mutate_input():
    row = dict(STRONG)
    snapshot = dict(row)
    identify_capability_gaps(row)
    assert row == snapshot


def test_identify_capability_strengths_returns_all_four_dimensions():
    from backend.src.financial_capability.fc_capability_rules import identify_capability_strengths

    strengths = identify_capability_strengths(STRONG)
    assert set(strengths) == {GROUP_SAVING, GROUP_BUDGETING, GROUP_KNOWLEDGE, GROUP_GOALS}
    assert all(len(items) >= 2 for items in strengths.values())


def test_extract_resilience_indicators_uses_supported_survey_variables():
    from backend.src.financial_capability.fc_capability_rules import extract_resilience_indicators
    from backend.src.financial_capability.fc_config import FEATURE_SPEC_BY_NAME

    full_strong = {**STRONG, "MANAGE1_3": 5, "SCFHORIZON": 4}
    items = extract_resilience_indicators(full_strong)
    assert len(items) == 6
    assert all(item.variable in FEATURE_SPEC_BY_NAME for item in items)
    assert all(item.status == "Supportive" for item in items)

    weak_items = extract_resilience_indicators({"SAVINGSRANGES": 1, "SAVEHABIT": 1})
    by_var = {item.variable: item for item in weak_items}
    assert by_var["SAVINGSRANGES"].status == "Watch"
    assert by_var["MANAGE1_1"].status == "Not provided"


def test_officer_guidance_and_questions_generated_from_actual_findings():
    from backend.src.financial_capability.fc_capability_rules import (
        generate_officer_guidance,
        generate_officer_questions,
    )

    weak_row = {**STRONG, "SAVEHABIT": 1, "SAVINGSRANGES": 1, "MANAGE1_2": 1}
    gaps = identify_capability_gaps(weak_row)
    guidance = generate_officer_guidance(gaps)
    questions = generate_officer_questions(gaps, weak_row)

    assert set(guidance) == {GROUP_SAVING, GROUP_BUDGETING, GROUP_KNOWLEDGE, GROUP_GOALS}
    assert "emergency reserve" in guidance[GROUP_SAVING].lower()
    assert any("savings or liquid reserves" in q.lower() for q in questions)
    assert any("monthly expense management" in q.lower() for q in questions)


def test_pdf_and_txt_reports_include_component_04_title_and_mandatory_disclaimer():
    from frontend.financial_capability.pdf_report import (
        MANDATORY_REPORT_DISCLAIMER,
        REPORT_TITLE,
        build_capability_assessment_pdf,
        build_capability_assessment_txt,
    )

    gaps = identify_capability_gaps(STRONG)
    dummy_result = {
        "predicted_class": "High",
        "confidence": 0.78,
        "probabilities": {"Low": 0.07, "Medium": 0.15, "High": 0.78},
    }
    pdf_bytes = build_capability_assessment_pdf(
        response=STRONG,
        result=dummy_result,
        gaps=gaps,
        shap_reasons=["Savings balance ($20,000 - $74,999) supported the High prediction."],
    )
    assert isinstance(pdf_bytes, bytes) and pdf_bytes.startswith(b"%PDF")

    txt_report = build_capability_assessment_txt(
        response=STRONG,
        result=dummy_result,
        gaps=gaps,
        shap_reasons=["Savings balance ($20,000 - $74,999) supported the High prediction."],
    )
    assert REPORT_TITLE in txt_report
    assert MANDATORY_REPORT_DISCLAIMER in txt_report
    assert "NOT loan repayment or default probability" in txt_report


def test_component_04_is_completely_independent_of_other_components():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    comp_dirs = [
        root / "backend" / "src" / "financial_capability",
        root / "frontend" / "financial_capability",
    ]
    forbidden = ("credit_risk", "financial_inclusion", "market_risk", "fraud_detection")
    for d in comp_dirs:
        for py_file in d.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for other in forbidden:
                assert other not in content, f"{py_file} references {other}"


