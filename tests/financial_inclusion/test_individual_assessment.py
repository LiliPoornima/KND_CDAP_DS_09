from datetime import date
import re

from streamlit.testing.v1 import AppTest

from frontend.financial_inclusion.sections.individual_assessment import (
    _dimension_rows,
    _dimension_extrema,
    _information_notes,
    _overall_dimension_score,
    _profile_individual,
    build_assessment_pdf,
)
from frontend.financial_inclusion.sections.individual_assessment import _profile_individual


def assessment_app() -> AppTest:
    return AppTest.from_string(
        "from frontend.financial_inclusion.sections.individual_assessment "
        "import render\nrender()"
    ).run()


def test_complete_form_submission_persists_saved_profile_and_pdf_control():
    app = assessment_app()
    answers = {
        "Does the respondent personally own a mobile phone?": "Yes",
        "Has the respondent used the internet recently?": "Yes",
        "Does the respondent have a financial account?": "Yes",
        "Has the respondent made or received digital payments?": "No",
        "Does the respondent use phone security protection?": "No",
        "Has the respondent saved money?": "Yes",
        "Ability to find emergency funds": "Somewhat difficult",
        "How long could expenses be covered?": "About 2 months",
    }
    for label, value in answers.items():
        next(widget for widget in app.selectbox if widget.label == label).set_value(value)

    app.button[0].click().run()
    saved = app.session_state["individual_assessment"]

    assert saved["result"]["status"] == "success"
    assert saved["result"]["profile"]["cluster_id"] in range(5)
    assert len(saved["result"]["dimension_scores"]) == 5
    assert any(
        widget.label == "Download assessment summary (PDF)"
        for widget in app.get("download_button")
    )
    assert len(app.get("plotly_chart")) == 2
    assert any("Recommended areas for follow-up" in item.value for item in app.subheader)
    assert not app.exception


def test_unavailable_form_answers_remain_missing_and_do_not_assign_profile():
    app = assessment_app()
    for label, value in {
        "Ability to find emergency funds": "Somewhat difficult",
        "How long could expenses be covered?": "About 2 months",
    }.items():
        next(widget for widget in app.selectbox if widget.label == label).set_value(value)

    app.button[0].click().run()
    saved = app.session_state["individual_assessment"]

    assert saved["entries"]["account"] is None
    assert saved["result"]["status"] == "incomplete"
    assert "profile" not in saved["result"]
    assert all(
        value is None
        for value in saved["result"]["dimension_scores"].values()
    )
    assert any(
        widget.label == "Download assessment summary (PDF)"
        for widget in app.get("download_button")
    )
    assert not app.exception


def test_partial_assessment_keeps_available_scores_and_withholds_overall_score():
    app = assessment_app()
    answers = {
        "Does the respondent personally own a mobile phone?": "Yes",
        "Has the respondent used the internet recently?": "Yes",
        "Does the respondent have a financial account?": "Yes",
        "Has the respondent made or received digital payments?": "Yes",
        "Does the respondent use phone security protection?": "Yes",
        "Has the respondent saved money?": "Yes",
        "Ability to find emergency funds": "Somewhat difficult",
        "How long could expenses be covered?": "Don't know",
    }
    for label, value in answers.items():
        next(widget for widget in app.selectbox if widget.label == label).set_value(value)

    app.button[0].click().run()
    result = app.session_state["individual_assessment"]["result"]

    assert result["status"] == "incomplete"
    assert result["dimension_scores"]["score_connectivity"] == 1.0
    assert result["dimension_scores"]["score_financial_resilience"] is None
    assert "profile" not in result
    assert not any("Overall dimension score" in item.label for item in app.metric)
    assert not app.exception


def test_dimension_summary_handles_ties_and_only_calculates_overall_when_complete():
    scores = {
        "score_connectivity": 0.8,
        "score_financial_access": 0.2,
        "score_digital_financial_usage": 0.2,
        "score_digital_safety": 0.5,
        "score_financial_resilience": 0.2,
    }
    rows = _dimension_rows(scores)
    strongest, weakest = _dimension_extrema(rows)

    assert [row["Dimension"] for row in strongest] == ["Connectivity"]
    assert [row["Dimension"] for row in weakest] == [
        "Financial access",
        "Digital financial usage",
        "Financial resilience",
    ]
    assert _overall_dimension_score(rows) == 38.0
    assert _overall_dimension_score(rows[:-1] + [{"Dimension": "Financial resilience", "Score": None}]) is None


def test_recommendations_preserve_ties_at_the_priority_boundary():
    _profile_individual()
    from importlib import import_module

    explain = import_module("fi_profile_explanation").explain_inclusion_profile
    tied_scores = {
        "score_connectivity": 0.5,
        "score_financial_access": 0.5,
        "score_digital_financial_usage": 0.5,
        "score_digital_safety": 0.5,
        "score_financial_resilience": 0.5,
    }

    explanation = explain(tied_scores, cluster_id=0)

    assert len(explanation["suggested_actions"]) == 5
    assert len(explanation["lowest_scoring_dimensions"]) == 5


def test_assessment_pdf_uses_saved_profile_scores_without_identifier_fields():
    indicators = {
        "con1": 1,
        "internet_use": 1,
        "account": 1,
        "anydigpayment": 0,
        "con18": 2,
        "saved": 1,
        "fin24a": 2,
        "fin24b": 3,
    }
    result = _profile_individual()(indicators)

    pdf = build_assessment_pdf(result, indicators, date(2026, 10, 28))

    assert pdf.startswith(b"%PDF-")
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) >= 1
    assert b"wpid_random" not in pdf
    assert [row["Score"] for row in _dimension_rows(result["dimension_scores"])] == [
        result["dimension_scores"][name]
        for name in (
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        )
    ]


def test_assessment_pdf_represents_unavailable_scores_and_keeps_nonresponses_distinct():
    entries = {
        "con1": None,
        "internet_use": None,
        "account": 0,
        "anydigpayment": 1,
        "con18": None,
        "saved": 1,
        "fin24a": 4,
        "fin24b": 6,
    }
    result = {
        "status": "incomplete",
        "dimension_scores": {
            "score_connectivity": None,
            "score_financial_access": 0.0,
            "score_digital_financial_usage": 1.0,
            "score_digital_safety": None,
            "score_financial_resilience": None,
        },
    }

    pdf = build_assessment_pdf(result, entries, date(2026, 10, 28))
    notes = _information_notes(entries)

    assert pdf.startswith(b"%PDF-")
    assert _dimension_rows(result["dimension_scores"])[0]["Score"] is None
    assert any("Don't know" in note for note in notes)
    assert any("Prefer not to answer" in note for note in notes)
    assert sum("Not available" in note for note in notes) == 3


def test_consolidated_financial_inclusion_dashboard_is_reachable_from_app_navigation():
    app = AppTest.from_file("frontend/app.py").run()
    assert not app.exception

    destination = app.switch_page(
        "pages/5_Financial_Inclusion_Decision_Support.py"
    ).run()

    assert not destination.exception
    assert any(
        "Component 02: Financial Inclusion Decision Support" in item.value
        for item in destination.title
    )
    assert destination.radio[0].options[-1] == "Decision Support & Action Planning"
