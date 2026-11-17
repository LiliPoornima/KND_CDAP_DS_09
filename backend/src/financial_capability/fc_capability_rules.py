"""Transparent, response-based capability-gap rules (NOT derived from SHAP).

Each rule uses the codebook meaning of a survey response. A response that
triggers an indicator is reported with the reason. The wording is
"Possible improvement area" - never a diagnosis of the person.

Thresholds are PROPOSED here from the response labels (for example the two or
three lowest agreement options) and must be agreed with the supervisor.
Gaps are never inferred from SHAP alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping

import pandas as pd

from .fc_config import (
    FEATURE_SPEC_BY_NAME,
    GROUP_BUDGETING,
    GROUP_GOALS,
    GROUP_KNOWLEDGE,
    GROUP_SAVING,
    LEARNING_CATEGORY,
)

STATUS_GAP = "Possible improvement area"
STATUS_NONE = "No improvement area indicated"
STATUS_INSUFFICIENT = "Insufficient information"
STRENGTH_STRONG = "Stronger signal (2 or more indicators)"
STRENGTH_WEAK = "Weaker signal (1 indicator)"


@dataclass(frozen=True)
class Indicator:
    """One response-based check: a variable, a test and a plain-English reason."""

    variable: str
    test: Callable[[float], bool]
    reason: str


def _le(limit: float) -> Callable[[float], bool]:
    return lambda v: v <= limit


def _ge(limit: float) -> Callable[[float], bool]:
    return lambda v: v >= limit


def _eq(target: float) -> Callable[[float], bool]:
    return lambda v: v == target


RULES: Dict[str, List[Indicator]] = {
    GROUP_SAVING: [
        Indicator("SAVEHABIT", _le(3),
                  "Saving is not a habit (answered Strongly disagree, Disagree or Disagree slightly)"),
        Indicator("SAVINGSRANGES", _le(2), "Savings today are $99 or less"),
        Indicator("FS1_7", _le(2),
                  "Reports knowing how to make themselves save 'Not at all' or 'Very little'"),
    ],
    GROUP_BUDGETING: [
        Indicator("MANAGE1_2", _le(2),
                  "Seldom or never stayed within a budget or spending plan (or not applicable)"),
        Indicator("MANAGE1_1", _le(2),
                  "Seldom or never paid all bills on time (or not applicable)"),
        Indicator("PROPPLAN_1", _le(2), "Does not consult a budget to see how much money is left"),
        Indicator("PROPPLAN_2", _le(2), "Does not consider steps needed to stick to a budget"),
        Indicator("FS1_6", _le(2),
                  "Reports knowing how to keep from overspending 'Not at all' or 'Very little'"),
    ],
    GROUP_KNOWLEDGE: [
        Indicator("SUBKNOWL1", _le(3), "Rates own financial knowledge 3 or lower on the 1-7 scale"),
        Indicator("LMscore", _le(1),
                  "Answered at most 1 of the 3 basic financial-knowledge questions correctly"),
        Indicator("FS1_3", _le(2),
                  "Reports knowing how to make complex financial decisions 'Not at all' or 'Very little'"),
        Indicator("FS2_3", _ge(4), "Often or always struggles to understand financial information"),
    ],
    GROUP_GOALS: [
        Indicator("FINGOALS", _eq(0), "No current or recent financial goal"),
        Indicator("PROPPLAN_3", _le(2), "Does not set financial goals for what to achieve with money"),
        Indicator("PROPPLAN_4", _le(2), "Does not prepare a clear plan of action for financial goals"),
        Indicator("GOALCONF", _le(2), "Not confident in own ability to achieve financial goals"),
        Indicator("ACT1_2", _le(2), "Follows through on own financial goals 'Not at all' or 'Very little'"),
    ],
}


@dataclass
class GapResult:
    """Outcome for one capability area."""

    area: str
    status: str
    learning_category: str
    strength: str = ""
    triggered: List[str] = field(default_factory=list)
    n_answered: int = 0
    n_indicators: int = 0

    @property
    def is_gap(self) -> bool:
        return self.status == STATUS_GAP


def _value(row: Mapping, variable: str):
    """Fetch a response as a float, or None if it is missing."""
    v = row.get(variable) if isinstance(row, Mapping) else row[variable]
    if v is None or pd.isna(v):
        return None
    return float(v)


def identify_capability_gaps(response_row: Mapping) -> List[GapResult]:
    """Apply every rule to one response row (cleaned codes, NaN/None = not answered).

    - Any triggered indicator -> 'Possible improvement area' (with the reasons).
    - No trigger and fewer than 2 answered indicators -> 'Insufficient information'.
    - Otherwise -> 'No improvement area indicated'.
    """
    results: List[GapResult] = []
    for area, indicators in RULES.items():
        triggered, answered = [], 0
        for ind in indicators:
            v = _value(response_row, ind.variable)
            if v is None:
                continue
            answered += 1
            if ind.test(v):
                label = FEATURE_SPEC_BY_NAME[ind.variable].label
                triggered.append(f"{label}: {ind.reason}")
        if triggered:
            status = STATUS_GAP
            strength = STRENGTH_STRONG if len(triggered) >= 2 else STRENGTH_WEAK
        elif answered < 2:
            status, strength = STATUS_INSUFFICIENT, ""
        else:
            status, strength = STATUS_NONE, ""
        results.append(GapResult(
            area=area, status=status, learning_category=LEARNING_CATEGORY[area],
            strength=strength, triggered=triggered, n_answered=answered,
            n_indicators=len(indicators),
        ))
    return results


def map_gaps_to_categories(gap_results: List[GapResult]) -> List[str]:
    """Learning-category LABELS for flagged areas, strongest signals first (labels only)."""
    flagged = [g for g in gap_results if g.is_gap]
    flagged.sort(key=lambda g: -len(g.triggered))
    return [g.learning_category for g in flagged]


# ---------------------------------------------------------------------------
# Component 04 Bank Officer Decision Support Helpers
# ---------------------------------------------------------------------------

STRENGTH_RULES: Dict[str, List[Indicator]] = {
    GROUP_SAVING: [
        Indicator("SAVEHABIT", _ge(4), "Putting money into savings is a regular habit"),
        Indicator("SAVINGSRANGES", _ge(4), "Maintains $1,000 or more in liquid savings today"),
        Indicator("AUTOMATED_1", _eq(1), "Uses automated retirement saving transfers"),
        Indicator("AUTOMATED_2", _eq(1), "Uses automated non-retirement saving transfers"),
        Indicator("FS1_7", _ge(4), "Strong ability to save consistently ('Very well' or 'Completely')"),
    ],
    GROUP_BUDGETING: [
        Indicator("MANAGE1_1", _ge(4), "Often or always paid all bills on time in the past 6 months"),
        Indicator("MANAGE1_2", _ge(4), "Often or always stayed within a budget or spending plan"),
        Indicator("MANAGE1_3", _ge(4), "Often or always paid credit card balances in full"),
        Indicator("PROPPLAN_1", _ge(4), "Regularly consults a budget to check remaining funds"),
        Indicator("PROPPLAN_2", _ge(4), "Actively considers steps needed to stick to a budget"),
        Indicator("FS1_6", _ge(4), "Strong self-discipline against overspending"),
    ],
    GROUP_KNOWLEDGE: [
        Indicator("SUBKNOWL1", _ge(5), "High self-rated financial knowledge (5–7 on 1–7 scale)"),
        Indicator("LMscore", _ge(2), "Answered 2 or 3 of 3 Lusardi-Mitchell knowledge questions correctly"),
        Indicator("FS1_3", _ge(4), "Confident in making complex financial decisions"),
        Indicator("FS2_3", _le(2), "Rarely or never struggles to understand financial information"),
    ],
    GROUP_GOALS: [
        Indicator("FINGOALS", _eq(1), "Maintains active current or recent financial goals"),
        Indicator("PROPPLAN_3", _ge(4), "Actively sets financial goals for what to achieve with money"),
        Indicator("PROPPLAN_4", _ge(4), "Prepares a clear plan of action for reaching financial goals"),
        Indicator("GOALCONF", _ge(3), "Somewhat or very confident in ability to achieve financial goals"),
        Indicator("ACT1_2", _ge(4), "Consistently follows through on personal financial goals"),
        Indicator("SCFHORIZON", _ge(3), "Plans finances over a multi-year horizon (next few years or longer)"),
    ],
}


@dataclass(frozen=True)
class ResilienceItem:
    """One supported financial resilience indicator from actual CFPB survey variables."""

    variable: str
    indicator_label: str
    applicant_response: str
    status: str  # "Supportive", "Watch", or "Not provided"
    officer_note: str


RESILIENCE_VARIABLE_SPECS = [
    (
        "SAVINGSRANGES",
        "Liquid Savings Buffer (Today)",
        lambda v: "Supportive" if v >= 4 else "Watch",
        lambda v: (
            "Reported liquid savings reserve ($1,000+) provides a buffer for unexpected expenses."
            if v >= 4
            else "Limited reported liquid savings ($999 or less); verify emergency reserve adequacy."
        ),
    ),
    (
        "SAVEHABIT",
        "Regular Saving Habit",
        lambda v: "Supportive" if v >= 4 else "Watch",
        lambda v: (
            "Applicant reports a consistent habit of setting money aside."
            if v >= 4
            else "Saving is not yet reported as a consistent habit."
        ),
    ),
    (
        "MANAGE1_1",
        "Timely Bill Payment (Past 6 Months)",
        lambda v: "Supportive" if v >= 4 else "Watch",
        lambda v: (
            "Consistent self-reported history of paying bills on time."
            if v >= 4
            else "Self-reported bill payment consistency warrants verification against bank statements."
        ),
    ),
    (
        "MANAGE1_2",
        "Budget & Expense Control (Past 6 Months)",
        lambda v: "Supportive" if v >= 4 else "Watch",
        lambda v: (
            "Regularly stays within a monthly spending plan or budget."
            if v >= 4
            else "Reported budget adherence is inconsistent; review monthly cash-flow margins."
        ),
    ),
    (
        "MANAGE1_3",
        "Credit Card Full Repayment",
        lambda v: "Supportive" if v >= 4 else "Watch",
        lambda v: (
            "Regularly pays off revolving credit balances in full."
            if v >= 4
            else "May carry revolving balances or not applicable; cross-check existing debt obligations."
        ),
    ),
    (
        "SCFHORIZON",
        "Financial Planning Time Horizon",
        lambda v: "Supportive" if v >= 3 else "Watch",
        lambda v: (
            "Plans ahead over a multi-year timeframe when making saving and spending decisions."
            if v >= 3
            else "Short-term planning horizon (next few months or next year); discuss loan tenure alignment."
        ),
    ),
]


OFFICER_GUIDANCE_BY_AREA: Dict[str, Dict[str, str]] = {
    GROUP_SAVING: {
        STATUS_GAP: (
            "Review applicant's emergency reserve buffer and ability to maintain a contingency fund "
            "alongside monthly loan installments."
        ),
        STATUS_NONE: (
            "Positive saving discipline indicated. Confirm reported liquid reserves during standard "
            "bank verification."
        ),
        STATUS_INSUFFICIENT: (
            "Saving indicators incomplete. Request clarification on current savings reserves during interview."
        ),
    },
    GROUP_BUDGETING: {
        STATUS_GAP: (
            "Discuss monthly income-expense tracking, discretionary spending flexibility, and punctual "
            "bill management before finalizing installment schedules."
        ),
        STATUS_NONE: (
            "Structured day-to-day money management indicated. Align proposed loan repayment schedule "
            "with existing household budget cycle."
        ),
        STATUS_INSUFFICIENT: (
            "Budgeting indicators incomplete. Review monthly cash-flow statements during affordability check."
        ),
    },
    GROUP_KNOWLEDGE: {
        STATUS_GAP: (
            "Explain loan interest terms, repayment schedule, inflation/rate sensitivity, and total "
            "borrowing cost clearly in plain language during the officer consultation."
        ),
        STATUS_NONE: (
            "Adequate financial literacy indicators observed. Standard loan disclosure and key-fact "
            "statement review recommended."
        ),
        STATUS_INSUFFICIENT: (
            "Financial knowledge indicators incomplete. Ensure loan terms and repayment obligations "
            "are clearly explained."
        ),
    },
    GROUP_GOALS: {
        STATUS_GAP: (
            "Clarify how the requested loan fits into the applicant's medium-to-long-term financial "
            "commitments and contingency planning."
        ),
        STATUS_NONE: (
            "Structured goal-setting and planning habits indicated. Verify alignment between loan "
            "purpose and stated financial horizon."
        ),
        STATUS_INSUFFICIENT: (
            "Goal planning indicators incomplete. Discuss the applicant's medium-term financial plan "
            "during the interview."
        ),
    },
}


def identify_capability_strengths(response_row: Mapping) -> Dict[str, List[str]]:
    """Return positive financial indicators observed in each capability dimension."""
    strengths: Dict[str, List[str]] = {}
    for area, indicators in STRENGTH_RULES.items():
        area_items: List[str] = []
        for ind in indicators:
            v = _value(response_row, ind.variable)
            if v is not None and ind.test(v):
                label = FEATURE_SPEC_BY_NAME[ind.variable].label
                area_items.append(f"{label}: {ind.reason}")
        strengths[area] = area_items
    return strengths


def extract_resilience_indicators(response_row: Mapping) -> List[ResilienceItem]:
    """Extract supported financial resilience indicators from actual approved survey features."""
    items: List[ResilienceItem] = []
    for var_name, label, status_fn, note_fn in RESILIENCE_VARIABLE_SPECS:
        spec = FEATURE_SPEC_BY_NAME[var_name]
        v = _value(response_row, var_name)
        if v is None:
            items.append(
                ResilienceItem(
                    variable=var_name,
                    indicator_label=label,
                    applicant_response="Not answered",
                    status="Not provided",
                    officer_note="Question not answered; verify directly during applicant interview.",
                )
            )
        else:
            code = int(round(v))
            ans_label = spec.options.get(code, str(code))
            items.append(
                ResilienceItem(
                    variable=var_name,
                    indicator_label=label,
                    applicant_response=ans_label,
                    status=status_fn(v),
                    officer_note=note_fn(v),
                )
            )
    return items


def generate_officer_guidance(gap_results: List[GapResult]) -> Dict[str, str]:
    """Return practical bank officer follow-up guidance for each capability dimension."""
    guidance: Dict[str, str] = {}
    for g in gap_results:
        area_map = OFFICER_GUIDANCE_BY_AREA.get(g.area, {})
        guidance[g.area] = area_map.get(
            g.status,
            "Review applicant responses alongside standard bank assessment checks.",
        )
    return guidance


def generate_officer_questions(
    gap_results: List[GapResult],
    response_row: Mapping,
) -> List[str]:
    """Generate short, neutral follow-up questions for the bank officer from actual findings."""
    questions: List[str] = []
    gap_by_area = {g.area: g for g in gap_results}

    sav_val = _value(response_row, "SAVINGSRANGES")
    save_hab = _value(response_row, "SAVEHABIT")
    if (
        (gap_by_area.get(GROUP_SAVING) and gap_by_area[GROUP_SAVING].is_gap)
        or (sav_val is not None and sav_val <= 3)
        or (save_hab is not None and save_hab <= 3)
    ):
        questions.append(
            "Does the applicant have savings or liquid reserves available to manage unexpected expenses alongside loan repayments?"
        )

    if gap_by_area.get(GROUP_BUDGETING) and gap_by_area[GROUP_BUDGETING].is_gap:
        questions.append(
            "Would further clarification about monthly expense management, household budgeting, and regular bill payment timing be useful?"
        )

    if gap_by_area.get(GROUP_KNOWLEDGE) and gap_by_area[GROUP_KNOWLEDGE].is_gap:
        questions.append(
            "Should the bank officer walk through the interest rate structure, repayment schedule, and total loan cost in additional detail?"
        )

    if gap_by_area.get(GROUP_GOALS) and gap_by_area[GROUP_GOALS].is_gap:
        questions.append(
            "Does the applicant have a structured action plan for managing future financial commitments over the full loan tenure?"
        )

    if not questions:
        questions.append(
            "How do the applicant's reported saving and budgeting habits align with the verified cash-flow and DTI figures in the bank's standard loan check?"
        )
        questions.append(
            "Are there any upcoming household or employment changes that could affect the applicant's financial planning horizon?"
        )

    return questions


