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

