
"""End-to-end individual financial inclusion profiling pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

DIMENSION_MODULE_DIR = (
    ROOT / "backend/src/financial_inclusion/02_dimension_creation"
)
CLUSTERING_MODULE_DIR = (
    ROOT / "backend/src/financial_inclusion/03_clustering"
)

for module_dir in (DIMENSION_MODULE_DIR, CLUSTERING_MODULE_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from fi_individual_score_calculation import calculate_dimension_scores
from fi_profile_inference import predict_inclusion_profile
from fi_profile_explanation import explain_inclusion_profile


REQUIRED_INDICATORS = [
    "con1",
    "internet_use",
    "account",
    "anydigpayment",
    "con18",
    "saved",
    "fin24a",
    "fin24b",
]


def profile_individual(
    indicators: dict[str, Any],
) -> dict[str, Any]:
    """Calculate scores, predict a profile, and return explanations.

    If one or more dimension scores cannot be calculated, inference is
    blocked. Missing indicators are not imputed.
    """
    if not isinstance(indicators, dict):
        raise TypeError("Indicators must be provided as a dictionary.")

    missing = sorted(set(REQUIRED_INDICATORS) - set(indicators))
    extra = sorted(set(indicators) - set(REQUIRED_INDICATORS))

    if missing or extra:
        raise ValueError(
            f"Indicator keys mismatch. Missing={missing}; extra={extra}"
        )

    score_result = calculate_dimension_scores(indicators)

    if not score_result["ready_for_inference"]:
        return {
            "status": "incomplete",
            "message": (
                "A complete profile cannot be assigned because one or "
                "more required dimension scores are unavailable."
            ),
            "dimension_scores": score_result["dimension_scores"],
            "components_available": score_result["components_available"],
            "unavailable_dimensions": score_result["unavailable_dimensions"],
            "ready_for_inference": False,
            "missing_data_policy": score_result["missing_data_policy"],
        }

    profile = predict_inclusion_profile(
        score_result["dimension_scores"]
    )

    explanation = explain_inclusion_profile(
        profile["dimension_scores"],
        profile["cluster_id"],
    )

    return {
        "status": "success",
        "ready_for_inference": True,
        "input_indicators": indicators.copy(),
        "dimension_scores": score_result["dimension_scores"],
        "components_available": score_result["components_available"],
        "profile": profile,
        "explanation": explanation,
        "missing_data_policy": score_result["missing_data_policy"],
        "interpretation_limit": (
            "This is a descriptive financial inclusion profile. It is "
            "not a creditworthiness, loan eligibility, default-risk, "
            "or causal assessment."
        ),
    }
