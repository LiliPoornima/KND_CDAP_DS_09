"""What-if analysis: change selected answers and compare model outputs.

This is a MODEL SIMULATION. It does not show that changing a real answer
would change a real-world outcome (the survey is cross-sectional).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .fc_config import CLASS_NAMES, FEATURE_SPEC_BY_NAME, GROUP_CONTEXT
from .fc_predictor import InvalidResponseError, ModelBundle, predict_assessment, validate_response

WHAT_IF_DISCLAIMER = (
    "This is a model-based scenario. It does not prove that changing this answer "
    "will cause a real-world outcome to change."
)

# Only capability questions can be changed; background fields stay fixed.
CHANGEABLE_FEATURES = tuple(
    name for name, spec in FEATURE_SPEC_BY_NAME.items() if spec.group != GROUP_CONTEXT
)


def run_what_if(model, response_row: Mapping[str, Any], changes: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare the original prediction with one where ``changes`` are applied.

    The original ``response_row`` is never mutated. Changes to the target,
    leakage fields, unknown fields, background fields or values outside the
    codebook are rejected.
    """
    if not changes:
        raise InvalidResponseError("Provide at least one change for the what-if analysis.")
    not_allowed = sorted(set(changes) - set(CHANGEABLE_FEATURES))
    if not_allowed:
        raise InvalidResponseError(
            f"These fields cannot be changed in what-if analysis: {not_allowed}")
    validate_response(changes)  # codebook value checks on the changed values

    original_input = dict(response_row)  # copy: keep the caller's row untouched
    scenario_input = {**original_input, **dict(changes)}

    before = predict_assessment(model, original_input)
    after = predict_assessment(model, scenario_input)
    delta = {c: after["probabilities"][c] - before["probabilities"][c] for c in CLASS_NAMES}
    return {
        "original": before,
        "what_if": after,
        "changes": {
            name: {
                "from": FEATURE_SPEC_BY_NAME[name].options.get(
                    original_input.get(name), "Not answered"),
                "to": FEATURE_SPEC_BY_NAME[name].options[value],
            } for name, value in changes.items()
        },
        "probability_change": delta,
        "class_changed": before["predicted_class"] != after["predicted_class"],
        "disclaimer": WHAT_IF_DISCLAIMER,
    }

