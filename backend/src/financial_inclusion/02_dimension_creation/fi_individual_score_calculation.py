
"""Calculate individual financial inclusion dimension scores.

Uses the current dimension-construction mappings. It does not impute
missing indicators. Inference should proceed only when all five
dimension scores are available.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


INDICATOR_KEYS = [
    "con1",
    "internet_use",
    "account",
    "anydigpayment",
    "con18",
    "saved",
    "fin24a",
    "fin24b",
]

SCORE_KEYS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def _numeric_or_missing(value: Any) -> float | None:
    """Return a finite numeric value, or None for missing/invalid input."""
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if not isinstance(value, Real):
        try:
            if str(value).strip() == "":
                return None
            value = float(value)
        except (TypeError, ValueError):
            return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def _binary_indicator(
    value: Any,
    *,
    yes_code: float,
    no_code: float,
) -> float | None:
    """Map only documented yes/no codes; other values remain missing."""
    number = _numeric_or_missing(value)

    if number == yes_code:
        return 1.0

    if number == no_code:
        return 0.0

    return None


def _existing_binary(value: Any) -> float | None:
    """Accept only an already-constructed 0/1 indicator."""
    number = _numeric_or_missing(value)

    if number in (0.0, 1.0):
        return number

    return None


def calculate_dimension_scores(
    indicators: dict[str, Any],
) -> dict[str, Any]:
    """Calculate the five dimension scores using current project rules.

    Connectivity uses the mean of available phone/internet components.
    Financial resilience requires all three of its components.
    No missing indicator is converted to zero or statistically imputed.
    """
    if not isinstance(indicators, dict):
        raise TypeError("indicators must be a dictionary.")

    missing_keys = sorted(set(INDICATOR_KEYS) - set(indicators))
    extra_keys = sorted(set(indicators) - set(INDICATOR_KEYS))

    if missing_keys or extra_keys:
        raise ValueError(
            f"Indicator keys mismatch. Missing={missing_keys}; "
            f"extra={extra_keys}"
        )

    # Dimension 1: Connectivity
    phone = _binary_indicator(
        indicators["con1"], yes_code=1, no_code=2
    )
    internet = _existing_binary(indicators["internet_use"])

    connectivity_components = [
        value for value in (phone, internet) if value is not None
    ]
    connectivity = (
        sum(connectivity_components) / len(connectivity_components)
        if connectivity_components
        else None
    )

    # Dimension 2: Financial Access
    financial_access = _existing_binary(indicators["account"])

    # Dimension 3: Digital Financial Usage
    digital_usage = _existing_binary(indicators["anydigpayment"])

    # Dimension 4: Digital Safety.
    # This is the phone-protection measure only; con21 is not used here.
    digital_safety = _binary_indicator(
        indicators["con18"], yes_code=1, no_code=2
    )

    # Dimension 5: Financial Resilience
    saved = _existing_binary(indicators["saved"])

    emergency_ease_mapping = {
        1.0: 0.0,
        2.0: 0.5,
        3.0: 1.0,
    }
    emergency_ease_code = _numeric_or_missing(indicators["fin24a"])
    emergency_ease = emergency_ease_mapping.get(emergency_ease_code)

    income_buffer_mapping = {
        1.0: 0.0,
        2.0: 1.0 / 3.0,
        3.0: 2.0 / 3.0,
        4.0: 1.0,
    }
    income_buffer_code = _numeric_or_missing(indicators["fin24b"])
    income_buffer = income_buffer_mapping.get(income_buffer_code)

    resilience_components = [saved, emergency_ease, income_buffer]
    resilience_complete = all(
        value is not None for value in resilience_components
    )

    resilience = (
        sum(resilience_components) / 3
        if resilience_complete
        else None
    )

    scores = {
        "score_connectivity": connectivity,
        "score_financial_access": financial_access,
        "score_digital_financial_usage": digital_usage,
        "score_digital_safety": digital_safety,
        "score_financial_resilience": resilience,
    }

    component_availability = {
        "score_connectivity": len(connectivity_components),
        "score_financial_access": int(financial_access is not None),
        "score_digital_financial_usage": int(digital_usage is not None),
        "score_digital_safety": int(digital_safety is not None),
        "score_financial_resilience": sum(
            value is not None for value in resilience_components
        ),
    }

    unavailable_dimensions = [
        score_name
        for score_name, value in scores.items()
        if value is None
    ]

    return {
        "dimension_scores": scores,
        "components_available": component_availability,
        "unavailable_dimensions": unavailable_dimensions,
        "ready_for_inference": not unavailable_dimensions,
        "missing_data_policy": (
            "No imputation. Connectivity averages available components; "
            "financial resilience requires all three components. "
            "Inference is blocked if any dimension score is unavailable."
        ),
    }
