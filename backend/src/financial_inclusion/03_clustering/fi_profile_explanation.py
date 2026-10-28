
"""Explain financial inclusion profiles and suggest evidence-linked actions.

This module does not train a model, change cluster assignments, or make
creditworthiness decisions. Recommendations are decision-support suggestions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

PROFILE_PATH = (
    ROOT
    / "outputs/financial_inclusion/reports/03_clustering/"
    "fi_clustering_profile_labels_5D_k5.json"
)

CLUSTER_INPUT_PATH = (
    ROOT
    / "data/financial_inclusion/processed/"
    "fi_clustering_input.csv"
)

LABELS_PATH = (
    ROOT
    / "outputs/financial_inclusion/metrics/"
    "fi_clustering_labels_5D_k5.csv"
)

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]

DIMENSION_NAMES = {
    "score_connectivity": "Connectivity",
    "score_financial_access": "Financial Access",
    "score_digital_financial_usage": "Digital Financial Usage",
    "score_digital_safety": "Digital Safety",
    "score_financial_resilience": "Financial Resilience",
}

# These are suggested actions, not causal claims or guaranteed solutions.
ACTION_GUIDANCE = {
    "score_connectivity": {
        "action": (
            "Assess barriers to reliable phone and internet access; "
            "consider accessible connectivity support where appropriate."
        ),
        "stakeholder": "Policymakers, NGOs, connectivity providers",
        "monitor": (
            "Phone/internet access indicators and connectivity score "
            "in a subsequent comparable assessment."
        ),
    },
    "score_financial_access": {
        "action": (
            "Investigate barriers to account access and review the "
            "accessibility, affordability, and availability of suitable services."
        ),
        "stakeholder": (
            "Financial service providers, policymakers, NGOs"
        ),
        "monitor": (
            "Account-access indicator and financial-access score "
            "in a subsequent comparable assessment."
        ),
    },
    "score_digital_financial_usage": {
        "action": (
            "Assess barriers to using digital financial payments, including "
            "awareness, usability, merchant acceptance, and transaction costs."
        ),
        "stakeholder": (
            "Financial service providers, merchants, NGOs"
        ),
        "monitor": (
            "Digital-payment usage indicator and digital-usage score "
            "in a subsequent comparable assessment."
        ),
    },
    "score_digital_safety": {
        "action": (
            "Consider practical phone-protection awareness and accessible "
            "guidance on safer device use."
        ),
        "stakeholder": (
            "Financial service providers, consumer-protection bodies, NGOs"
        ),
        "monitor": (
            "The available phone-protection indicator and digital-safety "
            "score in a subsequent comparable assessment."
        ),
    },
    "score_financial_resilience": {
        "action": (
            "Assess barriers to saving and financial coping capacity; "
            "consider accessible savings information and appropriate "
            "financial-support services."
        ),
        "stakeholder": (
            "Financial service providers, policymakers, NGOs"
        ),
        "monitor": (
            "The underlying resilience indicators and resilience score "
            "in a subsequent comparable assessment."
        ),
    },
}


def load_profile_definitions() -> dict[str, Any]:
    """Load and validate the final cluster-profile definitions."""
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Profile definitions not found: {PROFILE_PATH}"
        )

    with PROFILE_PATH.open("r", encoding="utf-8") as file:
        definitions = json.load(file)

    if definitions.get("status") != "final":
        raise ValueError("Cluster profile definitions are not marked final.")

    if definitions.get("labels_are_final") is not True:
        raise ValueError("Cluster labels are not marked final.")

    if definitions.get("candidate") != "5D_k5":
        raise ValueError("Unexpected clustering candidate.")

    if not isinstance(definitions.get("cluster_profiles"), dict):
        raise ValueError("Cluster profile definitions are missing.")

    return definitions


def load_cluster_centres() -> dict[int, dict[str, float]]:
    """Calculate descriptive mean scores from the saved labelled dataset.

    These means describe the existing analytical sample. They are not
    additional model outputs and are not causal explanations.
    """
    if not CLUSTER_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Clustering input not found: {CLUSTER_INPUT_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Saved cluster labels not found: {LABELS_PATH}"
        )

    data = pd.read_csv(CLUSTER_INPUT_PATH)
    labels = pd.read_csv(LABELS_PATH)

    if len(data) != len(labels):
        raise ValueError("Clustering input and labels row counts differ.")

    if not set(FEATURES).issubset(data.columns):
        raise ValueError("Clustering input is missing dimension scores.")

    if "cluster_id" not in labels.columns:
        raise ValueError("Saved labels do not contain cluster_id.")

    scores = data[FEATURES].apply(pd.to_numeric, errors="raise")

    if scores.isna().any().any():
        raise ValueError(
            "Dimension scores contain missing values; refusing to "
            "calculate incomplete cluster means."
        )

    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Dimension scores contain non-finite values.")

    if not scores.ge(0).all().all() or not scores.le(1).all().all():
        raise ValueError("Dimension scores must be between 0 and 1.")

    means = scores.assign(
        cluster_id=labels["cluster_id"].to_numpy()
    ).groupby("cluster_id")[FEATURES].mean()

    return {
        int(cluster_id): {
            feature: float(row[feature])
            for feature in FEATURES
        }
        for cluster_id, row in means.iterrows()
    }


def explain_inclusion_profile(
    dimension_scores: dict[str, float],
    cluster_id: int,
) -> dict[str, Any]:
    """Explain a saved cluster assignment using dimension-level evidence.

    The supplied cluster_id should come from predict_inclusion_profile().
    """
    definitions = load_profile_definitions()
    cluster_profiles = definitions["cluster_profiles"]
    cluster_key = str(cluster_id)

    if cluster_key not in cluster_profiles:
        raise ValueError(f"Unknown cluster_id: {cluster_id}")

    if set(dimension_scores) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(dimension_scores))
        extra = sorted(set(dimension_scores) - set(FEATURES))
        raise ValueError(
            f"Dimension score keys do not match. Missing={missing}; "
            f"extra={extra}"
        )

    scores: dict[str, float] = {}

    for feature in FEATURES:
        try:
            value = float(dimension_scores[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{feature} must be a finite numeric value."
            ) from exc

        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                f"{feature} must be between 0 and 1."
            )

        scores[feature] = value

    cluster_means = load_cluster_centres()
    if cluster_id not in cluster_means:
        raise ValueError(
            f"No observed cluster means for cluster {cluster_id}."
        )

    profile = cluster_profiles[cluster_key]
    dimension_evidence = []

    for feature in FEATURES:
        individual_score = scores[feature]
        cluster_mean = cluster_means[cluster_id][feature]
        difference = individual_score - cluster_mean

        if individual_score < 0.4:
            observation = "Low score; consider reviewing this dimension."
        elif individual_score < 0.7:
            observation = "Intermediate score; potential improvement area."
        else:
            observation = "Relatively high score on this dimension."

        dimension_evidence.append(
            {
                "dimension": DIMENSION_NAMES[feature],
                "feature": feature,
                "individual_score": round(individual_score, 4),
                "cluster_mean_score": round(cluster_mean, 4),
                "difference_from_cluster_mean": round(difference, 4),
                "observation": observation,
            }
        )

    # Rank dimensions by their individual score for review.
    # The 0.4 and 0.7 cutoffs are provisional display heuristics,
    # not validated policy thresholds.
    lowest_dimensions = sorted(
        dimension_evidence,
        key=lambda item: item["individual_score"],
    )

    def include_boundary_ties(limit: int) -> list[dict[str, Any]]:
        selected = lowest_dimensions[:limit]
        if not selected:
            return selected
        boundary_score = selected[-1]["individual_score"]
        selected.extend(
            item
            for item in lowest_dimensions[limit:]
            if np.isclose(
                item["individual_score"], boundary_score, rtol=0, atol=1e-9
            )
        )
        return selected

    priority_dimensions = include_boundary_ties(3)
    lowest_scoring = include_boundary_ties(2)
    suggested_actions = []
    for item in priority_dimensions:
        guidance = ACTION_GUIDANCE[item["feature"]]
        suggested_actions.append(
            {
                "dimension": item["dimension"],
                "reason": (
                    "Prioritised because its score is among this "
                    "individual's lowest dimension scores."
                ),
                **guidance,
            }
        )

    return {
        "cluster_id": cluster_id,
        "profile_name": profile["name"],
        "profile_description": profile["description"],
        "dimension_evidence": dimension_evidence,
        "lowest_scoring_dimensions": [item["dimension"] for item in lowest_scoring],
        "suggested_actions": suggested_actions,
        "interpretation_limits": [
            "Cluster assignment is descriptive, not causal.",
            "Cluster means describe the saved analytical sample.",
            "Recommendations are suggestions for stakeholder review, "
            "not guaranteed interventions.",
            "Digital safety represents the available phone-protection "
            "measure only.",
            "Score cutoffs are provisional display heuristics, not "
            "validated policy thresholds.",
            "Do not use this profile alone to determine creditworthiness, "
            "loan eligibility, or default risk.",
        ],
    }
