
"""Inference utilities for validated five-dimensional inclusion profiles."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

MODEL_FILE = (
    ROOT / "models/financial_inclusion/kmeans_5D_k5.joblib"
)
METADATA_FILE = (
    ROOT / "models/financial_inclusion/kmeans_5D_k5_metadata.json"
)
PROFILE_FILE = (
    ROOT / "outputs/financial_inclusion/reports/03_clustering/"
    "fi_clustering_profile_labels_5D_k5.json"
)

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def load_profile_artifacts():
    """Load and validate the saved model and its metadata."""
    for path in [MODEL_FILE, METADATA_FILE, PROFILE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    profile_metadata = json.loads(
        PROFILE_FILE.read_text(encoding="utf-8")
    )

    if metadata.get("validated") is not True:
        raise ValueError("KMeans artifact has not passed validation.")

    if metadata.get("candidate") != "5D_k5":
        raise ValueError("Unexpected clustering candidate.")

    if metadata.get("features") != FEATURES:
        raise ValueError("KMeans feature order does not match.")

    if profile_metadata.get("labels_are_final") is not True:
        raise ValueError("Final cluster labels are not confirmed.")

    model = joblib.load(MODEL_FILE)

    if model.n_clusters != 5:
        raise ValueError("Expected a five-cluster KMeans model.")

    return model, profile_metadata["cluster_profiles"]


def validate_dimension_scores(dimension_scores):
    """Validate all five required dimension scores in the range [0, 1]."""
    if not isinstance(dimension_scores, dict):
        raise TypeError("Dimension scores must be provided as a dictionary.")

    missing = [feature for feature in FEATURES if feature not in dimension_scores]
    extra = [feature for feature in dimension_scores if feature not in FEATURES]

    if missing:
        raise ValueError(f"Missing dimension scores: {missing}")

    if extra:
        raise ValueError(f"Unexpected dimension scores: {extra}")

    values = []
    for feature in FEATURES:
        try:
            value = float(dimension_scores[feature])
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be a numeric score.") from None

        if not np.isfinite(value):
            raise ValueError(f"{feature} must be finite.")

        if not 0 <= value <= 1:
            raise ValueError(f"{feature} must be between 0 and 1.")

        values.append(value)

    return pd.DataFrame([values], columns=FEATURES)


def predict_inclusion_profile(dimension_scores):
    """Return a profile assignment from the five dimension scores."""
    X = validate_dimension_scores(dimension_scores)
    model, profiles = load_profile_artifacts()

    cluster_id = int(model.predict(X)[0])
    profile = profiles.get(str(cluster_id))

    if profile is None:
        raise ValueError(
            f"No documented profile found for cluster {cluster_id}."
        )

    return {
        "cluster_id": cluster_id,
        "profile_name": profile["name"],
        "profile_description": profile["description"],
        "dimension_scores": {
            feature: float(X.iloc[0][feature]) for feature in FEATURES
        },
        "assignment_method": "Validated 5D_k5 KMeans on raw [0, 1] scores",
        "interpretation_limit": (
            "This is a descriptive financial inclusion profile, not a "
            "creditworthiness, loan-default, or causal assessment."
        ),
    }
