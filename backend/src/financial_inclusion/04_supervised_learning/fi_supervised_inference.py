
"""Reusable inference utilities for Component 02 supervised cluster prediction.

Uses the existing saved Logistic Regression pipeline without fitting or
modifying its preprocessing steps or classifier.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FEATURES = [
    "female",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "regionwb",
    "age",
]

CATEGORICAL_FEATURES = [
    "female",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "regionwb",
]

CLUSTER_LABELS = {
    0: "Digitally Active, Safety Gap",
    1: "Connected and Digitally Included",
    2: "Connected, Financially Excluded",
    3: "Multi-Dimensional Inclusion Gap",
    4: "Access Without Digital Usage",
}


def get_project_root() -> Path:
    """Resolve the repository root from this module's location."""
    return Path(__file__).resolve().parents[4]


def load_model(model_path: str | Path | None = None) -> Any:
    """Load the saved Logistic Regression pipeline; never fit it."""
    path = (
        Path(model_path)
        if model_path is not None
        else get_project_root()
        / "models"
        / "financial_inclusion"
        / "logistic_regression.joblib"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")

    pipeline = joblib.load(path)

    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a saved sklearn Pipeline.")

    required_steps = {"preprocessing", "model"}
    actual_steps = set(pipeline.named_steps)

    if not required_steps.issubset(actual_steps):
        raise ValueError(
            f"Unexpected pipeline steps. Found: {sorted(actual_steps)}"
        )

    actual_features = list(pipeline.feature_names_in_)

    if set(actual_features) != set(FEATURES):
        raise ValueError(
            "Saved pipeline features do not match the expected predictors. "
            f"Found: {actual_features}"
        )

    classes = set(
        int(value)
        for value in pipeline.named_steps["model"].classes_
    )

    if classes != set(CLUSTER_LABELS):
        raise ValueError(
            f"Unexpected cluster classes in saved model: {sorted(classes)}"
        )

    return pipeline


def validate_inputs(
    inputs: pd.DataFrame | dict[str, Any],
    pipeline: Any,
) -> pd.DataFrame:
    """Validate predictor columns and preserve the pipeline's input order.

    Missing values within supplied columns are allowed because the saved
    pipeline contains fitted imputers. Missing columns are not allowed.
    """
    if isinstance(inputs, dict):
        frame = pd.DataFrame([inputs])
    elif isinstance(inputs, pd.DataFrame):
        frame = inputs.copy()
    else:
        raise TypeError("Inputs must be a pandas DataFrame or a dictionary.")

    if frame.empty:
        raise ValueError("At least one input row is required.")

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate predictor columns: {duplicates}")

    expected = set(pipeline.feature_names_in_)
    provided = set(frame.columns)

    missing_columns = expected - provided
    unexpected_columns = provided - expected

    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}; "
            f"unexpected columns: {sorted(unexpected_columns)}"
        )

    # Always supply named columns in the saved pipeline's expected order.
    frame = frame.loc[:, list(pipeline.feature_names_in_)].copy()

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype(object)
        frame[column] = frame[column].where(
            frame[column].notna(), np.nan
        )

    try:
        frame["age"] = pd.to_numeric(frame["age"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "'age' must contain numeric values or missing values."
        ) from exc

    return frame


def predict_profiles(
    inputs: pd.DataFrame | dict[str, Any],
    pipeline: Any | None = None,
) -> pd.DataFrame:
    """Return predicted cluster IDs and per-class probabilities."""
    if pipeline is None:
        pipeline = load_model()

    frame = validate_inputs(inputs, pipeline)

    classifier = pipeline.named_steps["model"]
    predictions = pipeline.predict(frame)
    probabilities = pipeline.predict_proba(frame)
    classes = [int(value) for value in classifier.classes_]

    if probabilities.shape != (len(frame), len(classes)):
        raise ValueError(
            f"Unexpected probability shape: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("Model probabilities contain non-finite values.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Model probabilities do not sum to 1.")

    results = pd.DataFrame(index=frame.index)
    results["predicted_cluster_id"] = predictions.astype(int)
    results["predicted_cluster_label"] = (
        results["predicted_cluster_id"].map(CLUSTER_LABELS)
    )

    results["predicted_cluster_probability"] = [
        float(probabilities[row, classes.index(int(prediction))])
        for row, prediction in enumerate(predictions)
    ]

    for index, cluster_id in enumerate(classes):
        results[f"probability_cluster_{cluster_id}"] = probabilities[:, index]

    return results.reset_index(drop=True)


def get_transformed_features(
    inputs: pd.DataFrame | dict[str, Any],
    pipeline: Any | None = None,
) -> pd.DataFrame:
    """Return fitted preprocessing output with its encoded feature names."""
    if pipeline is None:
        pipeline = load_model()

    frame = validate_inputs(inputs, pipeline)
    preprocessor = pipeline.named_steps["preprocessing"]

    transformed = preprocessor.transform(frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    names = preprocessor.get_feature_names_out()

    if transformed.shape[1] != len(names):
        raise ValueError(
            "Transformed data width does not match feature-name count."
        )

    return pd.DataFrame(
        transformed,
        columns=names,
        index=frame.index,
    )
