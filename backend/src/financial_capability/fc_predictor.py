"""Saved-model loading, response validation and single-assessment prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from .fc_config import (
    CLASS_NAMES,
    FEATURE_NAMES,
    FEATURE_SPEC_BY_NAME,
    FORBIDDEN_MODEL_COLUMNS,
    METADATA_FILE,
    MODEL_DIR,
    SELECTED_MODEL_FILE,
)

TRAIN_COMMAND = "python -m backend.src.financial_capability.fc_run_pipeline"
LOW_CONFIDENCE_THRESHOLD = 0.50


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the saved model artifacts are missing."""


class InvalidResponseError(ValueError):
    """Raised when a response value or field is not allowed by the codebook."""


@dataclass
class ModelBundle:
    """A fitted pipeline (preprocessing + model) with its metadata."""

    pipeline: Any
    metadata: Dict[str, Any]


def artifacts_available(model_dir: Optional[Path] = None) -> bool:
    """True if the selected model and metadata files exist."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    return (d / SELECTED_MODEL_FILE).is_file() and (d / METADATA_FILE).is_file()


def load_metadata(model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the metadata JSON written at training time."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    path = d / METADATA_FILE
    if not path.is_file():
        raise ModelNotTrainedError(f"Model not trained yet. Run: {TRAIN_COMMAND}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle(model_dir: Optional[Path] = None) -> ModelBundle:
    """Load the saved selected model (never retrains)."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    if not artifacts_available(d):
        raise ModelNotTrainedError(f"Model not trained yet. Run: {TRAIN_COMMAND}")
    return ModelBundle(pipeline=joblib.load(d / SELECTED_MODEL_FILE), metadata=load_metadata(d))


def validate_response(response_row: Mapping[str, Any]) -> pd.DataFrame:
    """Validate a response and return a one-row DataFrame in FEATURE_NAMES order.

    Missing questions are allowed (NaN, imputed by the saved pipeline). Unknown
    fields, forbidden leakage fields and values outside the codebook are rejected.
    The input mapping is never modified.
    """
    forbidden = sorted(set(response_row) & set(FORBIDDEN_MODEL_COLUMNS))
    if forbidden:
        raise InvalidResponseError(f"Forbidden fields cannot be used as inputs: {forbidden}")
    unknown = sorted(set(response_row) - set(FEATURE_NAMES))
    if unknown:
        raise InvalidResponseError(f"Unknown fields: {unknown}")
    values: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        raw = response_row.get(name)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            values[name] = np.nan
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise InvalidResponseError(f"{name}: value {raw!r} must be a number.")
        if float(raw) not in {float(k) for k in FEATURE_SPEC_BY_NAME[name].options}:
            valid = list(FEATURE_SPEC_BY_NAME[name].options)
            raise InvalidResponseError(f"{name}: value {raw!r} is not valid. Allowed: {valid}")
        values[name] = float(raw)
    return pd.DataFrame([values], columns=list(FEATURE_NAMES))


def predict_assessment(model, response_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Predict the financial well-being category for one response.

    ``model`` is a fitted pipeline (or a ModelBundle). Returns the predicted
    class, all class probabilities and a low-confidence flag.
    """
    pipeline = model.pipeline if isinstance(model, ModelBundle) else model
    row = validate_response(response_row)
    proba = pipeline.predict_proba(row)[0]
    idx = int(np.argmax(proba))
    return {
        "predicted_class": CLASS_NAMES[idx],
        "predicted_index": idx,
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "confidence": float(proba[idx]),
        "low_confidence": bool(proba[idx] < LOW_CONFIDENCE_THRESHOLD),
        "row": row,
    }

