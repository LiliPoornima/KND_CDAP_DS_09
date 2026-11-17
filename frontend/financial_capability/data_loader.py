"""Cached loaders and shared helpers for the Component 05 dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Make ``backend`` importable when the page is run directly or through the main app.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.src.financial_capability import fc_predictor  # noqa: E402
from backend.src.financial_capability.fc_config import MODEL_DIR, SELECTED_MODEL_FILE  # noqa: E402
from backend.src.financial_capability.fc_explainability import FeatureExplainer  # noqa: E402

from frontend.financial_capability.constants import SESSION_RESPONSE_KEY  # noqa: E402


def _model_stamp() -> float:
    """File modification time, so caches refresh after the model is retrained."""
    path = MODEL_DIR / SELECTED_MODEL_FILE
    return path.stat().st_mtime if path.is_file() else 0.0


@st.cache_resource(show_spinner="Loading the saved Component 05 model...")
def _load_bundle(stamp: float):
    return fc_predictor.load_model_bundle()


@st.cache_resource(show_spinner="Preparing the SHAP explainer...")
def _load_explainer(stamp: float):
    return FeatureExplainer(_load_bundle(stamp).pipeline)


def get_bundle() -> Optional[fc_predictor.ModelBundle]:
    """The saved model bundle, or None if the model has not been trained yet."""
    if not fc_predictor.artifacts_available():
        return None
    return _load_bundle(_model_stamp())


def get_explainer() -> Optional[FeatureExplainer]:
    """Cached SHAP explainer for the saved model (None if not trained)."""
    if not fc_predictor.artifacts_available():
        return None
    return _load_explainer(_model_stamp())


def get_metadata() -> Optional[Dict[str, Any]]:
    """Training metadata (real metrics) or None."""
    bundle = get_bundle()
    return bundle.metadata if bundle else None


def show_not_trained() -> None:
    """Friendly empty state shown instead of any (fake) metric."""
    st.warning("**Model not trained yet.** No results are shown until the model has been trained "
               "on the CFPB data.")
    st.markdown("1. Place `NFWBS_PUF_2016_data.csv` in `data/financial_capability/raw/`.\n"
                "2. From the repository root run:")
    st.code(fc_predictor.TRAIN_COMMAND, language="powershell")


def require_bundle() -> Optional[fc_predictor.ModelBundle]:
    """Return the bundle, or show the empty state and return None."""
    bundle = get_bundle()
    if bundle is None:
        show_not_trained()
    return bundle


def current_response() -> Optional[Dict[str, Any]]:
    """The answers entered in the Individual Assessment (kept only in this browser session)."""
    return st.session_state.get(SESSION_RESPONSE_KEY)


def require_response() -> Optional[Dict[str, Any]]:
    """Return the session answers, or show a hint and return None."""
    resp = current_response()
    if not resp:
        st.info("No assessment yet. Open **Individual Assessment**, answer the questions "
                "(or use the example answers) and press *Run assessment*.")
    return resp

