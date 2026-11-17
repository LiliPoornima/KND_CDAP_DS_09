"""Prediction, artifact loading and what-if tests."""

import copy

import joblib
import json
import pytest

from backend.src.financial_capability.fc_config import (
    CLASS_NAMES,
    METADATA_FILE,
    SELECTED_MODEL_FILE,
)
from backend.src.financial_capability.fc_predictor import (
    InvalidResponseError,
    ModelNotTrainedError,
    artifacts_available,
    load_model_bundle,
    predict_assessment,
    validate_response,
)
from backend.src.financial_capability.fc_what_if import (
    CHANGEABLE_FEATURES,
    WHAT_IF_DISCLAIMER,
    run_what_if,
)


def test_prediction_returns_valid_class_and_probabilities(rf_model, sample_response):
    res = predict_assessment(rf_model, sample_response)
    assert res["predicted_class"] in CLASS_NAMES
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-6
    assert res["confidence"] == max(res["probabilities"].values())


def test_missing_answers_are_allowed(rf_model, sample_response):
    partial = {k: v for k, v in sample_response.items() if k in ("SAVEHABIT", "PPINCIMP")}
    assert predict_assessment(rf_model, partial)["predicted_class"] in CLASS_NAMES


@pytest.mark.parametrize("bad", [{"SAVEHABIT": 99}, {"SAVEHABIT": "high"}, {"GOALCONF": 0},
                                 {"NOT_A_FIELD": 1}, {"FWB1_1": 3}, {"SAVEHABIT": True}])
def test_invalid_responses_rejected(rf_model, sample_response, bad):
    with pytest.raises(InvalidResponseError):
        predict_assessment(rf_model, {**sample_response, **bad})


def test_validate_response_keeps_feature_order_and_input_untouched(sample_response):
    before = copy.deepcopy(sample_response)
    frame = validate_response(sample_response)
    assert frame.shape == (1, 31)
    assert sample_response == before


# ---------------------------------------------------------------- what-if
def test_what_if_compares_and_does_not_mutate_original(xgb_model, sample_response):
    original = copy.deepcopy(sample_response)
    out = run_what_if(xgb_model, sample_response, {"SAVEHABIT": 6})
    assert sample_response == original                       # original row unchanged
    assert out["original"]["row"].loc[0, "SAVEHABIT"] == 4   # original prediction used old value
    assert out["what_if"]["row"].loc[0, "SAVEHABIT"] == 6
    assert out["changes"]["SAVEHABIT"] == {"from": "Agree slightly", "to": "Strongly agree"}
    assert set(out["probability_change"]) == set(CLASS_NAMES)
    assert abs(sum(out["probability_change"].values())) < 1e-6
    assert out["disclaimer"] == WHAT_IF_DISCLAIMER and "does not prove" in out["disclaimer"]


def test_what_if_same_value_gives_no_change(rf_model, sample_response):
    out = run_what_if(rf_model, sample_response, {"SAVEHABIT": sample_response["SAVEHABIT"]})
    assert not out["class_changed"]
    assert all(abs(v) < 1e-9 for v in out["probability_change"].values())


@pytest.mark.parametrize("changes", [
    {"SAVEHABIT": 42},          # value outside codebook
    {"FWBscore": 80},           # target
    {"FWB1_2": 5},              # leakage field
    {"PPINCIMP": 9},            # background field is not changeable
    {"UNKNOWN": 1},
    {},
])
def test_what_if_rejects_invalid_changes(rf_model, sample_response, changes):
    original = copy.deepcopy(sample_response)
    with pytest.raises(InvalidResponseError):
        run_what_if(rf_model, sample_response, changes)
    assert sample_response == original


def test_changeable_fields_are_capability_questions_only():
    assert "SAVEHABIT" in CHANGEABLE_FEATURES and "PPINCIMP" not in CHANGEABLE_FEATURES


# ---------------------------------------------------------------- artifacts
def test_missing_artifacts_raise_helpful_error(tmp_path):
    assert not artifacts_available(tmp_path)
    with pytest.raises(ModelNotTrainedError, match="fc_run_pipeline"):
        load_model_bundle(tmp_path)


def test_saved_model_loads_and_predicts_without_retraining(rf_model, sample_response, tmp_path):
    joblib.dump(rf_model, tmp_path / SELECTED_MODEL_FILE)
    (tmp_path / METADATA_FILE).write_text(json.dumps({"selected_model": "Random Forest"}))
    assert artifacts_available(tmp_path)
    bundle = load_model_bundle(tmp_path)
    assert predict_assessment(bundle, sample_response)["predicted_class"] in CLASS_NAMES
    a = predict_assessment(rf_model, sample_response)["probabilities"]
    b = predict_assessment(bundle, sample_response)["probabilities"]
    assert a == pytest.approx(b)

