"""Leakage tests: forbidden score items / target-derived fields must never enter model inputs."""

import pytest

from backend.src.financial_capability.fc_config import (
    FEATURE_NAMES,
    FORBIDDEN_MODEL_COLUMNS,
    FWB_SCALE_ITEMS,
    OUTCOME_ADJACENT_FIELDS,
    SENSITIVE_FIELDS,
)
from backend.src.financial_capability.fc_feature_selection import (
    LeakageError,
    assert_no_forbidden_features,
    select_features_and_target,
)
from backend.src.financial_capability.fc_predictor import InvalidResponseError, validate_response


def test_ten_score_items_are_declared_forbidden():
    assert len(FWB_SCALE_ITEMS) == 10
    assert set(FWB_SCALE_ITEMS) <= set(FORBIDDEN_MODEL_COLUMNS)
    assert "FWBscore" in FORBIDDEN_MODEL_COLUMNS


def test_approved_features_contain_no_forbidden_field():
    assert not set(FEATURE_NAMES) & set(FORBIDDEN_MODEL_COLUMNS)
    assert_no_forbidden_features(FEATURE_NAMES)  # must not raise


def test_X_never_contains_forbidden_fields(raw_df):
    X, _ = select_features_and_target(raw_df)
    assert not set(X.columns) & set(FORBIDDEN_MODEL_COLUMNS)
    assert list(X.columns) == list(FEATURE_NAMES)


@pytest.mark.parametrize("bad", ["FWB1_1", "FWB2_4", "FWBscore", "ENDSMEET", "PPGENDER", "finalwt"])
def test_guard_fails_if_forbidden_field_enters_X(bad):
    with pytest.raises(LeakageError):
        assert_no_forbidden_features(list(FEATURE_NAMES) + [bad])


def test_outcome_and_sensitive_fields_excluded():
    assert set(OUTCOME_ADJACENT_FIELDS) <= set(FORBIDDEN_MODEL_COLUMNS)
    assert set(SENSITIVE_FIELDS) <= set(FORBIDDEN_MODEL_COLUMNS)


def test_target_not_correlated_by_copy(raw_df):
    X, y = select_features_and_target(raw_df)
    assert "target" not in X.columns
    assert len(X) == len(y)


def test_user_inputs_cannot_use_leakage_fields(sample_response):
    for bad in ("FWB1_3", "FWBscore"):
        with pytest.raises(InvalidResponseError, match="Forbidden"):
            validate_response({**sample_response, bad: 3})

