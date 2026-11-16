"""
Tests for credit-risk feature engineering.
"""

import pandas as pd
import pytest

from backend.src.credit_risk.give_me_some_credit.feature_engineering import (
    cr_create_credit_features,
)


@pytest.fixture
def sample_credit_data():
    """Create a small dataset for testing feature creation."""

    return pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1, 0],
            "NumberOfTime30-59DaysPastDueNotWorse": [1, 0, 2],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 1],
            "NumberOfTimes90DaysLate": [0, 2, 1],
            "NumberOfOpenCreditLinesAndLoans": [5, 8, 10],
            "NumberRealEstateLoansOrLines": [1, 2, 0],
            "NumberOfDependents": [0, 2, 1],
        }
    )


def test_creates_expected_features(sample_credit_data):
    """Check that all four derived features are created."""

    result = cr_create_credit_features(sample_credit_data)

    expected_columns = [
        "TotalDelinquencyCount",
        "SevereDelinquencyCount",
        "TotalOpenCreditAccounts",
        "HasDependents",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_calculates_total_delinquency_count(sample_credit_data):
    """Check the sum of the three delinquency-history columns."""

    result = cr_create_credit_features(sample_credit_data)

    assert result["TotalDelinquencyCount"].tolist() == [1, 3, 4]


def test_calculates_severe_delinquency_count(sample_credit_data):
    """Check the sum of the two more severe delinquency columns."""

    result = cr_create_credit_features(sample_credit_data)

    assert result["SevereDelinquencyCount"].tolist() == [0, 3, 2]


def test_calculates_total_open_credit_accounts(sample_credit_data):
    """Check the combined open-credit account count."""

    result = cr_create_credit_features(sample_credit_data)

    assert result["TotalOpenCreditAccounts"].tolist() == [6, 10, 10]


def test_creates_has_dependents_indicator(sample_credit_data):
    """Check that dependent counts are converted into binary values."""

    result = cr_create_credit_features(sample_credit_data)

    assert result["HasDependents"].tolist() == [0, 1, 1]


def test_preserves_original_data(sample_credit_data):
    """Check that the original DataFrame is not modified."""

    original = sample_credit_data.copy()

    cr_create_credit_features(sample_credit_data)

    pd.testing.assert_frame_equal(sample_credit_data, original)


def test_rejects_missing_required_column(sample_credit_data):
    """Check that missing required columns produce a clear error."""

    invalid_data = sample_credit_data.drop(
        columns=["NumberOfTimes90DaysLate"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        cr_create_credit_features(invalid_data)


def test_rejects_missing_values(sample_credit_data):
    """Check that missing values are not silently ignored."""

    invalid_data = sample_credit_data.copy()
    invalid_data.loc[0, "NumberOfDependents"] = float("nan")

    with pytest.raises(ValueError, match="contains missing values"):
        cr_create_credit_features(invalid_data)


def test_rejects_non_dataframe_input():
    """Check that the function requires a pandas DataFrame."""

    with pytest.raises(TypeError, match="pandas DataFrame"):
        cr_create_credit_features([1, 2, 3])