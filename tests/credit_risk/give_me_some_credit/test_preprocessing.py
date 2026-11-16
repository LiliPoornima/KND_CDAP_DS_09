"""Tests for Give Me Some Credit preprocessing helpers."""

import pandas as pd
import pytest

from backend.src.credit_risk.give_me_some_credit.preprocessing import (
    cr_apply_preprocessing,
    cr_fit_imputation,
)


def test_cr_fit_imputation_uses_numeric_median_and_categorical_mode():
    # Use a small dataset with both numeric and categorical missing values.
    df = pd.DataFrame(
        {
            "MonthlyIncome": [1000.0, None, 3000.0],
            "employment_type": ["full_time", None, "full_time"],
            "SeriousDlqin2yrs": [0, 1, 0],
        }
    )

    values = cr_fit_imputation(df, target_column="SeriousDlqin2yrs")

    # Median and mode should be learned from features, not the target.
    assert values["MonthlyIncome"] == 2000.0
    assert values["employment_type"] == "full_time"
    assert "SeriousDlqin2yrs" not in values


def test_cr_fit_imputation_handles_entirely_missing_features():
    # Specify the numeric dtype because an all-None column has no numeric type.
    df = pd.DataFrame(
        {
            "numeric_feature": pd.Series([None, None], dtype="float64"),
            "category_feature": [None, None],
        }
    )

    values = cr_fit_imputation(df)

    assert values["numeric_feature"] == 0.0
    assert values["category_feature"] == "__missing__"


def test_cr_fit_imputation_rejects_non_dataframe():
    # A list is not a supported tabular input.
    with pytest.raises(TypeError):
        cr_fit_imputation([1, 2, 3])


def test_cr_fit_imputation_rejects_empty_dataframe():
    # Empty input cannot provide meaningful imputation statistics.
    with pytest.raises(ValueError, match="empty DataFrame"):
        cr_fit_imputation(pd.DataFrame())


def test_cr_fit_imputation_rejects_missing_target_column():
    # A requested target must exist in the training data.
    df = pd.DataFrame({"age": [25, 40]})

    with pytest.raises(ValueError, match="does not exist"):
        cr_fit_imputation(df, target_column="SeriousDlqin2yrs")


def test_cr_fit_imputation_rejects_missing_target_labels():
    # Target labels must not be silently imputed.
    df = pd.DataFrame(
        {
            "age": [25, 40],
            "SeriousDlqin2yrs": [0, None],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        cr_fit_imputation(df, target_column="SeriousDlqin2yrs")


def test_cr_apply_preprocessing_uses_fitted_values_without_mutating_input():
    # Fit values on training data, then reuse them on separate data.
    train = pd.DataFrame({"age": [20.0, 40.0]})
    values = cr_fit_imputation(train)
    incoming = pd.DataFrame({"age": [None, 60.0]})
    original = incoming.copy(deep=True)

    processed = cr_apply_preprocessing(incoming, values)

    assert processed["age"].tolist() == [30.0, 60.0]
    pd.testing.assert_frame_equal(incoming, original)


def test_cr_apply_preprocessing_rejects_missing_expected_feature():
    # Applying training statistics requires the same expected feature columns.
    with pytest.raises(ValueError, match="missing features"):
        cr_apply_preprocessing(
            pd.DataFrame({"other_feature": [1]}),
            {"age": 30.0},
        )


def test_cr_apply_preprocessing_allows_unlabelled_prediction_data():
    # Prediction datasets may not contain the target column.
    values = cr_fit_imputation(pd.DataFrame({"age": [20.0, 40.0]}))
    prediction_data = pd.DataFrame({"age": [None, 50.0]})

    processed = cr_apply_preprocessing(prediction_data, values)

    assert processed["age"].tolist() == [30.0, 50.0]
