"""Tests for Give Me Some Credit experiment preparation."""

import pandas as pd
import pytest

from backend.src.credit_risk.give_me_some_credit.experiment import (
    cr_prepare_dataset,
)


def make_sample_data(rows=100):
    # Build a deterministic dataset with both target classes and missing values.
    return pd.DataFrame(
        {
            "Unnamed: 0": range(1, rows + 1),
            "MonthlyIncome": [None if i % 7 == 0 else 1000.0 + i for i in range(rows)],
            "age": [20 + (i % 50) for i in range(rows)],
            "SeriousDlqin2yrs": [i % 2 for i in range(rows)],
        }
    )


def test_cr_prepare_dataset_splits_data_and_imputes_without_leakage(tmp_path):
    # Prepare a sample dataset and save the processed splits to a temporary folder.
    train, test, values = cr_prepare_dataset(
        make_sample_data(),
        output_dir=tmp_path,
        test_size=0.2,
        random_state=42,
    )

    # The unique row index is removed and the target is retained separately.
    assert "Unnamed: 0" not in train.columns
    assert "SeriousDlqin2yrs" in train.columns
    assert len(train) == 80
    assert len(test) == 20

    # Missing numeric values are imputed using the training partition's median.
    assert train["MonthlyIncome"].isna().sum() == 0
    assert test["MonthlyIncome"].isna().sum() == 0
    assert values["MonthlyIncome"] == train["MonthlyIncome"].median()

    # Processed datasets are saved without modifying the source data.
    assert (tmp_path / "train_processed.csv").exists()
    assert (tmp_path / "test_processed.csv").exists()


def test_cr_prepare_dataset_preserves_target_proportions():
    # Stratification keeps both classes represented in train and test partitions.
    train, test, _ = cr_prepare_dataset(make_sample_data())

    assert set(train["SeriousDlqin2yrs"].unique()) == {0, 1}
    assert set(test["SeriousDlqin2yrs"].unique()) == {0, 1}


def test_cr_prepare_dataset_rejects_missing_target():
    # The target must be present to create a supervised train/test split.
    with pytest.raises(ValueError, match="does not exist"):
        cr_prepare_dataset(pd.DataFrame({"age": [20, 30]}))


def test_cr_prepare_dataset_rejects_missing_target_labels():
    # Missing labels must be resolved explicitly, not silently imputed.
    df = pd.DataFrame({"age": [20, 30], "SeriousDlqin2yrs": [0, None]})

    with pytest.raises(ValueError, match="missing values"):
        cr_prepare_dataset(df)


def test_cr_prepare_dataset_rejects_non_unique_row_index():
    # An index-like column must be unique before it is excluded from features.
    df = pd.DataFrame(
        {
            "Unnamed: 0": [1, 1, 2, 3],
            "age": [20, 30, 40, 50],
            "SeriousDlqin2yrs": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="not unique"):
        cr_prepare_dataset(df)


def test_cr_prepare_dataset_rejects_invalid_test_size():
    # The test fraction must be strictly between zero and one.
    with pytest.raises(ValueError, match="test_size"):
        cr_prepare_dataset(make_sample_data(), test_size=1.0)
