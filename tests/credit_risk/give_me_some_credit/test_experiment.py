"""Tests for Give Me Some Credit experiment preparation."""

import pandas as pd
import pytest

from backend.src.credit_risk.give_me_some_credit.experiment import (
    cr_prepare_dataset,
)


def make_sample_data(rows=100):
    """Build deterministic data with both target classes and missing values."""
    return pd.DataFrame(
        {
            "Unnamed: 0": range(1, rows + 1),
            "MonthlyIncome": [
                None if i % 7 == 0 else 1000.0 + i
                for i in range(rows)
            ],
            "age": [20 + (i % 50) for i in range(rows)],
            "SeriousDlqin2yrs": [i % 2 for i in range(rows)],
        }
    )


def test_cr_prepare_dataset_splits_data_and_imputes_without_leakage(
    tmp_path,
):
    """Check three-way splitting, imputation, and saved outputs."""

    source_data = make_sample_data()

    train, test, metadata = cr_prepare_dataset(
        source_data,
        output_dir=tmp_path,
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
    )

    validation = metadata["validation_processed"]
    values = metadata["imputation_values"]

    # The exported row index is removed and the target is retained.
    for partition in (train, validation, test):
        assert "Unnamed: 0" not in partition.columns
        assert "SeriousDlqin2yrs" in partition.columns

    # Confirm the expected 70/10/20 split of 100 rows.
    assert len(train) == 70
    assert len(validation) == 10
    assert len(test) == 20

    # Every partition must have identical columns in the same order.
    assert list(train.columns) == list(validation.columns)
    assert list(train.columns) == list(test.columns)

    # All missing predictor values must be imputed.
    for partition in (train, validation, test):
        assert not partition.drop(
            columns=["SeriousDlqin2yrs"]
        ).isna().any().any()

    # Target labels must remain unchanged and complete.
    for partition in (train, validation, test):
        assert partition["SeriousDlqin2yrs"].notna().all()
        assert set(partition["SeriousDlqin2yrs"].unique()) == {0, 1}

    # The original source DataFrame must not be modified.
    assert "Unnamed: 0" in source_data.columns
    assert source_data["MonthlyIncome"].isna().any()

    # All processed partitions must be saved.
    assert (tmp_path / "train_processed.csv").exists()
    assert (tmp_path / "validation_processed.csv").exists()
    assert (tmp_path / "test_processed.csv").exists()

    # Confirm imputation metadata is available.
    assert isinstance(values, dict)
    assert "MonthlyIncome" in values

    # The fitted imputation value must be based on training predictors.
    expected_training_median = source_data.loc[
        train.index, "MonthlyIncome"
    ].median()

    assert values["MonthlyIncome"] == expected_training_median

    # The imputed training values must contain no missing entries.
    assert train["MonthlyIncome"].notna().all()


def test_cr_prepare_dataset_preserves_target_proportions():
    """Stratification should preserve both target classes in all partitions."""

    train, test, metadata = cr_prepare_dataset(make_sample_data())
    validation = metadata["validation_processed"]

    for partition in (train, validation, test):
        assert set(partition["SeriousDlqin2yrs"].unique()) == {0, 1}

    # Class proportions should remain reasonably close across partitions.
    full_rate = make_sample_data()["SeriousDlqin2yrs"].mean()

    for partition in (train, validation, test):
        partition_rate = partition["SeriousDlqin2yrs"].mean()
        assert abs(partition_rate - full_rate) <= 0.05


def test_cr_prepare_dataset_rejects_missing_target():
    """A target column is required for supervised data preparation."""

    with pytest.raises(ValueError, match="does not exist"):
        cr_prepare_dataset(pd.DataFrame({"age": [20, 30]}))


def test_cr_prepare_dataset_rejects_missing_target_labels():
    """Missing target labels must not be silently imputed."""

    df = pd.DataFrame(
        {
            "age": [20, 30],
            "SeriousDlqin2yrs": [0, None],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        cr_prepare_dataset(df)


def test_cr_prepare_dataset_rejects_non_unique_row_index():
    """A non-unique exported index must not be silently excluded."""

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
    """The test fraction must be strictly between zero and one."""

    with pytest.raises(ValueError, match="test_size"):
        cr_prepare_dataset(
            make_sample_data(),
            test_size=1.0,
        )


def test_cr_prepare_dataset_rejects_invalid_validation_size():
    """The validation fraction must be strictly between zero and one."""

    with pytest.raises(ValueError, match="validation_size"):
        cr_prepare_dataset(
            make_sample_data(),
            validation_size=0.0,
        )


def test_cr_prepare_dataset_rejects_combined_sizes_at_least_one():
    """Training must retain a positive fraction of the data."""

    with pytest.raises(ValueError, match="must be less than 1"):
        cr_prepare_dataset(
            make_sample_data(),
            test_size=0.5,
            validation_size=0.5,
        )


def test_cr_prepare_dataset_rejects_single_class_target():
    """Stratified preparation requires two target classes."""

    df = pd.DataFrame(
        {
            "age": range(10),
            "SeriousDlqin2yrs": [0] * 10,
        }
    )

    with pytest.raises(ValueError, match="exactly two classes"):
        cr_prepare_dataset(df)