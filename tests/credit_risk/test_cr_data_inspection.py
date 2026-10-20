import pandas as pd
import pytest

# Import the inspection function being tested.
from backend.src.credit_risk.cr_data_inspection import cr_inspect_data


def test_cr_inspect_data_returns_summary():
    # Create a small dataset with one missing age value.
    # This allows us to verify the missing-value count and target distribution.
    df = pd.DataFrame(
        {
            "age": [25, None, 40],
            "SeriousDlqin2yrs": [0, 1, 0],
        }
    )

    # Inspect the dataset and specify the credit-risk target column.
    summary = cr_inspect_data(
        df,
        target_column="SeriousDlqin2yrs",
    )

    # Verify that the dataset dimensions are reported correctly.
    assert summary["rows"] == 3
    assert summary["columns"] == 2

    # Verify that the missing age value is detected.
    assert summary["missing_values"]["age"] == 1

    # Verify that the target distribution counts both classes correctly.
    assert summary["target_distribution"] == {"0": 2, "1": 1}


def test_cr_inspect_data_rejects_missing_target():
    # Create a dataset that does not contain the requested target column.
    df = pd.DataFrame({"age": [25, 40]})

    # Verify that the function raises ValueError for a missing target column.
    with pytest.raises(ValueError):
        cr_inspect_data(
            df,
            target_column="SeriousDlqin2yrs",
        )


def test_cr_inspect_data_rejects_empty_dataframe():
    # Create a DataFrame without rows or columns.
    df = pd.DataFrame()

    # Verify that empty datasets are rejected.
    with pytest.raises(ValueError):
        cr_inspect_data(df)


def test_cr_inspect_data_rejects_invalid_input():
    # A list is not a pandas DataFrame and should not be accepted.
    with pytest.raises(TypeError):
        cr_inspect_data([1, 2, 3])