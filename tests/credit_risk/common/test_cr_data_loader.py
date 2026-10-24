"""Tests for credit-risk data loading."""
import pandas as pd
import pytest

# Import the data loader function being tested.
from backend.src.credit_risk.common.cr_data_loader import cr_load_data


def test_cr_load_data_reads_csv(tmp_path):
    # Create a temporary CSV file so the test does not depend on the real dataset.
    # pytest automatically provides tmp_path as a temporary directory.
    csv_path = tmp_path / "sample.csv"

    # Create a small sample dataset with a feature and the credit-risk target.
    # SeriousDlqin2yrs indicates whether a borrower experienced serious delinquency.
    pd.DataFrame(
        {
            "age": [25, 40],
            "SeriousDlqin2yrs": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    # Load the sample CSV using the function being tested.
    df = cr_load_data(csv_path)

    # Verify that the loader returns two rows and two columns.
    assert df.shape == (2, 2)

    # Verify that the expected feature and target columns were loaded.
    assert "age" in df.columns
    assert "SeriousDlqin2yrs" in df.columns


def test_cr_load_data_missing_file():
    # Verify that the loader raises FileNotFoundError when the file is missing.
    with pytest.raises(FileNotFoundError):
        cr_load_data("nonexistent_dataset.csv")


def test_cr_load_data_rejects_directory(tmp_path):
    # A directory is not a valid CSV file, so the loader should reject it.
    with pytest.raises(IsADirectoryError):
        cr_load_data(tmp_path)


def test_cr_load_data_rejects_empty_csv(tmp_path):
    # Create a CSV file containing no content.
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    # Verify that an empty file produces a clear ValueError.
    with pytest.raises(ValueError):
        cr_load_data(csv_path)


def test_cr_load_data_rejects_csv_without_data_rows(tmp_path):
    # Create a CSV containing column headers but no data rows.
    csv_path = tmp_path / "headers_only.csv"
    csv_path.write_text(
        "age,SeriousDlqin2yrs\n",
        encoding="utf-8",
    )

    # Verify that a dataset with no observations is rejected.
    with pytest.raises(ValueError):
        cr_load_data(csv_path)