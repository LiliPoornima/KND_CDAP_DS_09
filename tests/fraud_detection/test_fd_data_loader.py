"""Tests for fraud-detection data loading."""

import dataclasses

import pandas as pd
import pytest

from backend.src.fraud_detection.fd_config import HANDBOOK_COLUMNS, HandbookExpectations
from backend.src.fraud_detection.fd_data_loader import (
    FraudDataValidationError,
    find_handbook_dir,
    load_handbook,
    validate_handbook,
)

TOY_EXPECTED = HandbookExpectations(
    n_rows=6, min_day=0, max_day=2, n_fraud=2,
    scenario_counts={0: 4, 1: 1, 2: 1}, n_files=3,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "TRANSACTION_ID": range(6),
        "TX_DATETIME": pd.to_datetime(["2018-04-01", "2018-04-01", "2018-04-02",
                                       "2018-04-02", "2018-04-03", "2018-04-03"]),
        "CUSTOMER_ID": [1, 2, 1, 2, 1, 2],
        "TERMINAL_ID": [1, 1, 2, 2, 3, 3],
        "TX_AMOUNT": [10.0] * 6,
        "TX_TIME_SECONDS": [0, 1, 86400, 86401, 172800, 172801],
        "TX_TIME_DAYS": [0, 0, 1, 1, 2, 2],
        "TX_FRAUD": [0, 0, 1, 0, 1, 0],
        "TX_FRAUD_SCENARIO": [0, 0, 1, 0, 2, 0],
    })[list(HANDBOOK_COLUMNS)]


def test_validate_passes_on_matching_data():
    validate_handbook(_toy_df(), TOY_EXPECTED, n_files=3)


def test_wrong_row_count_raises_clear_error():
    with pytest.raises(FraudDataValidationError, match="rows: found 5"):
        validate_handbook(_toy_df().iloc[:5], TOY_EXPECTED)


def test_wrong_file_count_raises():
    with pytest.raises(FraudDataValidationError, match="files: found 1"):
        validate_handbook(_toy_df(), TOY_EXPECTED, n_files=1)


def test_missing_column_raises():
    with pytest.raises(FraudDataValidationError, match="Missing columns"):
        validate_handbook(_toy_df().drop(columns=["TX_FRAUD"]), TOY_EXPECTED)


def test_single_daily_file_fails_default_expectations(tmp_path):
    _toy_df().to_pickle(tmp_path / "2018-04-01.pkl")
    with pytest.raises(FraudDataValidationError):
        load_handbook(raw_dir=tmp_path, use_cache=False)


def test_load_sorts_and_caches(tmp_path):
    df = _toy_df().sample(frac=1, random_state=0)
    df.to_pickle(tmp_path / "a.pkl")
    out = load_handbook(raw_dir=tmp_path, use_cache=True, cache_dir=tmp_path / "cache",
                        expected=dataclasses.replace(TOY_EXPECTED, n_files=1))
    assert out["TX_DATETIME"].is_monotonic_increasing
    assert (tmp_path / "cache" / "handbook_combined.parquet").is_file()


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_handbook_dir(tmp_path / "nope")


def test_real_handbook_matches_verified_expectations():
    try:
        find_handbook_dir()
    except FileNotFoundError:
        pytest.skip("Handbook data not present")
    df = load_handbook()  # raises if expectations are not met
    assert len(df) == 1_754_155
