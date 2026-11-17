"""Data tests: file exists, required columns exist, non-empty data, valid target."""

import pandas as pd
import pytest

from backend.src.financial_capability.fc_config import RAW_CSV_PATH, REQUIRED_COLUMNS
from backend.src.financial_capability.fc_data_loader import (
    FinancialCapabilityDataError,
    load_cfpb_data,
)
from backend.src.financial_capability.fc_validation import validate_dataset


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_cfpb_data(tmp_path / "nope.csv")


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(FinancialCapabilityDataError):
        load_cfpb_data(p)


def test_required_columns_enforced(raw_df):
    broken = raw_df.drop(columns=["SAVEHABIT"])
    with pytest.raises(FinancialCapabilityDataError, match="SAVEHABIT"):
        validate_dataset(broken)


def test_empty_dataframe_rejected():
    with pytest.raises(FinancialCapabilityDataError):
        validate_dataset(pd.DataFrame())


def test_report_counts_and_special_target_codes(raw_df):
    rep = validate_dataset(raw_df)
    assert rep.ok
    assert rep.n_rows == len(raw_df)
    assert rep.target_special_code_counts == {"-4": 1, "-1": 2}
    assert rep.n_valid_target_rows == len(raw_df) - 3
    assert rep.target_min >= 0


def test_out_of_codebook_values_are_reported(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "GOALCONF"] = 9  # not a valid GOALCONF code
    rep = validate_dataset(bad)
    assert rep.out_of_codebook_values.get("GOALCONF") == 1
    assert rep.warnings


def test_all_target_invalid_is_rejected(raw_df):
    bad = raw_df.copy()
    bad["FWBscore"] = -4
    with pytest.raises(FinancialCapabilityDataError, match="No valid target"):
        validate_dataset(bad)


# ---- checks against the REAL downloaded file (skipped if it is not present)
real = pytest.mark.skipif(not RAW_CSV_PATH.is_file(), reason="real CFPB CSV not downloaded")


@real
def test_real_file_has_all_required_columns_and_valid_target():
    df = load_cfpb_data()
    assert not df.empty
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    rep = validate_dataset(df)
    assert rep.ok and rep.n_valid_target_rows > 6000
    assert rep.out_of_codebook_values == {}

