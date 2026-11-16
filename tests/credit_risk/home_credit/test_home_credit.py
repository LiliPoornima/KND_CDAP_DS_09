"""Tests for Home Credit loading, leakage-safe features, and experiments."""

from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from backend.src.credit_risk.home_credit.data_loader import (
    cr_load_home_credit_table,
)
from backend.src.credit_risk.home_credit.experiment import (
    cr_prepare_home_credit_dataset_from_archive,
    cr_run_home_credit_baselines,
)
from backend.src.credit_risk.home_credit.feature_engineering import (
    cr_aggregate_home_credit_histories,
    cr_create_home_credit_features,
)
from backend.src.credit_risk.home_credit.preprocessing import (
    TARGET_COLUMN,
    cr_prepare_home_credit_dataset,
)
from backend.src.credit_risk.home_credit.training import (
    cr_train_home_credit_logistic_regression,
    cr_train_home_credit_random_forest,
)


def make_application_data(rows: int = 60) -> pd.DataFrame:
    row = np.arange(rows)
    return pd.DataFrame(
        {
            "SK_ID_CURR": 1000 + row,
            TARGET_COLUMN: row % 2,
            "CODE_GENDER": np.where((row // 2) % 2, "M", "F"),
            "DAYS_BIRTH": -(25 + row % 35) * 365,
            "DAYS_EMPLOYED": np.where(row == 0, 365243, -1000 - row),
            "AMT_CREDIT": 10_000 + row * 100,
            "AMT_INCOME_TOTAL": 4_000 + row * 50,
            "AMT_ANNUITY": 500 + row * 10,
            "AMT_GOODS_PRICE": 8_000 + row * 80,
            "EXT_SOURCE_1": 0.1 + (row % 7) / 10,
            "EXT_SOURCE_2": 0.2 + (row % 6) / 10,
            "EXT_SOURCE_3": 0.3 + (row % 5) / 10,
            "NAME_CONTRACT_TYPE": np.where(row % 3, "Cash loans", "Revolving loans"),
        }
    )


def make_history_tables() -> dict[str, pd.DataFrame]:
    """Build minimal linked tables with both in-time and future rows."""

    return {
        "application_train.csv": make_application_data(),
        "bureau.csv": pd.DataFrame(
            {
                "SK_ID_CURR": [1000, 1000],
                "SK_ID_BUREAU": [501, 502],
                "DAYS_CREDIT": [-10, 2],
                "DAYS_CREDIT_UPDATE": [-1, -1],
                "CREDIT_ACTIVE": ["Active", "Closed"],
                "CREDIT_DAY_OVERDUE": [2, 20],
                "CNT_CREDIT_PROLONG": [1, 1],
                "AMT_CREDIT_SUM": [100, 500],
                "AMT_CREDIT_SUM_DEBT": [40, 100],
                "AMT_CREDIT_SUM_OVERDUE": [10, 30],
            }
        ),
        "bureau_balance.csv": pd.DataFrame(
            {
                "SK_ID_BUREAU": [501, 501],
                "MONTHS_BALANCE": [-1, 1],
                "STATUS": ["2", "5"],
            }
        ),
        "previous_application.csv": pd.DataFrame(
            {
                "SK_ID_CURR": [1000, 1000],
                "DAYS_DECISION": [-4, 2],
                "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
                "AMT_APPLICATION": [100, 200],
                "AMT_CREDIT": [80, 180],
            }
        ),
        "installments_payments.csv": pd.DataFrame(
            {
                "SK_ID_CURR": [1000, 1000, 1000],
                "DAYS_INSTALMENT": [-10, -5, -2],
                "DAYS_ENTRY_PAYMENT": [-8, np.nan, 1],
                "AMT_INSTALMENT": [20, 30, 40],
                "AMT_PAYMENT": [20, np.nan, 40],
            }
        ),
        "POS_CASH_balance.csv": pd.DataFrame(
            {
                "SK_ID_CURR": [1000, 1000],
                "MONTHS_BALANCE": [-1, 1],
                "NAME_CONTRACT_STATUS": ["Active", "Completed"],
                "SK_DPD": [3, 0],
                "SK_DPD_DEF": [1, 0],
                "CNT_INSTALMENT_FUTURE": [2, 0],
            }
        ),
        "credit_card_balance.csv": pd.DataFrame(
            {
                "SK_ID_CURR": [1000, 1000],
                "MONTHS_BALANCE": [-2, 1],
                "AMT_BALANCE": [50, 100],
                "AMT_CREDIT_LIMIT_ACTUAL": [100, 100],
                "AMT_PAYMENT_TOTAL_CURRENT": [20, 100],
                "AMT_DRAWINGS_CURRENT": [10, 20],
                "SK_DPD": [2, 0],
                "SK_DPD_DEF": [1, 0],
            }
        ),
    }


def write_home_archive(path, tables=None):
    tables = make_history_tables() if tables is None else tables
    with ZipFile(path, "w") as archive:
        for name, frame in tables.items():
            archive.writestr(
                f"nested/{name}",
                frame.to_csv(index=False),
            )
    return path


def test_home_credit_loader_reads_nested_tables_and_chunks(tmp_path):
    archive_path = write_home_archive(tmp_path / "home-credit.zip")

    loaded = cr_load_home_credit_table(
        archive_path,
        table_name="application_train.csv",
        usecols=["SK_ID_CURR", TARGET_COLUMN],
    )
    chunks = list(
        cr_load_home_credit_table(
            archive_path,
            table_name="application_train.csv",
            chunksize=17,
            usecols=["SK_ID_CURR", TARGET_COLUMN],
        )
    )

    assert len(loaded) == 60
    assert list(loaded.columns) == ["SK_ID_CURR", TARGET_COLUMN]
    assert [len(chunk) for chunk in chunks] == [17, 17, 17, 9]
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), loaded)


def test_home_credit_loader_rejects_invalid_archive_requests(tmp_path):
    invalid_archive = tmp_path / "invalid.zip"
    invalid_archive.write_text("not a zip", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid ZIP"):
        cr_load_home_credit_table(invalid_archive)

    archive_path = write_home_archive(tmp_path / "valid.zip")
    with pytest.raises(ValueError, match="Unsupported Home Credit table"):
        cr_load_home_credit_table(archive_path, "unknown.csv")
    with pytest.raises(ValueError, match="positive integer"):
        cr_load_home_credit_table(archive_path, chunksize=0)
    with pytest.raises(FileNotFoundError, match="was not found"):
        cr_load_home_credit_table(archive_path, "sample_submission.csv")


def test_home_credit_features_build_ratios_and_handle_employment_sentinel():
    source = make_application_data(3)
    original = source.copy(deep=True)
    result = cr_create_home_credit_features(source)

    assert result.loc[0, "AGE_YEARS"] == pytest.approx(25 * 365 / 365.25)
    assert pd.isna(result.loc[0, "YEARS_EMPLOYED"])
    assert result.loc[1, "YEARS_EMPLOYED"] == pytest.approx(1001 / 365.25)
    assert result.loc[0, "CREDIT_TO_INCOME_RATIO"] == pytest.approx(
        source.loc[0, "AMT_CREDIT"] / source.loc[0, "AMT_INCOME_TOTAL"]
    )
    assert result.loc[0, "MEAN_EXTERNAL_SCORE"] == pytest.approx(0.2)
    pd.testing.assert_frame_equal(source, original)


def test_home_credit_preparation_separates_groups_and_excludes_identifiers(tmp_path):
    source = make_application_data()
    train, test, metadata = cr_prepare_home_credit_dataset(
        source,
        output_dir=tmp_path,
        random_state=42,
    )
    validation = metadata["validation_processed"]

    assert (len(train), len(validation), len(test)) == (42, 6, 12)
    assert metadata["split_sizes"] == {"train": 42, "validation": 6, "test": 12}
    for split in (train, validation, test):
        assert split[TARGET_COLUMN].isin([0, 1]).all()
        assert not {"SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"} & set(split)
        assert split.columns.tolist() == train.columns.tolist()
    groups = metadata["audit_groups"]
    assert groups["test"].index.equals(test.index)
    assert list(groups["test"].columns) == ["CODE_GENDER", "AGE_YEARS"]
    assert set(groups["test"]["CODE_GENDER"]) == {"F", "M"}
    assert (tmp_path / "train_processed.csv").is_file()
    assert (tmp_path / "validation_groups.csv").is_file()


def test_home_credit_history_aggregation_excludes_future_rows_and_combines_chunks(
    tmp_path,
):
    archive_path = write_home_archive(tmp_path / "home-history.zip")
    aggregates = cr_aggregate_home_credit_histories(archive_path, chunksize=1)

    applicant = aggregates.set_index("SK_ID_CURR").loc[1000]
    assert applicant["BUREAU_PRIOR_CREDIT_COUNT"] == 1
    assert applicant["BUREAU_CREDIT_AMOUNT_SUM"] == 100
    assert applicant["BUREAU_BALANCE_MONTH_COUNT"] == 1
    assert applicant["BUREAU_BALANCE_DPD_MONTH_COUNT"] == 1
    assert applicant["PREVIOUS_APPLICATION_COUNT"] == 1
    assert applicant["PREVIOUS_APPROVED_COUNT"] == 1
    assert applicant["INSTALLMENT_DUE_COUNT"] == 2
    assert applicant["INSTALLMENT_LATE_COUNT"] == 1
    assert applicant["INSTALLMENT_UNPAID_COUNT"] == 1
    assert applicant["POS_CASH_MONTH_COUNT"] == 1
    assert applicant["CREDIT_CARD_MONTH_COUNT"] == 1
    assert np.isfinite(aggregates.drop(columns="SK_ID_CURR").to_numpy()).all()


def test_home_credit_archive_preparation_attaches_history_and_writes_splits(tmp_path):
    archive_path = write_home_archive(tmp_path / "home-credit.zip")
    output_dir = tmp_path / "prepared"

    train, test, metadata = cr_prepare_home_credit_dataset_from_archive(
        archive_path,
        output_dir=output_dir,
        history_chunksize=1,
        random_state=42,
    )

    validation = metadata["validation_processed"]
    assert metadata["includes_prior_history"] is True
    assert metadata["history_chunksize"] == 1
    assert (len(train), len(validation), len(test)) == (42, 6, 12)
    assert "BUREAU_PRIOR_CREDIT_COUNT" in train
    assert not {"CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS", "SK_ID_CURR"} & set(train)
    assert metadata["audit_groups"]["test"].index.equals(test.index)
    assert (output_dir / "test_groups.csv").is_file()


def test_home_credit_models_fit_mixed_inputs_and_predict(tmp_path):
    archive_path = write_home_archive(tmp_path / "home-credit.zip")
    train, test, metadata = cr_prepare_home_credit_dataset_from_archive(
        archive_path,
        random_state=9,
    )
    validation = metadata["validation_processed"]
    # Add a category absent from training to exercise OneHotEncoder's
    # handle_unknown="ignore" behavior in both model pipelines.
    test = test.copy()
    test["NAME_CONTRACT_TYPE"] = "New product"
    predictors = [column for column in train.columns if column != TARGET_COLUMN]

    logistic = cr_train_home_credit_logistic_regression(train, solver="liblinear")
    forest = cr_train_home_credit_random_forest(train)

    for model in (logistic, forest):
        probabilities = model.predict_proba(test[predictors])
        assert probabilities.shape == (len(test), 2)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(logistic.classes_) == {0, 1}
    assert len(validation) == 6


def test_home_credit_baselines_select_validation_threshold_and_save_outputs(tmp_path):
    archive_path = write_home_archive(tmp_path / "home-credit.zip")
    train, test, metadata = cr_prepare_home_credit_dataset_from_archive(
        archive_path,
        random_state=21,
    )
    result = cr_run_home_credit_baselines(
        train,
        metadata["validation_processed"],
        test,
        thresholds=[0.3, 0.5, 0.7],
        output_dir=tmp_path / "results",
    )

    assert set(result["models"]) == {"Logistic Regression", "Random Forest"}
    assert set(result["comparison"].index) == set(result["models"])
    assert result["comparison"]["Validation-selected threshold"].isin(
        [0.3, 0.5, 0.7]
    ).all()
    assert (tmp_path / "results" / "model_comparison.csv").is_file()
    assert (tmp_path / "results" / "training_diagnostics.csv").is_file()
    assert (tmp_path / "results" / "random_forest_calibration_bins.csv").is_file()
