"""Tests for the Taiwan credit-card default workflow."""

from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from backend.src.credit_risk.credit_card_default.experiment import (
    cr_load_credit_card_default_data,
    cr_run_credit_card_baselines,
)
from backend.src.credit_risk.credit_card_default.feature_engineering import (
    cr_create_credit_card_features,
)
from backend.src.credit_risk.credit_card_default.preprocessing import (
    TARGET_COLUMN,
    cr_prepare_credit_card_dataset,
)


def make_credit_card_data(rows: int = 120) -> pd.DataFrame:
    """Return complete synthetic Taiwan rows with both target classes."""

    data: dict[str, object] = {
        "ID": np.arange(1, rows + 1),
        "SEX": np.where(np.arange(rows) % 2, 1, 2),
        "AGE": 25 + np.arange(rows) % 40,
        "MARRIAGE": 1 + np.arange(rows) % 3,
        "EDUCATION": 1 + np.arange(rows) % 3,
        "LIMIT_BAL": 20_000 + np.arange(rows) * 500,
        TARGET_COLUMN: np.arange(rows) % 2,
    }
    for month in range(6):
        status_column = "PAY_0" if month == 0 else f"PAY_{month + 1}"
        data[status_column] = (np.arange(rows) + month) % 4 - 1
        data[f"BILL_AMT{month + 1}"] = 100 + np.arange(rows) * (month + 1)
        data[f"PAY_AMT{month + 1}"] = 20 + np.arange(rows) % 70
    # This row has no positive bill balance, so its payment ratio is missing
    # before the train-fitted imputation step.
    for month in range(1, 7):
        data[f"BILL_AMT{month}"][0] = 0
    return pd.DataFrame(data)


def test_credit_card_features_summarize_repayment_history_without_mutation():
    source = pd.DataFrame(
        {
            **{column: [0, 0] for column in (
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
            )},
            **{column: [0, 0] for column in (
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
            )},
            **{column: [100, 0] for column in (
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
            )},
        }
    )
    source.loc[0, ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = [
        2, 1, 0, -1, 0, 0
    ]
    source.loc[0, ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]] = [50, 25, 10]

    result = cr_create_credit_card_features(source)

    assert result.loc[0, "BehavioralDelinquencyMonths"] == 2
    assert result.loc[0, "MaximumReportedDelayLevel"] == 2
    assert result.loc[0, "MonthsWithPositivePayment"] == 3
    assert result.loc[0, "MeanPositiveBillPaymentRatio"] == pytest.approx(
        (0.5 + 0.25 + 0.1) / 6
    )
    assert pd.isna(result.loc[1, "MeanPositiveBillPaymentRatio"])
    assert "BehavioralDelinquencyMonths" not in source.columns


def test_credit_card_features_validate_required_and_numeric_fields():
    with pytest.raises(ValueError, match="Missing required"):
        cr_create_credit_card_features(pd.DataFrame({"PAY_0": [0]}))

    partial = cr_create_credit_card_features(
        pd.DataFrame({"PAY_0": [1, -1]}),
        strict=False,
    )
    assert partial["BehavioralDelinquencyMonths"].tolist() == [1, 0]

    with pytest.raises(TypeError, match="must contain numeric"):
        cr_create_credit_card_features(pd.DataFrame({"PAY_0": ["late"]}), strict=False)

    with pytest.raises(ValueError, match="finite"):
        cr_create_credit_card_features(pd.DataFrame({"PAY_0": [np.inf]}), strict=False)


def test_credit_card_loader_reads_nested_zip_and_normalizes_embedded_header(
    tmp_path,
    monkeypatch,
):
    archive_path = tmp_path / "credit-card.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/default_credit_card.xls", b"mock workbook")

    workbook = pd.DataFrame(
        [
            ["ID", "SEX", "AGE", "default payment next month"],
            [1, 2, 35, 0],
            [2, 1, 42, 1],
        ],
        columns=["Column1", "Column2", "Column3", "Column4"],
    )

    def fake_read_excel(stream, *, sheet_name):
        assert stream.read() == b"mock workbook"
        assert sheet_name == 0
        return workbook.copy()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    result = cr_load_credit_card_default_data(archive_path)

    assert list(result.columns) == ["ID", "SEX", "AGE", TARGET_COLUMN]
    assert result[TARGET_COLUMN].tolist() == [0, 1]
    assert result["ID"].tolist() == [1, 2]


def test_credit_card_preparation_splits_and_keeps_audit_groups_separate(tmp_path):
    source = make_credit_card_data()
    original_columns = list(source.columns)

    train, test, metadata = cr_prepare_credit_card_dataset(
        source,
        output_dir=tmp_path,
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
    )
    validation = metadata["validation_processed"]

    assert (len(train), len(validation), len(test)) == (84, 12, 24)
    assert set(metadata["audit_group_columns"]) == {
        "SEX", "AGE", "MARRIAGE", "EDUCATION"
    }
    for split in (train, validation, test):
        assert "ID" not in split
        assert not {"SEX", "AGE", "MARRIAGE", "EDUCATION"} & set(split.columns)
        assert split[TARGET_COLUMN].isin([0, 1]).all()
        assert not split.drop(columns=TARGET_COLUMN).isna().any().any()
    assert metadata["audit_groups"]["test"].index.equals(test.index)
    assert train["MeanPositiveBillPaymentRatio"].median() == pytest.approx(
        metadata["imputation_values"]["MeanPositiveBillPaymentRatio"]
    )
    assert list(source.columns) == original_columns
    for name in (
        "train_processed.csv",
        "validation_processed.csv",
        "test_processed.csv",
        "train_groups.csv",
        "validation_groups.csv",
        "test_groups.csv",
    ):
        assert (tmp_path / name).is_file()


def test_credit_card_baselines_select_threshold_and_save_research_outputs(tmp_path):
    train, test, metadata = cr_prepare_credit_card_dataset(
        make_credit_card_data(),
        random_state=7,
    )
    result = cr_run_credit_card_baselines(
        train,
        metadata["validation_processed"],
        test,
        thresholds=[0.3, 0.5, 0.7],
        test_groups=metadata["audit_groups"]["test"][ ["SEX"] ],
        output_dir=tmp_path,
    )

    assert set(result["models"]) == {"Logistic Regression", "Random Forest"}
    assert set(result["comparison"].index) == set(result["models"])
    assert result["comparison"]["Validation-selected threshold"].isin(
        [0.3, 0.5, 0.7]
    ).all()
    assert set(result["fairness"]["Random Forest"]) == {"SEX"}
    assert (tmp_path / "model_comparison.csv").is_file()
    assert (tmp_path / "random_forest_sex_fairness.csv").is_file()
    assert (tmp_path / "logistic_regression_performance_confidence_intervals.csv").is_file()
