"""Tests for baseline credit-risk modelling."""

import pandas as pd
import pytest

from backend.src.credit_risk.give_me_some_credit.model import (
    cr_analyze_thresholds,
    cr_compare_models,
    cr_evaluate_model,
    cr_train_logistic_regression,
    cr_train_random_forest,
)


@pytest.fixture
def sample_training_data():
    """Create a synthetic binary classification dataset."""

    return pd.DataFrame(
        {
            "age": [22, 25, 28, 35, 40, 45, 50, 55, 60, 65],
            "MonthlyIncome": [
                1800, 2200, 2500, 3000, 3500,
                4000, 4500, 5000, 5500, 6000,
            ],
            "DebtRatio": [
                0.9, 0.8, 0.7, 0.6, 0.5,
                0.4, 0.3, 0.2, 0.2, 0.1,
            ],
            "SeriousDlqin2yrs": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        }
    )


def test_train_logistic_regression_returns_model(sample_training_data):
    """Check that training returns a fitted pipeline."""

    model = cr_train_logistic_regression(sample_training_data)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_evaluation_returns_expected_metrics(sample_training_data):
    """Check that evaluation returns the required metrics."""

    model = cr_train_logistic_regression(sample_training_data)

    metrics = cr_evaluate_model(model, sample_training_data)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics


def test_training_rejects_missing_target(sample_training_data):
    """Check that training requires the target column."""

    invalid_data = sample_training_data.drop(
        columns=["SeriousDlqin2yrs"]
    )

    with pytest.raises(ValueError, match="Target column"):
        cr_train_logistic_regression(invalid_data)


def test_training_rejects_missing_predictor_values(sample_training_data):
    """Check that missing predictors are rejected."""

    invalid_data = sample_training_data.copy()
    invalid_data.loc[0, "age"] = float("nan")

    with pytest.raises(ValueError, match="predictors contain missing values"):
        cr_train_logistic_regression(invalid_data)


def test_train_random_forest_returns_model(sample_training_data):
    """Check that Random Forest training returns a fitted model."""

    model = cr_train_random_forest(sample_training_data)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_compare_models_returns_expected_metrics(sample_training_data):
    """Check that comparison includes both models and all metrics."""

    comparison = cr_compare_models(
        train_df=sample_training_data,
        test_df=sample_training_data,
    )

    assert "Logistic Regression" in comparison.index
    assert "Random Forest" in comparison.index

    expected_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC (Average Precision)",
    ]

    for column in expected_columns:
        assert column in comparison.columns


def test_compare_models_saves_results(
    sample_training_data,
    tmp_path,
):
    """Check that the comparison table can be saved to CSV."""

    cr_compare_models(
        train_df=sample_training_data,
        test_df=sample_training_data,
        output_dir=tmp_path,
    )

    assert (tmp_path / "model_comparison.csv").exists()

def test_analyze_thresholds_returns_expected_metrics(sample_training_data):
    """Threshold analysis should return precision, recall, and F1-score."""

    train_df = sample_training_data.copy()

    # Train the model on training data.
    model = cr_train_logistic_regression(train_df)

    # Use separate data to demonstrate threshold analysis.
    validation_df = train_df.copy()

    results = cr_analyze_thresholds(
        model,
        validation_df,
        thresholds=[0.30, 0.50, 0.70],
    )

    assert list(results.columns) == [
        "Threshold",
        "Precision",
        "Recall",
        "F1-score",
    ]

    assert results["Threshold"].tolist() == [0.30, 0.50, 0.70]

    for metric in ["Precision", "Recall", "F1-score"]:
        assert results[metric].between(0, 1).all()


def test_analyze_thresholds_rejects_missing_target(sample_training_data):
    """Threshold analysis must reject data without the target column."""

    train_df = sample_training_data.copy()
    model = cr_train_logistic_regression(train_df)

    validation_df = train_df.drop(columns=["SeriousDlqin2yrs"])

    with pytest.raises(ValueError, match="Target column"):
        cr_analyze_thresholds(model, validation_df)