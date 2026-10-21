"""
Credit-risk model training, evaluation, comparison, and threshold analysis.

This module provides:
1. Logistic Regression baseline training.
2. Random Forest baseline training.
3. Evaluation on held-out test data.
4. Comparison of classification performance.
5. Decision-threshold analysis using validation data.

Models are fitted on training data only. Validation data is used for
threshold analysis, while the held-out test dataset is reserved for
final evaluation.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "SeriousDlqin2yrs"


def _validate_training_data(
    train_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate training data and separate predictors from the target."""

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("Training data must be a pandas DataFrame.")

    if train_df.empty:
        raise ValueError("Training data must not be empty.")

    if not train_df.columns.is_unique:
        raise ValueError("Training data must have unique column names.")

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if train_df[target_column].isna().any():
        raise ValueError("Training target contains missing values.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    if X_train.shape[1] == 0:
        raise ValueError("Training data must contain predictor columns.")

    if X_train.isna().any().any():
        raise ValueError("Training predictors contain missing values.")

    if not all(
        pd.api.types.is_numeric_dtype(X_train[column])
        for column in X_train.columns
    ):
        raise ValueError("All training predictors must be numeric.")

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "The target must contain both binary classes: 0 and 1."
        )

    return X_train, y_train


def cr_train_logistic_regression(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Pipeline:
    """
    Train a baseline Logistic Regression classifier.

    StandardScaler and LogisticRegression are fitted on training data.
    Balanced class weights account for the imbalanced target classes.
    """

    X_train, y_train = _validate_training_data(
        train_df,
        target_column,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def cr_train_random_forest(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Pipeline:
    """
    Train a Random Forest classifier for credit-risk prediction.

    The classifier uses balanced class weights and is fitted on
    training data only.
    """

    X_train, y_train = _validate_training_data(
        train_df,
        target_column,
    )

    model = Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )

    model.fit(X_train, y_train)

    return model


def cr_evaluate_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate a trained classifier on held-out test data.

    Returns classification metrics, ROC-AUC, average precision,
    the confusion matrix, and the classification report.

    If output_dir is supplied, scalar metrics are saved to CSV.
    """

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("Test data must be a pandas DataFrame.")

    if test_df.empty:
        raise ValueError("Test data must not be empty.")

    if not test_df.columns.is_unique:
        raise ValueError("Test data must have unique column names.")

    if target_column not in test_df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if test_df[target_column].isna().any():
        raise ValueError("Test target contains missing values.")

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    if X_test.shape[1] == 0:
        raise ValueError("Test data must contain predictor columns.")

    if X_test.isna().any().any():
        raise ValueError("Test predictors contain missing values.")

    if not all(
        pd.api.types.is_numeric_dtype(X_test[column])
        for column in X_test.columns
    ):
        raise ValueError("All test predictors must be numeric.")

    # Check predictor names and order against those used during fitting.
    expected_columns = getattr(model, "feature_names_in_", None)

    if expected_columns is not None:
        if list(X_test.columns) != list(expected_columns):
            raise ValueError(
                "Test predictor columns must match the model's "
                "training columns in both names and order."
            )

    if set(y_test.unique()) != {0, 1}:
        raise ValueError(
            "Test target must contain both binary classes: 0 and 1."
        )

    # Generate predictions without fitting on test data.
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        ),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, y_probability)
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No serious delinquency",
                "Serious delinquency",
            ],
            zero_division=0,
            output_dict=True,
        ),
    }

    # Save scalar evaluation metrics if requested.
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {
                "confusion_matrix",
                "classification_report",
            }
        }

        pd.DataFrame([scalar_metrics]).to_csv(
            destination / "model_metrics.csv",
            index=False,
        )

    return metrics


def cr_compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Train and compare Logistic Regression and Random Forest.

    Both models are fitted on the same training data and evaluated
    on the same held-out test data.

    Returns a DataFrame containing:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - PR-AUC (Average Precision)

    If output_dir is supplied, the comparison table is saved to CSV.
    """

    # Validate both datasets before training.
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("Training data must be a pandas DataFrame.")

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("Test data must be a pandas DataFrame.")

    if train_df.empty:
        raise ValueError("Training data must not be empty.")

    if test_df.empty:
        raise ValueError("Test data must not be empty.")

    if not train_df.columns.is_unique:
        raise ValueError("Training data must have unique column names.")

    if not test_df.columns.is_unique:
        raise ValueError("Test data must have unique column names.")

    if target_column not in train_df.columns:
        raise ValueError(
            f"Training target '{target_column}' does not exist."
        )

    if target_column not in test_df.columns:
        raise ValueError(
            f"Test target '{target_column}' does not exist."
        )

    # Confirm that training and test predictors have identical columns.
    train_columns = [
        column for column in train_df.columns
        if column != target_column
    ]

    test_columns = [
        column for column in test_df.columns
        if column != target_column
    ]

    if train_columns != test_columns:
        raise ValueError(
            "Training and test predictor columns must match "
            "in both names and order."
        )

    # Train both models using the training partition only.
    models = {
        "Logistic Regression": cr_train_logistic_regression(
            train_df,
            target_column=target_column,
        ),
        "Random Forest": cr_train_random_forest(
            train_df,
            target_column=target_column,
        ),
    }

    results = []

    for model_name, model in models.items():
        # Evaluate both models on exactly the same test partition.
        metrics = cr_evaluate_model(
            model,
            test_df,
            target_column=target_column,
        )

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC (Average Precision)": (
                    metrics["pr_auc_average_precision"]
                ),
            }
        )

    comparison = pd.DataFrame(results).set_index("Model")

    # Save the comparison table to the requested output directory.
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        comparison.to_csv(
            destination / "model_comparison.csv"
        )

    return comparison


def cr_analyze_thresholds(
    model: Any,
    validation_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate a fitted classification model at different thresholds.

    The model must already be fitted on training data.
    The validation dataset must not have been used to fit the model.

    Args:
        model: A fitted classification model supporting predict_proba().
        validation_df: Validation data containing predictors and target.
        target_column: Name of the binary target column.
        thresholds: Candidate probability thresholds to evaluate.

    Returns:
        A DataFrame containing precision, recall, and F1-score
        for each candidate threshold.
    """

    # Validate the validation dataset.
    if not isinstance(validation_df, pd.DataFrame):
        raise TypeError("validation_df must be a pandas DataFrame.")

    if validation_df.empty:
        raise ValueError("validation_df must not be empty.")

    if not validation_df.columns.is_unique:
        raise ValueError(
            "Validation data must have unique column names."
        )

    if target_column not in validation_df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if validation_df[target_column].isna().any():
        raise ValueError("The target column contains missing values.")

    # Separate predictors from the target.
    X_validation = validation_df.drop(columns=[target_column])
    y_validation = validation_df[target_column]

    if X_validation.empty:
        raise ValueError(
            "Validation data must contain predictor columns."
        )

    if X_validation.isna().any().any():
        raise ValueError(
            "Validation predictors contain missing values."
        )

    if not all(
        pd.api.types.is_numeric_dtype(X_validation[column])
        for column in X_validation.columns
    ):
        raise ValueError(
            "All validation predictors must be numeric."
        )

    if set(y_validation.unique()) != {0, 1}:
        raise ValueError(
            "Validation target must contain both binary classes: 0 and 1."
        )

    # Check that validation features match the model's training features.
    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        expected_features = list(expected_features)

        if set(X_validation.columns) != set(expected_features):
            raise ValueError(
                "Validation predictors do not match the model's features."
            )

        # Match the feature order expected by the fitted model.
        X_validation = X_validation[expected_features]

    # Use a default threshold grid if none is supplied.
    if thresholds is None:
        thresholds = [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ]

    if len(thresholds) == 0:
        raise ValueError(
            "At least one threshold must be provided."
        )

    if any(
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0.0 < threshold < 1.0
        for threshold in thresholds
    ):
        raise ValueError(
            "Every threshold must be a number strictly between 0 and 1."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The model must support predict_proba()."
        )

    # Obtain probabilities for the positive class (serious delinquency).
    probabilities = model.predict_proba(X_validation)[:, 1]

    # Evaluate each threshold independently.
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "Threshold": float(threshold),
                "Precision": precision_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "F1-score": f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)