"""Reproducible data preparation for Give Me Some Credit experiments."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.src.credit_risk.common.cr_data_loader import cr_load_data
from backend.src.credit_risk.give_me_some_credit.preprocessing import (
    cr_apply_preprocessing,
    cr_fit_imputation,
)
from backend.src.credit_risk.give_me_some_credit.feature_engineering import (
    cr_create_credit_features,
)

TARGET_COLUMN = "SeriousDlqin2yrs"
INDEX_COLUMN = "Unnamed: 0"


def cr_prepare_dataset(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split labelled data into training, validation, and test partitions.

    Fit imputation using training predictors only, then apply the fitted
    values to all partitions.

    By default, the resulting partitions contain 70% training data,
    10% validation data, and 20% test data.

    Returns:
        A tuple containing:
        - Processed training DataFrame.
        - Processed test DataFrame.
        - Metadata dictionary containing imputation values and the
          processed validation DataFrame.
    """

    # Validate the input DataFrame.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot prepare an empty DataFrame.")

    if not df.columns.is_unique:
        raise ValueError("Input data must have unique column names.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df[target_column].isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be greater than 0 and less than 1."
        )

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be greater than 0 and less than 1."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    if df[target_column].nunique() != 2:
        raise ValueError(
            "Target column must contain exactly two classes."
        )

    # Copy the data so the original DataFrame is not modified.
    data = df.copy()

    # Remove the exported row index only when it is unique.
    if INDEX_COLUMN in data.columns:
        if not data[INDEX_COLUMN].is_unique:
            raise ValueError(
                f"'{INDEX_COLUMN}' exists but is not unique; "
                "verify its meaning before excluding it."
            )

        data = data.drop(columns=[INDEX_COLUMN])

    # Separate predictors and target before splitting.
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[1] == 0:
        raise ValueError("Input data must contain predictor columns.")

    # First, separate the final test partition.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Calculate validation size relative to the remaining development data.
    relative_validation_size = validation_size / (1 - test_size)

    # Split the development data into training and validation partitions.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_development,
    )

    # Learn imputation values from training predictors only.
    imputation_values = cr_fit_imputation(X_train)

    # Apply the same fitted imputation values to every partition.
    X_train_processed = cr_apply_preprocessing(
        X_train,
        imputation_values,
    )

    X_validation_processed = cr_apply_preprocessing(
        X_validation,
        imputation_values,
    )

    X_test_processed = cr_apply_preprocessing(
        X_test,
        imputation_values,
    )

    # Create the same deterministic features in each partition.
    X_train_engineered = cr_create_credit_features(
        X_train_processed,
        strict=False,
    )

    X_validation_engineered = cr_create_credit_features(
        X_validation_processed,
        strict=False,
    )

    X_test_engineered = cr_create_credit_features(
        X_test_processed,
        strict=False,
    )

    # Restore the correct target labels.
    train_processed = X_train_engineered.copy()
    train_processed[target_column] = y_train

    validation_processed = X_validation_engineered.copy()
    validation_processed[target_column] = y_validation

    test_processed = X_test_engineered.copy()
    test_processed[target_column] = y_test

    # Verify that all partitions have the same columns in the same order.
    if not (
        list(train_processed.columns)
        == list(validation_processed.columns)
        == list(test_processed.columns)
    ):
        raise ValueError(
            "Training, validation, and test columns do not match."
        )

    # Save all processed partitions when an output directory is provided.
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        train_processed.to_csv(
            destination / "train_processed.csv",
            index=False,
        )

        validation_processed.to_csv(
            destination / "validation_processed.csv",
            index=False,
        )

        test_processed.to_csv(
            destination / "test_processed.csv",
            index=False,
        )

    # Preserve the existing three-value return structure.
    metadata = {
        "imputation_values": imputation_values,
        "validation_processed": validation_processed,
    }

    return train_processed, test_processed, metadata


def cr_prepare_dataset_from_csv(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the source CSV and prepare three processed data partitions."""

    # Reuse the common loader for consistent file validation.
    raw_data = cr_load_data(input_path)

    return cr_prepare_dataset(
        raw_data,
        output_dir=output_dir,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )