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
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split labelled data, fit imputation on training data, and transform
    both partitions.

    The returned training and test DataFrames contain processed features,
    engineered features when their source columns exist, and the target.

    Imputation values are learned from training predictors only to prevent
    information from the test partition leaking into preprocessing.
    """

    # Validate the input DataFrame.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot prepare an empty DataFrame.")

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

    if df[target_column].nunique() < 2:
        raise ValueError(
            "Target column must contain at least two classes."
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

    # Preserve target class proportions in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Learn imputation values using training predictors only.
    imputation_values = cr_fit_imputation(X_train)

    # Apply the same fitted imputation values to both partitions.
    X_train_processed = cr_apply_preprocessing(
        X_train,
        imputation_values,
    )

    X_test_processed = cr_apply_preprocessing(
        X_test,
        imputation_values,
    )

    # Create the same deterministic features in each partition.
    # The feature function uses available source columns and does not
    # require the target column.
    X_train_engineered = cr_create_credit_features(
        X_train_processed,
        strict=False,
    )

    X_test_engineered = cr_create_credit_features(
        X_test_processed,
        strict=False,
    )

    # Restore the correct target labels.
    train_processed = X_train_engineered.copy()
    train_processed[target_column] = y_train

    test_processed = X_test_engineered.copy()
    test_processed[target_column] = y_test

    # Save the processed datasets when an output directory is provided.
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        train_processed.to_csv(
            destination / "train_processed.csv",
            index=False,
        )

        test_processed.to_csv(
            destination / "test_processed.csv",
            index=False,
        )

    return train_processed, test_processed, imputation_values


def cr_prepare_dataset_from_csv(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the source CSV and prepare processed training and test files."""

    # Reuse the common loader for consistent file validation.
    raw_data = cr_load_data(input_path)

    return cr_prepare_dataset(
        raw_data,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
    )