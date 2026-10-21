"""Reusable preprocessing helpers for Give Me Some Credit data."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Reject invalid or empty inputs before preprocessing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot preprocess an empty DataFrame.")

    # Duplicate column names make feature selection and imputation ambiguous.
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")


def _validate_target(
    df: pd.DataFrame,
    target_column: str | None,
) -> None:
    """Ensure a requested target exists and has no missing labels."""
    if target_column is None:
        return

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    # Never impute target labels; missing labels should be handled explicitly.
    if df[target_column].isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )


def cr_fit_imputation(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Learn feature imputation values from a training DataFrame.

    Numeric features use their median. Non-numeric features use their mode.
    Entirely missing numeric columns use 0; entirely missing non-numeric
    columns use the explicit '__missing__' category.

    Fit these values on the training split only, then reuse them for
    validation and test data to avoid data leakage. The target column, if
    provided, is validated but never included in the returned values.
    """
    _validate_dataframe(df)
    _validate_target(df, target_column)

    imputation_values: dict[str, Any] = {}

    # Learn one replacement value per feature from the supplied training data.
    for column in df.columns:
        if column == target_column:
            continue

        feature = df[column]

        if is_numeric_dtype(feature):
            # A median is robust to extreme numeric values.
            median = feature.median()
            imputation_values[column] = (
                float(median) if pd.notna(median) else 0.0
            )
        else:
            # Use the most frequent category; provide a fallback for all-NA data.
            modes = feature.mode(dropna=True)
            imputation_values[column] = (
                modes.iloc[0] if not modes.empty else "__missing__"
            )

    return imputation_values


def cr_apply_preprocessing(
    df: pd.DataFrame,
    imputation_values: Mapping[str, Any],
    target_column: str | None = None,
) -> pd.DataFrame:
    """Apply previously learned imputation values without refitting them.

    The input DataFrame is not modified. If a target column is supplied, it
    must exist and contain no missing labels. Every feature used to fit the
    imputation values must also be present in the DataFrame being transformed.
    """
    _validate_dataframe(df)
    _validate_target(df, target_column)

    if not isinstance(imputation_values, Mapping):
        raise TypeError("imputation_values must be a mapping of columns to values.")

    # Missing expected features usually indicate a train/prediction schema mismatch.
    missing_features = [
        column for column in imputation_values if column not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "Input DataFrame is missing features used during fitting: "
            + ", ".join(missing_features)
        )

    # Work on a copy so callers can safely retain the original raw DataFrame.
    processed = df.copy()

    # Apply training-derived values only; do not calculate new statistics here.
    for column, value in imputation_values.items():
        if column == target_column:
            continue
        processed[column] = processed[column].fillna(value)

    return processed
