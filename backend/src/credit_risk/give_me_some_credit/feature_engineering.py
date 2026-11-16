"""
Feature engineering for the Give Me Some Credit dataset.

Create additional credit-risk predictors from existing borrower data.
The original DataFrame is never modified.
"""

import pandas as pd


DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

SEVERE_DELINQUENCY_COLUMNS = [
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

OPEN_ACCOUNT_COLUMNS = [
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
]

DEPENDENT_COLUMN = "NumberOfDependents"


def cr_create_credit_features(
    df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Create credit-risk features from available source columns.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed predictor data.
    strict : bool, default=True
        If True, raise ValueError when any required feature source column
        is missing. If False, create features only when their source
        columns are available.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with applicable engineered features.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    if not df.columns.is_unique:
        raise ValueError("DataFrame must have unique column names.")

    # Required columns for the full Give Me Some Credit dataset.
    all_required_columns = (
        DELINQUENCY_COLUMNS
        + OPEN_ACCOUNT_COLUMNS
        + [DEPENDENT_COLUMN]
    )

    missing_columns = [
        column
        for column in all_required_columns
        if column not in df.columns
    ]

    # Preserve the expected validation behavior by default.
    if strict and missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    def validate_columns(columns: list[str]) -> bool:
        """Validate available source columns and report their availability."""

        if not all(column in result.columns for column in columns):
            return False

        for column in columns:
            if not pd.api.types.is_numeric_dtype(result[column]):
                raise TypeError(
                    f"Column '{column}' must contain numeric values."
                )

            if result[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing values. "
                    "Preprocess missing values before feature engineering."
                )

        return True

    # Total count across all three delinquency-history columns.
    if validate_columns(DELINQUENCY_COLUMNS):
        result["TotalDelinquencyCount"] = result[
            DELINQUENCY_COLUMNS
        ].sum(axis=1)

    # Count of 60-89-days-past-due and 90-days-late events.
    if validate_columns(SEVERE_DELINQUENCY_COLUMNS):
        result["SevereDelinquencyCount"] = result[
            SEVERE_DELINQUENCY_COLUMNS
        ].sum(axis=1)

    # Combined number of open credit lines and real-estate accounts.
    if validate_columns(OPEN_ACCOUNT_COLUMNS):
        result["TotalOpenCreditAccounts"] = result[
            OPEN_ACCOUNT_COLUMNS
        ].sum(axis=1)

    # Binary indicator for borrowers who have dependents.
    if validate_columns([DEPENDENT_COLUMN]):
        result["HasDependents"] = (
            result[DEPENDENT_COLUMN] > 0
        ).astype(int)

    return result