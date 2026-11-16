"""Credit-risk dataset inspection."""
import pandas as pd


def cr_inspect_data(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict:
    """
    Inspect the structure and quality of a credit-risk dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.
    target_column : str or None
        Optional name of the target column.

    Returns
    -------
    dict
        Dataset dimensions, column names, data types, missing-value
        counts, duplicate-row count, and optional target distribution.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty or the target column is missing.
    """

    # Confirm that the input is a pandas DataFrame.
    # This prevents the function from processing an unexpected input type.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Reject datasets without observations because they cannot be inspected
    # meaningfully for missing values, duplicates, or target distribution.
    if df.empty:
        raise ValueError("Cannot inspect an empty DataFrame.")

    # Check that the requested target column exists before analysing it.
    # The target column is SeriousDlqin2yrs for our primary dataset.
    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    # Count missing values in each column.
    # Keep only columns containing missing values to make the summary concise.
    missing_counts = df.isna().sum()

    # Create a summary of the dataset's dimensions and quality.
    # Convert pandas and NumPy values to standard Python types where needed.
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in missing_counts.items()
            if count > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # If a target column was provided, count the observations for each label.
    # dropna=False ensures that missing target values are included in the report.
    if target_column is not None:
        summary["target_distribution"] = {
            str(label): int(count)
            for label, count in (
                df[target_column].value_counts(dropna=False).items()
            )
        }

    # Return the inspection results for reporting and further analysis.
    return summary