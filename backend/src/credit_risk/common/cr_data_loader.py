"""Credit-risk dataset loading."""
from pathlib import Path

import pandas as pd


def cr_load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load a credit-risk dataset from a CSV file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    IsADirectoryError
        If the supplied path points to a directory.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """

    # Convert the supplied file path into a Path object.
    # This allows us to check the path and work with it consistently.
    path = Path(file_path)

    # Check whether the dataset exists before attempting to read it.
    # Raise a clear error if the path does not point to an existing file.
    if not path.exists():
        raise FileNotFoundError(
            f"Credit-risk dataset not found: {path}"
        )

    # Ensure the supplied path points to a file rather than a directory.
    # pandas cannot load a directory as a CSV dataset.
    if path.is_dir():
        raise IsADirectoryError(
            f"Expected a CSV file, but received a directory: {path}"
        )

    try:
        # Read the CSV file and convert its contents into a pandas DataFrame.
        # A DataFrame allows us to inspect, preprocess, and analyse the data.
        df = pd.read_csv(path)

    # Handle a completely empty file that contains no readable CSV data.
    # Preserve the original exception as the cause for debugging.
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The dataset file is empty: {path}"
        ) from exc

    # Handle CSV files with malformed row structures or parsing problems.
    # Return a consistent, understandable error to the caller.
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Unable to parse the dataset as a valid CSV: {path}"
        ) from exc

    # A CSV may contain column headers but no actual data rows.
    # Reject such datasets because they cannot be used for analysis.
    if df.empty:
        raise ValueError(
            f"The dataset contains no data rows: {path}"
        )

    # Return the successfully loaded dataset for downstream processing.
    return df