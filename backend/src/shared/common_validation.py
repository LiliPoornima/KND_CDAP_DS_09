"""Shared input validation helpers for project data pipelines.

The project defines several repeated validation patterns in its data-prep and
feature-engineering modules (for example ensuring required columns exist, that
DataFrames are not empty, and that target columns are present without missing
labels). These helpers centralize that logic while keeping the public API small,
flexible, and friendly to the project's existing validation patterns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd


__all__ = [
    "ensure_dataframe",
    "require_columns",
    "validate_columns",
    "validate_not_none",
    "validate_string",
    "validate_number",
    "validate_probability",
    "validate_path",
    "validate_target_column",
]


def _normalise_columns(columns: Iterable[str] | str | None) -> list[str]:
    """Return a list of non-empty column names from a supported input."""
    if columns is None:
        return []

    if isinstance(columns, str):
        columns = [columns]

    result: list[str] = []
    for column in columns:
        if not isinstance(column, str):
            raise TypeError("Column names must be strings.")
        if not column:
            raise ValueError("Column names cannot be empty strings.")
        result.append(column)

    return result


def validate_not_none(value: Any, *, name: str = "value") -> Any:
    """Reject None values early with a clear error message."""
    if value is None:
        raise ValueError(f"{name} cannot be None.")
    return value


def validate_string(
    value: Any,
    *,
    name: str = "value",
    allow_blank: bool = False,
) -> str:
    """Validate and return a string value."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not allow_blank and not value.strip():
        raise ValueError(f"{name} cannot be blank.")
    return value


def validate_number(
    value: Any,
    *,
    name: str = "value",
    minimum: float | None = None,
    maximum: float | None = None,
    allow_zero: bool = True,
    allow_negative: bool = True,
) -> float:
    """Validate numeric scalar input against optional bounds."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    numeric = float(value)

    if not allow_zero and numeric == 0:
        raise ValueError(f"{name} cannot be zero.")
    if not allow_negative and numeric < 0:
        raise ValueError(f"{name} cannot be negative.")

    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")

    return numeric


def validate_probability(
    value: Any,
    *,
    name: str = "value",
) -> float:
    """Validate a probability in the inclusive range [0, 1]."""
    return validate_number(
        value,
        name=name,
        minimum=0.0,
        maximum=1.0,
        allow_zero=True,
        allow_negative=False,
    )


def validate_path(path: str | Path | None, *, name: str = "path") -> Path:
    """Validate and normalise a file-system path."""
    if path is None:
        raise ValueError(f"{name} cannot be None.")

    value = Path(path)
    if not str(value):
        raise ValueError(f"{name} cannot be blank.")
    return value


def ensure_dataframe(
    df: Any,
    *,
    name: str = "DataFrame",
    allow_empty: bool = False,
    require_unique_columns: bool = True,
    required_columns: Iterable[str] | str | None = None,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Validate a pandas DataFrame and optionally enforce required columns.

    This helper is intentionally permissive: it raises clear errors for the common
    misuse patterns seen across the project, but it does not mutate the supplied
    DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if df.empty and not allow_empty:
        raise ValueError(f"{name} cannot be empty.")
    if require_unique_columns and not df.columns.is_unique:
        raise ValueError(f"{name} column names must be unique.")

    if required_columns is not None:
        require_columns(df, required_columns, dataset_name=name)

    if target_column is not None:
        validate_target_column(df, target_column)

    return df


def require_columns(
    df: Any,
    columns: Iterable[str] | str | None,
    *,
    dataset_name: str = "DataFrame",
) -> None:
    """Raise ValueError when any required columns are missing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{dataset_name} must be a pandas DataFrame.")

    required = _normalise_columns(columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def validate_columns(
    df: Any,
    columns: Iterable[str] | str | None,
    *,
    dataset_name: str = "DataFrame",
) -> bool:
    """Return True if all columns are present; otherwise raise ValueError.

    This mirrors the common pattern used throughout the repository: the caller
    passes a DataFrame and a list of required columns; column absence is treated
    as a validation failure.
    """
    require_columns(df, columns, dataset_name=dataset_name)
    return True


def validate_target_column(
    df: pd.DataFrame,
    target_column: str | None,
    *,
    name: str = "target column",
) -> None:
    """Ensure a target column exists and contains no missing labels."""
    if target_column is None:
        return

    validate_string(target_column, name=name)
    if target_column not in df.columns:
        raise ValueError(f"{name} '{target_column}' does not exist.")

    if df[target_column].isna().any():
        raise ValueError(f"{name} '{target_column}' contains missing values.")


def validate_mapping(
    value: Any,
    *,
    name: str = "mapping",
    allow_empty: bool = True,
) -> Mapping[str, Any]:
    """Validate that an object is a mapping and optionally that it is non-empty."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if not allow_empty and not value:
        raise ValueError(f"{name} cannot be empty.")
    return value
