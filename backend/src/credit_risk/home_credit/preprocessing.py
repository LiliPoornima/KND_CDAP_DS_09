"""Leakage-aware application splits and audit-group separation."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.src.credit_risk.home_credit.feature_engineering import (
    cr_create_home_credit_features,
)


TARGET_COLUMN = "TARGET"
IDENTIFIER_COLUMN = "SK_ID_CURR"
AUDIT_GROUP_COLUMNS = ("CODE_GENDER", "DAYS_BIRTH")


def cr_prepare_home_credit_dataset(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split labeled application data and preserve audit fields separately.

    Feature imputation and categorical encoding are intentionally deferred to
    model pipelines, which will fit those transformations on training rows
    only. The internal application test partition is held out from selection.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if not df.columns.is_unique:
        raise ValueError("Input data must have unique column names.")
    if not df.index.is_unique:
        raise ValueError("Input row indexes must be unique.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")
    if df[target_column].isna().any():
        raise ValueError("Target column contains missing values.")
    if set(df[target_column].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes: 0 and 1.")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split sizes must be greater than 0 and less than 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    data = df.copy()
    if IDENTIFIER_COLUMN in data.columns:
        if data[IDENTIFIER_COLUMN].isna().any():
            raise ValueError(f"{IDENTIFIER_COLUMN} must not contain missing values.")
        if not data[IDENTIFIER_COLUMN].is_unique:
            raise ValueError(f"{IDENTIFIER_COLUMN} must be unique per application.")

    target = data[target_column].astype(int)
    row_ids = data.index.to_numpy()
    development_ids, test_ids = train_test_split(
        row_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=target.to_numpy(),
    )
    relative_validation_size = validation_size / (1 - test_size)
    train_ids, validation_ids = train_test_split(
        development_ids,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=target.loc[development_ids].to_numpy(),
    )

    def prepare_split(row_subset: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_split = data.loc[row_subset].copy()
        audit_data = pd.DataFrame(index=raw_split.index)
        if "CODE_GENDER" in raw_split.columns:
            audit_data["CODE_GENDER"] = raw_split["CODE_GENDER"]
        if "DAYS_BIRTH" in raw_split.columns:
            audit_data["AGE_YEARS"] = (
                -pd.to_numeric(raw_split["DAYS_BIRTH"], errors="coerce")
                / 365.25
            )

        processed = cr_create_home_credit_features(
            raw_split.drop(columns=[target_column]),
            strict=True,
        )
        # The target, application identifier, and direct audit attributes
        # never enter the predictor matrix.
        excluded = [target_column, IDENTIFIER_COLUMN, "CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"]
        predictors = processed.drop(
            columns=[column for column in excluded if column in processed.columns]
        )
        predictors[target_column] = target.loc[row_subset]
        return predictors, audit_data

    train_processed, train_groups = prepare_split(train_ids)
    validation_processed, validation_groups = prepare_split(validation_ids)
    test_processed, test_groups = prepare_split(test_ids)

    columns = list(train_processed.columns)
    if not (
        list(validation_processed.columns) == columns
        and list(test_processed.columns) == columns
    ):
        raise ValueError("Processed split columns do not match.")

    audit_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    metadata: dict[str, Any] = {
        "validation_processed": validation_processed,
        "audit_groups": audit_groups,
        "audit_group_columns": list(train_groups.columns),
        "feature_columns": columns[:-1],
        "target_column": target_column,
        "split_sizes": {
            "train": len(train_processed),
            "validation": len(validation_processed),
            "test": len(test_processed),
        },
    }

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        train_processed.to_csv(destination / "train_processed.csv", index=False)
        validation_processed.to_csv(
            destination / "validation_processed.csv",
            index=False,
        )
        test_processed.to_csv(destination / "test_processed.csv", index=False)
        for split_name, group_frame in audit_groups.items():
            group_frame.to_csv(
                destination / f"{split_name}_groups.csv",
                index=False,
            )

    return train_processed, test_processed, metadata
