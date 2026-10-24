"""Leakage-aware split and training-fitted imputation for Taiwan data."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.src.credit_risk.credit_card_default.feature_engineering import (
    cr_create_credit_card_features,
)


TARGET_COLUMN = "default.payment.next.month"
IDENTIFIER_COLUMNS = ["ID"]
AUDIT_GROUP_COLUMNS = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]


def cr_prepare_credit_card_dataset(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create train/validation/test partitions and retain group labels apart.

    ``ID`` and documented demographic/audit columns are excluded from model
    predictors by default. The group columns are returned separately for
    fairness analysis. Imputation values are fitted using training rows only.
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
    group_columns = [
        column for column in AUDIT_GROUP_COLUMNS if column in data.columns
    ]
    group_data = data[group_columns].copy()
    target = data[target_column].astype(int)
    excluded = [
        column
        for column in [target_column, *IDENTIFIER_COLUMNS, *group_columns]
        if column in data.columns
    ]
    predictors = data.drop(columns=excluded)
    if predictors.empty:
        raise ValueError("No predictor columns remain after excluding audit fields.")
    non_numeric = [
        column
        for column in predictors.columns
        if not pd.api.types.is_numeric_dtype(predictors[column])
    ]
    if non_numeric:
        raise ValueError(f"Model predictors must be numeric: {non_numeric}")

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

    # Create deterministic row-level history summaries after splitting.
    train_features = cr_create_credit_card_features(
        predictors.loc[train_ids],
        strict=True,
    )
    validation_features = cr_create_credit_card_features(
        predictors.loc[validation_ids],
        strict=True,
    )
    test_features = cr_create_credit_card_features(
        predictors.loc[test_ids],
        strict=True,
    )

    # Fit medians from training rows only, including derived ratio features.
    imputation_values = train_features.median()
    imputation_values = imputation_values.fillna(0.0)
    train_features = train_features.fillna(imputation_values)
    validation_features = validation_features.fillna(imputation_values)
    test_features = test_features.fillna(imputation_values)

    def add_target(features: pd.DataFrame, row_subset: Any) -> pd.DataFrame:
        result = features.copy()
        result[target_column] = target.loc[row_subset]
        return result

    train_processed = add_target(train_features, train_ids)
    validation_processed = add_target(validation_features, validation_ids)
    test_processed = add_target(test_features, test_ids)

    expected_columns = list(train_processed.columns)
    if not (
        list(validation_processed.columns) == expected_columns
        and list(test_processed.columns) == expected_columns
    ):
        raise ValueError("Processed split columns do not match.")

    audit_groups = {
        "train": group_data.loc[train_ids].copy(),
        "validation": group_data.loc[validation_ids].copy(),
        "test": group_data.loc[test_ids].copy(),
    }
    metadata: dict[str, Any] = {
        "validation_processed": validation_processed,
        "audit_groups": audit_groups,
        "audit_group_columns": group_columns,
        "excluded_predictor_columns": excluded,
        "imputation_values": imputation_values.to_dict(),
        "feature_columns": expected_columns[:-1],
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
