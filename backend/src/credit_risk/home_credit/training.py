"""Train leakage-safe Home Credit baselines on mixed-type features."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import TARGET_COLUMN


def _validate_home_credit_training_data(
    train_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("Training data must be a pandas DataFrame.")
    if train_df.empty:
        raise ValueError("Training data must not be empty.")
    if not train_df.columns.is_unique:
        raise ValueError("Training data must have unique column names.")
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")
    if train_df[target_column].isna().any():
        raise ValueError("Training target contains missing values.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    if X_train.empty:
        raise ValueError("Training data must contain predictor columns.")
    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain both classes: 0 and 1.")

    numeric_columns = [
        column
        for column in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[column])
    ]
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]
    return X_train, y_train.astype(int), numeric_columns, categorical_columns


def _build_home_credit_pipeline(
    train_df: pd.DataFrame,
    *,
    target_column: str,
    classifier: Any,
    scale_numeric: bool,
    fit: bool = True,
) -> Pipeline:
    X_train, y_train, numeric_columns, categorical_columns = (
        _validate_home_credit_training_data(train_df, target_column)
    )
    transformers = []

    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        ]
        if scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler(with_mean=False))
            )
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=20,
                        sparse_output=True,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    if fit:
        model.fit(X_train, y_train)
    return model


def cr_train_home_credit_logistic_regression(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    *,
    solver: str = "newton-cholesky",
    max_iter: int = 1000,
    tolerance: float = 1e-4,
) -> Pipeline:
    """Fit Logistic Regression with train-only imputation and encoding.

    Solver controls are exposed for reproducible convergence diagnostics on
    the high-row-count, sparse Home Credit design matrix.
    """

    if solver not in {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}:
        raise ValueError("Unsupported Logistic Regression solver.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not 0 < tolerance < 1
    ):
        raise ValueError("tolerance must be a number between 0 and 1.")

    return _build_home_credit_pipeline(
        train_df,
        target_column=target_column,
        classifier=LogisticRegression(
            class_weight="balanced",
            solver=solver,
            max_iter=max_iter,
            tol=tolerance,
            random_state=42,
        ),
        scale_numeric=True,
    )


def cr_train_home_credit_random_forest(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Pipeline:
    """Fit a balanced Random Forest with train-only preprocessing."""

    return _build_home_credit_pipeline(
        train_df,
        target_column=target_column,
        classifier=RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        scale_numeric=False,
    )


def cr_create_home_credit_random_forest(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Pipeline:
    """Create the standard Random Forest pipeline without fitting it."""

    return _build_home_credit_pipeline(
        train_df,
        target_column=target_column,
        classifier=RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        scale_numeric=False,
        fit=False,
    )
