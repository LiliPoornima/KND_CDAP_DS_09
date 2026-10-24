"""Credit-card baseline training using the shared credit-risk estimators."""

from typing import Any

import pandas as pd

from backend.src.credit_risk.give_me_some_credit.model import (
    cr_train_logistic_regression,
    cr_train_random_forest,
)

from .preprocessing import TARGET_COLUMN


def cr_train_credit_card_logistic_regression(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Any:
    """Fit the shared Logistic Regression baseline on training rows."""

    return cr_train_logistic_regression(
        train_df,
        target_column=target_column,
    )


def cr_train_credit_card_random_forest(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Any:
    """Fit the shared Random Forest baseline on training rows."""

    return cr_train_random_forest(
        train_df,
        target_column=target_column,
    )
