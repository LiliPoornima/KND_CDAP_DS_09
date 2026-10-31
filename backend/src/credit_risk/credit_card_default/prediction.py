"""Interactive single-applicant prediction for the Taiwan card dataset."""

from pathlib import Path

import pandas as pd

from backend.src.credit_risk.credit_card_default.feature_engineering import (
    cr_create_credit_card_features,
)
from backend.src.credit_risk.credit_card_default.training import (
    cr_train_credit_card_random_forest,
)
from backend.src.credit_risk.credit_card_default.preprocessing import TARGET_COLUMN


def cr_prepare_credit_card_prediction(raw_input: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Engineer documented behavioral summaries and match the fitted schema."""
    processed = cr_create_credit_card_features(raw_input, strict=True)
    missing = [column for column in feature_columns if column not in processed]
    if missing:
        raise ValueError(f"Prediction input is missing model features: {missing}")
    return processed.loc[:, feature_columns]


def cr_train_credit_card_prediction_model(training_path: str | Path):
    """Fit the research Random Forest on the existing training partition."""
    training = pd.read_csv(training_path)
    return cr_train_credit_card_random_forest(training, target_column=TARGET_COLUMN)
