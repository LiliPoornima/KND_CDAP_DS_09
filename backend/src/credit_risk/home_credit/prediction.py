"""Application-only Home Credit assessment for a research prototype."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
    cr_evaluate_binary_probabilities,
)
from backend.src.credit_risk.home_credit.feature_engineering import (
    cr_create_home_credit_features,
)
from backend.src.credit_risk.home_credit.preprocessing import TARGET_COLUMN
from backend.src.credit_risk.home_credit.training import (
    cr_train_home_credit_random_forest,
)
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_analyze_group_fairness,
    cr_bin_age_groups,
)


APPLICATION_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "CREDIT_TO_INCOME_RATIO",
    "ANNUITY_TO_INCOME_RATIO",
    "CREDIT_TO_GOODS_RATIO",
    "ANNUITY_TO_CREDIT_RATIO",
    "YEARS_EMPLOYED",
]

# Used only to audit the held-out performance for applicants with no linked
# bureau, previous-application, installment, POS-cash, or card history.
LINKED_HISTORY_COLUMNS = [
    "BUREAU_PRIOR_CREDIT_COUNT",
    "PREVIOUS_APPLICATION_COUNT",
    "INSTALLMENT_DUE_COUNT",
    "POS_CASH_MONTH_COUNT",
    "CREDIT_CARD_MONTH_COUNT",
]


def cr_prepare_home_credit_application(raw_input: pd.DataFrame) -> pd.DataFrame:
    """Create the model's application-time features from questionnaire answers."""
    if not isinstance(raw_input, pd.DataFrame) or len(raw_input) != 1:
        raise ValueError("Provide exactly one Home Credit application row.")
    # The model uses only these application fields, so optional source fields
    # such as birth date and external bureau scores are not required at scoring.
    engineered = cr_create_home_credit_features(raw_input, strict=False)
    missing = [column for column in APPLICATION_FEATURES if column not in engineered]
    if missing:
        raise ValueError(f"Application input is missing model features: {missing}")
    return engineered.loc[:, APPLICATION_FEATURES]


def cr_predict_home_credit_application(bundle: dict[str, Any], applicant: pd.DataFrame) -> float:
    """Return one validation-calibrated payment-difficulty probability."""
    if not isinstance(applicant, pd.DataFrame) or len(applicant) != 1:
        raise ValueError("Provide exactly one prepared applicant row.")
    if list(applicant.columns) != APPLICATION_FEATURES:
        raise ValueError("Applicant columns do not match the application model schema.")
    raw_probability = float(bundle["model"].predict_proba(applicant)[0, 1])
    clipped = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
    log_odds = np.array([[np.log(clipped / (1 - clipped))]])
    return float(bundle["calibrator"].predict_proba(log_odds)[0, 1])


def cr_fit_home_credit_application_model(
    prepared_dir: str | Path,
) -> dict[str, Any]:
    """Fit and evaluate the application-only baseline using saved splits.

    The validation-selected threshold and held-out metrics are both specific to
    this 20-feature model. No linked-history feature is given to the estimator.
    """
    source = Path(prepared_dir)
    usecols = list(dict.fromkeys([
        *APPLICATION_FEATURES,
        *LINKED_HISTORY_COLUMNS,
        TARGET_COLUMN,
    ]))
    train = pd.read_csv(source / "train_processed.csv", usecols=usecols)
    validation = pd.read_csv(source / "validation_processed.csv", usecols=usecols)
    test = pd.read_csv(source / "test_processed.csv", usecols=usecols)
    test_groups = pd.read_csv(source / "test_groups.csv")
    if len(test_groups) != len(test):
        raise ValueError("Home Credit test groups do not align with the test split.")

    model = cr_train_home_credit_random_forest(
        train[APPLICATION_FEATURES + [TARGET_COLUMN]],
        target_column=TARGET_COLUMN,
    )
    calibration_features, threshold_features, calibration_target, threshold_target = (
        train_test_split(
            validation[APPLICATION_FEATURES],
            validation[TARGET_COLUMN],
            test_size=0.5,
            random_state=42,
            stratify=validation[TARGET_COLUMN],
        )
    )

    def _log_odds(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    calibration_raw_probability = model.predict_proba(calibration_features)[:, 1]
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(_log_odds(calibration_raw_probability), calibration_target)

    def calibrated_probability(features: pd.DataFrame) -> np.ndarray:
        raw_probability = model.predict_proba(features)[:, 1]
        return calibrator.predict_proba(_log_odds(raw_probability))[:, 1]

    validation_probability = calibrated_probability(threshold_features)
    threshold_table = cr_analyze_probability_thresholds(
        threshold_target, validation_probability
    )
    selected = threshold_table.sort_values(
        ["F1-score", "Precision", "Threshold"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    threshold = float(selected["Threshold"])
    test_probability = calibrated_probability(test[APPLICATION_FEATURES])
    test_metrics = cr_evaluate_binary_probabilities(
        test[TARGET_COLUMN], test_probability, threshold=threshold,
        target_names=("No recorded difficulty", "Recorded difficulty"),
    )
    test_prediction = (test_probability >= threshold).astype(int)
    fairness_results = {}
    for attribute, groups in (
        ("gender", test_groups["CODE_GENDER"]),
        ("age", cr_bin_age_groups(test_groups["AGE_YEARS"])),
    ):
        fairness_results[attribute] = cr_analyze_group_fairness(
            test[TARGET_COLUMN], test_prediction, groups,
            attribute_name=attribute,
        )

    no_history = test[LINKED_HISTORY_COLUMNS].fillna(0).eq(0).all(axis=1)
    no_history_frame = test.loc[no_history]
    no_history_metrics = None
    if no_history_frame[TARGET_COLUMN].nunique() == 2:
        no_history_metrics = cr_evaluate_binary_probabilities(
            no_history_frame[TARGET_COLUMN],
            test_probability[no_history.to_numpy()],
            threshold=threshold,
            target_names=("No recorded difficulty", "Recorded difficulty"),
        )

    return {
        "model": model,
        "calibrator": calibrator,
        "threshold": threshold,
        "validation_f1": float(selected["F1-score"]),
        "test_metrics": test_metrics,
        "fairness_results": fairness_results,
        "test_rows": len(test),
        "no_history_rows": len(no_history_frame),
        "no_history_events": int(no_history_frame[TARGET_COLUMN].sum()),
        "no_history_metrics": no_history_metrics,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "calibration_rows": len(calibration_features),
        "threshold_selection_rows": len(threshold_features),
    }
