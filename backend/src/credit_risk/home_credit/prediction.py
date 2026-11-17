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
    cr_bootstrap_group_fairness,
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

APPLICATION_AMOUNT_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
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
    training_amount_ranges = {
        column: {
            "lower_supported": float(train[column].quantile(0.01)),
            "upper_supported": float(train[column].quantile(0.99)),
        }
        for column in APPLICATION_AMOUNT_COLUMNS
    }
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
    fairness_uncertainty = {}
    for attribute, groups in (
        ("gender", test_groups["CODE_GENDER"]),
        ("age", cr_bin_age_groups(test_groups["AGE_YEARS"])),
    ):
        fairness_results[attribute] = cr_analyze_group_fairness(
            test[TARGET_COLUMN], test_prediction, groups,
            attribute_name=attribute,
        )
        fairness_uncertainty[attribute] = cr_bootstrap_group_fairness(
            test[TARGET_COLUMN], test_prediction, groups,
            attribute_name=attribute, n_bootstrap=1000, random_state=42,
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
        "fairness_uncertainty": fairness_uncertainty,
        "training_amount_ranges": training_amount_ranges,
        "test_rows": len(test),
        "no_history_rows": len(no_history_frame),
        "no_history_events": int(no_history_frame[TARGET_COLUMN].sum()),
        "no_history_metrics": no_history_metrics,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "calibration_rows": len(calibration_features),
        "threshold_selection_rows": len(threshold_features),
    }


def cr_compute_home_credit_counterfactuals(
    bundle: dict[str, Any],
    raw_input: pd.DataFrame,
    target_probability: float | None = None,
) -> list[dict[str, Any]]:
    """Identify actionable parameter changes that reduce payment-difficulty probability below threshold.

    When an applicant is predicted as 'NO' (probability >= threshold), this function
    explores feasible financial and application adjustments (loan amount, monthly installment,
    household income, collateral) to find the minimum changes needed to achieve 'YES' (loan approval).
    """
    if not isinstance(raw_input, pd.DataFrame) or len(raw_input) != 1:
        raise ValueError("Provide exactly one Home Credit application row.")

    threshold = float(bundle["threshold"])
    target_cutoff = target_probability if target_probability is not None else threshold

    applicant = cr_prepare_home_credit_application(raw_input)
    current_prob = cr_predict_home_credit_application(bundle, applicant)

    # If applicant is already below threshold, no recourse needed
    if current_prob < target_cutoff:
        return []

    counterfactuals: list[dict[str, Any]] = []
    current_credit = float(raw_input.loc[0, "AMT_CREDIT"])
    current_annuity = float(raw_input.loc[0, "AMT_ANNUITY"])
    current_income = float(raw_input.loc[0, "AMT_INCOME_TOTAL"])

    # 1. Option: Reduce monthly installment (tenure extension)
    best_annuity = None
    for factor in np.linspace(0.95, 0.30, 27):
        test_annuity = round(current_annuity * factor, -2)
        if test_annuity <= 0:
            continue
        test_raw = raw_input.copy()
        test_raw["AMT_ANNUITY"] = test_annuity
        p = cr_predict_home_credit_application(
            bundle, cr_prepare_home_credit_application(test_raw)
        )
        if p < target_cutoff:
            best_annuity = (test_annuity, p)
            break
    if best_annuity:
        test_annuity, p = best_annuity
        diff = current_annuity - test_annuity
        pct = (diff / current_annuity) * 100
        counterfactuals.append(
            {
                "parameter": "Monthly Installment (Tenure Extension)",
                "feature_name": "AMT_ANNUITY",
                "current_value": current_annuity,
                "suggested_value": test_annuity,
                "change_amount": -diff,
                "change_percent": -pct,
                "new_probability": p,
                "new_decision": "YES",
                "action": (
                    f"Extend repayment tenure so the monthly installment drops from "
                    f"{current_annuity:,.0f} to {test_annuity:,.0f} (a reduction of "
                    f"{diff:,.0f} or {pct:.0f}%)."
                ),
            }
        )

    # 2. Option: Reduce requested loan amount (with proportional installment)
    best_credit = None
    for factor in np.linspace(0.95, 0.25, 29):
        test_credit = round(current_credit * factor, -3)
        if test_credit <= 0:
            continue
        test_annuity = round(current_annuity * (test_credit / current_credit), -2)
        test_raw = raw_input.copy()
        test_raw["AMT_CREDIT"] = test_credit
        test_raw["AMT_ANNUITY"] = test_annuity
        p = cr_predict_home_credit_application(
            bundle, cr_prepare_home_credit_application(test_raw)
        )
        if p < target_cutoff:
            best_credit = (test_credit, test_annuity, p)
            break
    if best_credit:
        test_credit, test_annuity, p = best_credit
        diff = current_credit - test_credit
        pct = (diff / current_credit) * 100
        counterfactuals.append(
            {
                "parameter": "Requested Loan Amount",
                "feature_name": "AMT_CREDIT",
                "current_value": current_credit,
                "suggested_value": test_credit,
                "change_amount": -diff,
                "change_percent": -pct,
                "new_probability": p,
                "new_decision": "YES",
                "action": (
                    f"Reduce the loan request from {current_credit:,.0f} to "
                    f"{test_credit:,.0f} (down by {diff:,.0f} or {pct:.0f}%), with "
                    f"proportional installment ~{test_annuity:,.0f}."
                ),
            }
        )

    # 3. Option: Increase annual income / Add co-borrower
    best_income = None
    for factor in np.linspace(1.10, 2.50, 29):
        test_income = round(current_income * factor, -3)
        test_raw = raw_input.copy()
        test_raw["AMT_INCOME_TOTAL"] = test_income
        p = cr_predict_home_credit_application(
            bundle, cr_prepare_home_credit_application(test_raw)
        )
        if p < target_cutoff:
            best_income = (test_income, p)
            break
    if best_income:
        test_income, p = best_income
        diff = test_income - current_income
        pct = (diff / current_income) * 100
        counterfactuals.append(
            {
                "parameter": "Annual Income / Co-Applicant",
                "feature_name": "AMT_INCOME_TOTAL",
                "current_value": current_income,
                "suggested_value": test_income,
                "change_amount": diff,
                "change_percent": pct,
                "new_probability": p,
                "new_decision": "YES",
                "action": (
                    f"Add a co-borrower or verify additional household income of at least "
                    f"{diff:,.0f}/year (raising total verified income to {test_income:,.0f})."
                ),
            }
        )

    # 4. Option: Combined moderate loan adjustment & tenure extension
    if not best_credit or best_credit[0] < current_credit * 0.70:
        for credit_factor, ann_factor in [(0.85, 0.75), (0.80, 0.70), (0.75, 0.65)]:
            test_credit = round(current_credit * credit_factor, -3)
            test_annuity = round(current_annuity * ann_factor, -2)
            test_raw = raw_input.copy()
            test_raw["AMT_CREDIT"] = test_credit
            test_raw["AMT_ANNUITY"] = test_annuity
            p = cr_predict_home_credit_application(
                bundle, cr_prepare_home_credit_application(test_raw)
            )
            if p < target_cutoff:
                counterfactuals.append(
                    {
                        "parameter": "Combined: Restructured Loan & Extended Tenure",
                        "feature_name": "AMT_CREDIT_AND_ANNUITY",
                        "current_value": f"{current_credit:,.0f} / {current_annuity:,.0f}",
                        "suggested_value": f"{test_credit:,.0f} / {test_annuity:,.0f}",
                        "change_amount": -(current_credit - test_credit),
                        "change_percent": -((1 - credit_factor) * 100),
                        "new_probability": p,
                        "new_decision": "YES",
                        "action": (
                            f"Restructure the request: reduce loan amount to {test_credit:,.0f} "
                            f"(-{(1-credit_factor)*100:.0f}%) and lower installment to "
                            f"{test_annuity:,.0f} (-{(1-ann_factor)*100:.0f}%)."
                        ),
                    }
                )
                break

    # 5. Option: Asset / Collateral pledge (property or vehicle)
    if raw_input.loc[0, "FLAG_OWN_REALTY"] == "N":
        test_raw = raw_input.copy()
        test_raw["FLAG_OWN_REALTY"] = "Y"
        p = cr_predict_home_credit_application(
            bundle, cr_prepare_home_credit_application(test_raw)
        )
        if p < target_cutoff:
            counterfactuals.append(
                {
                    "parameter": "Pledge Real Estate Collateral",
                    "feature_name": "FLAG_OWN_REALTY",
                    "current_value": "No property pledged",
                    "suggested_value": "Pledge real estate",
                    "change_amount": 0,
                    "change_percent": 0,
                    "new_probability": p,
                    "new_decision": "YES",
                    "action": "Pledge verified residential/commercial real estate as collateral security.",
                }
            )

    if raw_input.loc[0, "FLAG_OWN_CAR"] == "N":
        test_raw = raw_input.copy()
        test_raw["FLAG_OWN_CAR"] = "Y"
        p = cr_predict_home_credit_application(
            bundle, cr_prepare_home_credit_application(test_raw)
        )
        if p < target_cutoff:
            counterfactuals.append(
                {
                    "parameter": "Pledge Vehicle Collateral",
                    "feature_name": "FLAG_OWN_CAR",
                    "current_value": "No vehicle pledged",
                    "suggested_value": "Pledge vehicle",
                    "change_amount": 0,
                    "change_percent": 0,
                    "new_probability": p,
                    "new_decision": "YES",
                    "action": "Pledge verified vehicle ownership as collateral security.",
                }
            )

    return counterfactuals
