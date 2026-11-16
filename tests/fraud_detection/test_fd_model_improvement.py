"""Tests for candidate models and detailed evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_config import MODEL_FEATURES
from backend.src.fraud_detection.fd_model_improvement import (
    compute_detailed_metrics,
    evaluate_fixed_threshold,
    predict_model_scores,
    select_validation_threshold,
    train_candidate_model,
)


def _toy_data(n=200):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        feat: rng.normal(size=n).astype(np.float32) for feat in MODEL_FEATURES
    })
    y = (rng.random(n) < 0.1).astype(int)
    # Ensure at least 5 positives
    y[:5] = 1
    return df, y


@pytest.mark.parametrize("model_name", [
    "logistic_regression",
    "lightgbm_default",
    "lightgbm_tuned",
    "xgboost",
    "catboost",
])
def test_all_candidate_models_train_and_predict(model_name):
    X, y = _toy_data(150)
    model = train_candidate_model(model_name, X, y, seed=1, features=MODEL_FEATURES)
    scores = predict_model_scores(model, X)
    assert len(scores) == len(X)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert not np.isnan(scores).any()


def test_compute_detailed_metrics_known_case():
    y = np.array([0] * 100 + [1] * 10)
    p = np.concatenate([np.linspace(0.01, 0.99, 100), np.linspace(0.80, 0.999, 10)])
    m = compute_detailed_metrics(y, p, target_fpr=0.01)

    assert m["n_tx"] == 110
    assert m["n_fraud"] == 10
    assert m["prevalence"] == pytest.approx(10 / 110)
    assert m["tp"] + m["fn"] == 10
    assert m["fp"] + m["tn"] == 100
    assert m["flagged"] == m["tp"] + m["fp"]
    assert m["actual_fpr"] == pytest.approx(m["fp"] / 100)
    assert m["precision_at_1fpr"] == pytest.approx(m["tp"] / m["flagged"])
    # Verify both oracle_* and legacy keys exist and match
    assert m["oracle_threshold"] == m["threshold"]
    assert m["oracle_recall_at_1fpr"] == m["recall_at_1fpr"]
    assert m["oracle_actual_fpr"] == m["actual_fpr"]


def test_compute_detailed_metrics_missing_class():
    y_zeros = np.zeros(50, dtype=int)
    p = np.random.random(50)
    m = compute_detailed_metrics(y_zeros, p)
    assert np.isnan(m["recall_at_1fpr"])
    assert np.isnan(m["precision_at_1fpr"])
    assert np.isnan(m["oracle_recall_at_1fpr"])


def test_select_validation_threshold_uses_only_negatives():
    y_val = np.array([0] * 100 + [1] * 10)
    scores_val = np.linspace(0.0, 1.0, 110)
    # Expected threshold is 99th percentile of the 100 negative scores
    neg_scores = scores_val[y_val == 0]
    expected_thr = float(np.quantile(neg_scores, 0.99))
    thr = select_validation_threshold(y_val, scores_val, target_fpr=0.01)
    assert thr == pytest.approx(expected_thr)


def test_fixed_threshold_selection_never_accesses_test_labels():
    """Verify that changing test labels has zero impact on the validation-derived threshold."""
    rng = np.random.default_rng(123)
    y_val = (rng.random(200) < 0.05).astype(int)
    scores_val = rng.uniform(0.0, 1.0, 200)

    val_thr = select_validation_threshold(y_val, scores_val, target_fpr=0.01)

    # Change external/test labels completely
    y_test_clean = (rng.random(500) < 0.05).astype(int)
    y_test_perturbed = 1 - y_test_clean
    scores_test = rng.uniform(0.0, 1.0, 500)

    # Selected validation threshold remains identical
    assert select_validation_threshold(y_val, scores_val, target_fpr=0.01) == val_thr

    # Applying the locked threshold to test stream evaluates blindly
    eval_clean = evaluate_fixed_threshold(y_test_clean, scores_test, val_thr)
    eval_perturbed = evaluate_fixed_threshold(y_test_perturbed, scores_test, val_thr)
    assert eval_clean["fixed_threshold"] == val_thr
    assert eval_perturbed["fixed_threshold"] == val_thr


def test_evaluate_fixed_threshold_confusion_matrix_and_fpr():
    y_test = np.array([0] * 200 + [1] * 20)
    # 20 negatives scored > 0.5, 180 negatives scored <= 0.5
    # 15 positives scored > 0.5, 5 positives scored <= 0.5
    scores = np.zeros(220)
    scores[:20] = 0.8  # FP: 20
    scores[20:200] = 0.2  # TN: 180
    scores[200:215] = 0.9  # TP: 15
    scores[215:220] = 0.1  # FN: 5

    res = evaluate_fixed_threshold(y_test, scores, threshold=0.5)
    assert res["fixed_tp"] == 15
    assert res["fixed_fp"] == 20
    assert res["fixed_tn"] == 180
    assert res["fixed_fn"] == 5
    assert res["fixed_flagged"] == 35
    assert res["fixed_recall"] == pytest.approx(15 / 20)
    # Actual FPR MUST have legitimate transactions (TN + FP = 200) as denominator
    assert res["fixed_actual_fpr"] == pytest.approx(20 / 200)
    assert res["fixed_precision"] == pytest.approx(15 / 35)

