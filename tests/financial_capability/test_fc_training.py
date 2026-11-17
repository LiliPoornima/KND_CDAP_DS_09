"""Training and evaluation tests (small fixture, fast)."""

import numpy as np
import pytest

from backend.src.financial_capability.fc_evaluation import (
    confusion_frame,
    evaluate_model,
    majority_baseline,
    metrics_to_frame,
    per_class_frame,
    subgroup_metrics,
)
from backend.src.financial_capability.fc_train import (
    select_best_model,
    tune_random_forest,
    tune_xgboost,
)


@pytest.mark.parametrize("fixture_name", ["rf_model", "xgb_model"])
def test_both_models_train_and_predict_valid_classes(fixture_name, request, splits):
    model = request.getfixturevalue(fixture_name)
    _, Xte, _, _ = splits
    pred = model.predict(Xte)
    assert set(np.unique(pred)) <= {0, 1, 2}
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(Xte), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_models_beat_the_majority_baseline_on_learnable_data(rf_model, splits):
    Xtr, Xte, ytr, yte = splits
    base = majority_baseline(Xtr, ytr, Xte, yte)
    assert evaluate_model(rf_model, Xte, yte)["accuracy"] > base["accuracy"]


def test_evaluation_metrics_exist_and_confusion_matrix_shape(rf_model, xgb_model, splits):
    _, Xte, _, yte = splits
    results = {}
    for name, model in (("Random Forest", rf_model), ("XGBoost", xgb_model)):
        m = evaluate_model(model, Xte, yte)
        results[name] = m
        for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                    "macro_f1", "roc_auc_ovr_macro"):
            assert 0.0 <= m[key] <= 1.0
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (3, 3) and cm.sum() == len(yte)
        assert set(m["per_class"]) == {"Low", "Medium", "High"}
        assert sum(v["support"] for v in m["per_class"].values()) == len(yte)
        assert confusion_frame(m).shape == (3, 3)
    assert len(metrics_to_frame(results)) == 2
    assert len(per_class_frame(results)) == 6


def test_subgroup_metrics_skip_tiny_groups(rf_model, splits):
    _, Xte, _, yte = splits
    sub = subgroup_metrics(rf_model, Xte, yte, min_size=10_000)
    assert sub["accuracy"].isna().all() and (sub["note"] != "").all()
    sub2 = subgroup_metrics(rf_model, Xte, yte, min_size=1)
    assert sub2["accuracy"].notna().any()


def test_cross_validation_runs_on_training_data_only(splits):
    Xtr, _, ytr, _ = splits
    rf = tune_random_forest(Xtr, ytr, folds=2, grid={"n_estimators": [20], "max_depth": [3, 6]})
    xgb = tune_xgboost(Xtr, ytr, folds=2, grid={"n_estimators": [20], "max_depth": [2]})
    assert len(rf) == 2 and len(xgb) == 1
    assert rf["cv_macro_f1_mean"].is_monotonic_decreasing
    assert select_best_model(rf, xgb) in {"Random Forest", "XGBoost"}

