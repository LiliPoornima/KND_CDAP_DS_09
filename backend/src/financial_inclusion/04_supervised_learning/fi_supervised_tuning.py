"""Tune Component 02 demographic classifiers without changing its target.

The existing baseline artifacts are left untouched. This experiment uses the
same fixed stratified 80:20 split and tunes only inside the training partition.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT / "outputs/financial_inclusion/reports/04_supervised_learning/fi_supervised_dataset_audit.csv"
OUT_DIR = ROOT / "outputs/financial_inclusion/experiments/supervised_tuning"
MODEL_DIR = ROOT / "models/financial_inclusion/supervised_tuned"
FEATURES = ["female", "age", "educ", "inc_q", "emp_in", "urbanicity", "regionwb"]
CATEGORICAL = ["female", "educ", "inc_q", "emp_in", "urbanicity", "regionwb"]
NUMERIC = ["age"]
LABELS = [0, 1, 2, 3, 4]
SEED = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
    ])


def metrics(y_true, pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, labels=LABELS, average="weighted", zero_division=0),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, low_memory=False)
    X, y = data[FEATURES].copy(), data["cluster_id"].astype(int)
    for col in CATEGORICAL:
        X[col] = X[col].astype("object")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    definitions = {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, solver="lbfgs", random_state=SEED),
            {"model__C": np.logspace(-2, 2, 9), "model__class_weight": [None, "balanced"]},
            4,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"model__n_estimators": [300, 500], "model__max_depth": [None, 12, 24],
             "model__min_samples_leaf": [1, 3, 8], "model__max_features": ["sqrt", 0.7],
             "model__class_weight": [None, "balanced", "balanced_subsample"]},
            4,
        ),
        "xgboost": (
            XGBClassifier(objective="multi:softprob", num_class=5, eval_metric="mlogloss",
                          tree_method="hist", random_state=SEED, n_jobs=1),
            {"model__n_estimators": [250, 400], "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.03, 0.06, 0.1], "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.8, 1.0], "model__min_child_weight": [1, 5],
             "model__reg_lambda": [1, 5]},
            3,
        ),
    }
    all_metrics, reports, matrices, configs = [], [], [], {}
    for name, (estimator, params, n_iter) in definitions.items():
        search = RandomizedSearchCV(
            Pipeline([("preprocessing", build_preprocessor()), ("model", estimator)]),
            params, n_iter=n_iter, scoring="balanced_accuracy", cv=cv,
            random_state=SEED, n_jobs=1, refit=True, verbose=1,
        )
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        row = {"model": name, **metrics(y_test, pred), "cv_best_balanced_accuracy": search.best_score_}
        all_metrics.append(row)
        reports.append(f"=== {name} ===\n{classification_report(y_test, pred, labels=LABELS, zero_division=0)}")
        matrices.append(f"=== {name} ===\nrows=actual, columns=predicted; labels={LABELS}\n{confusion_matrix(y_test, pred, labels=LABELS)}")
        configs[name] = {"best_params": search.best_params_, "best_cv_balanced_accuracy": float(search.best_score_)}
        joblib.dump(search.best_estimator_, MODEL_DIR / f"{name}.joblib")
        fitted = search.best_estimator_
        encoded_names = fitted.named_steps["preprocessing"].get_feature_names_out()
        fitted_model = fitted.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importance = np.asarray(fitted_model.feature_importances_, dtype=float)
        else:
            importance = np.mean(np.abs(fitted_model.coef_), axis=0)
        pd.DataFrame({"encoded_feature": encoded_names, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(OUT_DIR / f"{name}_feature_importance.csv", index=False)

    results = pd.DataFrame(all_metrics).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    results.to_csv(OUT_DIR / "metrics.csv", index=False)
    (OUT_DIR / "classification_reports.txt").write_text("\n\n".join(reports), encoding="utf-8")
    (OUT_DIR / "confusion_matrices.txt").write_text("\n\n".join(matrices), encoding="utf-8")
    metadata = {
        "target": "cluster_id", "features": FEATURES, "class_labels": LABELS,
        "split": {"test_size": 0.2, "random_state": SEED, "stratified": True},
        "tuning": {"cv": "StratifiedKFold(n_splits=2, shuffle=True, random_state=42)", "scoring": "balanced_accuracy"},
        "train_rows": len(X_train), "test_rows": len(X_test),
        "class_counts_full": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "experiments": configs,
        "catboost": "not run: package unavailable at inspection time",
        "resampling": "not used: categorical one-hot representation makes synthetic SMOTE questionable; class weights tuned instead",
    }
    (OUT_DIR / "experiment_details.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
