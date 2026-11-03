"""Generate checked SHAP outputs for the tuned logistic model only."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from fi_shap_explanations import explain_predictions


def main():
    source = ROOT / "outputs/financial_inclusion/reports/04_supervised_learning/fi_supervised_dataset_audit.csv"
    model_path = ROOT / "models/financial_inclusion/supervised_tuned/logistic_regression.joblib"
    out = ROOT / "outputs/financial_inclusion/experiments/supervised_tuning/shap"
    df = pd.read_csv(source, low_memory=False)
    features = ["female", "age", "educ", "inc_q", "emp_in", "urbanicity", "regionwb"]
    categorical = ["female", "educ", "inc_q", "emp_in", "urbanicity", "regionwb"]
    X, y = df[features].copy(), df["cluster_id"].astype(int)
    for col in categorical:
        X[col] = X[col].astype(object)
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    bg_idx, _ = train_test_split(np.arange(len(X_train)), train_size=min(100, len(X_train)), random_state=42, stratify=y_train)
    sample_idx, _ = train_test_split(np.arange(len(X_test)), train_size=min(100, len(X_test)), random_state=42, stratify=y_test)
    pipeline = __import__("joblib").load(model_path)
    result = explain_predictions(X_test.iloc[sample_idx], X_train.iloc[bg_idx], pipeline)
    raw, grouped = result["raw_shap_values"], result["grouped_shap_values"]
    expected_raw = (len(sample_idx), len(result["encoded_feature_names"]), 5)
    expected_grouped = (len(sample_idx), len(features), 5)
    if raw.shape != expected_raw or grouped.shape != expected_grouped:
        raise ValueError(f"Unexpected SHAP shapes: {raw.shape}, {grouped.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(grouped).all():
        raise ValueError("SHAP values contain non-finite values")
    grouped_error = float(np.max(np.abs(grouped.sum(axis=1) - raw.sum(axis=1))))
    if result["max_additivity_error"] > 1e-4 or grouped_error > 1e-8:
        raise ValueError("SHAP additivity or grouped contribution check failed")
    out.mkdir(parents=True, exist_ok=True)
    result["global_importance"].to_csv(out / "global_importance.csv", index=False)
    result["local_explanations"].to_csv(out / "local_explanations.csv", index=False)
    result["predictions"].to_csv(out / "predictions.csv", index=False)
    (out / "validation.json").write_text(json.dumps({
        "model": str(model_path.relative_to(ROOT)), "shap_version": shap.__version__,
        "class_ids": result["class_ids"], "explained_rows": len(sample_idx),
        "background_rows": len(bg_idx), "raw_shape": list(raw.shape),
        "grouped_shape": list(grouped.shape), "max_additivity_error": result["max_additivity_error"],
        "max_grouped_sum_error": grouped_error, "checks_passed": True,
        "explained_output": result["explained_output"],
    }, indent=2), encoding="utf-8")
    print(f"SHAP validated; raw={raw.shape}, grouped={grouped.shape}, additivity={result['max_additivity_error']:.3g}")


if __name__ == "__main__":
    main()
