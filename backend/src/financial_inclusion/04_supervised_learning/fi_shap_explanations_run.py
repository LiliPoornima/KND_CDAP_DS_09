
"""Run validated SHAP analysis and export Component 02 results.

This script loads saved artifacts only. It does not train or fit models.
"""

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

from fi_supervised_inference import load_model
from fi_shap_explanations import explain_predictions


RANDOM_STATE = 42
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 100

FEATURES = [
    "female", "educ", "inc_q", "emp_in",
    "urbanicity", "regionwb", "age",
]
CATEGORICAL = [
    "female", "educ", "inc_q", "emp_in", "urbanicity", "regionwb",
]


def main():
    data_path = (
        ROOT / "outputs" / "financial_inclusion" / "reports"
        / "04_supervised_learning" / "fi_supervised_dataset_audit.csv"
    )
    output_dir = (
        ROOT / "outputs" / "financial_inclusion" / "reports"
        / "05_shap_explainability"
    )

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset audit file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    required = set(FEATURES + ["cluster_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    X = df[FEATURES].copy()
    y = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)

    if y.isna().any() or set(y.unique()) != {0, 1, 2, 3, 4}:
        raise ValueError("Unexpected or missing cluster labels.")

    for column in CATEGORICAL:
        X[column] = X[column].astype(object)
        X[column] = X[column].where(X[column].notna(), np.nan)

    # Reconstruct the documented stratified split; no estimator is fitted.
    all_indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    background_size = min(BACKGROUND_SIZE, len(X_train))
    explain_size = min(EXPLAIN_SIZE, len(X_test))

    if background_size < 10 or explain_size < 10:
        raise ValueError("Not enough rows for a useful SHAP sample.")

    background_positions, _ = train_test_split(
        np.arange(len(X_train)),
        train_size=background_size,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    explain_positions, _ = train_test_split(
        np.arange(len(X_test)),
        train_size=explain_size,
        random_state=RANDOM_STATE,
        stratify=y_test,
    )

    background = X_train.iloc[background_positions].copy()
    sample = X_test.iloc[explain_positions].copy()

    # Keep original audit-file row positions for reproducibility.
    source_row_indices = X.index[test_idx[explain_positions]].to_numpy()

    pipeline = load_model()
    result = explain_predictions(sample, background, pipeline)

    raw_values = result["raw_shap_values"]
    grouped_values = result["grouped_shap_values"]
    class_ids = result["class_ids"]
    encoded_names = result["encoded_feature_names"]

    expected_raw_shape = (len(sample), len(encoded_names), len(class_ids))
    expected_grouped_shape = (len(sample), len(FEATURES), len(class_ids))

    if raw_values.shape != expected_raw_shape:
        raise AssertionError(
            f"Raw SHAP shape {raw_values.shape}; expected {expected_raw_shape}"
        )
    if grouped_values.shape != expected_grouped_shape:
        raise AssertionError(
            "Grouped SHAP shape does not match sample, predictors and classes."
        )
    if not np.isfinite(raw_values).all():
        raise AssertionError("Raw SHAP values contain non-finite values.")
    if not np.isfinite(grouped_values).all():
        raise AssertionError("Grouped SHAP values contain non-finite values.")
    if result["max_additivity_error"] >= 1e-4:
        raise AssertionError("SHAP additivity validation failed.")

    global_importance = result["global_importance"].copy()
    local = result["local_explanations"].copy()
    predictions = result["predictions"].copy()

    # Add pseudonymous row positions, not respondent identifiers or raw inputs.
    local["source_row_index"] = np.repeat(source_row_indices, len(FEATURES))
    predictions["source_row_index"] = source_row_indices

    # Ranking within each cluster; larger values mean larger absolute
    # contributions to that class's decision score in this sample.
    global_importance["rank_within_cluster"] = (
        global_importance.groupby("cluster_id")[
            "mean_absolute_shap_decision_score"
        ]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    # Aggregate contribution check: grouped contributions should preserve
    # the sum of encoded contributions for every row and class.
    grouped_sums = grouped_values.sum(axis=1)
    raw_sums = raw_values.sum(axis=1)
    grouped_sum_error = float(np.max(np.abs(grouped_sums - raw_sums)))

    if grouped_sum_error > 1e-8:
        raise AssertionError(
            f"Grouped contribution conservation failed: {grouped_sum_error}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    global_path = output_dir / "fi_shap_global_importance.csv"
    local_path = output_dir / "fi_shap_local_explanations.csv"
    predictions_path = output_dir / "fi_shap_predictions.csv"
    metadata_path = output_dir / "fi_shap_validation_summary.json"

    global_importance.to_csv(global_path, index=False)
    local.to_csv(local_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = {
        "analysis": "Saved Logistic Regression pipeline SHAP explanations",
        "shap_version": shap.__version__,
        "model_artifact": "models/financial_inclusion/logistic_regression.joblib",
        "target": "cluster_id",
        "explained_output": result["explained_output"],
        "random_state": RANDOM_STATE,
        "background_rows": len(background),
        "explained_rows": len(sample),
        "train_rows_in_reconstructed_split": len(X_train),
        "test_rows_in_reconstructed_split": len(X_test),
        "class_ids": class_ids,
        "original_predictors": FEATURES,
        "encoded_feature_count": len(encoded_names),
        "raw_shap_shape": list(raw_values.shape),
        "grouped_shap_shape": list(grouped_values.shape),
        "global_importance_rows": len(global_importance),
        "local_explanation_rows": len(local),
        "prediction_rows": len(predictions),
        "max_additivity_error": result["max_additivity_error"],
        "max_grouped_sum_error": grouped_sum_error,
        "checks_passed": {
            "shapes": True,
            "finite_shap_values": True,
            "additivity": True,
            "grouped_contribution_conservation": True,
            "all_five_classes": class_ids == [0, 1, 2, 3, 4],
        },
        "interpretation_limit": (
            "SHAP values explain Logistic Regression decision scores, not "
            "probabilities or causal reasons for financial exclusion. "
            "Global rankings are based on this sampled test subset."
        ),
        "split_provenance_note": (
            "The split indices were reconstructed from the saved audit CSV "
            "using the documented seed and stratification. Exact equivalence "
            "to the original split requires matching source row order and "
            "pre-split filtering."
        ),
        "outputs": [
            str(global_path.relative_to(ROOT)),
            str(local_path.relative_to(ROOT)),
            str(predictions_path.relative_to(ROOT)),
            str(metadata_path.relative_to(ROOT)),
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("SHAP VERSION:", shap.__version__)
    print("BACKGROUND ROWS:", len(background))
    print("EXPLAINED ROWS:", len(sample))
    print("RAW SHAP SHAPE:", raw_values.shape)
    print("GROUPED SHAP SHAPE:", grouped_values.shape)
    print("CLASSES:", class_ids)
    print("MAX ADDITIVITY ERROR:", result["max_additivity_error"])
    print("MAX GROUPED SUM ERROR:", grouped_sum_error)
    print("GLOBAL IMPORTANCE ROWS:", len(global_importance))
    print("LOCAL EXPLANATION ROWS:", len(local))
    print("PREDICTION ROWS:", len(predictions))
    print("ALL VALIDATION CHECKS: PASS")
    print("\nOUTPUT FILES:")
    for path in (global_path, local_path, predictions_path, metadata_path):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
