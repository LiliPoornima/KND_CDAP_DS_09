
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = (
    ROOT / "outputs" / "financial_inclusion"
    / "reports" / "04_supervised_learning" / "fi_supervised_dataset_audit.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "financial_inclusion" / "metrics"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUTPUT_DIR / "fi_supervised_baseline_metrics.csv"
REPORT_PATH = OUTPUT_DIR / "fi_supervised_classification_reports.txt"
CONFUSION_PATH = OUTPUT_DIR / "fi_supervised_confusion_matrices.txt"

TARGET = "cluster_id"

CATEGORICAL_FEATURES = [
    "female",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "regionwb",
]

NUMERIC_FEATURES = ["age"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CLUSTER_LABELS = [0, 1, 2, 3]


# --------------------------------------------------
# 2. MAIN FUNCTION
# --------------------------------------------------

def main():

    df = pd.read_csv(DATA_PATH, low_memory=False)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Preserve categorical variables as categories.
    # Convert pandas missing values to np.nan-compatible
    # missing values for scikit-learn imputers.
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
        X[col] = X[col].where(X[col].notna(), float("nan"))

    # Same split for every model.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 3. PREPROCESSING
    # --------------------------------------------------

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
    ])

    # --------------------------------------------------
    # 4. CLASSIFICATION MODELS
    # --------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    reports = []
    matrices = []

    print("=== DATASET AND SPLIT ===")
    print("Total rows:", len(df))
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Predictors:", FEATURES)

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index().to_string())

    # --------------------------------------------------
    # 5. MAJORITY-CLASS BASELINE
    # --------------------------------------------------

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    dummy_predictions = dummy.predict(X_test)

    dummy_metrics = {
        "model": "Majority Baseline",
        "accuracy": accuracy_score(y_test, dummy_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, dummy_predictions
        ),
        "macro_f1": f1_score(
            y_test,
            dummy_predictions,
            labels=CLUSTER_LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_test,
            dummy_predictions,
            labels=CLUSTER_LABELS,
            average="weighted",
            zero_division=0,
        ),
    }

    results.append(dummy_metrics)

    print("\n=== Majority Baseline ===")
    for key, value in dummy_metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

    # --------------------------------------------------
    # 6. TRAIN AND EVALUATE MODELS
    # --------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=CLUSTER_LABELS,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_test,
                predictions,
                labels=CLUSTER_LABELS,
                average="weighted",
                zero_division=0,
            ),
        }

        results.append(metrics)

        report = classification_report(
            y_test,
            predictions,
            labels=CLUSTER_LABELS,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=CLUSTER_LABELS,
        )

        reports.append(
            f"\n=== {name} ===\n{report}"
        )

        matrices.append(
            f"\n=== {name} ===\n"
            "Rows = actual clusters; columns = predicted clusters\n"
            f"Cluster order: {CLUSTER_LABELS}\n{matrix}"
        )

        print(f"\n=== {name} ===")

        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.4f}")

        print("\nClassification report:")
        print(report)

        print("Confusion matrix:")
        print(matrix)

    # --------------------------------------------------
    # 7. SAVE COMPARISON RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False,
    )

    results_df.to_csv(RESULT_PATH, index=False)

    REPORT_PATH.write_text(
        "\n".join(reports),
        encoding="utf-8",
    )

    CONFUSION_PATH.write_text(
        "\n".join(matrices),
        encoding="utf-8",
    )

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    print("\nSaved metrics:", RESULT_PATH)
    print("Saved reports:", REPORT_PATH)
    print("Saved confusion matrices:", CONFUSION_PATH)


if __name__ == "__main__":
    main()
