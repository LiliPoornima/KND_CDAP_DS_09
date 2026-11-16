
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    ROOT / "outputs" / "financial_inclusion" / "reports"
    / "04_supervised_learning" / "fi_supervised_dataset_audit.csv"
)

METRICS_PATH = (
    ROOT / "outputs" / "financial_inclusion" / "metrics"
    / "fi_supervised_baseline_metrics.csv"
)

EXPECTED_COUNTS = {
    0: 10107,
    1: 23574,
    2: 9005,
    3: 8781,
    4: 5036,
}


def main():
    print("=== SUPERVISED DATASET AUDIT ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("Rows:", len(df))
    print("Unique target classes:", sorted(df["cluster_id"].dropna().unique()))
    print("Missing target values:", df["cluster_id"].isna().sum())
    print("Duplicate rows:", df.duplicated().sum())

    actual_counts = (
        df["cluster_id"].value_counts().sort_index().to_dict()
    )
    actual_counts = {int(k): int(v) for k, v in actual_counts.items()}

    print("\nTarget distribution:")
    print(pd.Series(actual_counts).to_string())

    print("\nTarget distribution matches expected counts:",
          actual_counts == EXPECTED_COUNTS)

    print("\n=== SUPERVISED METRICS ===")
    metrics = pd.read_csv(METRICS_PATH)
    print(metrics.to_string(index=False))

    expected_models = {
        "Majority Baseline",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }

    actual_models = set(metrics["model"].astype(str))

    print("\nExpected model rows present:",
          expected_models.issubset(actual_models))

    print("\nAudit completed. Review any False result above.")


if __name__ == "__main__":
    main()
