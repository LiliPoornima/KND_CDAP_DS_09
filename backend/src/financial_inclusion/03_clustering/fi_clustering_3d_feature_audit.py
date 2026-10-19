
"""Audit features for three-dimension clustering."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    ROOT / "data" / "financial_inclusion" / "processed"
    / "fi_clustering_input_3d.csv"
)

REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports" / "03_clustering"
METRICS_DIR = ROOT / "outputs" / "financial_inclusion" / "metrics"

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    X = df[FEATURES]

    if X.isna().any().any():
        raise ValueError("Missing feature values found.")

    if not X.apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError("Feature scores must be within [0, 1].")

    # Descriptive statistics and unique-value counts.
    descriptive = X.describe().T
    descriptive["unique_values"] = X.nunique()
    descriptive["skewness"] = X.skew()
    descriptive["binary_feature"] = X.nunique().eq(2)

    descriptive_file = (
        REPORT_DIR / "fi_clustering_3d_feature_statistics.csv"
    )
    descriptive.to_csv(descriptive_file)

    # Correlation matrix.
    correlation = X.corr()
    correlation_file = (
        REPORT_DIR / "fi_clustering_3d_feature_correlations.csv"
    )
    correlation.to_csv(correlation_file)

    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "respondents": int(len(df)),
        "economies": int(df["economy"].nunique()),
        "features": FEATURES,
        "feature_statistics_file": str(descriptive_file.relative_to(ROOT)),
        "feature_correlations_file": str(correlation_file.relative_to(ROOT)),
        "missing_values": int(X.isna().sum().sum()),
        "source_data_modified": False,
    }

    summary_file = REPORT_DIR / "fi_clustering_3d_feature_audit.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Three-dimension feature audit completed.")
    print("\n=== FEATURE STATISTICS ===")
    print(descriptive.round(4).to_string())
    print("\n=== FEATURE CORRELATIONS ===")
    print(correlation.round(4).to_string())
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
