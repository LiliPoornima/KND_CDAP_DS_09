
"""Audit feature distributions before clustering."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_clustering_input.csv"
)

REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports" / "03_clustering"

SUMMARY_FILE = REPORT_DIR / "fi_clustering_feature_audit_summary.json"
DISTRIBUTION_FILE = REPORT_DIR / "fi_clustering_feature_distributions.csv"

SCORE_COLUMNS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    missing = [col for col in SCORE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    features = df[SCORE_COLUMNS].copy()

    if features.isna().any().any():
        raise ValueError("Missing values detected in clustering features.")

    if not features.apply(
        lambda column: pd.api.types.is_numeric_dtype(column)
    ).all():
        raise TypeError("All clustering features must be numeric.")

    if not features.apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise ValueError("Scores must be within [0, 1].")

    # Descriptive statistics for each score.
    distribution = features.describe().T
    distribution["unique_values"] = features.nunique()
    distribution["variance"] = features.var()
    distribution["skewness"] = features.skew()

    distribution.index.name = "feature"
    distribution = distribution.reset_index()

    distribution.to_csv(DISTRIBUTION_FILE, index=False)

    # Exact score frequencies are useful for detecting highly discrete scores.
    frequency_records = []

    for column in SCORE_COLUMNS:
        counts = features[column].value_counts().sort_index()

        for value, count in counts.items():
            frequency_records.append(
                {
                    "feature": column,
                    "score_value": float(value),
                    "respondents": int(count),
                    "percent": round(float(count / len(features) * 100), 3),
                }
            )

    frequency_file = REPORT_DIR / "fi_clustering_feature_frequencies.csv"
    pd.DataFrame(frequency_records).to_csv(
        frequency_file, index=False
    )

    # Correlation is descriptive, not proof of redundancy.
    correlation = features.corr().round(4)

    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "respondents": int(len(features)),
        "feature_count": len(SCORE_COLUMNS),
        "features": SCORE_COLUMNS,
        "missing_feature_values": int(features.isna().sum().sum()),
        "all_scores_within_zero_one": True,
        "feature_statistics": {
            column: {
                "mean": round(float(features[column].mean()), 4),
                "std": round(float(features[column].std()), 4),
                "min": round(float(features[column].min()), 4),
                "median": round(float(features[column].median()), 4),
                "max": round(float(features[column].max()), 4),
                "unique_values": int(features[column].nunique()),
                "variance": round(float(features[column].var()), 6),
                "skewness": round(float(features[column].skew()), 4),
            }
            for column in SCORE_COLUMNS
        },
        "correlation_matrix": correlation.to_dict(),
        "distribution_report": str(
            DISTRIBUTION_FILE.relative_to(ROOT)
        ),
        "frequency_report": str(frequency_file.relative_to(ROOT)),
        "standardisation_applied": False,
        "input_data_modified": False,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Clustering feature audit completed.")
    print("\n=== FEATURE DISTRIBUTIONS ===")
    print(
        distribution[
            [
                "feature",
                "mean",
                "std",
                "min",
                "50%",
                "max",
                "unique_values",
                "variance",
                "skewness",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== FEATURE CORRELATIONS ===")
    print(correlation.to_string())

    print("\nNo scaling applied and input data unchanged.")
    print(f"\nSummary: {SUMMARY_FILE}")
    print(f"Distribution report: {DISTRIBUTION_FILE}")
    print(f"Frequency report: {frequency_file}")


if __name__ == "__main__":
    main()
