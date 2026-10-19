
"""Audit provisional clustering labels and dimension profiles."""

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

LABELS_FILE = (
    ROOT
    / "outputs"
    / "financial_inclusion"
    / "metrics"
    / "fi_clustering_sample_labels.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "financial_inclusion" / "metrics"

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels not found: {LABELS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    labels = pd.read_csv(LABELS_FILE, low_memory=False)

    required_label_columns = [
        "wpid_random",
        "economy",
        "provisional_cluster",
    ]

    missing = [
        col for col in required_label_columns if col not in labels.columns
    ]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    # Ensure the labels belong to the correct respondents.
    if labels["wpid_random"].duplicated().any():
        raise ValueError("Duplicate respondent IDs found in labels.")

    if len(labels) != len(df):
        raise ValueError(
            "Label count differs from clustering input row count."
        )

    merged = df.merge(
        labels[["wpid_random", "provisional_cluster"]],
        on="wpid_random",
        how="left",
        validate="one_to_one",
    )

    if merged["provisional_cluster"].isna().any():
        raise ValueError("Some respondents have no cluster label.")

    if merged[FEATURES].isna().any().any():
        raise ValueError("Missing dimension scores found.")

    # Cluster sizes and percentages.
    cluster_sizes = (
        merged["provisional_cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="respondents")
    )

    cluster_sizes["percent"] = (
        100 * cluster_sizes["respondents"] / len(merged)
    ).round(2)

    cluster_sizes.to_csv(
        OUTPUT_DIR / "fi_cluster_sizes.csv",
        index=False,
    )

    # Mean dimension score for each cluster.
    profile_means = (
        merged.groupby("provisional_cluster")[FEATURES]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"provisional_cluster": "cluster"})
    )

    profile_means.to_csv(
        OUTPUT_DIR / "fi_cluster_profile_means.csv",
        index=False,
    )

    # Country representation in each cluster.
    country_counts = (
        merged.groupby(["provisional_cluster", "economy"])
        .size()
        .rename("respondents")
        .reset_index()
        .rename(columns={"provisional_cluster": "cluster"})
    )

    country_counts.to_csv(
        OUTPUT_DIR / "fi_cluster_country_counts.csv",
        index=False,
    )

    print("Cluster profile audit completed.")

    print("\n=== CLUSTER SIZES ===")
    print(cluster_sizes.to_string(index=False))

    print("\n=== MEAN DIMENSION SCORES BY CLUSTER ===")
    print(profile_means.to_string(index=False))

    print("\n=== COUNTRIES REPRESENTED ===")
    print(
        country_counts.groupby("cluster")["economy"]
        .nunique()
        .rename("economies")
        .to_string()
    )

    print("\nReports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
