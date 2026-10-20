
"""Compare KMeans cluster sizes and profiles for k=2..6."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[4]

INPUTS = {
    "3D": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input_3d.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
        ],
    },
    "5D": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        ],
    },
}

OUTPUT_DIR = ROOT / "outputs/financial_inclusion"
REPORT_DIR = OUTPUT_DIR / "reports" / "03_clustering"
METRICS_DIR = OUTPUT_DIR / "metrics"

RANDOM_STATE = 42
K_VALUES = range(2, 7)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    profile_rows = []
    size_rows = []

    for dataset_name, config in INPUTS.items():
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        features = config["features"]
        X = df[features].to_numpy(dtype=float)

        if not pd.notna(X).all():
            raise ValueError(f"{dataset_name}: missing feature values.")

        for k in K_VALUES:
            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(X)

            counts = pd.Series(labels).value_counts().sort_index()
            silhouette = silhouette_score(
                X,
                labels,
                sample_size=min(5000, len(X)),
                random_state=RANDOM_STATE,
            )

            summary_rows.append({
                "dataset": dataset_name,
                "k": k,
                "respondents": len(df),
                "economies": df["economy"].nunique(),
                "silhouette": silhouette,
                "smallest_cluster_size": int(counts.min()),
                "smallest_cluster_percent": float(
                    counts.min() / len(df) * 100
                ),
                "largest_cluster_percent": float(
                    counts.max() / len(df) * 100
                ),
            })

            profile = df[features].copy()
            profile["cluster"] = labels
            means = profile.groupby("cluster")[features].mean()

            for cluster_id, row in means.iterrows():
                profile_rows.append({
                    "dataset": dataset_name,
                    "k": k,
                    "cluster": int(cluster_id),
                    "respondents": int(counts.loc[cluster_id]),
                    "percent_of_dataset": round(
                        counts.loc[cluster_id] / len(df) * 100, 2
                    ),
                    **row.to_dict(),
                })

            for cluster_id, count in counts.items():
                size_rows.append({
                    "dataset": dataset_name,
                    "k": k,
                    "cluster": int(cluster_id),
                    "respondents": int(count),
                    "percent_of_dataset": round(
                        count / len(df) * 100, 2
                    ),
                })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "k"]
    )
    profiles_df = pd.DataFrame(profile_rows).sort_values(
        ["dataset", "k", "cluster"]
    )
    sizes_df = pd.DataFrame(size_rows).sort_values(
        ["dataset", "k", "cluster"]
    )

    summary_file = REPORT_DIR / "fi_clustering_k_sensitivity_summary.csv"
    profiles_file = REPORT_DIR / "fi_clustering_k_sensitivity_profiles.csv"
    sizes_file = METRICS_DIR / "fi_clustering_k_sensitivity_sizes.csv"

    summary_df.to_csv(summary_file, index=False)
    profiles_df.to_csv(profiles_file, index=False)
    sizes_df.to_csv(sizes_file, index=False)

    print("KMeans k-sensitivity audit completed.")
    print("\n=== SUMMARY: k=2..6 ===")
    print(summary_df.round(4).to_string(index=False))

    print("\n=== CLUSTER PROFILES ===")
    print(profiles_df.round(4).to_string(index=False))

    print(f"\nSummary: {summary_file}")
    print(f"Profiles: {profiles_file}")
    print(f"Cluster sizes: {sizes_file}")
    print("\nNo source data modified; no final model selected.")


if __name__ == "__main__":
    main()
