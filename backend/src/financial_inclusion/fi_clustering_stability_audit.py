
"""Assess KMeans clustering stability across random seeds."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[3]

INPUTS = {
    "3D_k5": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input_3d.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
        ],
        "k": 5,
    },
    "5D_k4": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        ],
        "k": 4,
    },
    "5D_k5": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        ],
        "k": 5,
    },
}

REPORT_DIR = ROOT / "outputs/financial_inclusion/reports"
METRICS_DIR = ROOT / "outputs/financial_inclusion/metrics"

SEEDS = [11, 21, 32, 42, 53, 64, 75, 86, 97, 108]
SILHOUETTE_SAMPLE_SIZE = 5000


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    stability_rows = []
    size_rows = []

    for candidate, config in INPUTS.items():
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        features = config["features"]
        X = df[features].to_numpy(dtype=float)
        k = config["k"]

        if not np.isfinite(X).all():
            raise ValueError(f"{candidate}: missing or non-finite values.")

        labels_by_seed = {}

        for seed in SEEDS:
            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=seed,
            )
            labels = model.fit_predict(X)
            labels_by_seed[seed] = labels

            counts = pd.Series(labels).value_counts()
            silhouette = silhouette_score(
                X,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X)),
                random_state=42,
            )

            size_rows.append({
                "candidate": candidate,
                "seed": seed,
                "respondents": len(df),
                "silhouette": silhouette,
                "smallest_cluster_size": int(counts.min()),
                "smallest_cluster_percent": (
                    float(counts.min()) / len(df) * 100
                ),
                "largest_cluster_percent": (
                    float(counts.max()) / len(df) * 100
                ),
            })

        pairwise_ari = [
            adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])
            for a, b in combinations(SEEDS, 2)
        ]

        stability_rows.append({
            "candidate": candidate,
            "k": k,
            "respondents": len(df),
            "economies": int(df["economy"].nunique()),
            "seed_runs": len(SEEDS),
            "pairwise_comparisons": len(pairwise_ari),
            "mean_pairwise_ari": float(np.mean(pairwise_ari)),
            "min_pairwise_ari": float(np.min(pairwise_ari)),
            "max_pairwise_ari": float(np.max(pairwise_ari)),
        })

    stability_df = pd.DataFrame(stability_rows)
    size_df = pd.DataFrame(size_rows)

    stability_file = REPORT_DIR / "fi_clustering_stability_summary.csv"
    size_file = METRICS_DIR / "fi_clustering_stability_runs.csv"

    stability_df.to_csv(stability_file, index=False)
    size_df.to_csv(size_file, index=False)

    print("Clustering stability audit completed.")
    print("\n=== PAIRWISE ARI STABILITY ===")
    print(stability_df.round(4).to_string(index=False))

    print("\n=== SILHOUETTE AND CLUSTER SIZE ACROSS SEEDS ===")
    print(
        size_df.groupby("candidate")[
            [
                "silhouette",
                "smallest_cluster_percent",
                "largest_cluster_percent",
            ]
        ].agg(["mean", "std", "min", "max"]).round(4).to_string()
    )

    print(f"\nStability summary: {stability_file}")
    print(f"Per-seed results: {size_file}")
    print("\nNo source data modified; no final model selected.")


if __name__ == "__main__":
    main()
