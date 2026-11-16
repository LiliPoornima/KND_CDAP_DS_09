
"""Audit cluster sizes and profiles for raw KMeans/GMM k=6 candidates."""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

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

REPORT_DIR = ROOT / "outputs/financial_inclusion/reports/03_clustering"
METRICS_DIR = ROOT / "outputs/financial_inclusion/metrics"

RANDOM_STATE = 42
K = 6


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    size_tables = []
    profile_tables = []
    agreement = {}

    for dataset_name, config in INPUTS.items():
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        features = config["features"]
        X = df[features].to_numpy(dtype=float)

        if not pd.notna(X).all():
            raise ValueError(f"{dataset_name}: missing feature values.")

        kmeans = KMeans(
            n_clusters=K,
            n_init=10,
            random_state=RANDOM_STATE,
        )
        kmeans_labels = kmeans.fit_predict(X)

        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=2,
            random_state=RANDOM_STATE,
        )
        gmm_labels = gmm.fit_predict(X)

        agreement[dataset_name] = {
            "adjusted_rand_index_kmeans_vs_gmm": float(
                adjusted_rand_score(kmeans_labels, gmm_labels)
            ),
            "kmeans_clusters": int(pd.Series(kmeans_labels).nunique()),
            "gmm_clusters": int(pd.Series(gmm_labels).nunique()),
            "respondents": int(len(df)),
            "economies": int(df["economy"].nunique()),
        }

        for algorithm, labels in [
            ("KMeans", kmeans_labels),
            ("GMM", gmm_labels),
        ]:
            temp = df[["economy"]].copy()
            temp["cluster"] = labels

            sizes = (
                temp.groupby("cluster")
                .size()
                .rename("respondents")
                .reset_index()
            )
            sizes["dataset"] = dataset_name
            sizes["algorithm"] = algorithm
            sizes["percent_of_dataset"] = (
                100 * sizes["respondents"] / len(df)
            ).round(2)
            sizes["economies_represented"] = sizes["cluster"].map(
                temp.groupby("cluster")["economy"].nunique()
            )
            size_tables.append(sizes)

            profile = df[features].copy()
            profile["cluster"] = labels

            means = profile.groupby("cluster")[features].mean().reset_index()
            means["dataset"] = dataset_name
            means["algorithm"] = algorithm
            profile_tables.append(means)

    sizes_df = pd.concat(size_tables, ignore_index=True)
    profiles_df = pd.concat(profile_tables, ignore_index=True)

    size_file = METRICS_DIR / "fi_clustering_candidate_cluster_sizes.csv"
    profile_file = REPORT_DIR / "fi_clustering_candidate_profiles.csv"
    agreement_file = REPORT_DIR / "fi_clustering_algorithm_agreement.json"

    sizes_df.to_csv(size_file, index=False)
    profiles_df.to_csv(profile_file, index=False)
    agreement_file.write_text(
        json.dumps(agreement, indent=2),
        encoding="utf-8",
    )

    print("Candidate cluster profile audit completed.")
    print("\n=== CLUSTER SIZES ===")
    print(
        sizes_df[
            [
                "dataset", "algorithm", "cluster",
                "respondents", "percent_of_dataset",
                "economies_represented",
            ]
        ].to_string(index=False)
    )

    print("\n=== CLUSTER FEATURE MEANS ===")
    print(profiles_df.round(4).to_string(index=False))

    print("\n=== KMEANS VS GMM AGREEMENT ===")
    print(json.dumps(agreement, indent=2))

    print(f"\nCluster sizes: {size_file}")
    print(f"Cluster profiles: {profile_file}")
    print(f"Algorithm agreement: {agreement_file}")
    print("\nNo source data modified; no final model selected.")


if __name__ == "__main__":
    main()

