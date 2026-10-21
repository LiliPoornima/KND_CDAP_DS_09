
"""Audit cluster sizes and profiles for selected clustering candidates."""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[4]

FEATURES_3D = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
]

FEATURES_5D = FEATURES_3D + [
    "score_digital_safety",
    "score_financial_resilience",
]

CANDIDATES = {
    "3D_k5": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input_3d.csv",
        "features": FEATURES_3D,
        "k": 5,
    },
    "5D_k4": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": FEATURES_5D,
        "k": 4,
    },
    "5D_k5": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": FEATURES_5D,
        "k": 5,
    },
    "5D_k6": {
        "path": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": FEATURES_5D,
        "k": 6,
    },
}

REPORT_DIR = ROOT / "outputs/financial_inclusion/reports/03_clustering"
METRICS_DIR = ROOT / "outputs/financial_inclusion/metrics"

RANDOM_STATE = 42


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    size_tables = []
    profile_tables = []
    agreement = {}

    for candidate_name, config in CANDIDATES.items():
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        features = config["features"]
        k = config["k"]

        missing_columns = [
            col for col in ["economy", *features]
            if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{candidate_name}: missing columns {missing_columns}"
            )

        X = df[features].to_numpy(dtype=float)
        if not pd.notna(X).all():
            raise ValueError(f"{candidate_name}: missing feature values.")

        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE,
        )
        kmeans_labels = kmeans.fit_predict(X)

        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=2,
            random_state=RANDOM_STATE,
        )
        gmm_labels = gmm.fit_predict(X)

        agreement[candidate_name] = {
            "k": k,
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
                .agg(
                    respondents=("economy", "size"),
                    economies_represented=("economy", "nunique"),
                )
                .reset_index()
            )
            sizes["candidate"] = candidate_name
            sizes["algorithm"] = algorithm
            sizes["percent_of_dataset"] = (
                100 * sizes["respondents"] / len(df)
            ).round(2)
            size_tables.append(sizes)

            profile = df[features].copy()
            profile["cluster"] = labels

            means = (
                profile.groupby("cluster")[features]
                .mean()
                .reset_index()
            )
            means["candidate"] = candidate_name
            means["algorithm"] = algorithm
            means["respondents"] = (
                profile.groupby("cluster").size().reindex(means["cluster"]).to_numpy()
            )
            means["percent_of_dataset"] = (
                100 * means["respondents"] / len(df)
            ).round(2)
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
                "candidate", "algorithm", "cluster",
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
