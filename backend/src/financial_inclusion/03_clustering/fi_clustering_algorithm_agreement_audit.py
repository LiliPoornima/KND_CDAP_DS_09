
"""Audit agreement between raw-feature KMeans and GMM clustering."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_clustering_input.csv"
)

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    X = df[FEATURES].to_numpy(dtype=float)

    if not pd.notna(X).all():
        raise ValueError("Input contains missing values.")

    print("RAW FEATURES: KMeans vs GMM")
    print(f"Respondents: {len(df)}")
    print()

    for k in range(2, 7):
        kmeans_labels = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        ).fit_predict(X)

        gmm_labels = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=2,
            random_state=42,
        ).fit_predict(X)

        ari = adjusted_rand_score(kmeans_labels, gmm_labels)

        print(
            f"k={k}: ARI={ari:.6f}, "
            f"identical_partition={ari == 1.0}"
        )


if __name__ == "__main__":
    main()
