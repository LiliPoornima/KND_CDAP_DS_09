
"""Audit economy representation across selected KMeans candidates."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]

CANDIDATES = {
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
}

REPORT_DIR = ROOT / "outputs/financial_inclusion/reports"
METRICS_DIR = ROOT / "outputs/financial_inclusion/metrics"
RANDOM_STATE = 42


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    economy_rows = []
    cluster_rows = []
    summary_rows = []

    for candidate, config in CANDIDATES.items():
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        features = config["features"]
        k = config["k"]

        required = ["wpid_random", "economy", *features]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{candidate}: missing columns {missing}")

        if df["wpid_random"].isna().any() or df["wpid_random"].duplicated().any():
            raise ValueError(f"{candidate}: respondent IDs are not unique.")

        X = df[features].to_numpy(dtype=float)
        if not pd.notna(X).all():
            raise ValueError(f"{candidate}: missing feature values.")

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)

        assigned = df[["wpid_random", "economy"]].copy()
        assigned["cluster"] = labels
        assigned["candidate"] = candidate

        # Economy-by-cluster counts and within-economy percentages.
        counts = (
            assigned.groupby(["economy", "cluster"])
            .size()
            .rename("respondents")
            .reset_index()
        )

        counts["economy_total"] = counts.groupby("economy")[
            "respondents"
        ].transform("sum")
        counts["within_economy_percent"] = (
            100 * counts["respondents"] / counts["economy_total"]
        ).round(2)
        counts["candidate"] = candidate
        economy_rows.extend(counts.to_dict("records"))

        # Economy representation per cluster.
        for cluster_id, group in assigned.groupby("cluster"):
            cluster_rows.append({
                "candidate": candidate,
                "cluster": int(cluster_id),
                "respondents": int(len(group)),
                "percent_of_sample": round(
                    len(group) / len(assigned) * 100, 2
                ),
                "economies_represented": int(group["economy"].nunique()),
                "largest_economy_share_percent": round(
                    100 * group["economy"].value_counts(normalize=True).max(),
                    2,
                ),
            })

        economy_totals = assigned.groupby("economy").size()
        economy_cluster_counts = assigned.groupby("economy")["cluster"].nunique()

        summary_rows.append({
            "candidate": candidate,
            "respondents": int(len(assigned)),
            "economies": int(assigned["economy"].nunique()),
            "economies_spread_across_multiple_clusters": int(
                (economy_cluster_counts > 1).sum()
            ),
            "economies_in_one_cluster_only": int(
                (economy_cluster_counts == 1).sum()
            ),
            "smallest_economy_sample": int(economy_totals.min()),
            "median_economy_sample": float(economy_totals.median()),
            "largest_economy_sample": int(economy_totals.max()),
        })

    economy_df = pd.DataFrame(economy_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    summary_df = pd.DataFrame(summary_rows)

    economy_file = REPORT_DIR / "fi_clustering_economy_cluster_distribution.csv"
    cluster_file = REPORT_DIR / "fi_clustering_cluster_economy_coverage.csv"
    summary_file = REPORT_DIR / "fi_clustering_economy_distribution_summary.csv"

    economy_df.to_csv(economy_file, index=False)
    cluster_df.to_csv(cluster_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print("Economy distribution audit completed.")
    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== CLUSTER ECONOMY COVERAGE ===")
    print(cluster_df.to_string(index=False))

    print(f"\nEconomy-cluster distribution: {economy_file}")
    print(f"Cluster coverage: {cluster_file}")
    print(f"Summary: {summary_file}")
    print("\nNo source data modified; no final model selected.")


if __name__ == "__main__":
    main()
