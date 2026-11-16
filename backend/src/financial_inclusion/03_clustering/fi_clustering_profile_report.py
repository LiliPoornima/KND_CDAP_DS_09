
"""Generate interpretable financial inclusion cluster profiles."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

INPUTS = {
    "5D_k4": {
        "path": ROOT / "outputs/financial_inclusion/metrics/fi_clustering_labels_5D_k4.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        ],
    },
        "5D_k5": {
        "path": ROOT / "outputs/financial_inclusion/metrics/fi_clustering_labels_5D_k5.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        ],
    },
    "3D_k5": {
        "path": ROOT / "outputs/financial_inclusion/metrics/fi_clustering_labels_3D_k5.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
        ],
    },
}

REPORT_DIR = ROOT / "outputs/financial_inclusion/reports/03_clustering"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profile_rows = []
    economy_rows = []

    for candidate, config in INPUTS.items():
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(f"Labels file not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        features = config["features"]

        required = [
            "wpid_random",
            "economy",
            "cluster_id",
            *features,
        ]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"{candidate}: missing columns {missing}")

        if df["wpid_random"].isna().any() or df["wpid_random"].duplicated().any():
            raise ValueError(f"{candidate}: respondent IDs must be unique.")

        if df[features].isna().any().any():
            raise ValueError(f"{candidate}: feature scores contain missing values.")

        total = len(df)

        for cluster_id, group in df.groupby("cluster_id", sort=True):
            row = {
                "candidate": candidate,
                "cluster_id": int(cluster_id),
                "respondents": int(len(group)),
                "percent_of_candidate": round(
                    100 * len(group) / total, 2
                ),
                "economies_represented": int(group["economy"].nunique()),
            }

            for feature in features:
                row[f"{feature}_mean"] = round(
                    float(group[feature].mean()), 4
                )
                row[f"{feature}_median"] = round(
                    float(group[feature].median()), 4
                )

            profile_rows.append(row)

            economy_counts = group["economy"].value_counts()

            economy_rows.append({
                "candidate": candidate,
                "cluster_id": int(cluster_id),
                "largest_economy": str(economy_counts.index[0]),
                "largest_economy_respondents": int(economy_counts.iloc[0]),
                "largest_economy_percent_of_cluster": round(
                    100 * economy_counts.iloc[0] / len(group), 2
                ),
            })

    profiles = pd.DataFrame(profile_rows)
    economy_summary = pd.DataFrame(economy_rows)

    profile_file = REPORT_DIR / "fi_clustering_interpretable_profiles.csv"
    economy_file = REPORT_DIR / "fi_clustering_profile_economy_check.csv"

    profiles.to_csv(profile_file, index=False)
    economy_summary.to_csv(economy_file, index=False)

    print("Cluster profile reports created.")

    print("\n=== CLUSTER PROFILES ===")
    print(profiles.round(4).to_string(index=False))

    print("\n=== LARGEST ECONOMY WITHIN EACH CLUSTER ===")
    print(economy_summary.to_string(index=False))

    print(f"\nProfiles: {profile_file}")
    print(f"Economy check: {economy_file}")
    print("\nSource data unchanged; profile names not assigned automatically.")


if __name__ == "__main__":
    main()
