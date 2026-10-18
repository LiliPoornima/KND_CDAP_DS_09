
"""Save reproducible labels for the selected clustering candidates."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]

CANDIDATES = {
    "5D_k4": {
        "input": ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
            "score_digital_safety",
            "score_financial_resilience",
        ],
        "k": 4,
    },
    "3D_k5": {
        "input": ROOT / "data/financial_inclusion/processed/fi_clustering_input_3d.csv",
        "features": [
            "score_connectivity",
            "score_financial_access",
            "score_digital_financial_usage",
        ],
        "k": 5,
    },
}

OUTPUT_DIR = ROOT / "outputs/financial_inclusion/metrics"
RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, config in CANDIDATES.items():
        input_file = config["input"]

        if not input_file.exists():
            raise FileNotFoundError(f"Input not found: {input_file}")

        df = pd.read_csv(input_file, low_memory=False)
        features = config["features"]

        required = ["wpid_random", "economy", *features]
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

        if df["wpid_random"].isna().any() or df["wpid_random"].duplicated().any():
            raise ValueError(f"{name}: respondent IDs must be unique.")

        X = df[features].to_numpy(dtype=float)

        if not pd.notna(X).all():
            raise ValueError(f"{name}: feature values contain missing data.")

        model = KMeans(
            n_clusters=config["k"],
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)

        output = df[["wpid_random", "economy"]].copy()
        output["cluster_id"] = labels
        output["candidate"] = name

        for feature in features:
            output[feature] = df[feature].to_numpy()

        output_file = OUTPUT_DIR / f"fi_clustering_labels_{name}.csv"
        output.to_csv(output_file, index=False)

        print(f"\n{name}")
        print(f"Saved: {output_file}")
        print(f"Respondents: {len(output)}")
        print(f"Economies: {output['economy'].nunique()}")
        print("\nCluster distribution:")
        print(
            output["cluster_id"]
            .value_counts()
            .sort_index()
            .rename("respondents")
            .to_string()
        )

    print("\nCandidate labels saved. No source data modified.")
    print("Candidate selection remains provisional.")


if __name__ == "__main__":
    main()
