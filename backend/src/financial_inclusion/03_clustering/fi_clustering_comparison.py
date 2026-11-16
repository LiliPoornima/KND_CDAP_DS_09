
"""Compare clustering algorithms across three- and five-dimension inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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

RESULTS_FILE = METRICS_DIR / "fi_clustering_3d_5d_comparison.csv"
SUMMARY_FILE = REPORT_DIR / "fi_clustering_3d_5d_summary.json"

ID_COLUMN = "wpid_random"
COUNTRY_COLUMN = "economy"
K_VALUES = range(2, 7)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 5000
HIERARCHICAL_SAMPLE_SIZE = 5000


def evaluate(X, labels):
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        raise ValueError("Invalid cluster count for metric calculation.")

    return {
        "silhouette": float(
            silhouette_score(
                X,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X)),
                random_state=RANDOM_STATE,
            )
        ),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    summary_inputs = {}

    for dataset_name, config in INPUTS.items():
        input_file = config["path"]
        features = config["features"]

        if not input_file.exists():
            raise FileNotFoundError(f"Input not found: {input_file}")

        df = pd.read_csv(input_file, low_memory=False)
        required = [ID_COLUMN, COUNTRY_COLUMN, *features]
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(f"{dataset_name}: missing columns {missing}")

        if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
            raise ValueError(f"{dataset_name}: respondent IDs must be unique.")

        X_raw = df[features].to_numpy(dtype=float)

        if not np.isfinite(X_raw).all():
            raise ValueError(f"{dataset_name}: non-finite feature values.")

        if ((X_raw < 0) | (X_raw > 1)).any():
            raise ValueError(f"{dataset_name}: scores outside [0, 1].")

        X_standardised = StandardScaler().fit_transform(X_raw)

        summary_inputs[dataset_name] = {
            "input_file": str(input_file.relative_to(ROOT)),
            "respondents": int(len(df)),
            "economies": int(df[COUNTRY_COLUMN].nunique()),
            "features": features,
        }

        for scaling_name, X in [
            ("raw", X_raw),
            ("standardised", X_standardised),
        ]:
            for k in K_VALUES:
                # K-Means and GMM on the full eligible sample.
                models = [
                    (
                        "KMeans",
                        KMeans(
                            n_clusters=k,
                            n_init=10,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                    (
                        "GMM",
                        GaussianMixture(
                            n_components=k,
                            covariance_type="full",
                            reg_covar=1e-5,
                            n_init=2,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]

                for algorithm, model in models:
                    labels = model.fit_predict(X)

                    if len(np.unique(labels)) < 2:
                        continue

                    metrics = evaluate(X, labels)

                    results.append({
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "scaling": scaling_name,
                        "k": k,
                        "sample_size": len(df),
                        "cluster_count": len(np.unique(labels)),
                        **metrics,
                        "bic": float(model.bic(X)) if algorithm == "GMM" else None,
                        "aic": float(model.aic(X)) if algorithm == "GMM" else None,
                    })

            # Hierarchical clustering uses a fixed reproducible sample.
            rng = np.random.default_rng(RANDOM_STATE)
            sample_size = min(HIERARCHICAL_SAMPLE_SIZE, len(df))
            sample_indices = np.sort(
                rng.choice(len(df), size=sample_size, replace=False)
            )
            X_sample = X[sample_indices]

            for k in K_VALUES:
                model = AgglomerativeClustering(
                    n_clusters=k,
                    linkage="ward",
                )
                labels = model.fit_predict(X_sample)
                metrics = evaluate(X_sample, labels)

                results.append({
                    "dataset": dataset_name,
                    "algorithm": "Hierarchical_Ward",
                    "scaling": scaling_name,
                    "k": k,
                    "sample_size": sample_size,
                    "cluster_count": len(np.unique(labels)),
                    **metrics,
                    "bic": None,
                    "aic": None,
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "silhouette", ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_FILE, index=False)

    summary = {
        "inputs": summary_inputs,
        "k_values_tested": list(K_VALUES),
        "random_state": RANDOM_STATE,
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
        "hierarchical_sample_size": HIERARCHICAL_SAMPLE_SIZE,
        "results_file": str(RESULTS_FILE.relative_to(ROOT)),
        "final_model_selected": False,
        "stability_analysis_completed": True,
        "warning": (
            "Silhouette alone does not select the final model. "
            "Review cluster sizes, stability, profile interpretability, "
            "economy representation, and GMM BIC/AIC where applicable. "
            "The 3D and 5D candidates use different eligible populations, "
            "so compare them cautiously."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("3D vs 5D clustering comparison completed.")
    print("\n=== TOP 15 RESULTS BY SILHOUETTE ===")
    print(
        results_df.head(15)[
            [
                "dataset", "algorithm", "scaling", "k", "sample_size",
                "silhouette", "davies_bouldin", "calinski_harabasz",
                "bic", "aic",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== BEST RESULT PER DATASET ===")
    best = results_df.loc[
        results_df.groupby("dataset")["silhouette"].idxmax()
    ]
    print(
        best[
            ["dataset", "algorithm", "scaling", "k", "sample_size",
             "silhouette", "davies_bouldin", "calinski_harabasz"]
        ].round(4).to_string(index=False)
    )

    print(f"\nResults: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print("\nNo final model selected; no source data modified.")


if __name__ == "__main__":
    main()
