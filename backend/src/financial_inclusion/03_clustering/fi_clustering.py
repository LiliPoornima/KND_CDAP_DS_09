
"""Compare clustering algorithms for financial inclusion profiles."""

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

INPUT_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_clustering_input.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "financial_inclusion"
REPORT_DIR = OUTPUT_DIR / "reports" / "03_clustering"
METRICS_DIR = OUTPUT_DIR / "metrics"

RESULTS_FILE = METRICS_DIR / "fi_clustering_model_comparison.csv"
LABELS_FILE = METRICS_DIR / "fi_clustering_sample_labels.csv"
SUMMARY_FILE = REPORT_DIR / "fi_clustering_summary.json"

ID_COLUMN = "wpid_random"
COUNTRY_COLUMN = "economy"

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]

K_VALUES = range(2, 7)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 5000
HIERARCHICAL_SAMPLE_SIZE = 5000


def calculate_metrics(X, labels, random_state=RANDOM_STATE):
    cluster_count = len(np.unique(labels))

    if cluster_count < 2 or cluster_count >= len(labels):
        raise ValueError("Invalid number of clusters for evaluation.")

    return {
        "silhouette_score": float(
            silhouette_score(
                X,
                labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X)),
                random_state=random_state,
            )
        ),
        "davies_bouldin_score": float(
            davies_bouldin_score(X, labels)
        ),
        "calinski_harabasz_score": float(
            calinski_harabasz_score(X, labels)
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [ID_COLUMN, COUNTRY_COLUMN, *FEATURES]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_raw = df[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(X_raw).all():
        raise ValueError("Features contain missing or non-finite values.")

    if ((X_raw < 0) | (X_raw > 1)).any():
        raise ValueError("Feature scores must be within [0, 1].")

    # Compare raw 0-1 scores against standardised scores.
    X_standardised = StandardScaler().fit_transform(X_raw)

    experiments = [
        ("raw", X_raw),
        ("standardised", X_standardised),
    ]

    results = []
    best_labels = None
    best_key = None
    best_silhouette = -np.inf

    # Full eligible sample: K-Means and GMM.
    for scaling_name, X in experiments:
        for k in K_VALUES:
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
                unique_clusters = np.unique(labels)

                if len(unique_clusters) < 2:
                    continue

                metrics = calculate_metrics(X, labels)

                row = {
                    "algorithm": algorithm,
                    "scaling": scaling_name,
                    "k": k,
                    "sample_size": len(df),
                    "cluster_count": len(unique_clusters),
                    **metrics,
                }

                if algorithm == "GMM":
                    row["bic"] = float(model.bic(X))
                    row["aic"] = float(model.aic(X))
                else:
                    row["bic"] = np.nan
                    row["aic"] = np.nan

                results.append(row)

                # Provisional candidate only; not final model selection.
                if metrics["silhouette_score"] > best_silhouette:
                    best_silhouette = metrics["silhouette_score"]
                    best_labels = labels.copy()
                    best_key = row.copy()

    # Hierarchical clustering is computationally expensive.
    # Use the same fixed sample for raw and standardised experiments.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(HIERARCHICAL_SAMPLE_SIZE, len(df))
    sample_indices = np.sort(
        rng.choice(len(df), size=sample_size, replace=False)
    )

    for scaling_name, X in experiments:
        X_sample = X[sample_indices]

        for k in K_VALUES:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
            )
            labels = model.fit_predict(X_sample)

            metrics = calculate_metrics(X_sample, labels)

            results.append(
                {
                    "algorithm": "Hierarchical_Ward",
                    "scaling": scaling_name,
                    "k": k,
                    "sample_size": sample_size,
                    "cluster_count": len(np.unique(labels)),
                    **metrics,
                    "bic": np.nan,
                    "aic": np.nan,
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "silhouette_score", ascending=False
    )
    results_df.to_csv(RESULTS_FILE, index=False)

    # Save candidate labels only if a candidate was produced.
    # Labels are provisional; final model selection needs further review.
    if best_labels is not None:
        label_output = df[[ID_COLUMN, COUNTRY_COLUMN]].copy()
        label_output["provisional_cluster"] = best_labels
        label_output.to_csv(LABELS_FILE, index=False)

    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "eligible_respondents": int(len(df)),
        "economies": int(df[COUNTRY_COLUMN].nunique()),
        "features": FEATURES,
        "k_values_tested": list(K_VALUES),
        "random_state": RANDOM_STATE,
        "full_sample_algorithms": ["KMeans", "GMM"],
        "hierarchical_sample_size": sample_size,
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
        "scaling_comparison": ["raw", "standardised"],
        "comparison_file": str(RESULTS_FILE.relative_to(ROOT)),
        "provisional_labels_file": str(LABELS_FILE.relative_to(ROOT)),
        "provisional_best_by_silhouette": best_key,
        "final_model_selected": False,
        "stability_analysis_completed": False,
        "input_data_modified": False,
        "note": (
            "Silhouette is only one selection criterion. Review cluster "
            "sizes, metric trade-offs, GMM BIC/AIC, stability, and "
            "interpretability before selecting a final solution."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Clustering model comparison completed.")
    print("\n=== TOP 10 RESULTS BY SILHOUETTE ===")
    print(
        results_df.head(10)[
            [
                "algorithm",
                "scaling",
                "k",
                "sample_size",
                "silhouette_score",
                "davies_bouldin_score",
                "calinski_harabasz_score",
                "bic",
                "aic",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nResults: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Provisional labels: {LABELS_FILE}")


if __name__ == "__main__":
    main()
