"""Held-out and repeated-seed validation of the fixed 5D_k5 solution.

This experiment writes only to its own directory. The existing production
K-Means artifact and reference labels are loaded and never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
INPUT = ROOT / "data/financial_inclusion/processed/fi_final_analytical_dataset.csv"
CLUSTER_INPUT = ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv"
LABELS = ROOT / "outputs/financial_inclusion/metrics/fi_clustering_labels_5D_k5.csv"
MODEL_PATH = ROOT / "models/financial_inclusion/kmeans_5D_k5.joblib"
META_PATH = ROOT / "models/financial_inclusion/kmeans_5D_k5_metadata.json"
PROFILE_PATH = ROOT / "outputs/financial_inclusion/reports/03_clustering/fi_clustering_profile_labels_5D_k5.json"
FEATURES = [
    "score_connectivity", "score_financial_access",
    "score_digital_financial_usage", "score_digital_safety",
    "score_financial_resilience",
]
SEEDS = [42, 123, 2025, 2026, 2027]
PROFILE_NAMES = {
    0: "Digitally Active, Safety Gap",
    1: "Connected and Digitally Included",
    2: "Connected, Financially Excluded",
    3: "Multi-Dimensional Inclusion Gap",
    4: "Access Without Digital Usage",
}


def reference_predictions(reference_model, X):
    return reference_model.predict(X).astype(int)


def align_labels(fitted_model, reference_model):
    """Map newly fitted cluster IDs to reference IDs via centroid distance."""
    cost = cdist(fitted_model.cluster_centers_, reference_model.cluster_centers_)
    new_ids, ref_ids = linear_sum_assignment(cost)
    mapping = {int(new): int(ref) for new, ref in zip(new_ids, ref_ids)}
    return mapping, float(cost[new_ids, ref_ids].sum())


def run_seed(X, reference, seed):
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=seed, shuffle=True,
    )
    # This is a fresh model fitted only on training respondents.
    fitted = KMeans(
        n_clusters=5, init="k-means++", n_init=10, max_iter=300,
        tol=1e-4, algorithm="lloyd", random_state=seed,
    )
    fitted.fit(X[train_idx])
    raw_pred = fitted.predict(X[test_idx]).astype(int)
    reference_test = reference[test_idx]
    ari = float(adjusted_rand_score(reference_test, raw_pred))
    mapping, centroid_mapping_cost = align_labels(fitted, REFERENCE_MODEL)
    aligned_pred = np.array([mapping[int(label)] for label in raw_pred], dtype=int)
    test_silhouette = float(silhouette_score(X[test_idx], raw_pred, metric="euclidean"))

    distances = fitted.transform(X[test_idx])
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    margin = ordered[:, 1] - ordered[:, 0]
    dist_frame = pd.DataFrame({
        "seed": seed,
        "aligned_cluster_id": aligned_pred,
        "nearest_centroid_distance": nearest,
        "nearest_second_distance_margin": margin,
    })
    distributions = []
    profiles = []
    for cluster_id in range(5):
        predicted_mask = aligned_pred == cluster_id
        reference_mask = reference_test == cluster_id
        distributions.append({
            "seed": seed,
            "cluster_id": cluster_id,
            "profile_name": PROFILE_NAMES[cluster_id],
            "full_reference_count": int(np.sum(reference == cluster_id)),
            "full_reference_proportion": float(np.mean(reference == cluster_id)),
            "test_reference_count": int(reference_mask.sum()),
            "test_reference_proportion": float(reference_mask.mean()),
            "test_holdout_count": int(predicted_mask.sum()),
            "test_holdout_proportion": float(predicted_mask.mean()),
        })
        for feature_idx, feature in enumerate(FEATURES):
            pred_values = X[test_idx][predicted_mask, feature_idx]
            ref_values = X[test_idx][reference_mask, feature_idx]
            profiles.append({
                "seed": seed, "cluster_id": cluster_id,
                "profile_name": PROFILE_NAMES[cluster_id], "dimension": feature,
                "full_reference_mean": float(np.mean(X[reference == cluster_id, feature_idx])),
                "full_reference_median": float(np.median(X[reference == cluster_id, feature_idx])),
                "test_reference_mean": float(np.mean(ref_values)) if len(ref_values) else np.nan,
                "test_reference_median": float(np.median(ref_values)) if len(ref_values) else np.nan,
                "test_holdout_mean": float(np.mean(pred_values)) if len(pred_values) else np.nan,
                "test_holdout_median": float(np.median(pred_values)) if len(pred_values) else np.nan,
            })
    distance_summary = dist_frame.groupby(["seed", "aligned_cluster_id"], as_index=False).agg(
        respondents=("nearest_centroid_distance", "size"),
        nearest_distance_mean=("nearest_centroid_distance", "mean"),
        nearest_distance_median=("nearest_centroid_distance", "median"),
        nearest_second_margin_mean=("nearest_second_distance_margin", "mean"),
        nearest_second_margin_median=("nearest_second_distance_margin", "median"),
    ).rename(columns={"aligned_cluster_id": "cluster_id"})
    return {
        "seed": seed,
        "train_size": len(train_idx), "test_size": len(test_idx),
        "test_silhouette": test_silhouette, "reference_ari": ari,
        "centroid_alignment_cost": centroid_mapping_cost,
        "mapping": mapping,
        "distributions": distributions, "profiles": profiles,
        "distances": distance_summary,
    }


def save_figures(metrics, distribution, profiles):
    average_dist = distribution.groupby("cluster_id", as_index=False).agg(
        full_reference_proportion=("full_reference_proportion", "first"),
        test_reference_proportion=("test_reference_proportion", "mean"),
        test_holdout_proportion=("test_holdout_proportion", "mean"),
    )
    x = np.arange(5)
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, average_dist.full_reference_proportion, width, label="Full-data reference")
    ax.bar(x, average_dist.test_reference_proportion, width, label="Reference labels, test respondents")
    ax.bar(x + width, average_dist.test_holdout_proportion, width, label="Train-only K-Means, test respondents")
    ax.set_xticks(x, [f"{i}: {PROFILE_NAMES[i]}" for i in x], rotation=18, ha="right")
    ax.set_ylabel("Proportion of respondents")
    ax.set_title("Cluster distribution: reference and held-out assignments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cluster_distribution_comparison.png", dpi=180)
    plt.close(fig)

    average_profiles = profiles.groupby(["cluster_id", "dimension"], as_index=False).agg(
        full_reference_mean=("full_reference_mean", "first"),
        test_reference_mean=("test_reference_mean", "mean"),
        test_holdout_mean=("test_holdout_mean", "mean"),
    )
    dimensions = FEATURES
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for ax, cluster_id in zip(axes, range(5)):
        rows = average_profiles[average_profiles.cluster_id == cluster_id].set_index("dimension").loc[dimensions]
        positions = np.arange(len(dimensions))
        ax.plot(positions, rows.full_reference_mean, marker="o", label="Full reference")
        ax.plot(positions, rows.test_holdout_mean, marker="s", label="Held-out train-only")
        ax.set_xticks(positions, [f.replace("score_", "").replace("_", "\n") for f in dimensions], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{cluster_id}: {PROFILE_NAMES[cluster_id]}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Dimension score mean")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Dimension profiles: full reference vs held-out assignments")
    fig.tight_layout()
    fig.savefig(OUT / "dimension_profile_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([metrics.test_silhouette, metrics.reference_ari], tick_labels=["Test silhouette", "ARI vs reference"])
    ax.set_title("Repeated hold-out metrics by seed")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "repeated_holdout_metrics.png", dpi=180)
    plt.close(fig)


def main():
    global REFERENCE_MODEL
    for path in [INPUT, CLUSTER_INPUT, LABELS, MODEL_PATH, META_PATH, PROFILE_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = pd.read_csv(INPUT, usecols=["wpid_random", "economy", *FEATURES], low_memory=False)
    clustering_input = pd.read_csv(CLUSTER_INPUT, usecols=["wpid_random", *FEATURES], low_memory=False)
    saved_labels = pd.read_csv(LABELS, low_memory=False)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    profile_metadata = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if metadata.get("features") != FEATURES or metadata.get("n_clusters") != 5 or not metadata.get("validated"):
        raise ValueError("Reference model metadata does not match the validated 5D_k5 specification")
    if profile_metadata.get("labels_are_final") is not True:
        raise ValueError("Reference profile labels are not marked final")
    if not source.wpid_random.is_unique or not saved_labels.wpid_random.is_unique:
        raise ValueError("Respondent IDs must be unique")
    if set(source.wpid_random) != set(clustering_input.wpid_random):
        raise ValueError("Analytical dataset and clustering input cohorts differ")
    checked_scores = source[["wpid_random", *FEATURES]].merge(
        clustering_input, on="wpid_random", how="inner", validate="one_to_one",
        suffixes=("_analysis", "_clustering"),
    )
    if any(not np.array_equal(checked_scores[f"{feature}_analysis"].to_numpy(),
                              checked_scores[f"{feature}_clustering"].to_numpy())
           for feature in FEATURES):
        raise ValueError("Analytical dimension scores do not exactly match clustering inputs")
    joined = source.merge(
        saved_labels[["wpid_random", "cluster_id"]], on="wpid_random",
        how="inner", validate="one_to_one", suffixes=("", "_saved"),
    )
    if len(joined) != len(source) or len(joined) != int(metadata["training_rows"]):
        raise ValueError("Input/reference label cohort size mismatch")
    X = joined[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all() or np.any((X < 0) | (X > 1)):
        raise ValueError("Dimension inputs must be finite values in [0, 1]")
    REFERENCE_MODEL = joblib.load(MODEL_PATH)
    if REFERENCE_MODEL.get_params()["n_clusters"] != 5 or REFERENCE_MODEL.get_params()["n_init"] != 10:
        raise ValueError("Saved reference estimator settings differ from expected settings")
    reference = joined.cluster_id.to_numpy(dtype=int)
    predicted_reference = reference_predictions(REFERENCE_MODEL, X)
    if not np.array_equal(predicted_reference, reference):
        raise ValueError("Saved reference model predictions do not reproduce the saved labels")

    runs = [run_seed(X, reference, seed) for seed in SEEDS]
    metric_rows = [{k: v for k, v in run.items() if k not in {"mapping", "distributions", "profiles", "distances"}} | {
        "cluster_id_alignment": json.dumps(run["mapping"], sort_keys=True),
    } for run in runs]
    metrics = pd.DataFrame(metric_rows)
    distribution = pd.DataFrame([row for run in runs for row in run["distributions"]])
    profiles = pd.DataFrame([row for run in runs for row in run["profiles"]])
    distance = pd.concat([run["distances"] for run in runs], ignore_index=True)
    summary = metrics[["test_silhouette", "reference_ari"]].agg(["mean", "std", "min", "max"]).T.reset_index(names="metric")
    metrics.to_csv(OUT / "heldout_metrics_by_seed.csv", index=False)
    summary.to_csv(OUT / "heldout_metric_summary.csv", index=False)
    distribution.to_csv(OUT / "cluster_distribution_comparison.csv", index=False)
    profiles.to_csv(OUT / "dimension_profile_comparison_by_seed.csv", index=False)
    profiles.groupby(["cluster_id", "profile_name", "dimension"], as_index=False).agg(
        full_reference_mean=("full_reference_mean", "first"),
        full_reference_median=("full_reference_median", "first"),
        test_reference_mean=("test_reference_mean", "mean"),
        test_reference_median=("test_reference_median", "mean"),
        heldout_mean_across_seeds=("test_holdout_mean", "mean"),
        heldout_mean_sd_across_seeds=("test_holdout_mean", "std"),
        heldout_median_across_seeds=("test_holdout_median", "mean"),
    ).to_csv(OUT / "dimension_profile_comparison_summary.csv", index=False)
    distance.to_csv(OUT / "centroid_distance_summary.csv", index=False)
    save_figures(metrics, distribution, profiles)

    overall = {
        "dataset_rows": len(joined), "economies": int(joined.economy.nunique()),
        "features_in_order": FEATURES, "profile_names": PROFILE_NAMES,
        "input_dataset": str(INPUT.relative_to(ROOT)),
        "exact_score_match_to_clustering_input": True,
        "reference_model_parameters": REFERENCE_MODEL.get_params(),
        "reference_training_rows": int(metadata["training_rows"]),
        "seeds": SEEDS, "split": "unstratified random 80:20 split; unsupervised target is not used to stratify",
        "reference_predictions_exactly_match_saved_labels": True,
        "test_set_used_for_fit": False,
        "label_alignment": "Hungarian assignment minimizing Euclidean distance between fitted training-only centroids and existing reference centroids; ARI uses unaligned labels and is permutation invariant.",
        "ari_interpretation": "Agreement with the existing full-data reference clustering, not ground-truth classification accuracy.",
        "test_silhouette": "Calculated on all held-out rows and raw five-dimensional scores with training-only predicted labels.",
    }
    (OUT / "experiment_details.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved held-out K-Means validation artifacts in {OUT}")


if __name__ == "__main__":
    main()
