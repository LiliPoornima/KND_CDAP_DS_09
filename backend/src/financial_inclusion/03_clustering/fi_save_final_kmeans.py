
"""Validate and save the reproducible final 5D k=5 KMeans model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    ROOT / "data/financial_inclusion/processed/fi_clustering_input.csv"
)
LABELS_FILE = (
    ROOT / "outputs/financial_inclusion/metrics/"
    "fi_clustering_labels_5D_k5.csv"
)
PROFILE_FILE = (
    ROOT / "outputs/financial_inclusion/reports/03_clustering/"
    "fi_clustering_profile_labels_5D_k5.json"
)

MODEL_DIR = ROOT / "models/financial_inclusion"
MODEL_FILE = MODEL_DIR / "kmeans_5D_k5.joblib"
METADATA_FILE = MODEL_DIR / "kmeans_5D_k5_metadata.json"

FEATURES = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]

RANDOM_STATE = 42
N_CLUSTERS = 5


def main():
    for path in [INPUT_FILE, LABELS_FILE, PROFILE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    data = pd.read_csv(INPUT_FILE)
    saved = pd.read_csv(LABELS_FILE)
    profiles = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))

    if profiles.get("candidate") != "5D_k5":
        raise ValueError("Profile metadata is not for 5D_k5.")

    if profiles.get("labels_are_final") is not True:
        raise ValueError("Profile metadata does not confirm final labels.")

    required = ["wpid_random", *FEATURES]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing clustering input columns: {missing}")

    if not {"wpid_random", "cluster_id", "candidate"}.issubset(saved.columns):
        raise ValueError("Saved label file is missing required columns.")

    if not data["wpid_random"].is_unique:
        raise ValueError("Clustering input IDs are not unique.")

    if not saved["wpid_random"].is_unique:
        raise ValueError("Saved label IDs are not unique.")

    if not data["wpid_random"].equals(saved["wpid_random"]):
        raise ValueError(
            "Input and saved labels have different IDs or row order."
        )

    if not saved["candidate"].eq("5D_k5").all():
        raise ValueError("Saved labels contain another candidate.")

    X = data[FEATURES].to_numpy(dtype=float)
    expected = saved["cluster_id"].to_numpy(dtype=int)

    if not np.isfinite(X).all():
        raise ValueError("Clustering features contain invalid values.")

    if ((X < 0) | (X > 1)).any():
        raise ValueError("Dimension scores must be between 0 and 1.")

    model = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=10,
        random_state=RANDOM_STATE,
    )
    predictions = model.fit_predict(X)

    exact_matches = int(np.sum(predictions == expected))
    ari = float(adjusted_rand_score(expected, predictions))

    print("Rows:", len(data))
    print("Exact cluster ID matches:", exact_matches)
    print("Label mismatches:", len(data) - exact_matches)
    print("Adjusted Rand Index:", ari)

    # Never save the artifact unless exact label IDs are reproduced.
    if not np.array_equal(predictions, expected):
        raise RuntimeError(
            "Validation failed. Model artifact was not saved."
        )

    # Verify that each model cluster has a documented profile label.
    documented_ids = {int(k) for k in profiles["cluster_profiles"]}
    predicted_ids = set(np.unique(predictions).tolist())

    if documented_ids != predicted_ids:
        raise ValueError(
            f"Profile IDs {documented_ids} do not match "
            f"model IDs {predicted_ids}."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "candidate": "5D_k5",
        "algorithm": "KMeans",
        "n_clusters": N_CLUSTERS,
        "n_init": 10,
        "random_state": RANDOM_STATE,
        "scaling": "raw dimension scores in [0, 1]",
        "features": FEATURES,
        "training_rows": int(len(data)),
        "exact_saved_label_matches": exact_matches,
        "label_mismatches": int(len(data) - exact_matches),
        "adjusted_rand_index": ari,
        "profile_metadata_file": str(PROFILE_FILE.relative_to(ROOT)),
        "validated": True,
        "intended_use": (
            "Assign inclusion profiles from the five validated "
            "dimension scores; not a creditworthiness or default model."
        ),
    }

    # Save only after every validation check has passed.
    joblib.dump(model, MODEL_FILE)
    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL SAVED:", MODEL_FILE)
    print("METADATA SAVED:", METADATA_FILE)
    print("VALIDATION: PASS")


if __name__ == "__main__":
    main()
