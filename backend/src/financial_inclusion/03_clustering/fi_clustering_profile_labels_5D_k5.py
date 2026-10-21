"""Document final names for the selected five-dimensional k=5 clusters."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

OUTPUT_FILE = (
    ROOT / "outputs" / "financial_inclusion" / "reports" / "03_clustering"
    / "fi_clustering_profile_labels_5D_k5.json"
)

PROFILE_LABELS = {
    "0": {
        "name": "Digitally Active, Safety Gap",
        "description": (
            "High account access and digital financial usage, "
            "but a low phone-protection score."
        ),
    },
    "1": {
        "name": "Connected and Digitally Included",
        "description": (
            "High connectivity, account access, digital financial usage, "
            "and phone-protection scores."
        ),
    },
    "2": {
        "name": "Connected, Financially Excluded",
        "description": (
            "High connectivity and phone-protection scores, "
            "but no observed account access or digital financial usage."
        ),
    },
    "3": {
        "name": "Multi-Dimensional Inclusion Gap",
        "description": (
            "Relatively low connectivity, with no observed account access, "
            "digital financial usage, or phone protection."
        ),
    },
    "4": {
        "name": "Access Without Digital Usage",
        "description": (
            "Observed account access, but no observed digital financial "
            "payment usage; the phone-protection score is intermediate."
        ),
    },
}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "candidate": "5D_k5",
        "algorithm": "KMeans",
        "k": 5,
        "random_state": 42,
        "status": "final",
        "labels_are_final": True,
        "warning": (
            "Cluster IDs are specific to this fitted model. Names describe "
            "observed cluster averages, not causal effects or definitive "
            "classifications of individuals or economies. The digital-safety "
            "dimension reflects the available phone-protection measure only."
        ),
        "cluster_profiles": PROFILE_LABELS,
    }

    OUTPUT_FILE.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )

    print("Final cluster profile labels saved.")


print(f"File: {OUTPUT_FILE}")
print(f"Profiles documented: {len(PROFILE_LABELS)}")
print("Candidate: 5D_k5")
print("Final profile-label metadata recorded.")


if __name__ == "__main__":
    main()
