
"""Document provisional names for the five-dimensional k=4 clusters."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

OUTPUT_FILE = (
    ROOT / "outputs" / "financial_inclusion" / "reports" / "03_clustering"
    / "fi_clustering_profile_labels_5D_k4.json"
)

PROFILE_LABELS = {
    "0": {
        "name": "High Usage, Lower Phone Protection",
        "description": (
            "High account access and digital payment usage, "
            "with a low phone-protection score."
        ),
    },
    "1": {
        "name": "Broad Digital-Financial Inclusion",
        "description": (
            "High connectivity, account access, digital payment usage, "
            "and phone-protection scores."
        ),
    },
    "2": {
        "name": "Connected but Financially Underserved",
        "description": (
            "High connectivity and phone-protection scores, "
            "but low account access and digital payment usage."
        ),
    },
    "3": {
        "name": "Lower Digital-Financial Inclusion",
        "description": (
            "Relatively low connectivity, account access, "
            "digital payment usage, and phone-protection scores."
        ),
    },
}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "candidate": "5D_k4",
        "algorithm": "KMeans",
        "k": 4,
        "random_state": 42,
        "status": "provisional",
        "labels_are_final": False,
        "warning": (
            "Cluster IDs are specific to this fitted model. "
            "Names describe observed cluster averages, not causal effects "
            "or definitive classifications of individuals or economies."
        ),
        "cluster_profiles": PROFILE_LABELS,
    }

    OUTPUT_FILE.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )

    print("Provisional cluster profile labels saved.")
    print(f"File: {OUTPUT_FILE}")
    print(f"Profiles documented: {len(PROFILE_LABELS)}")
    print("Final model selection remains pending.")


if __name__ == "__main__":
    main()
