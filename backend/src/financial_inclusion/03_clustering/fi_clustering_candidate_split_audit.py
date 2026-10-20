
"""Compare respondent-level cluster splits across 5D KMeans candidates."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "outputs/financial_inclusion/metrics"
REPORT_DIR = ROOT / "outputs/financial_inclusion/reports/03_clustering"

CANDIDATES = {
    "5D_k4": METRICS_DIR / "fi_clustering_labels_5D_k4.csv",
    "5D_k5": METRICS_DIR / "fi_clustering_labels_5D_k5.csv",
    "5D_k6": METRICS_DIR / "fi_clustering_labels_5D_k6.csv",
}

ID_COL = "wpid_random"
CLUSTER_COL = "cluster_id"


def load_labels(name, path):
    df = pd.read_csv(path, low_memory=False)
    required = [ID_COL, "economy", CLUSTER_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    if df[ID_COL].isna().any() or df[ID_COL].duplicated().any():
        raise ValueError(f"{name}: respondent IDs must be unique and non-missing.")

    return df[[ID_COL, "economy", CLUSTER_COL]].rename(
        columns={CLUSTER_COL: name}
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    labels = {
        name: load_labels(name, path)
        for name, path in CANDIDATES.items()
    }

    common_ids = set.intersection(
        *(set(df[ID_COL]) for df in labels.values())
    )
    print(f"Common respondents across all candidates: {len(common_ids)}")

    merged = None

    for name, df in labels.items():
        subset = df[df[ID_COL].isin(common_ids)].copy()
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(
                subset,
                on=ID_COL,
                how="inner",
                validate="one_to_one",
                suffixes=("", "_duplicate"),
            )
            economy_duplicate = "economy_duplicate"
            if economy_duplicate in merged.columns:
                if not merged["economy"].equals(merged[economy_duplicate]):
                    raise ValueError(f"Economy mismatch in {name}.")
                merged = merged.drop(columns=[economy_duplicate])

    if merged.empty:
        raise ValueError("No common respondents across candidate labels.")

    comparisons = [
        ("5D_k4", "5D_k5"),
        ("5D_k4", "5D_k6"),
        ("5D_k5", "5D_k6"),
    ]

    print("\n=== PARTITION AGREEMENT ===")
    agreement_rows = []

    for left, right in comparisons:
        ari = adjusted_rand_score(merged[left], merged[right])
        print(f"{left} vs {right}: ARI = {ari:.6f}")
        agreement_rows.append({
            "candidate_a": left,
            "candidate_b": right,
            "respondents_compared": len(merged),
            "adjusted_rand_index": ari,
        })

    agreement_df = pd.DataFrame(agreement_rows)
    agreement_file = REPORT_DIR / "fi_clustering_candidate_split_agreement.csv"
    agreement_df.to_csv(agreement_file, index=False)

    for left, right in comparisons:
        cross = pd.crosstab(
            merged[left],
            merged[right],
            margins=True,
            margins_name="TOTAL",
        )
        print(f"\n=== {left} -> {right} CROSS-TAB ===")
        print(cross.to_string())

        output_file = REPORT_DIR / (
            f"fi_clustering_split_{left}_to_{right}.csv"
        )
        cross.to_csv(output_file)

    print(f"\nAgreement report: {agreement_file}")
    print("Source data unchanged; no final model selected.")


if __name__ == "__main__":
    main()
