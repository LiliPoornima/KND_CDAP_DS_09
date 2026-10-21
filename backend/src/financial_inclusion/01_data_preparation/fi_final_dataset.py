
"""Create a compact final analytical dataset for Component 02.

Existing source datasets and clustering outputs are read-only.
The final dataset is saved to a new CSV file.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = ROOT / "data" / "financial_inclusion" / "processed"
METRICS_DIR = ROOT / "outputs" / "financial_inclusion" / "metrics"

DIMENSION_FILE = DATA_DIR / "fi_dimension_dataset.csv"
CLUSTERING_INPUT_FILE = DATA_DIR / "fi_clustering_input.csv"
LABEL_FILE = METRICS_DIR / "fi_clustering_labels_5D_k5.csv"

OUTPUT_FILE = DATA_DIR / "fi_final_analytical_dataset.csv"

# Respondent context needed for demographic and economy-level analysis.
CONTEXT_COLUMNS = [
    "wpid_random",
    "economy",
    "economycode",
    "regionwb",
    "wgt",
    "female",
    "age",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
]

# Original variables used to construct the five dimensions.
SOURCE_INDICATORS = [
    "con1",
    "internet_use",
    "account",
    "account_fin",
    "anydigpayment",
    "con18",
    "con21",
    "saved",
    "fin24a",
    "fin24b",
]

# Constructed indicators and the separate scam-risk indicator.
CONSTRUCTED_INDICATORS = [
    "dim_connect_phone",
    "dim_connect_internet",
    "dim_access_account",
    "dim_access_financial_institution",
    "dim_usage_any_digital_payment",
    "dim_safety_phone_protection",
    "risk_scam_exposure",
    "dim_resilience_saved",
    "dim_resilience_emergency_ease",
    "dim_resilience_income_buffer",
]

SCORE_COLUMNS = [
    "score_connectivity",
    "score_connectivity_components_available",
    "score_financial_access",
    "score_financial_access_components_available",
    "score_digital_financial_usage",
    "score_digital_financial_usage_components_available",
    "score_digital_safety",
    "score_digital_safety_components_available",
    "score_financial_resilience",
    "score_financial_resilience_components_available",
]


def require_columns(df, columns, dataset_name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def main():
    for path in [DIMENSION_FILE, CLUSTERING_INPUT_FILE, LABEL_FILE]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Check that the existing 5D_k5 labels file has been created."
            )

    dimensions = pd.read_csv(DIMENSION_FILE, low_memory=False)
    clustering = pd.read_csv(CLUSTERING_INPUT_FILE, low_memory=False)
    labels = pd.read_csv(LABEL_FILE, low_memory=False)

    require_columns(
        dimensions,
        CONTEXT_COLUMNS + SOURCE_INDICATORS
        + CONSTRUCTED_INDICATORS + SCORE_COLUMNS,
        "Dimension dataset",
    )
    require_columns(
        clustering,
        ["wpid_random", "economy"]
        + [column for column in SCORE_COLUMNS if not column.endswith(
            "_components_available"
        )],
        "Clustering input",
    )
    require_columns(
        labels,
        ["wpid_random", "economy", "cluster_id", "candidate"],
        "5D_k5 labels",
    )

    # Check respondent identifiers before merging.
    for name, df in [
        ("dimension dataset", dimensions),
        ("clustering input", clustering),
        ("5D_k5 labels", labels),
    ]:
        if df["wpid_random"].isna().any():
            raise ValueError(f"{name}: missing respondent IDs.")
        if df["wpid_random"].duplicated().any():
            raise ValueError(f"{name}: duplicate respondent IDs.")

    labels = labels.loc[labels["candidate"] == "5D_k5"].copy()

    if labels.empty:
        raise ValueError("No rows found for candidate 5D_k5.")

    if set(clustering["wpid_random"]) != set(labels["wpid_random"]):
        raise ValueError(
            "Clustering input and 5D_k5 label respondent IDs do not match."
        )

    # Use the existing five-dimensional clustering population.
    # Do not fit KMeans again or alter the original data.
    cohort_ids = clustering[["wpid_random", "economy"]].copy()

    final_df = cohort_ids.merge(
        dimensions[
            CONTEXT_COLUMNS
            + SOURCE_INDICATORS
            + CONSTRUCTED_INDICATORS
            + SCORE_COLUMNS
        ],
        on=["wpid_random", "economy"],
        how="left",
        validate="one_to_one",
        indicator="_dimension_match",
    )

    if not (final_df["_dimension_match"] == "both").all():
        unmatched = int((final_df["_dimension_match"] != "both").sum())
        raise ValueError(
            f"{unmatched} clustering respondents were not found "
            "in the dimension dataset."
        )

    final_df = final_df.drop(columns="_dimension_match")

    final_df = final_df.merge(
        labels[["wpid_random", "cluster_id"]],
        on="wpid_random",
        how="left",
        validate="one_to_one",
        indicator="_label_match",
    )

    if not (final_df["_label_match"] == "both").all():
        unmatched = int((final_df["_label_match"] != "both").sum())
        raise ValueError(f"{unmatched} respondents have no cluster label.")

    final_df = final_df.drop(columns="_label_match")

    if final_df["cluster_id"].isna().any():
        raise ValueError("Missing cluster labels found.")

    # Ensure the dimension scores used here match the clustering input.
    score_features = [
        "score_connectivity",
        "score_financial_access",
        "score_digital_financial_usage",
        "score_digital_safety",
        "score_financial_resilience",
    ]

    check = final_df[
        ["wpid_random"] + score_features
    ].merge(
        clustering[["wpid_random"] + score_features],
        on="wpid_random",
        suffixes=("_final", "_clustering"),
        validate="one_to_one",
    )

    for feature in score_features:
        if not (
            check[f"{feature}_final"].reset_index(drop=True).equals(
                check[f"{feature}_clustering"].reset_index(drop=True)
            )
        ):
            # CSV float parsing can create tiny precision differences.
            if not (
                check[f"{feature}_final"].sub(
                    check[f"{feature}_clustering"]
                ).abs().fillna(0).le(1e-10).all()
            ):
                raise ValueError(
                    f"Final dataset score mismatch: {feature}"
                )

    # Final integrity checks.
    if final_df["wpid_random"].duplicated().any():
        raise ValueError("Duplicate respondent IDs in final dataset.")

    if len(final_df) != len(clustering):
        raise ValueError("Final dataset row count differs from clustering input.")

    if final_df[score_features].isna().any().any():
        raise ValueError("Missing values found in clustering score columns.")

    # Save to a NEW filename; no existing CSV is overwritten.
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFINAL ANALYTICAL DATASET CREATED")
    print(f"Output: {OUTPUT_FILE.relative_to(ROOT)}")
    print(f"Rows: {len(final_df):,}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"Economies: {final_df['economy'].nunique()}")
    print(f"Unique respondents: {final_df['wpid_random'].nunique():,}")
    print("\nCluster distribution:")
    print(final_df["cluster_id"].value_counts().sort_index().to_string())
    print("\nMissing values in final dataset:")
    print(final_df.isna().sum().loc[lambda s: s > 0].to_string())
    print("\nOriginal datasets were not modified.")


if __name__ == "__main__":
    main()
