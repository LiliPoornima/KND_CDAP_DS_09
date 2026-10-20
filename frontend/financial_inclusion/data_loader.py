
"""Load the final analytical dataset for the Component 02 dashboard."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_final_analytical_dataset.csv"
)

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

DIMENSION_COLUMNS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]

REQUIRED_COLUMNS = (
    CONTEXT_COLUMNS
    + DIMENSION_COLUMNS
    + ["cluster_id"]
)


def load_final_dataset() -> pd.DataFrame:
    """Load and validate the final respondent-level analytical dataset."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Final analytical dataset not found: {DATASET_PATH}. "
            "Run the Component 02 final dataset preparation script first."
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Final analytical dataset is missing columns: {missing_columns}"
        )

    if df["wpid_random"].isna().any():
        raise ValueError("The dataset contains missing respondent IDs.")

    if df["wpid_random"].duplicated().any():
        raise ValueError("The dataset contains duplicate respondent IDs.")

    if df["cluster_id"].isna().any():
        raise ValueError("The dataset contains missing cluster labels.")

    observed_clusters = set(df["cluster_id"].unique())
    expected_clusters = {0, 1, 2, 3, 4}

    if observed_clusters != expected_clusters:
        raise ValueError(
            "Expected the final 5D_k5 cluster IDs {0, 1, 2, 3, 4}; "
            f"found {sorted(observed_clusters)}."
        )

    return df


def calculate_dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dimension statistics directly from respondent data."""
    rows = []

    for column in DIMENSION_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")

        rows.append(
            {
                "variable": column,
                "mean": values.mean(),
                "median": values.median(),
                "distinct_nonmissing_values": values.nunique(),
                "missing_percent": values.isna().mean() * 100,
            }
        )

    return pd.DataFrame(rows)


def calculate_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final cluster sizes and dimension means from the dataset."""
    profile = (
        df.groupby("cluster_id")
        .agg(
            respondents=("wpid_random", "nunique"),
            economies_represented=("economy", "nunique"),
            **{
                f"{column}_mean": (column, "mean")
                for column in DIMENSION_COLUMNS
            },
        )
        .reset_index()
    )

    total = profile["respondents"].sum()
    profile["percent_of_candidate"] = (
        profile["respondents"] / total * 100
    )

    return profile


def calculate_cluster_demographics(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calculate demographic composition by cluster, excluding missing values."""
    if column not in df.columns:
        raise ValueError(f"Unknown demographic column: {column}")

    result = (
        df.dropna(subset=[column])
        .groupby(["cluster_id", column])
        .size()
        .rename("respondents")
        .reset_index()
    )

    denominators = result.groupby("cluster_id")["respondents"].transform("sum")
    result["percent_within_cluster"] = (
        result["respondents"] / denominators * 100
    )

    return result
