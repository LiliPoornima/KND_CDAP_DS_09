
"""Load existing Component 02 research outputs for Streamlit."""

import json
from pathlib import Path

import pandas as pd


# frontend/financial_inclusion/data_loader.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports"
METRICS_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "metrics"

DIMENSION_DIR = REPORTS_DIR / "02_dimension_creation"
CLUSTERING_DIR = REPORTS_DIR / "03_clustering"
SUPERVISED_DIR = REPORTS_DIR / "04_supervised_learning"


def load_csv(path: Path) -> pd.DataFrame:
    """Load an existing CSV research output."""
    if not path.is_file():
        raise FileNotFoundError(f"Research CSV not found: {path}")

    return pd.read_csv(path, low_memory=False)


def load_json(path: Path) -> dict:
    """Load an existing JSON research output."""
    if not path.is_file():
        raise FileNotFoundError(f"Research JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dimension_summary() -> dict:
    """Load the dimension creation summary."""
    return load_json(
        DIMENSION_DIR / "fi_dimension_creation_report.json"
    )


def load_dimension_distributions() -> pd.DataFrame:
    """Load the dimension score distribution report."""
    return load_csv(
        DIMENSION_DIR / "fi_dimension_score_distributions.csv"
    )


def load_dimension_coverage() -> dict:
    """Load dimension availability and missingness information."""
    return load_json(
        DIMENSION_DIR / "fi_dimension_availability_summary.json"
    )


def load_clustering_summary() -> dict:
    """Load the clustering summary."""
    return load_json(
        CLUSTERING_DIR / "fi_clustering_summary.json"
    )


def load_cluster_profiles() -> pd.DataFrame:
    """Load interpretable cluster profiles."""
    return load_csv(
        CLUSTERING_DIR / "fi_clustering_interpretable_profiles.csv"
    )


def load_cluster_labels() -> dict:
    """Load proposed cluster profile labels and descriptions."""
    return load_json(
        CLUSTERING_DIR / "fi_clustering_profile_labels_5D_k4.json"
    )


def load_clustering_coverage() -> pd.DataFrame:
    """Load cluster/economy coverage checks."""
    return load_csv(
        CLUSTERING_DIR / "fi_clustering_profile_economy_check.csv"
    )


def load_supervised_target_audit() -> pd.DataFrame:
    """Load the supervised target audit."""
    return load_csv(
        SUPERVISED_DIR / "fi_supervised_target_audit.csv"
    )


def load_model_metrics() -> pd.DataFrame:
    """Load saved model evaluation metrics."""
    return load_csv(
        METRICS_DIR / "fi_supervised_baseline_metrics.csv"
    )


def load_classification_reports() -> str:
    """Load saved classification reports."""
    path = METRICS_DIR / "fi_supervised_classification_reports.txt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Classification reports not found: {path}"
        )

    return path.read_text(encoding="utf-8")


def load_confusion_matrices() -> str:
    """Load saved confusion matrices."""
    path = METRICS_DIR / "fi_supervised_confusion_matrices.txt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Confusion matrices not found: {path}"
        )

    return path.read_text(encoding="utf-8")
