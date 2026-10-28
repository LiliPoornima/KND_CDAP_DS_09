"""Labels shared by the Component 02 dashboard."""

from .data_loader import DIMENSION_COLUMNS, DIMENSION_LABELS

CLUSTER_NAMES = {
    0: "Digitally Active, Safety Gap",
    1: "Connected and Digitally Included",
    2: "Connected, Financially Excluded",
    3: "Multi-Dimensional Inclusion Gap",
    4: "Access Without Digital Usage",
}
CLUSTER_IDS = tuple(CLUSTER_NAMES)
CLUSTER_LABELS = {key: f"{key} — {value}" for key, value in CLUSTER_NAMES.items()}
DIMENSIONS = tuple(DIMENSION_LABELS[column] for column in DIMENSION_COLUMNS)
SECTIONS = (
    "Executive overview", "Profile explorer", "Global cluster distribution",
    "Cluster comparison", "Explainable AI", "Demographic insights",
    "Income group analysis", "Region analysis", "Gap detector",
    "Policy recommendations",
)
