
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = (
    ROOT / "data" / "financial_inclusion"
    / "processed" / "fi_dimension_dataset.csv"
)

LABEL_PATH = (
    ROOT / "outputs" / "financial_inclusion"
    / "metrics" / "fi_clustering_labels_5D_k5.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "financial_inclusion"
    / "reports" / "04_supervised_learning" / "fi_supervised_target_audit.csv"
)

predictors = [
    "female",
    "age",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "regionwb",
]

df = pd.read_csv(DATA_PATH, low_memory=False)
labels = pd.read_csv(LABEL_PATH, low_memory=False)

print("=== INPUT CHECK ===")
print("Dimension dataset:", df.shape)
print("Cluster label dataset:", labels.shape)
print("Label columns:", labels.columns.tolist())

# Find a common respondent identifier.
id_candidates = ["wpid_random"]
common_id = next(
    (
        col for col in id_candidates
        if col in df.columns and col in labels.columns
    ),
    None,
)

if common_id is None:
    raise ValueError(
        "No common respondent ID found. "
        "Check the column names before proceeding."
    )

print("Join key:", common_id)
print("Unique IDs in dimensions:", df[common_id].nunique())
print("Unique IDs in labels:", labels[common_id].nunique())

# Detect the cluster label column; do not assume its name.
possible_targets = [
    col for col in labels.columns
    if "cluster" in col.lower() or "label" in col.lower()
]
print("Possible target columns:", possible_targets)

if len(possible_targets) != 1:
    raise ValueError(
        "Could not identify one unambiguous cluster target. "
        "Use the printed column names to select it."
    )

target = possible_targets[0]

merged = df.merge(
    labels[[common_id, target]],
    on=common_id,
    how="inner",
    validate="one_to_one",
)

print("\n=== MERGE CHECK ===")
print("Matched respondents:", len(merged))
print("Target:", target)
print("Target distribution:")
print(merged[target].value_counts(dropna=False).sort_index())

rows = []
for col in predictors:
    rows.append({
        "variable": col,
        "found": col in merged.columns,
        "missing_percent": (
            round(merged[col].isna().mean() * 100, 2)
            if col in merged.columns else None
        ),
        "unique_values": (
            int(merged[col].nunique(dropna=True))
            if col in merged.columns else None
        ),
    })

audit = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT_PATH, index=False)

print("\n=== PREDICTOR AUDIT ===")
print(audit.to_string(index=False))
print("\nSaved:", OUTPUT_PATH)
