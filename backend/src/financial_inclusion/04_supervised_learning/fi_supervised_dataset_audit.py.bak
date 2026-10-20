
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = (
    ROOT / "data" / "financial_inclusion"
    / "processed" / "fi_dimension_dataset.csv"
)
LABEL_PATH = (
    ROOT / "outputs" / "financial_inclusion"
    / "metrics" / "fi_clustering_labels_5D_k4.csv"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "financial_inclusion"
    / "reports" / "04_supervised_learning" / "fi_supervised_dataset_audit.csv"
)

ID = "wpid_random"
TARGET = "cluster_id"

PREDICTORS = [
    "female", "age", "educ", "inc_q",
    "emp_in", "urbanicity", "regionwb"
]

df = pd.read_csv(DATA_PATH, low_memory=False)
labels = pd.read_csv(LABEL_PATH, low_memory=False)

# Join respondents to their candidate cluster labels.
data = df.merge(
    labels[[ID, TARGET]],
    on=ID,
    how="inner",
    validate="one_to_one"
)

# Retain only the intended target and demographic predictors.
data = data[[ID, TARGET] + PREDICTORS].copy()

print("=== DATASET SUMMARY ===")
print("Dataset shape:", data.shape)
print("Duplicate respondent IDs:", data[ID].duplicated().sum())
print("Missing target values:", data[TARGET].isna().sum())

print("\n=== CLUSTER DISTRIBUTION ===")
counts = data[TARGET].value_counts().sort_index()
summary = pd.DataFrame({
    "respondents": counts,
    "percentage": (counts / len(data) * 100).round(2)
})
print(summary)

print("\n=== PREDICTOR CHECK ===")
print(data[PREDICTORS].isna().sum().to_string())

print("\n=== PREDICTOR VALUE CHECK ===")
for col in PREDICTORS:
    values = sorted(data[col].dropna().unique().tolist())
    print(f"{col}: {values[:20]}")

print("\n=== COMPLETE-CASE CHECK ===")
complete = data.dropna(subset=[TARGET] + PREDICTORS)
print("Rows before:", len(data))
print("Rows after complete-case filtering:", len(complete))
print("Rows excluded:", len(data) - len(complete))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
