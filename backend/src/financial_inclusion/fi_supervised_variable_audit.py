
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data" / "financial_inclusion"
    / "processed" / "fi_analysis_dataset.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "financial_inclusion"
    / "reports" / "fi_supervised_variable_audit.csv"
)

# Candidate predictors from the codebook
# Survey weight is audited separately, not used as an ordinary predictor.
variables = [
    "female",
    "age",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "regionwb",
    "economy",
    "economycode",
    "wgt",
]

df = pd.read_csv(DATA_PATH, low_memory=False)

rows = []

for col in variables:
    if col not in df.columns:
        rows.append({
            "variable": col,
            "status": "NOT_FOUND",
        })
        continue

    s = df[col]

    rows.append({
        "variable": col,
        "status": "FOUND",
        "dtype": str(s.dtype),
        "missing_count": int(s.isna().sum()),
        "missing_percent": round(s.isna().mean() * 100, 2),
        "unique_values": int(s.nunique(dropna=True)),
        "value_counts": str(
            s.value_counts(dropna=False).to_dict()
        ),
    })

audit = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT_PATH, index=False)

print("=== SUPERVISED PREDICTOR AUDIT ===")
print(audit.to_string(index=False))
print(f"\nSaved: {OUTPUT_PATH}")
