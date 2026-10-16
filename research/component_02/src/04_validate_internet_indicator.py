
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "research/component_02/data/raw/"
    "findex_microdata_2025_labelled_update112425.csv"
)

OUTPUT_DIR = Path("research/component_02/reports/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Check the variables we need.
variables = ["internet_use", "con24", "con25", "con26", "con27"]

print("=== INTERNET VARIABLE AVAILABILITY ===")
for col in variables:
    if col not in df.columns:
        print(f"{col}: NOT FOUND")
        continue

    print(f"\n--- {col} ---")
    print("Missing:", int(df[col].isna().sum()))
    print("Missing %:", round(df[col].isna().mean() * 100, 3))
    print(df[col].value_counts(dropna=False).sort_index().to_string())

# Export distributions for later evidence and reporting.
distribution_rows = []

for col in variables:
    if col not in df.columns:
        continue

    counts = df[col].value_counts(dropna=False)

    for value, count in counts.items():
        distribution_rows.append({
            "variable": col,
            "observed_value": (
                "<MISSING>" if pd.isna(value) else value
            ),
            "count": int(count),
            "percentage": round(count / len(df) * 100, 3),
        })

pd.DataFrame(distribution_rows).to_csv(
    OUTPUT_DIR / "internet_variable_distributions.csv",
    index=False
)

# Compare the existing constructed variable with questionnaire responses.
for questionnaire_var in ["con24", "con25"]:
    if (
        "internet_use" in df.columns
        and questionnaire_var in df.columns
    ):
        left = df["internet_use"].astype("string").fillna("<MISSING>")
        right = df[questionnaire_var].astype("string").fillna("<MISSING>")

        cross_tab = pd.crosstab(
            left,
            right,
            margins=True
        )

        print(f"\n=== internet_use vs {questionnaire_var} ===")
        print(cross_tab.to_string())

        cross_tab.to_csv(
            OUTPUT_DIR / f"internet_use_vs_{questionnaire_var}.csv"
        )

print("\nValidation completed.")
print("No original data was changed.")
