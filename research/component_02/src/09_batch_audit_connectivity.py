
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

# Candidate variables for the Connectivity dimension
candidates = [
    "con1", "con9", "con11", "con12",
    "con13", "con14", "con16",
    "con24", "con25", "con26", "con27",
    "internet_use"
]

df = pd.read_csv(INPUT_FILE, low_memory=False)

available = [v for v in candidates if v in df.columns]
not_found = [v for v in candidates if v not in df.columns]

print("=== 1. VARIABLE AVAILABILITY ===")
print("Available:", available)
print("Not found:", not_found)

# Overall summary
summary_rows = []

for col in available:
    summary_rows.append({
        "variable": col,
        "data_type": str(df[col].dtype),
        "total_rows": len(df),
        "missing_count": int(df[col].isna().sum()),
        "missing_percentage": round(
            df[col].isna().mean() * 100, 3
        ),
        "unique_nonmissing_values": int(df[col].nunique())
    })

summary = pd.DataFrame(summary_rows)

print("\n=== 2. OVERALL SUMMARY ===")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "connectivity_batch_summary.csv",
    index=False
)

# Value counts for all available variables
distribution_rows = []

for col in available:
    counts = df[col].value_counts(dropna=False)

    for value, count in counts.items():
        distribution_rows.append({
            "variable": col,
            "observed_value": (
                "<MISSING>" if pd.isna(value) else str(value)
            ),
            "count": int(count),
            "percentage_of_all_rows": round(
                count / len(df) * 100, 3
            )
        })

distribution = pd.DataFrame(distribution_rows)

distribution.to_csv(
    OUTPUT_DIR / "connectivity_batch_value_counts.csv",
    index=False
)

# Missingness by economy
if "economy" in df.columns:
    missing_rows = []

    for col in available:
        rates = df.groupby("economy")[col].agg(
            total="size",
            missing=lambda x: x.isna().sum()
        )

        rates["variable"] = col
        rates["missing_percentage"] = (
            rates["missing"] / rates["total"] * 100
        ).round(3)

        missing_rows.append(rates.reset_index())

    economy_missingness = pd.concat(
        missing_rows, ignore_index=True
    )

    economy_missingness.to_csv(
        OUTPUT_DIR / "connectivity_missingness_by_economy.csv",
        index=False
    )

    print("\n=== 3. ECONOMY-LEVEL MISSINGNESS EXAMPLES ===")
    print(
        economy_missingness
        .sort_values("missing_percentage", ascending=False)
        .head(20)
        .to_string(index=False)
    )

print("\nBatch audit completed.")
print("Original dataset was not modified.")
print("Outputs saved in:", OUTPUT_DIR.resolve())
