
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "research/component_02/data/raw/"
    "findex_microdata_2025_labelled_update112425.csv"
)

OUTPUT_DIR = Path("research/component_02/reports/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

required = ["con1", "con9", "economy"]
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 1. Overall distribution
print("=== 1. CON9 OVERALL DISTRIBUTION ===")
print("Total rows:", len(df))
print("Missing:", int(df["con9"].isna().sum()))
print(
    "Missing percentage:",
    round(df["con9"].isna().mean() * 100, 3),
    "%"
)

counts = df["con9"].value_counts(dropna=False).sort_index()
distribution = counts.rename("count").to_frame()
distribution["percentage_of_all_rows"] = (
    distribution["count"] / len(df) * 100
).round(3)

print(distribution.to_string())

distribution.to_csv(
    OUTPUT_DIR / "con9_value_distribution.csv"
)

# 2. Compare phone ownership and phone type
print("\n=== 2. CON1 VS CON9 ===")

con1_label = df["con1"].astype("string").fillna("<MISSING>")
con9_label = df["con9"].astype("string").fillna("<MISSING>")

cross_tab = pd.crosstab(
    con1_label,
    con9_label,
    margins=True
)

print(cross_tab.to_string())

cross_tab.to_csv(
    OUTPUT_DIR / "con1_vs_con9_crosstab.csv"
)

# 3. Missingness by phone ownership code
print("\n=== 3. CON9 MISSINGNESS BY CON1 ===")

missing_by_con1 = (
    df.assign(con9_missing=df["con9"].isna())
    .groupby("con1", dropna=False)
    .agg(
        respondents=("con9_missing", "size"),
        con9_missing_count=("con9_missing", "sum"),
        con9_missing_rate=("con9_missing", "mean")
    )
)

missing_by_con1["con9_missing_percentage"] = (
    missing_by_con1["con9_missing_rate"] * 100
).round(3)

print(missing_by_con1.to_string())

missing_by_con1.to_csv(
    OUTPUT_DIR / "con9_missingness_by_con1.csv"
)

# 4. Missingness by economy
print("\n=== 4. ECONOMY-LEVEL CON9 MISSINGNESS ===")

by_economy = (
    df.assign(con9_missing=df["con9"].isna())
    .groupby("economy")
    .agg(
        respondents=("con9_missing", "size"),
        missing_count=("con9_missing", "sum"),
        missing_rate=("con9_missing", "mean")
    )
)

by_economy["missing_percentage"] = (
    by_economy["missing_rate"] * 100
).round(3)

by_economy = by_economy.sort_values(
    "missing_percentage", ascending=False
)

print(by_economy.head(20).to_string())

by_economy.to_csv(
    OUTPUT_DIR / "con9_missingness_by_economy.csv"
)

print("\nValidation completed.")
print("Original dataset was not modified.")
