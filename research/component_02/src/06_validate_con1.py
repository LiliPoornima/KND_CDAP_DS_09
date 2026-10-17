
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

required_columns = ["con1", "economy"]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 1. Overall distribution
print("=== 1. CON1 OVERALL DISTRIBUTION ===")
print("Total rows:", len(df))
print("Data type:", df["con1"].dtype)
print("Missing values:", int(df["con1"].isna().sum()))
print(
    "Missing percentage:",
    round(df["con1"].isna().mean() * 100, 3),
    "%"
)

counts = df["con1"].value_counts(dropna=False).sort_index()
distribution = counts.rename("count").to_frame()
distribution["percentage"] = (
    distribution["count"] / len(df) * 100
).round(3)

print(distribution.to_string())

distribution.to_csv(
    OUTPUT_DIR / "con1_value_distribution.csv"
)

# 2. Distribution by economy
print("\n=== 2. CON1 DISTRIBUTION BY ECONOMY ===")

by_economy = pd.crosstab(
    df["economy"],
    df["con1"],
    dropna=False,
    margins=False
)

print("Number of economies:", df["economy"].nunique())
print("Cross-tab shape:", by_economy.shape)
print("\nFirst 15 economies:")
print(by_economy.head(15).to_string())

by_economy.to_csv(
    OUTPUT_DIR / "con1_distribution_by_economy.csv"
)

# 3. Check whether con1 is missing in particular economies
missing_by_economy = (
    df.assign(con1_missing=df["con1"].isna())
    .groupby("economy")
    .agg(
        respondents=("con1_missing", "size"),
        missing_count=("con1_missing", "sum"),
        missing_percentage=("con1_missing", "mean")
    )
)

missing_by_economy["missing_percentage"] = (
    missing_by_economy["missing_percentage"] * 100
).round(3)

print("\n=== 3. ECONOMY-LEVEL MISSINGNESS ===")
print(
    missing_by_economy
    .sort_values("missing_percentage", ascending=False)
    .head(15)
    .to_string()
)

missing_by_economy.to_csv(
    OUTPUT_DIR / "con1_missingness_by_economy.csv"
)

print("\nValidation completed.")
print("Original dataset was not modified.")
