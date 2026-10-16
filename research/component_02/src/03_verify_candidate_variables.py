
from pathlib import Path
import pandas as pd

# Run this script from the repository root.
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

print("=" * 70)
print("DATASET")
print("=" * 70)
print("Shape:", df.shape)

# 1. Check the actual internet-related columns.
print("\n" + "=" * 70)
print("INTERNET-RELATED COLUMNS")
print("=" * 70)

internet_columns = [
    col for col in df.columns
    if "internet" in col.lower()
    or col.lower() in ["con24", "con25", "con26", "con27"]
]

print(internet_columns)

# 2. Print observed codes, including missing values.
variables_to_check = [
    "con24", "con25", "con26", "con27",
    "account_fin", "account_mob", "account",
    "dig_account", "anydigpayment", "saved", "borrowed"
]

for col in variables_to_check:
    print(f"\n--- {col} ---")

    if col not in df.columns:
        print("NOT FOUND")
        continue

    counts = df[col].value_counts(dropna=False).sort_index()
    percentages = (
        df[col].value_counts(dropna=False, normalize=True)
        .sort_index() * 100
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(3)
    })
    print(result.to_string())

# 3. Examine the relationship between con24 and con25.
if "con24" in df.columns and "con25" in df.columns:
    print("\n" + "=" * 70)
    print("CROSS-TABULATION: con24 vs con25")
    print("=" * 70)

    cross_tab = pd.crosstab(
        df["con24"],
        df["con25"],
        dropna=False,
        margins=True
    )
    print(cross_tab.to_string())

    cross_tab.to_csv(
        OUTPUT_DIR / "con24_con25_crosstab.csv"
    )

# 4. Check missingness of key variables by economy.
print("\n" + "=" * 70)
print("MISSINGNESS BY ECONOMY")
print("=" * 70)

key_variables = [
    "account_fin", "account_mob", "account",
    "dig_account", "anydigpayment", "saved", "borrowed"
]

available = [
    col for col in key_variables
    if col in df.columns
]

if "economy" in df.columns and available:
    missing_by_economy = df.groupby("economy")[available].agg(
        lambda s: round(s.isna().mean() * 100, 2)
    )

    print(missing_by_economy.head(15).to_string())

    missing_by_economy.to_csv(
        OUTPUT_DIR / "candidate_missingness_by_economy.csv"
    )

    missing_by_economy.to_csv(
        OUTPUT_DIR / "candidate_missingness_by_economy.csv"
    )

print("\nAudit complete. No values were changed.")
