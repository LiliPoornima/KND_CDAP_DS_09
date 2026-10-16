from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "research/component_02/data/raw/"
    "findex_microdata_2025_labelled_update112425.csv"
)

OUTPUT_DIR = Path("research/component_02/reports/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

required = ["internet_use", "con24", "con25"]
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("=== 1. THREE-WAY CROSS-TABULATION ===")

three_way = (
    df.groupby(
        ["internet_use", "con24", "con25"],
        dropna=False
    )
    .size()
    .reset_index(name="count")
)

print(three_way.to_string(index=False))

three_way.to_csv(
    OUTPUT_DIR / "internet_three_way_crosstab.csv",
    index=False
)

print("\n=== 2. CHECK CON24 AND CON25 AGAINST internet_use ===")

# Expected indicator based on the questionnaire responses:
# Yes if con24 == 1 OR con25 == 1.
# This is only a diagnostic hypothesis, not a final recoding rule.

expected = (
    df["con24"].eq(1)
    | df["con25"].eq(1)
).astype("int8")

mismatch = df["internet_use"].ne(expected)

print("Total rows:", len(df))
print("Rows not matching proposed rule:", int(mismatch.sum()))
print(
    "Percentage not matching:",
    round(mismatch.mean() * 100, 3),
    "%"
)

print("\nMismatch details:")
print(
    df.loc[mismatch, ["internet_use", "con24", "con25"]]
    .value_counts(dropna=False)
    .to_string()
)

df.loc[
    mismatch, ["internet_use", "con24", "con25"]
].to_csv(
    OUTPUT_DIR / "internet_indicator_mismatch_rows.csv",
    index=False
)

print("\n=== 3. COUNTS BY con24 ===")

summary = (
    df.groupby("con24", dropna=False)
    .agg(
        total_rows=("internet_use", "size"),
        internet_use_0=("internet_use", lambda x: int((x == 0).sum())),
        internet_use_1=("internet_use", lambda x: int((x == 1).sum())),
        con25_yes=("con25", lambda x: int((x == 1).sum())),
    )
)

print(summary.to_string())
summary.to_csv(
    OUTPUT_DIR / "internet_indicator_by_con24.csv"
)

print("\nDiagnostic completed. Original dataset was not modified.")
