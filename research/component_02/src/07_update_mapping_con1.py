
from pathlib import Path
import pandas as pd

MAPPING_FILE = Path(
    "research/component_02/references/"
    "component02_variable_mapping.csv"
)

if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Mapping file not found: {MAPPING_FILE.resolve()}"
    )

df = pd.read_csv(MAPPING_FILE, dtype=str)

required = [
    "variable", "dimension", "indicator_role",
    "codebook_definition", "valid_yes_code",
    "valid_no_code", "skip_logic", "missing_value_rule",
    "decision", "reason"
]

if not all(col in df.columns for col in required):
    raise ValueError("Mapping CSV columns do not match the expected structure.")

if not df["variable"].eq("con1").any():
    raise ValueError("con1 row not found in mapping register.")

mask = df["variable"].eq("con1")

df.loc[mask, "dimension"] = "Connectivity"
df.loc[mask, "indicator_role"] = "Mobile phone ownership/access candidate"
df.loc[mask, "codebook_definition"] = (
    "Has a mobile phone used to make and receive personal calls"
)
df.loc[mask, "valid_yes_code"] = "1"
df.loc[mask, "valid_no_code"] = "2"
df.loc[mask, "skip_logic"] = "No skip logic established in this validation"
df.loc[mask, "missing_value_rule"] = (
    "No missing values observed; preserve codes 3=Don't know "
    "and 4=Refused separately from No"
)
df.loc[mask, "decision"] = "Validated candidate"
df.loc[mask, "reason"] = (
    "Codebook definition and observed codes agree; 144090 rows, "
    "0 missing; codes 1=Yes, 2=No, 3=Don't know, 4=Refused"
)

df.to_csv(MAPPING_FILE, index=False)

print("Mapping register updated for con1.")
print(
    df.loc[mask, [
        "variable", "codebook_definition",
        "valid_yes_code", "valid_no_code",
        "decision", "reason"
    ]].to_string(index=False)
)
print("\nSaved:", MAPPING_FILE.resolve())
