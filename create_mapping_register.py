from pathlib import Path
import pandas as pd

p = Path("research/component_02/references/component02_variable_mapping.csv")
p.parent.mkdir(parents=True, exist_ok=True)

if p.exists():
    print("File already exists. No changes made.")
else:
    cols = [
        "variable", "dimension", "indicator_role",
        "codebook_definition", "valid_yes_code", "valid_no_code",
        "skip_logic", "missing_value_rule", "decision", "reason"
    ]

    rows = [
        {
            "variable": "internet_use",
            "dimension": "Connectivity",
            "indicator_role": "Primary indicator candidate",
            "codebook_definition": "Constructed indicator; official documentation pending",
            "valid_yes_code": 1,
            "valid_no_code": 0,
            "skip_logic": "Compare with con24 and con25",
            "missing_value_rule": "Check missingness before final approval",
            "decision": "Pending",
            "reason": "Tested rule con24 == 1 OR con25 == 1 matched all 144090 rows; zero mismatches"
        },
        {
            "variable": "con1",
            "dimension": "Connectivity",
            "indicator_role": "Phone ownership/access candidate",
            "codebook_definition": "Pending verification",
            "valid_yes_code": "Pending",
            "valid_no_code": "Pending",
            "skip_logic": "Pending review",
            "missing_value_rule": "Do not recode until verified",
            "decision": "Pending",
            "reason": "Requires codebook and observed-value validation"
        },
        {
            "variable": "con24",
            "dimension": "Connectivity",
            "indicator_role": "Questionnaire source variable",
            "codebook_definition": "Past-7-day internet use; supplied codebook excerpt lists 1=Yes, 2=No, 8=Don't know, 9=Refused",
            "valid_yes_code": 1,
            "valid_no_code": 2,
            "skip_logic": "Review con25 for follow-up responses",
            "missing_value_rule": "Do not recode observed values 3 and 4",
            "decision": "Pending",
            "reason": "Observed codes 3 and 4 remain unexplained"
        },
        {
            "variable": "con25",
            "dimension": "Connectivity",
            "indicator_role": "Questionnaire follow-up variable",
            "codebook_definition": "Exact official definition pending",
            "valid_yes_code": 1,
            "valid_no_code": 2,
            "skip_logic": "Missing when con24=1 in observed data",
            "missing_value_rule": "Preserve missingness until verified",
            "decision": "Pending",
            "reason": "Observed conditional skip pattern requires documentation review"
        }
    ]

    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    print("Created initial mapping register.")

print("Path:", p.resolve())
print("Exists:", p.exists())
if p.exists():
    print(pd.read_csv(p).to_string(index=False))
