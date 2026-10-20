
"""Audit missing values across Component 02 candidate variables."""

import json
from pathlib import Path

import pandas as pd

from fi_data_loader import load_microdata

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "outputs" / "financial_inclusion" / "reports" / "01_data_preparation"
REGISTER_PATH = REPORT_DIR / "fi_codebook_verification_register.csv"


def run_missingness_audit() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTER_PATH.exists():
        raise FileNotFoundError(
            f"Codebook register not found: {REGISTER_PATH}"
        )

    df = load_microdata()
    register = pd.read_csv(REGISTER_PATH)

    variables = register["variable"].dropna().tolist()
    missing_columns = sorted(set(variables) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Variables missing from source data: {missing_columns}"
        )

    # Overall missingness for each candidate variable.
    overall_rows = []

    for _, item in register.iterrows():
        variable = item["variable"]
        missing_count = int(df[variable].isna().sum())

        overall_rows.append({
            "variable": variable,
            "dimension": item["dimension"],
            "verification_status": item["verification_status"],
            "row_count": int(len(df)),
            "missing_count": missing_count,
            "non_missing_count": int(df[variable].notna().sum()),
            "missing_percent": round(
                missing_count / len(df) * 100, 2
            ),
            "primary_model_status": item["primary_model_status"],
        })

    overall = pd.DataFrame(overall_rows).sort_values(
        ["dimension", "missing_percent"],
        ascending=[True, False],
    )

    overall_path = REPORT_DIR / "fi_missingness_by_variable.csv"
    overall.to_csv(overall_path, index=False)

    # Missingness by economy, preserving the original missing values.
    economy_rows = []

    for economy, group in df.groupby("economy", dropna=False):
        for variable in variables:
            missing_count = int(group[variable].isna().sum())

            economy_rows.append({
                "economy": economy,
                "variable": variable,
                "row_count": int(len(group)),
                "missing_count": missing_count,
                "non_missing_count": int(group[variable].notna().sum()),
                "missing_percent": round(
                    missing_count / len(group) * 100, 2
                ) if len(group) else None,
            })

    by_economy = pd.DataFrame(economy_rows)
    economy_path = REPORT_DIR / "fi_missingness_by_economy.csv"
    by_economy.to_csv(economy_path, index=False)

    dimension_summary = (
        overall.groupby("dimension")
        .agg(
            variable_count=("variable", "count"),
            variables_with_missing=("missing_count",
                                    lambda values: int((values > 0).sum())),
            mean_missing_percent=("missing_percent", "mean"),
            max_missing_percent=("missing_percent", "max"),
        )
        .round(2)
        .reset_index()
    )

    dimension_path = REPORT_DIR / "fi_missingness_by_dimension.csv"
    dimension_summary.to_csv(dimension_path, index=False)

    summary = {
        "row_count": int(len(df)),
        "economy_count": int(df["economy"].nunique(dropna=True)),
        "candidate_variable_count": int(len(variables)),
        "variables_with_any_missing": int(
            (overall["missing_count"] > 0).sum()
        ),
        "overall_missing_cells_in_candidate_variables": int(
            overall["missing_count"].sum()
        ),
        "outputs": {
            "by_variable": str(overall_path.relative_to(PROJECT_ROOT)),
            "by_economy": str(economy_path.relative_to(PROJECT_ROOT)),
            "by_dimension": str(dimension_path.relative_to(PROJECT_ROOT)),
        },
        "raw_data_modified": False,
        "note": (
            "Missing values are counted, not imputed. This report does not "
            "classify missing values as errors or structural skips."
        ),
    }

    summary_path = REPORT_DIR / "fi_missingness_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nMissingness by dimension:")
    print(dimension_summary.to_string(index=False))

    return summary


if __name__ == "__main__":
    run_missingness_audit()
