
from pathlib import Path
import json

import pandas as pd


# Project root: KND_CDAP_DS_09
ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_analysis_dataset.csv"
)

OUTPUT_DIR = ROOT / "data" / "financial_inclusion" / "processed"
REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "fi_dimension_dataset.csv"
REPORT_FILE = REPORT_DIR / "fi_dimension_creation_report.json"


def require_columns(df, columns):
    """Check that every required source column exists."""
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(f"Required columns not found: {missing}")


def binary_from_codes(series, yes_code=1, no_code=2):
    """
    Recode verified source responses:
    yes_code -> 1
    no_code  -> 0
    all other values -> missing
    """
    result = pd.Series(float("nan"), index=series.index, dtype="float64")

    result.loc[series == yes_code] = 1
    result.loc[series == no_code] = 0

    return result


def existing_binary(series):
    """
    Preserve valid constructed binary indicators (0/1).
    Other values remain missing.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin([0, 1]))


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [
        "wpid_random",
        "economy",
        "wgt",
        "con1",
        "internet_use",
        "account",
        "account_fin",
        "anydigpayment",
        "con18",
        "con21",
        "saved",
        "fin24a",
        "fin24b",
    ]

    require_columns(df, required)

    original_rows = len(df)
    original_columns = len(df.columns)
    original_ids_unique = bool(df["wpid_random"].is_unique)

    if df["wpid_random"].isna().any():
        raise ValueError("Missing respondent IDs found in input dataset.")

    if not original_ids_unique:
        raise ValueError("Duplicate respondent IDs found in input dataset.")

    # Preserve all original source columns and their values.
    out = df.copy()

    # =========================================================
    # DIMENSION 1: CONNECTIVITY
    # =========================================================

    # con1: 1 = yes, 2 = no
    out["dim_connect_phone"] = binary_from_codes(df["con1"])

    # internet_use: existing constructed 0/1 variable
    out["dim_connect_internet"] = existing_binary(df["internet_use"])

    # =========================================================
    # DIMENSION 2: FINANCIAL ACCESS
    # =========================================================

    # Main dimension indicator.
    out["dim_access_account"] = existing_binary(df["account"])

    # Retained separately for audit/comparison.
    # Not included as a second component in the score.
    out["dim_access_financial_institution"] = existing_binary(
        df["account_fin"]
    )

    # =========================================================
    # DIMENSION 3: DIGITAL FINANCIAL USAGE
    # =========================================================

    # Existing constructed 0/1 variable.
    # Missing values are not converted to zero.
    out["dim_usage_any_digital_payment"] = existing_binary(
        df["anydigpayment"]
    )

    # =========================================================
    # DIMENSION 4: DIGITAL SAFETY / RISK
    # =========================================================

    # con18: phone security protection, 1 = yes, 2 = no.
    out["dim_safety_phone_protection"] = binary_from_codes(df["con18"])

    # con21: scam exposure.
    # This is a risk indicator, not a direct safety-capability measure.
    out["risk_scam_exposure"] = binary_from_codes(df["con21"])

    # =========================================================
    # DIMENSION 5: FINANCIAL RESILIENCE
    # =========================================================

    # Existing constructed 0/1 saving indicator.
    out["dim_resilience_saved"] = existing_binary(df["saved"])

    # fin24a:
    # 1 = very difficult
    # 2 = somewhat difficult
    # 3 = no difficulty
    # 4/5 = don't know/refused
    #
    # Provisional normalized ordinal mapping.
    out["dim_resilience_emergency_ease"] = (
        pd.to_numeric(df["fin24a"], errors="coerce")
        .map({
            1: 0.0,
            2: 0.5,
            3: 1.0,
        })
    )

    # fin24b:
    # 1 = less than two weeks
    # 2 = about one month
    # 3 = about two months
    # 4 = more than two months
    # 5/6 = don't know/refused
    #
    # Provisional normalized ordinal mapping.
    out["dim_resilience_income_buffer"] = (
        pd.to_numeric(df["fin24b"], errors="coerce")
        .map({
            1: 0.0,
            2: 1 / 3,
            3: 2 / 3,
            4: 1.0,
        })
    )

    # =========================================================
    # DIMENSION SCORE CONSTRUCTION
    # =========================================================

    dimension_components = {
        "score_connectivity": [
            "dim_connect_phone",
            "dim_connect_internet",
        ],
        "score_financial_access": [
            "dim_access_account",
        ],
        "score_digital_financial_usage": [
            "dim_usage_any_digital_payment",
        ],
        "score_digital_safety": [
            "dim_safety_phone_protection",
        ],
        "score_financial_resilience": [
            "dim_resilience_saved",
            "dim_resilience_emergency_ease",
            "dim_resilience_income_buffer",
        ],
    }

    for score_name, components in dimension_components.items():

        # Default: mean of available components.
        # This remains appropriate for the two-component
        # connectivity score under the current provisional policy.
        out[score_name] = out[components].mean(axis=1)

        out[f"{score_name}_components_available"] = (
            out[components].notna().sum(axis=1)
        )

    # ---------------------------------------------------------
    # IMPORTANT CORRECTION:
    # Financial resilience requires all three components.
    # If any component is missing, the composite score is missing.
    # ---------------------------------------------------------

    resilience_components = [
        "dim_resilience_saved",
        "dim_resilience_emergency_ease",
        "dim_resilience_income_buffer",
    ]

    complete_resilience = (
        out[resilience_components].notna().all(axis=1)
    )

    out["score_financial_resilience"] = (
        out[resilience_components]
        .mean(axis=1)
        .where(complete_resilience)
    )

    out["score_financial_resilience_components_available"] = (
        out[resilience_components].notna().sum(axis=1)
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    if len(out) != original_rows:
        raise AssertionError("Row count changed unexpectedly.")

    if not out["wpid_random"].is_unique:
        raise AssertionError("Respondent IDs are not unique after processing.")

    if not out["wpid_random"].equals(df["wpid_random"]):
        raise AssertionError("Respondent ID order changed unexpectedly.")

    # Verify original columns are preserved without value changes.
    source_columns_preserved = all(
        out[column].equals(df[column])
        for column in df.columns
    )

    if not source_columns_preserved:
        raise AssertionError("One or more original source columns changed.")

    score_columns = list(dimension_components.keys())

    component_columns = sorted({
        component
        for components in dimension_components.values()
        for component in components
    })

    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "input_rows": int(original_rows),
        "output_rows": int(len(out)),
        "input_columns": int(original_columns),
        "output_columns": int(len(out.columns)),
        "respondent_ids_unique_before": original_ids_unique,
        "respondent_ids_unique_after": bool(out["wpid_random"].is_unique),
        "source_columns_preserved": bool(source_columns_preserved),
        "rows_dropped": 0,
        "imputation_performed": False,
        "unresolved_variables_excluded": ["con13", "con24"],
        "resilience_requires_all_components": True,
        "score_missing_percent": {
            column: round(
                float(out[column].isna().mean() * 100), 2
            )
            for column in score_columns
        },
        "component_availability": {
            column: {
                "available": int(out[column].notna().sum()),
                "missing": int(out[column].isna().sum()),
                "missing_percent": round(
                    float(out[column].isna().mean() * 100), 2
                ),
            }
            for column in component_columns
        },
        "resilience_component_completeness": {
            "all_three_components_available": int(
                complete_resilience.sum()
            ),
            "one_or_more_components_missing": int(
                (~complete_resilience).sum()
            ),
        },
        "important_note": (
            "Dimension scores are provisional. Financial resilience "
            "scores are calculated only when all three components are "
            "available. Missing values are not imputed. Scam exposure "
            "is retained separately from safety capability. Country "
            "coverage and construct validity must be reviewed before "
            "cross-country ranking or clustering."
        ),
    }

    out.to_csv(OUTPUT_FILE, index=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Dimension construction completed.")
    print(f"Input rows: {original_rows}")
    print(f"Output rows: {len(out)}")
    print(f"Original source columns preserved: {source_columns_preserved}")
    print(f"Output dataset: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")

    print("\nDimension score missingness (%):")

    for column, missing_percent in summary["score_missing_percent"].items():
        print(f"{column}: {missing_percent}%")

    print("\nFinancial resilience completeness:")
    print(
        "All three components available:",
        summary["resilience_component_completeness"][
            "all_three_components_available"
        ],
    )
    print(
        "One or more components missing:",
        summary["resilience_component_completeness"][
            "one_or_more_components_missing"
        ],
    )


if __name__ == "__main__":
    main()
