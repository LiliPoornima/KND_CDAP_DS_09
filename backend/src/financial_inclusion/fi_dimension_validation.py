
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT / "data" / "financial_inclusion" /
    "processed" / "fi_dimension_dataset.csv"
)

REPORT_DIR = ROOT / "outputs" / "financial_inclusion" / "reports"

SUMMARY_FILE = REPORT_DIR / "fi_dimension_validation_summary.json"
COUNTRY_FILE = REPORT_DIR / "fi_dimension_country_coverage.csv"
DISTRIBUTION_FILE = REPORT_DIR / "fi_dimension_score_distributions.csv"


DIMENSIONS = {
    "Connectivity": [
        "dim_connect_phone",
        "dim_connect_internet",
        "score_connectivity",
    ],
    "Financial Access": [
        "dim_access_account",
        "dim_access_financial_institution",
        "score_financial_access",
    ],
    "Digital Financial Usage": [
        "dim_usage_any_digital_payment",
        "score_digital_financial_usage",
    ],
    "Digital Safety": [
        "dim_safety_phone_protection",
        "risk_scam_exposure",
        "score_digital_safety",
    ],
    "Financial Resilience": [
        "dim_resilience_saved",
        "dim_resilience_emergency_ease",
        "dim_resilience_income_buffer",
        "score_financial_resilience",
    ],
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = ["wpid_random", "economy", "wgt"]
    for columns in DIMENSIONS.values():
        required.extend(columns)

    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df["wpid_random"].isna().any():
        raise ValueError("Missing respondent IDs found.")

    if df["wpid_random"].duplicated().any():
        raise ValueError("Duplicate respondent IDs found.")

    if df["economy"].isna().any():
        raise ValueError("Missing economy values found.")

    country_rows = []
    distribution_rows = []
    dimension_summary = {}

    for dimension, columns in DIMENSIONS.items():
        for column in columns:
            series = df[column]
            valid = series.dropna()

            country_rows.append({
                "dimension": dimension,
                "variable": column,
                "economies_total": int(df["economy"].nunique()),
                "economies_with_data": int(
                    (df.groupby("economy")[column].count() > 0).sum()
                ),
                "economies_with_80pct_or_more_missing": int(
                    (
                        df.groupby("economy")[column]
                        .apply(lambda x: x.isna().mean() >= 0.80)
                    ).sum()
                ),
                "non_missing_n": int(series.notna().sum()),
                "missing_n": int(series.isna().sum()),
                "missing_percent": round(
                    float(series.isna().mean() * 100), 2
                ),
            })

            distribution_rows.append({
                "dimension": dimension,
                "variable": column,
                "distinct_nonmissing_values": int(valid.nunique()),
                "minimum": (
                    float(valid.min()) if len(valid) else None
                ),
                "maximum": (
                    float(valid.max()) if len(valid) else None
                ),
                "mean": (
                    round(float(valid.mean()), 4) if len(valid) else None
                ),
                "median": (
                    round(float(valid.median()), 4) if len(valid) else None
                ),
            })

        score_column = [
            c for c in columns if c.startswith("score_")
        ][0]

        dimension_summary[dimension] = {
            "score_column": score_column,
            "score_missing_percent": round(
                float(df[score_column].isna().mean() * 100), 2
            ),
            "score_outside_0_1": int(
                (
                    df[score_column].notna()
                    & ~df[score_column].between(0, 1)
                ).sum()
            ),
        }

    country_df = pd.DataFrame(country_rows)
    distribution_df = pd.DataFrame(distribution_rows)

    country_df.to_csv(COUNTRY_FILE, index=False)
    distribution_df.to_csv(DISTRIBUTION_FILE, index=False)

    summary = {
        "rows": int(len(df)),
        "unique_respondents": int(df["wpid_random"].nunique()),
        "economies": int(df["economy"].nunique()),
        "dimensions": dimension_summary,
        "source_file": str(INPUT_FILE.relative_to(ROOT)),
        "note": (
            "Coverage and score-range audit only. Passing this audit "
            "does not prove construct validity or cross-country "
            "comparability."
        ),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Dimension validation audit completed.")
    print(f"Rows: {len(df)}")
    print(f"Economies: {df['economy'].nunique()}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Country coverage: {COUNTRY_FILE}")
    print(f"Distributions: {DISTRIBUTION_FILE}")
    print("\nDimension summary:")
    print(json.dumps(dimension_summary, indent=2))


if __name__ == "__main__":
    main()
