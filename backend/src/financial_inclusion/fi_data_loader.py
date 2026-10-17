
"""Load the Global Findex 2025 microdata for Component 02."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "financial_inclusion"
    / "raw"
    / "findex_microdata_2025_labelled_update112425.csv"
)


def load_microdata(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw survey CSV without changing survey response codes."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Global Findex CSV was not found: {csv_path}"
        )

    df = pd.read_csv(csv_path, low_memory=False)

    if df.empty:
        raise ValueError("The Global Findex dataset contains no rows.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicated}")

    return df

