
"""
Component 02: Initial dataset structure validation.

"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path, index_col=None) -> pd.DataFrame:
    """Load a supported dataset format into a pandas DataFrame."""

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, index_col=index_col)

    if suffix == ".txt":
        return pd.read_csv(file_path, sep=None, engine="python",
                           index_col=index_col)

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    if suffix == ".dta":
        return pd.read_stata(file_path)

    if suffix == ".sav":
        return pd.read_spss(file_path)

    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(file_path, index_col=index_col)

    raise ValueError(
        f"Unsupported file format: {suffix}. "
        "Supported formats: CSV, TXT, Parquet, DTA, SAV, XLSX, XLSM."
    )


def validate_structure(df: pd.DataFrame) -> dict:
    """Create a summary of the dataset structure without exposing row data."""

    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100) if len(df) else 0

    missingness = pd.DataFrame({
        "column_name": df.columns.astype(str),
        "missing_count": missing_count.to_numpy(),
        "missing_percent": missing_percent,
    })

    column_inventory = pd.DataFrame({
        "column_name": df.columns.astype(str),
        "data_type": [str(dtype) for dtype in df.dtypes],
        "unique_non_missing_values": [
            int(df.iloc[:, i].nunique(dropna=True))
            for i in range(df.shape[1])
        ],
    })

    duplicate_rows = int(df.duplicated().sum())
    duplicate_column_names = int(df.columns.duplicated().sum())
    all_missing_columns = [
        str(df.columns[i])
        for i in range(df.shape[1])
        if df.iloc[:, i].isna().all()
    ]

    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "index_is_unique": bool(df.index.is_unique),
        "duplicate_index_count": int(df.index.duplicated().sum()),
        "duplicate_row_count": duplicate_rows,
        "duplicate_column_name_count": duplicate_column_names,
        "total_missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing_values": int((missing_count > 0).sum()),
        "columns_with_all_values_missing": all_missing_columns,
        "memory_usage_mb": round(
            float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 2
        ),
    }

    return summary, column_inventory, missingness


def main():
    parser = argparse.ArgumentParser(
        description="Validate the structure of a Component 02 dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source dataset file."
    )

    parser.add_argument(
        "--index-col",
        default=None,
        help=(
            "Optional column to use as the DataFrame index. "
            "Use this only if the file contains a saved index column."
        )
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset file was not found: {input_path}"
        )

    index_col = args.index_col
    if index_col is not None and index_col.isdigit():
        index_col = int(index_col)

    output_dir = Path(__file__).resolve().parents[1] / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path, index_col=index_col)
    summary, inventory, missingness = validate_structure(df)

    summary["source_file_name"] = input_path.name

    with (output_dir / "structure_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    inventory.to_csv(
        output_dir / "column_inventory.csv", index=False
    )

    missingness.to_csv(
        output_dir / "missingness_summary.csv", index=False
    )

    print("\n=== COMPONENT 02: DATASET STRUCTURE AUDIT ===")
    print(f"Source file: {input_path.name}")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Columns: {summary['column_count']:,}")
    print(f"Unique DataFrame index: {summary['index_is_unique']}")
    print(f"Duplicate index values: {summary['duplicate_index_count']:,}")
    print(f"Duplicate rows: {summary['duplicate_row_count']:,}")
    print(
        "Duplicate column names: "
        f"{summary['duplicate_column_name_count']:,}"
    )
    print(f"Total missing cells: {summary['total_missing_cells']:,}")
    print(
        "Columns with missing values: "
        f"{summary['columns_with_missing_values']:,}"
    )
    print(
        "Columns entirely missing: "
        f"{len(summary['columns_with_all_values_missing']):,}"
    )
    print(f"Memory usage: {summary['memory_usage_mb']} MB")

    print("\n=== GENERATED AUDIT FILES ===")
    print(output_dir / "structure_summary.json")
    print(output_dir / "column_inventory.csv")
    print(output_dir / "missingness_summary.csv")


if __name__ == "__main__":
    main()
