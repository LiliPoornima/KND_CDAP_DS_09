"""Fraud-detection data loading."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .fd_config import (
    COL_DATETIME,
    COL_DAY,
    COL_FRAUD,
    COL_SCENARIO,
    HANDBOOK_ALT_RAW_DIR,
    HANDBOOK_COLUMNS,
    HANDBOOK_EXPECTATIONS,
    HANDBOOK_RAW_DIR,
    IEEE_RAW_DIR,
    PROCESSED_DIR,
    HandbookExpectations,
    get_logger,
)

logger = get_logger(__name__)

CACHE_NAME = "handbook_combined.parquet"


class FraudDataValidationError(ValueError):
    """Raised when the loaded Handbook data does not match the verified expectations."""


def find_handbook_dir(raw_dir: Optional[Path] = None) -> Path:
    """Locate the folder holding the daily ``.pkl`` files."""
    candidates = [raw_dir] if raw_dir else [HANDBOOK_RAW_DIR, HANDBOOK_ALT_RAW_DIR]
    for cand in candidates:
        if cand is not None and cand.is_dir() and any(cand.glob("*.pkl")):
            return cand
    raise FileNotFoundError(
        "No Handbook .pkl files found. Looked in: "
        + ", ".join(str(c) for c in candidates if c is not None)
    )


def validate_handbook(
    df: pd.DataFrame,
    expected: HandbookExpectations = HANDBOOK_EXPECTATIONS,
    n_files: Optional[int] = None,
) -> None:
    """Validate a combined Handbook dataframe; raise ``FraudDataValidationError`` on mismatch."""
    problems: List[str] = []
    missing = [c for c in HANDBOOK_COLUMNS if c not in df.columns]
    if missing:
        raise FraudDataValidationError(f"Missing columns: {missing}")
    if n_files is not None and n_files != expected.n_files:
        problems.append(f"files: found {n_files}, expected {expected.n_files}")
    if len(df) != expected.n_rows:
        problems.append(f"rows: found {len(df):,}, expected {expected.n_rows:,}")
    dmin, dmax = int(df[COL_DAY].min()), int(df[COL_DAY].max())
    if (dmin, dmax) != (expected.min_day, expected.max_day):
        problems.append(f"TX_TIME_DAYS range: found {dmin}..{dmax}, expected "
                        f"{expected.min_day}..{expected.max_day}")
    n_fraud = int(df[COL_FRAUD].sum())
    if n_fraud != expected.n_fraud:
        problems.append(f"TX_FRAUD sum: found {n_fraud:,}, expected {expected.n_fraud:,}")
    counts = df[COL_SCENARIO].value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    if counts != expected.scenario_counts:
        problems.append(f"TX_FRAUD_SCENARIO counts: found {dict(sorted(counts.items()))}, "
                        f"expected {dict(sorted(expected.scenario_counts.items()))}")
    if problems:
        raise FraudDataValidationError(
            "Handbook data failed validation (a single daily file is not enough; all daily "
            "files are required):\n  - " + "\n  - ".join(problems)
        )


def load_handbook(
    raw_dir: Optional[Path] = None,
    use_cache: bool = True,
    cache_dir: Path = PROCESSED_DIR,
    expected: HandbookExpectations = HANDBOOK_EXPECTATIONS,
    validate: bool = True,
) -> pd.DataFrame:
    """Load, concatenate, sort and validate the full Handbook dataset.

    The result is sorted by ``TX_DATETIME`` (stable) with a fresh RangeIndex and is
    cached as parquet under ``data/fraud_detection/processed/`` (git-ignored).
    """
    cache_path = Path(cache_dir) / CACHE_NAME
    if use_cache and cache_path.is_file():
        logger.info("Loading cached Handbook data from %s", cache_path)
        df = pd.read_parquet(cache_path)
        if validate:
            validate_handbook(df, expected)
        return df

    folder = find_handbook_dir(raw_dir)
    files = sorted(folder.glob("*.pkl"))
    logger.info("Reading %d Handbook files from %s", len(files), folder)
    frames = [pd.read_pickle(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.sort_values([COL_DATETIME, "TRANSACTION_ID"], kind="stable").reset_index(drop=True)
    if validate:
        validate_handbook(df, expected, n_files=len(files))

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached combined data to %s", cache_path)
    return df


def load_ieee_cis(path: Optional[Path] = None) -> pd.DataFrame:
    """Load IEEE-CIS ``train_transaction.csv`` (manual Kaggle download; never downloaded here)."""
    path = Path(path) if path else IEEE_RAW_DIR / "train_transaction.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download train_transaction.csv manually from Kaggle "
            "(IEEE-CIS Fraud Detection competition, login required) and place it there."
        )
    return pd.read_csv(path)


def load_baf(*args, **kwargs) -> pd.DataFrame:
    """Stub for the optional Bank Account Fraud backup dataset (not implemented)."""
    raise NotImplementedError("BAF loader is an optional backup and is not implemented.")
