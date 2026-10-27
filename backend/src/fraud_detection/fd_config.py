"""Fraud-detection configuration: constants, paths and dataclasses.

Single source of truth for Component 04. Nothing here imports other components.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------- paths
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = REPO_ROOT / "data" / "fraud_detection"
RAW_DIR: Path = DATA_DIR / "raw"
HANDBOOK_RAW_DIR: Path = RAW_DIR / "handbook"
# Location where the GitHub download currently sits (used if ``handbook`` is absent/empty).
HANDBOOK_ALT_RAW_DIR: Path = RAW_DIR / "simulated-data-raw-main" / "data"
IEEE_RAW_DIR: Path = RAW_DIR / "ieee_cis"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = REPO_ROOT / "outputs" / "fraud_detection"
MODEL_DIR: Path = REPO_ROOT / "models" / "fraud_detection"

# --------------------------------------------------------------------------- columns
COL_ID = "TRANSACTION_ID"
COL_DATETIME = "TX_DATETIME"
COL_CUSTOMER = "CUSTOMER_ID"
COL_TERMINAL = "TERMINAL_ID"
COL_AMOUNT = "TX_AMOUNT"
COL_SECONDS = "TX_TIME_SECONDS"
COL_DAY = "TX_TIME_DAYS"
COL_FRAUD = "TX_FRAUD"
COL_SCENARIO = "TX_FRAUD_SCENARIO"

HANDBOOK_COLUMNS: Tuple[str, ...] = (
    COL_ID, COL_DATETIME, COL_CUSTOMER, COL_TERMINAL, COL_AMOUNT,
    COL_SECONDS, COL_DAY, COL_FRAUD, COL_SCENARIO,
)

# Columns that must NEVER be fed to the model directly (leakage / identifiers).
FORBIDDEN_MODEL_COLUMNS: Tuple[str, ...] = (
    COL_SCENARIO, COL_DAY, COL_SECONDS, COL_ID, COL_CUSTOMER, COL_TERMINAL,
)

MODEL_FEATURES: Tuple[str, ...] = (
    "TX_AMOUNT", "hour", "dow", "amt_ratio", "t_risk", "t_vol",
    "c_mean_1D", "c_cnt_1D", "c_mean_7D", "c_cnt_7D", "c_mean_30D", "c_cnt_30D",
)

# --------------------------------------------------------------------------- time design
DELAYS: Tuple[int, ...] = (7, 14, 28)
WINDOW_DAYS: int = 7
SHORT_WINDOW_DAYS: int = 3
WARMUP_END_DAY: int = 34          # days 0..34 are feature history only
TRAIN_START: int = 35
TRAIN_END: int = 62               # inclusive
GAP_START: int = TRAIN_END + 1    # 63
REFERENCE_DAYS: int = 21
LAST_TEST_DAY: int = 181          # day 182 belongs to an incomplete week and is not used
DEFAULT_DRIFT_ONSET_D7: int = 105
DEFAULT_DRIFT_ONSET_OTHER: int = 112
TERMINAL_RISK_WINDOW: int = 28

# --------------------------------------------------------------------------- drift scenarios
SCENARIO_CONTROL = "control"
SCENARIO_S1 = "S1"
SCENARIO_S2 = "S2"
SCENARIO_S3 = "S3"
DRIFT_SCENARIOS: Tuple[str, ...] = (
    SCENARIO_CONTROL,
    SCENARIO_S1,
    SCENARIO_S2,
    SCENARIO_S3,
)

# --------------------------------------------------------------------------- data expectations
@dataclass(frozen=True)
class HandbookExpectations:
    """Verified properties of the full concatenated Handbook dataset."""

    n_rows: int = 1_754_155
    min_day: int = 0
    max_day: int = 182
    n_fraud: int = 14_681
    scenario_counts: Dict[int, int] = field(
        default_factory=lambda: {0: 1_739_474, 1: 973, 2: 9_077, 3: 4_631}
    )
    n_files: int = 183


HANDBOOK_EXPECTATIONS = HandbookExpectations()


# --------------------------------------------------------------------------- splits
@dataclass(frozen=True)
class SplitConfig:
    """Day boundaries (all inclusive) for one label-delay setting."""

    delay: int
    train_start: int
    train_end: int
    gap_start: int
    gap_end: int
    test_start: int
    test_end: int
    reference_start: int
    reference_end: int
    drift_onset: int


def build_split_config(delay: int, drift_onset: Optional[int] = None) -> SplitConfig:
    """Build the split for a label delay following the spec (section 5).

    test_start = ceil((63 + d) / 7) * 7 ; reference = 21 days from test_start;
    drift onset = 105 for d = 7, otherwise 112 (overridable).
    """
    if delay <= 0:
        raise ValueError(f"delay must be positive, got {delay}")
    test_start = math.ceil((GAP_START + delay) / 7) * 7
    if drift_onset is None:
        drift_onset = DEFAULT_DRIFT_ONSET_D7 if delay == 7 else DEFAULT_DRIFT_ONSET_OTHER
    return SplitConfig(
        delay=delay,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        gap_start=GAP_START,
        gap_end=test_start - 1,
        test_start=test_start,
        test_end=LAST_TEST_DAY,
        reference_start=test_start,
        reference_end=test_start + REFERENCE_DAYS - 1,
        drift_onset=drift_onset,
    )


# --------------------------------------------------------------------------- model / metrics
TARGET_FPR: float = 0.01          # primary metric: recall at 1% FPR
CUSTOMER_WINDOWS: Tuple[str, ...] = ("1D", "7D", "30D")


@dataclass(frozen=True)
class ModelParams:
    """LightGBM hyper-parameters (spec section 7)."""

    learning_rate: float = 0.05
    num_leaves: int = 31
    n_estimators: int = 200
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 20.0
    n_jobs: int = -1


MODEL_PARAMS = ModelParams()


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configuration is left to the entry point)."""
    return logging.getLogger(name)
