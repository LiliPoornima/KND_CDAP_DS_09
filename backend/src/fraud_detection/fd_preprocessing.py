"""Fraud-detection data preprocessing."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .fd_config import COL_DAY, SplitConfig, build_split_config

Window = Tuple[int, int]  # (first_day, last_day) inclusive


def matured_mask(
    df: pd.DataFrame, t: int, d: int, day_col: str = COL_DAY
) -> pd.Series:
    """Boolean mask of rows whose label is available at decision time ``t``.

    LEAKAGE RULE: a label may be used at time ``t`` only if ``label_day <= t - d``.
    This is the ONLY place that encodes the rule; use it everywhere.
    """
    return df[day_col] <= (t - d)


def matured_cutoff(t: int, d: int) -> int:
    """Last day whose labels are matured at time ``t`` with delay ``d``."""
    return t - d


def make_windows(start: int, end: int, size: int = 7) -> List[Window]:
    """Consecutive non-overlapping windows of ``size`` days within [start, end] (inclusive).

    Incomplete trailing windows are dropped.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    return [(lo, lo + size - 1) for lo in range(start, end - size + 2, size)]


def window_mask(df: pd.DataFrame, window: Window, day_col: str = COL_DAY) -> pd.Series:
    """Mask of rows whose day lies in the inclusive ``window``."""
    lo, hi = window
    return (df[day_col] >= lo) & (df[day_col] <= hi)


def week_index(window: Window, size: int = 7) -> int:
    """Week number (day // 7) of a window start; week 10 starts on day 70."""
    return window[0] // 7


def build_splits(
    df: pd.DataFrame,
    delay: int,
    drift_onset: Optional[int] = None,
    day_col: str = COL_DAY,
) -> dict:
    """Return train / gap / test / reference row masks plus the ``SplitConfig``.

    Train = days 35..62, gap = days up to test_start-1, test = test_start..181,
    reference = 21 days from test_start (trusted, pre-drift).
    """
    cfg: SplitConfig = build_split_config(delay, drift_onset)
    day = df[day_col]
    return {
        "config": cfg,
        "train": (day >= cfg.train_start) & (day <= cfg.train_end),
        "gap": (day >= cfg.gap_start) & (day <= cfg.gap_end),
        "test": (day >= cfg.test_start) & (day <= cfg.test_end),
        "reference": (day >= cfg.reference_start) & (day <= cfg.reference_end),
    }
