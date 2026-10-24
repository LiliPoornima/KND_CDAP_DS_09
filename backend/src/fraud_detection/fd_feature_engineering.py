"""Fraud-detection feature engineering (leakage-safe, delay-aware).

Model features: TX_AMOUNT, hour, dow, amt_ratio, t_risk, t_vol and, for w in (1D, 7D, 30D),
c_mean_w and c_cnt_w.

* Customer features use only the PREVIOUS ``w`` of the same customer (``closed='left'``),
  so the current transaction is never included. They use amounts only, no labels.
* Terminal risk uses LABELS, so it is delayed: for a transaction on day D the statistics are
  computed from labels up to day ``D - delay`` (cumulative-sum difference over a 28-day span),
  which is exactly the ``label_day <= t - d`` rule of ``fd_preprocessing.matured_mask``.

After ANY drift injection (amounts, times or labels change) ALL features must be rebuilt by
calling :func:`build_features` again on the modified frame.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .fd_config import (
    COL_AMOUNT, COL_CUSTOMER, COL_DAY, COL_DATETIME, COL_FRAUD, COL_TERMINAL,
    CUSTOMER_WINDOWS, MODEL_FEATURES, TERMINAL_RISK_WINDOW, get_logger,
)

logger = get_logger(__name__)


def add_customer_features(
    df: pd.DataFrame, windows: Sequence[str] = CUSTOMER_WINDOWS
) -> pd.DataFrame:
    """Add rolling customer mean/count over the previous window (current row excluded).

    Returns a new frame; row order of ``df`` is preserved.
    """
    n = len(df)
    tmp = pd.DataFrame({
        COL_CUSTOMER: df[COL_CUSTOMER].to_numpy(),
        COL_DATETIME: df[COL_DATETIME].to_numpy(),
        COL_AMOUNT: df[COL_AMOUNT].to_numpy(),
        "pos": np.arange(n),
    }).sort_values([COL_CUSTOMER, COL_DATETIME], kind="stable")
    pos = tmp["pos"].to_numpy()
    indexed = tmp.set_index(COL_DATETIME)
    out = df.copy()
    for w in windows:
        roll = indexed.groupby(COL_CUSTOMER)[COL_AMOUNT].rolling(w, closed="left")
        mean = np.full(n, np.nan, dtype=np.float32)
        cnt = np.zeros(n, dtype=np.float32)
        mean[pos] = roll.mean().to_numpy()
        cnt[pos] = np.nan_to_num(roll.count().to_numpy(), nan=0.0)
        out[f"c_mean_{w}"] = mean
        out[f"c_cnt_{w}"] = cnt
    return out


def add_terminal_features(
    df: pd.DataFrame, delay: int, window: int = TERMINAL_RISK_WINDOW
) -> pd.DataFrame:
    """Add delayed terminal risk (``t_risk``) and volume (``t_vol``).

    For a transaction on day D the statistics cover the ``window`` days ending at day
    ``D - delay`` (labels of those days have matured by day D).
    """
    if delay < 0:
        raise ValueError("delay must be non-negative")
    days = df[COL_DAY].to_numpy().astype(np.int64)
    term = df[COL_TERMINAL].to_numpy().astype(np.int64)
    fraud = df[COL_FRAUD].to_numpy()
    nd, nt = int(days.max()) + 1, int(term.max()) + 1

    tx = np.zeros((nt, nd))
    fr = np.zeros((nt, nd))
    np.add.at(tx, (term, days), 1)
    np.add.at(fr, (term, days), fraud)
    ctx, cfr = tx.cumsum(1), fr.cumsum(1)

    def cum(a: np.ndarray, d: int) -> np.ndarray:
        return a[:, d] if d >= 0 else np.zeros(a.shape[0])

    risk = np.zeros((nt, nd), dtype=np.float32)
    vol = np.zeros((nt, nd), dtype=np.float32)
    for D in range(nd):  # loop over days only (<= 183 iterations), vectorised over terminals
        hi, lo = D - delay, D - delay - window
        f = cum(cfr, hi) - cum(cfr, lo)
        t = cum(ctx, hi) - cum(ctx, lo)
        vol[:, D] = t
        risk[:, D] = np.where(t > 0, f / np.maximum(t, 1), 0)

    out = df.copy()
    out["t_risk"] = risk[term, days]
    out["t_vol"] = vol[term, days]
    return out


def build_features(
    df: pd.DataFrame, delay: int, terminal_window: int = TERMINAL_RISK_WINDOW
) -> pd.DataFrame:
    """Build ALL model features for a (possibly drift-injected) frame.

    ``df`` needs TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, TX_TIME_DAYS, TX_FRAUD.
    The input is not modified; a new frame with feature columns is returned.
    """
    out = df.reset_index(drop=True)
    out = add_customer_features(out)
    out["hour"] = out[COL_DATETIME].dt.hour.astype(np.float32)
    out["dow"] = out[COL_DATETIME].dt.day_of_week.astype(np.float32)
    amt = out[COL_AMOUNT].to_numpy(dtype=np.float64)
    c30 = out["c_mean_30D"].to_numpy(dtype=np.float64)
    out["amt_ratio"] = (amt / (np.where(np.isnan(c30), amt, c30) + 1e-6)).astype(np.float32)
    out = add_terminal_features(out, delay, terminal_window)
    out[COL_AMOUNT] = out[COL_AMOUNT].astype(np.float32)
    missing = [c for c in MODEL_FEATURES if c not in out.columns]
    if missing:  # pragma: no cover - defensive
        raise RuntimeError(f"Feature build incomplete, missing: {missing}")
    return out
