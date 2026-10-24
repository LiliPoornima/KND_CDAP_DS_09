"""Extended leakage-safe features for fraud detection.

Builds upon fd_feature_engineering.py with additional domain-informed, leakage-safe features:
1. c_std_30D: Rolling standard deviation of customer transaction amount (past 30D, closed='left').
2. c_zscore_30D: Outlier score (TX_AMOUNT - c_mean_30D) / (c_std_30D + 1e-4).
3. c_cnt_1h: Short-term transaction burst count in the past 1 hour (closed='left').
4. c_amt_diff_30D: Absolute deviation TX_AMOUNT - c_mean_30D.
5. t_risk_smooth: Empirical Bayes smoothed terminal risk (using only matured labels).
6. amt_log: Log-transformed amount log1p(TX_AMOUNT).

All customer features strictly exclude the current row (closed='left').
All terminal features strictly observe the delay rule: label_day <= t - delay.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fd_config import (
    COL_AMOUNT,
    COL_CUSTOMER,
    COL_DATETIME,
    COL_DAY,
    COL_FRAUD,
    COL_TERMINAL,
    CUSTOMER_WINDOWS,
    MODEL_FEATURES,
    PROCESSED_DIR,
    TERMINAL_RISK_WINDOW,
    TRAIN_END,
    TRAIN_START,
    get_logger,
)
from . import fd_feature_engineering
from .fd_feature_engineering import build_features

logger = get_logger(__name__)

EXTENDED_FEATURE_NAMES: Tuple[str, ...] = (
    "c_std_30D",
    "c_zscore_30D",
    "c_cnt_1h",
    "c_amt_diff_30D",
    "t_risk_smooth",
    "amt_log",
)

ALL_EXTENDED_FEATURES: Tuple[str, ...] = MODEL_FEATURES + EXTENDED_FEATURE_NAMES


def add_extended_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling std, z-score, burst count and amount difference (closed='left')."""
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

    # 1. 30D rolling std
    roll_30d = indexed.groupby(COL_CUSTOMER)[COL_AMOUNT].rolling("30D", closed="left")
    std_30d = np.zeros(n, dtype=np.float32)
    std_30d[pos] = np.nan_to_num(roll_30d.std().to_numpy(), nan=0.0)
    out["c_std_30D"] = std_30d

    # 2. 1h rolling count (burst card testing)
    roll_1h = indexed.groupby(COL_CUSTOMER)[COL_AMOUNT].rolling("1h", closed="left")
    cnt_1h = np.zeros(n, dtype=np.float32)
    cnt_1h[pos] = np.nan_to_num(roll_1h.count().to_numpy(), nan=0.0)
    out["c_cnt_1h"] = cnt_1h

    # 3. Deviation & z-score vs customer 30D mean
    amt = out[COL_AMOUNT].to_numpy(dtype=np.float64)
    mean_30d = out["c_mean_30D"].to_numpy(dtype=np.float64)
    c_mean_safe = np.where(np.isnan(mean_30d), amt, mean_30d)
    diff = amt - c_mean_safe
    out["c_amt_diff_30D"] = diff.astype(np.float32)

    std_safe = np.where(std_30d > 1e-4, std_30d, 1.0)
    zscore = np.where(np.isnan(mean_30d), 0.0, diff / std_safe)
    out["c_zscore_30D"] = np.clip(zscore, -5.0, 20.0).astype(np.float32)

    # 4. Log amount
    out["amt_log"] = np.log1p(np.maximum(amt, 0.0)).astype(np.float32)

    return out


def compute_expanding_delay_aware_prior(
    df: pd.DataFrame,
    delay: int,
    train_end: int = TRAIN_END,
) -> np.ndarray:
    """Compute strictly causal, delay-aware historical fraud prior for each day.

    For historical decision time D:
    - If D - delay < 0 (early warmup before any matured labels): uses the first observed
      valid matured rate or 0.005 default.
    - If D <= train_end: uses strictly matured transactions on day <= D - delay.
    - If D > train_end (validation and test period): frozen at (train_end - delay) so
      that validation and test labels are NEVER read.

    Guarantees:
    - Earlier training rows never receive later training labels (zero intra-training lookahead).
    - Validation and test labels are strictly excluded.
    """
    days = df[COL_DAY].to_numpy().astype(np.int64)
    fraud = df[COL_FRAUD].to_numpy()
    if len(days) == 0:
        return np.empty(0, dtype=np.float64)

    nd = int(days.max()) + 1
    daily_tx = np.zeros(nd)
    daily_fr = np.zeros(nd)
    np.add.at(daily_tx, days, 1)
    np.add.at(daily_fr, days, fraud)

    ctx = daily_tx.cumsum()
    cfr = daily_fr.cumsum()
    max_matured = max(0, train_end - delay)

    valid_idx = np.where(ctx > 0)[0]
    if len(valid_idx) > 0 and ctx[valid_idx[0]] > 0:
        first_valid = float(cfr[valid_idx[0]] / ctx[valid_idx[0]])
        if first_valid <= 0.0 and cfr[valid_idx[-1]] > 0:
            first_valid = float(cfr[valid_idx[-1]] / ctx[valid_idx[-1]])
    else:
        first_valid = 0.005
    if first_valid <= 0.0:
        first_valid = 0.005

    priors = np.zeros(nd, dtype=np.float64)
    for D in range(nd):
        m = D - delay
        if m < 0:
            priors[D] = first_valid
        else:
            eff = min(m, max_matured)
            rate = float(cfr[eff] / ctx[eff]) if ctx[eff] > 0 else first_valid
            priors[D] = rate if rate > 0.0 else first_valid
    return priors


compute_delay_aware_prior = compute_expanding_delay_aware_prior


def compute_pretraining_warmup_prior(
    df: pd.DataFrame,
    delay: int = 7,
    train_start: int = TRAIN_START,
) -> float:
    """Estimate empirical fraud prior strictly on matured pre-training transactions.

    Uses transactions on day <= (train_start - delay).
    At train_start (Day 35), these labels are already matured and known before
    any training transaction is observed. Completely independent of all training,
    validation, and test labels.
    """
    cutoff = train_start - delay
    matured = df[df[COL_DAY] <= cutoff]
    if len(matured) == 0:
        raise ValueError(f"No matured transactions found on day <= {cutoff}")
    rate = float(matured[COL_FRAUD].mean())
    if rate <= 0.0 or rate >= 1.0:
        raise ValueError(f"Invalid pre-training prior rate: {rate}")
    return rate


def compute_training_prior(
    df: pd.DataFrame,
    delay: int = 7,
    train_start: int = TRAIN_START,
    train_end: int = TRAIN_END,
    mode: str = "expanding",
) -> Union[np.ndarray, float]:
    """Estimate empirical fraud prior strictly without lookahead or test leakage.

    Modes:
    - 'expanding' (default): Returns daily causal prior array (nd,) where each day D
      uses only labels matured by D - delay, frozen at train_end - delay.
    - 'warmup': Returns scalar prior estimated strictly from matured transactions
      prior to train_start (day <= train_start - delay).
    - 'training_window': Legacy mean on [train_start, train_end].
    """
    if mode == "expanding":
        return compute_expanding_delay_aware_prior(df, delay=delay, train_end=train_end)
    elif mode == "warmup":
        return compute_pretraining_warmup_prior(df, delay=delay, train_start=train_start)
    elif mode == "training_window":
        train_mask = (df[COL_DAY] >= train_start) & (df[COL_DAY] <= train_end)
        train_df = df[train_mask]
        if len(train_df) == 0:
            raise ValueError(f"No transactions found in training window [{train_start}, {train_end}]")
        return float(train_df[COL_FRAUD].mean())
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'expanding', 'warmup', or 'training_window'.")


def add_smoothed_terminal_risk(
    df: pd.DataFrame,
    delay: int,
    prior_fraud_rate: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
    window: int = TERMINAL_RISK_WINDOW,
    prior_weight: float = 10.0,
    train_end: int = TRAIN_END,
) -> pd.DataFrame:
    """Add empirical Bayes smoothed terminal risk using strictly matured labels and causal prior.

    Formula: (fraud_count + prior_D * prior_weight) / (tx_count + prior_weight)

    If prior_fraud_rate is None, compute_expanding_delay_aware_prior() is used, ensuring:
    - Earlier training rows never receive information from later training labels.
    - Validation and test labels are frozen at train_end - delay and never read.
    """
    if prior_fraud_rate is None:
        prior_fraud_rate = compute_expanding_delay_aware_prior(df, delay=delay, train_end=train_end)

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

    smooth_risk = np.zeros((nt, nd), dtype=np.float32)
    is_array_prior = isinstance(prior_fraud_rate, (np.ndarray, Sequence, pd.Series))

    if is_array_prior:
        p_arr = np.asarray(prior_fraud_rate, dtype=np.float64)
        if len(p_arr) < nd:
            raise ValueError(f"prior array length {len(p_arr)} < max day + 1 ({nd})")
        if (p_arr <= 0.0).any() or (p_arr >= 1.0).any():
            raise ValueError("All elements in prior_fraud_rate array must be in (0, 1)")
    else:
        p_val = float(prior_fraud_rate)
        if p_val <= 0.0 or p_val >= 1.0:
            raise ValueError(f"prior_fraud_rate must be in (0, 1), got {p_val}")

    for D in range(nd):
        hi, lo = D - delay, D - delay - window
        f = cum(cfr, hi) - cum(cfr, lo)
        t = cum(ctx, hi) - cum(ctx, lo)
        p = p_arr[D] if is_array_prior else p_val
        smooth_risk[:, D] = (f + p * prior_weight) / (t + prior_weight)

    out = df.copy()
    out["t_risk_smooth"] = smooth_risk[term, days]
    return out


def build_extended_features(
    df: pd.DataFrame,
    delay: int,
    prior_fraud_rate: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
    terminal_window: int = TERMINAL_RISK_WINDOW,
    train_end: int = TRAIN_END,
) -> pd.DataFrame:
    """Build base 12 features + 6 extended features for df.

    If prior_fraud_rate is not provided, the strictly causal expanding delay-aware
    prior is used by default.
    """
    base_df = build_features(df, delay, terminal_window)
    out = add_extended_customer_features(base_df)
    out = add_smoothed_terminal_risk(
        out,
        delay,
        prior_fraud_rate=prior_fraud_rate,
        window=terminal_window,
        train_end=train_end,
    )
    return out


def extended_features_fingerprint(
    df: pd.DataFrame,
    delay: int,
    prior_fraud_rate: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
    terminal_window: int = TERMINAL_RISK_WINDOW,
    base_module_file: Optional[Path] = None,
) -> str:
    """Fingerprint covering raw inputs, parameters, prior rate, and the code of both feature files."""
    h = hashlib.sha1()
    for col in (COL_CUSTOMER, COL_TERMINAL, COL_AMOUNT, COL_DAY, COL_FRAUD):
        h.update(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    h.update(np.ascontiguousarray(df[COL_DATETIME].to_numpy().astype("datetime64[ns]")).tobytes())
    if prior_fraud_rate is None or isinstance(prior_fraud_rate, (np.ndarray, Sequence, pd.Series)):
        p_arr = (
            compute_expanding_delay_aware_prior(df, delay)
            if prior_fraud_rate is None
            else np.asarray(prior_fraud_rate, dtype=np.float64)
        )
        h.update(f"|expanding_prior|{len(p_arr)}|".encode())
        h.update(np.ascontiguousarray(p_arr).tobytes())
    else:
        h.update(f"|scalar_prior|{float(prior_fraud_rate):.8f}|".encode())
    h.update(f"|extended|{len(df)}|{delay}|{terminal_window}|".encode())
    h.update(Path(__file__).read_bytes())

    # Include base feature engineering source code so changes invalidate the cache
    base_path = Path(base_module_file) if base_module_file else Path(fd_feature_engineering.__file__)
    if base_path.is_file():
        h.update(base_path.read_bytes())
    return h.hexdigest()[:16]


def load_or_build_extended_features(
    df: pd.DataFrame,
    delay: int,
    prior_fraud_rate: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
    cache_dir: Optional[Path] = None,
    tag: str = "extended",
    terminal_window: int = TERMINAL_RISK_WINDOW,
) -> pd.DataFrame:
    """Return extended features with content-fingerprinted parquet caching."""
    cache_dir = PROCESSED_DIR if cache_dir is None else Path(cache_dir)
    fp = extended_features_fingerprint(df, delay, prior_fraud_rate, terminal_window)
    path = cache_dir / f"features_{tag}_d{delay}_{fp}.parquet"
    if path.is_file():
        logger.info("Loading cached extended features %s", path.name)
        return pd.read_parquet(path)
    out = build_extended_features(
        df, delay, prior_fraud_rate=prior_fraud_rate, terminal_window=terminal_window
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info("Built and cached extended features %s", path.name)
    return out
