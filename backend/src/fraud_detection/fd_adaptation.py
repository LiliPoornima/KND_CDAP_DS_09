"""Attribution-Guided Adaptation (AGA) (Spec Section 11).

When an adaptation is triggered at decision time t with label delay d:
1. Determine D = top-k features by |delta SHAP| (or PSI / Random in ablations).
2. Build candidate models using ONLY matured labels (label_day <= t - d) over the last 28 matured days:
   (a) FULL: retrain on all features.
   (b) DROP_D: retrain without features in D.
   (c) RECENCY: retrain on all features with exponentially increasing sample weights for recent days.
3. Select the candidate with the highest Recall@1%FPR on the most recent 7 matured days
   (held out from candidate training).
4. Return chosen model + record (candidate name, D, training/validation windows, validation score).

LEAKAGE ENFORCEMENT: All training rows and validation rows strictly satisfy label_day <= t - d.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fd_config import COL_DAY, COL_FRAUD, MODEL_FEATURES, MODEL_PARAMS, ModelParams, get_logger
from .fd_evaluation import recall_at_fpr
from .fd_preprocessing import matured_cutoff
from .fd_training import predict_scores, train_model

logger = get_logger(__name__)

CANDIDATE_FULL = "FULL"
CANDIDATE_DROP_D = "DROP_D"
CANDIDATE_RECENCY = "RECENCY"
CANDIDATES: Tuple[str, ...] = (CANDIDATE_FULL, CANDIDATE_DROP_D, CANDIDATE_RECENCY)


@dataclass
class AdaptationResult:
    """Record of an Attribution-Guided Adaptation execution."""

    decision_day: int
    matured_cutoff: int
    train_start_day: int
    train_end_day: int
    val_start_day: int
    val_end_day: int
    feature_set_D: List[str]
    selection_mode: str
    winning_candidate: str
    validation_scores: Dict[str, float]
    selected_model: Any
    features_used: List[str]


def select_top_k_features(
    deltas: Dict[str, float],
    k: int = 3,
    mode: str = "shap",
    seed: int = 42,
    all_features: Sequence[str] = MODEL_FEATURES,
) -> List[str]:
    """Select top-k features for D by |delta SHAP|, PSI, or random sampling."""
    feat_list = list(all_features)
    k = min(k, len(feat_list))

    if mode.lower() == "shap":
        # Extract delta values (supports keys like 'delta_<feat>' or raw '<feat>')
        clean_deltas = {}
        for f in feat_list:
            clean_deltas[f] = abs(deltas.get(f"delta_{f}", deltas.get(f, 0.0)))
        sorted_feats = sorted(clean_deltas.keys(), key=lambda f: clean_deltas[f], reverse=True)
        return sorted_feats[:k]

    elif mode.lower() == "psi":
        clean_psi = {}
        for f in feat_list:
            clean_psi[f] = abs(deltas.get(f"psi_{f}", deltas.get(f, 0.0)))
        sorted_feats = sorted(clean_psi.keys(), key=lambda f: clean_psi[f], reverse=True)
        return sorted_feats[:k]

    elif mode.lower() == "random":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(feat_list, size=k, replace=False)
        return list(chosen)

    else:
        raise ValueError(f"Unknown feature selection mode '{mode}'. Expected: 'shap', 'psi', 'random'")


def build_matured_slices(
    df: pd.DataFrame,
    decision_day: int,
    delay: int,
    matured_span_days: int = 28,
    val_holdout_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, int, int]:
    """Slice the last 28 matured days into training (21 days) and holdout validation (7 days).

    Strictly satisfies: all rows in df_train and df_val have label_day <= decision_day - delay.
    """
    cutoff = matured_cutoff(decision_day, delay)
    val_end = cutoff
    val_start = cutoff - val_holdout_days + 1
    train_end = val_start - 1
    train_start = cutoff - matured_span_days + 1

    train_mask = (df[COL_DAY] >= train_start) & (df[COL_DAY] <= train_end)
    val_mask = (df[COL_DAY] >= val_start) & (df[COL_DAY] <= val_end)

    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)

    if len(df_train) == 0:
        raise ValueError(f"No training rows available for days {train_start}..{train_end}")
    if len(df_val) == 0:
        raise ValueError(f"No validation rows available for days {val_start}..{val_end}")

    return df_train, df_val, train_start, train_end, val_start, val_end


def compute_recency_weights(
    days: np.ndarray,
    half_life_days: float = 7.0,
) -> np.ndarray:
    """Calculate exponential recency sample weights for recent days."""
    max_d = float(np.max(days))
    decay_rate = np.log(2.0) / half_life_days
    weights = np.exp(-(max_d - days) * decay_rate)
    # Normalize to mean weight 1.0
    return weights / np.mean(weights)


def train_candidates(
    df_train: pd.DataFrame,
    D: Sequence[str],
    seed: int = 1,
    all_features: Sequence[str] = MODEL_FEATURES,
    params: ModelParams = MODEL_PARAMS,
    recency_half_life: float = 7.0,
) -> Dict[str, Tuple[Any, List[str]]]:
    """Train the 3 candidate models: FULL, DROP_D, and RECENCY."""
    all_feats = list(all_features)
    drop_feats = [f for f in all_feats if f not in D]
    if not drop_feats:
        # Fallback to all features if all were dropped
        drop_feats = all_feats

    # Candidate (a): FULL
    m_full = train_model(df_train, seed=seed, features=all_feats, params=params)

    # Candidate (b): DROP_D
    m_drop = train_model(df_train, seed=seed, features=drop_feats, params=params)

    # Candidate (c): RECENCY
    weights = compute_recency_weights(df_train[COL_DAY].to_numpy(), half_life_days=recency_half_life)
    m_rec = train_model(df_train, seed=seed, features=all_feats, sample_weight=weights, params=params)

    return {
        CANDIDATE_FULL: (m_full, all_feats),
        CANDIDATE_DROP_D: (m_drop, drop_feats),
        CANDIDATE_RECENCY: (m_rec, all_feats),
    }


def evaluate_candidates(
    candidates: Dict[str, Tuple[Any, List[str]]],
    df_val: pd.DataFrame,
    fpr: float = 0.01,
) -> Tuple[str, Dict[str, float]]:
    """Evaluate candidates on holdout matured validation set and select the winner."""
    y_val = df_val[COL_FRAUD].to_numpy()
    val_scores: Dict[str, float] = {}

    for name, (model, _) in candidates.items():
        p_val = predict_scores(model, df_val)
        rec = recall_at_fpr(y_val, p_val, fpr=fpr)
        val_scores[name] = float(rec) if np.isfinite(rec) else 0.0

    # Winner is candidate with highest validation recall@1%FPR
    winner = max(val_scores.keys(), key=lambda k: val_scores[k])
    return winner, val_scores


def run_aga(
    df: pd.DataFrame,
    decision_day: int,
    delay: int,
    deltas: Dict[str, float],
    k: int = 3,
    selection_mode: str = "shap",
    seed: int = 1,
    params: ModelParams = MODEL_PARAMS,
    forced_candidate: Optional[str] = None,
) -> AdaptationResult:
    """Run full Attribution-Guided Adaptation (AGA) procedure.

    Returns the selected winning model and complete audit record.
    If ``forced_candidate`` is specified, deploys that candidate directly (for P1-a, P1-b, P1-c).
    """
    cutoff = matured_cutoff(decision_day, delay)
    D = select_top_k_features(deltas, k=k, mode=selection_mode, seed=seed)

    df_train, df_val, tr_start, tr_end, val_start, val_end = build_matured_slices(
        df=df,
        decision_day=decision_day,
        delay=delay,
    )

    candidates = train_candidates(
        df_train=df_train,
        D=D,
        seed=seed,
        params=params,
    )

    winner, val_scores = evaluate_candidates(candidates, df_val)

    # Allow policy override for standalone variants (P1-a, P1-b, P1-c)
    if forced_candidate is not None:
        chosen_candidate = forced_candidate
    else:
        chosen_candidate = winner

    selected_model, features_used = candidates[chosen_candidate]

    logger.info(
        "AGA completed at day %d (cutoff %d): Selected '%s' (val recall: %.4f) [D: %s]",
        decision_day,
        cutoff,
        chosen_candidate,
        val_scores.get(chosen_candidate, 0.0),
        ", ".join(D),
    )

    return AdaptationResult(
        decision_day=decision_day,
        matured_cutoff=cutoff,
        train_start_day=tr_start,
        train_end_day=tr_end,
        val_start_day=val_start,
        val_end_day=val_end,
        feature_set_D=D,
        selection_mode=selection_mode,
        winning_candidate=chosen_candidate,
        validation_scores=val_scores,
        selected_model=selected_model,
        features_used=features_used,
    )
