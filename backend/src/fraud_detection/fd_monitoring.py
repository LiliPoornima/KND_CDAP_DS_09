"""Fraud-detection label-free monitoring signals (Spec Section 9).

Computed per window against a trusted REFERENCE window without labels:
- PSI per feature: 10 quantile bins from the reference, clip window values to reference range,
  fill NaN with -1, epsilon 1e-6 in bins. Reports mean PSI and max PSI.
- Score JS: scipy.spatial.distance.jensenshannon between binned model-score histograms
  (bins = reference quantiles).
- Entropy: mean binary entropy of predicted probabilities (clip 1e-6); relative change vs reference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .fd_config import MODEL_FEATURES, get_logger

logger = get_logger(__name__)


def calculate_feature_psi(
    ref_vals: np.ndarray,
    win_vals: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Calculate Population Stability Index (PSI) for a single feature.

    1. Derive 10 quantile bins from the reference data.
    2. Fill NaN with -1.0.
    3. Clip window values to reference range [min, max].
    4. Apply epsilon smoothing (1e-6) to bin proportions.
    5. PSI = sum((win_p - ref_p) * ln(win_p / ref_p)).
    """
    ref_arr = np.asarray(ref_vals, dtype=np.float64)
    win_arr = np.asarray(win_vals, dtype=np.float64)

    ref_clean = np.nan_to_num(ref_arr, nan=-1.0)
    win_clean = np.nan_to_num(win_arr, nan=-1.0)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(ref_clean, quantiles)

    # Collapse duplicate edges if data has low cardinality
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 2:
        return 0.0
    bin_edges = unique_edges

    # Clip window values to reference range
    win_clipped = np.clip(win_clean, bin_edges[0], bin_edges[-1])

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    win_counts, _ = np.histogram(win_clipped, bins=bin_edges)

    ref_dist = ref_counts / max(len(ref_clean), 1)
    win_dist = win_counts / max(len(win_clean), 1)

    # Apply epsilon smoothing
    ref_dist = np.maximum(ref_dist, eps)
    win_dist = np.maximum(win_dist, eps)

    # PSI calculation
    psi = np.sum((win_dist - ref_dist) * np.log(win_dist / ref_dist))
    return float(max(psi, 0.0))


def calculate_features_psi(
    ref_df: pd.DataFrame,
    win_df: pd.DataFrame,
    features: Sequence[str] = MODEL_FEATURES,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Calculate PSI for each feature, returning per-feature PSI, mean_psi, and max_psi."""
    results: Dict[str, float] = {}
    psi_values = []

    for feat in features:
        if feat not in ref_df.columns or feat not in win_df.columns:
            logger.warning("Feature '%s' missing from data for PSI calculation", feat)
            continue
        psi_val = calculate_feature_psi(
            ref_df[feat].to_numpy(),
            win_df[feat].to_numpy(),
            n_bins=n_bins,
            eps=eps,
        )
        results[f"psi_{feat}"] = psi_val
        psi_values.append(psi_val)

    if psi_values:
        results["mean_psi"] = float(np.mean(psi_values))
        results["max_psi"] = float(np.max(psi_values))
    else:
        results["mean_psi"] = 0.0
        results["max_psi"] = 0.0

    return results


def calculate_score_js(
    ref_scores: np.ndarray,
    win_scores: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Calculate Jensen-Shannon distance between model score histograms.

    Bins are derived from reference score quantiles, window scores are clipped to
    reference range, and smoothed with epsilon.
    """
    s_ref = np.asarray(ref_scores, dtype=np.float64)
    s_win = np.asarray(win_scores, dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(s_ref, quantiles))
    if len(bin_edges) < 2:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    win_clipped = np.clip(s_win, bin_edges[0], bin_edges[-1])

    ref_counts, _ = np.histogram(s_ref, bins=bin_edges)
    win_counts, _ = np.histogram(win_clipped, bins=bin_edges)

    ref_p = (ref_counts + eps) / (ref_counts.sum() + len(ref_counts) * eps)
    win_p = (win_counts + eps) / (win_counts.sum() + len(win_counts) * eps)

    js_dist = jensenshannon(ref_p, win_p)
    return float(js_dist)


def calculate_binary_entropy(
    scores: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Mean binary Shannon entropy of predicted probabilities.

    H(p) = - [p * ln(p) + (1 - p) * ln(1 - p)], with p clipped to [eps, 1 - eps].
    """
    p = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    entropy = - (p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(entropy))


def calculate_entropy_relative_change(
    ref_scores: np.ndarray,
    win_scores: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate relative entropy change of window probabilities vs reference.

    rel_change = (mean_entropy_win - mean_entropy_ref) / mean_entropy_ref.
    """
    ref_ent = calculate_binary_entropy(ref_scores, eps=eps)
    win_ent = calculate_binary_entropy(win_scores, eps=eps)
    if ref_ent <= 0.0:
        return 0.0
    return float((win_ent - ref_ent) / ref_ent)


def compute_monitoring_signals(
    ref_df: pd.DataFrame,
    ref_scores: np.ndarray,
    win_df: pd.DataFrame,
    win_scores: np.ndarray,
    features: Sequence[str] = MODEL_FEATURES,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute all label-free monitoring signals for a given window against reference."""
    psi_dict = calculate_features_psi(ref_df, win_df, features=features, n_bins=n_bins, eps=eps)
    score_js = calculate_score_js(ref_scores, win_scores, n_bins=n_bins, eps=eps)
    win_entropy = calculate_binary_entropy(win_scores, eps=eps)
    ref_entropy = calculate_binary_entropy(ref_scores, eps=eps)
    rel_entropy = (win_entropy - ref_entropy) / ref_entropy if ref_entropy > 0 else 0.0

    signals: Dict[str, float] = {
        "mean_psi": psi_dict["mean_psi"],
        "max_psi": psi_dict["max_psi"],
        "score_js": score_js,
        "entropy_mean": win_entropy,
        "entropy_ref": ref_entropy,
        "entropy_rel_change": rel_entropy,
    }
    # Include individual feature PSIs
    signals.update(psi_dict)
    return signals


def run_monitoring_experiment(
    delay: int = 7,
    onset_day: int = 105,
    seed: int = 1,
    output_dir: Optional[Any] = None,
) -> Dict[str, pd.DataFrame]:
    """Run Phase 4 weekly monitoring signals and SHAP explainability across all scenarios.

    Computes weekly PSI, Score JS, Entropy, and TreeSHAP shifts vs trusted reference.
    Evaluates the verification alarm rule (exceeding Control max over weeks 15-25).
    Saves results to outputs/fraud_detection/signals/.
    """
    from pathlib import Path
    from .fd_config import (
        OUTPUT_DIR,
        SCENARIO_CONTROL,
        SCENARIO_S1,
        SCENARIO_S2,
        SCENARIO_S3,
    )
    from .fd_data_loader import load_handbook
    from .fd_drift_injection import inject_drift
    from .fd_explainability import build_tree_explainer, calculate_shap_profile, compute_explainability_signals
    from .fd_feature_engineering import build_features
    from .fd_preprocessing import build_splits, make_windows
    from .fd_training import predict_scores, train_model

    target_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "signals"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Handbook data for monitoring experiment...")
    raw = load_handbook()

    # Train base model on control stream (days 35..62)
    logger.info("Training base LightGBM model on days 35..62 (delay=%d, seed=%d)...", delay, seed)
    ctrl_raw = inject_drift(raw, SCENARIO_CONTROL, onset_day=onset_day, seed=seed)
    ctrl_feat = build_features(ctrl_raw, delay=delay)
    sp = build_splits(ctrl_feat, delay=delay)
    cfg = sp["config"]
    model = train_model(ctrl_feat[sp["train"]], seed=seed)
    explainer = build_tree_explainer(model)

    # Establish trusted reference window (days 70..90)
    test_ctrl = ctrl_feat[sp["test"]].reset_index(drop=True)
    ref_mask = (test_ctrl.TX_TIME_DAYS >= cfg.reference_start) & (test_ctrl.TX_TIME_DAYS <= cfg.reference_end)
    ref_df = test_ctrl[ref_mask].reset_index(drop=True)
    ref_scores = predict_scores(model, ref_df)
    ref_profile = calculate_shap_profile(explainer, ref_df, seed=42)

    logger.info(
        "Reference established on days %d..%d (%d rows, %d features)",
        cfg.reference_start,
        cfg.reference_end,
        len(ref_df),
        len(MODEL_FEATURES),
    )

    windows = list(make_windows(cfg.test_start, cfg.test_end, 7))
    all_scenario_dfs: Dict[str, pd.DataFrame] = {}

    for sc in (SCENARIO_CONTROL, SCENARIO_S1, SCENARIO_S2, SCENARIO_S3):
        logger.info("Computing monitoring signals for scenario: %s", sc)
        injected = inject_drift(raw, sc, onset_day=onset_day, seed=seed)
        feat = build_features(injected, delay=delay)
        test = feat[sp["test"]].reset_index(drop=True)
        scores = predict_scores(model, test)

        sc_rows = []
        for win in windows:
            w_idx = win[0] // 7
            wm = (test.TX_TIME_DAYS >= win[0]) & (test.TX_TIME_DAYS <= win[1])
            w_df = test[wm].reset_index(drop=True)
            w_scores = scores[wm]

            # Label-free monitoring signals
            mon_signals = compute_monitoring_signals(
                ref_df=ref_df,
                ref_scores=ref_scores,
                win_df=w_df,
                win_scores=w_scores,
                features=MODEL_FEATURES,
            )

            # SHAP explainability signals
            exp_signals = compute_explainability_signals(
                explainer=explainer,
                win_df=w_df,
                ref_profile=ref_profile,
                features=MODEL_FEATURES,
                max_samples=10000,
                seed=42,
            )

            row = {
                "week": w_idx,
                "day_start": win[0],
                "day_end": win[1],
                "scenario": sc,
                "delay": delay,
                "seed": seed,
            }
            row.update(mon_signals)
            row["shap_shift"] = exp_signals["shap_shift"]
            for f in MODEL_FEATURES:
                row[f"delta_{f}"] = exp_signals[f"delta_{f}"]

            sc_rows.append(row)

        df_sc = pd.DataFrame(sc_rows)
        df_sc.to_csv(target_dir / f"weekly_signals_{sc.lower()}_d{delay}_seed{seed}.csv", index=False)
        all_scenario_dfs[sc] = df_sc

    # Evaluate alarm thresholds based on Control post-onset weeks 15-25
    ctrl_df = all_scenario_dfs[SCENARIO_CONTROL]
    ctrl_post = ctrl_df[ctrl_df.week >= 15]

    thresholds = {
        "mean_psi": float(ctrl_post["mean_psi"].max()),
        "max_psi": float(ctrl_post["max_psi"].max()),
        "score_js": float(ctrl_post["score_js"].max()),
        "entropy_rel_change": float(ctrl_post["entropy_rel_change"].abs().max()),
        "shap_shift": float(ctrl_post["shap_shift"].max()),
    }

    alarm_records = []
    for sc in (SCENARIO_CONTROL, SCENARIO_S1, SCENARIO_S2, SCENARIO_S3):
        sc_df = all_scenario_dfs[sc]
        post_df = sc_df[sc_df.week >= 15]
        rec: Dict[str, Any] = {"scenario": sc}

        for sig in ("mean_psi", "max_psi", "score_js", "shap_shift"):
            thr = thresholds[sig]
            alarm_rows = post_df[post_df[sig] > thr]
            rec[f"alarm_week_{sig}"] = int(alarm_rows["week"].iloc[0]) if len(alarm_rows) > 0 else "None"

        # Entropy relative change alarm (direction-independent magnitude)
        ent_thr = thresholds["entropy_rel_change"]
        ent_rows = post_df[post_df["entropy_rel_change"].abs() > ent_thr]
        rec["alarm_week_entropy"] = int(ent_rows["week"].iloc[0]) if len(ent_rows) > 0 else "None"

        alarm_records.append(rec)

    alarm_df = pd.DataFrame(alarm_records)
    alarm_df.to_csv(target_dir / f"alarm_summary_d{delay}_seed{seed}.csv", index=False)

    threshold_df = pd.DataFrame([thresholds])
    threshold_df.to_csv(target_dir / f"control_thresholds_d{delay}_seed{seed}.csv", index=False)

    logger.info("Phase 4 monitoring experiment complete. Outputs saved to %s", target_dir)
    return {
        "alarms": alarm_df,
        "thresholds": threshold_df,
        "control": all_scenario_dfs[SCENARIO_CONTROL],
        "s1": all_scenario_dfs[SCENARIO_S1],
        "s2": all_scenario_dfs[SCENARIO_S2],
        "s3": all_scenario_dfs[SCENARIO_S3],
    }


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run label-free monitoring and explainability.")
    parser.add_argument("--delay", type=int, default=7, help="Label delay in days (default: 7)")
    parser.add_argument("--onset", type=int, default=105, help="Drift onset day (default: 105)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    args = parser.parse_args()

    results = run_monitoring_experiment(delay=args.delay, onset_day=args.onset, seed=args.seed)
    print("\n=== Control Max Thresholds (Weeks 15-25) ===")
    print(results["thresholds"].to_string(index=False))
    print("\n=== First Alarm Week by Signal and Scenario ===")
    print(results["alarms"].to_string(index=False))
