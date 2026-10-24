"""Baselines and ablation benchmarking framework (Spec Section 12).

Implements comparative adaptation baselines and ablation variants:
- B0: Static model (never retrained).
- B1: Periodic retrain every 7 days using all available matured labels (last 28 matured days).
- B2: Conventional feature PSI trigger (mean_psi >= 0.005 or max_psi >= 0.05).
- B3: NannyML-CBPE-style confidence-based performance estimate.
- B4: Explanation-Shift trigger (shap_shift >= locked threshold).
- B5: PRODEM-style error-prediction meta-model trained on matured labels.
- P1: Proposed EWS-triggered Attribution-Guided Adaptation (AGA).
- P1-noSHAP: EWS-triggered AGA with feature selection D guided by PSI.
- P1-a, P1-b, P1-c: EWS-triggered standalone policies (FULL, DROP_D, RECENCY).

LEAKAGE ENFORCEMENT: Every method uses only labels that have matured at decision time (day <= t - delay).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .fd_adaptation import (
    CANDIDATE_DROP_D,
    CANDIDATE_FULL,
    CANDIDATE_RECENCY,
    run_aga,
)
from .fd_config import (
    COL_DAY,
    COL_FRAUD,
    MODEL_FEATURES,
    MODEL_PARAMS,
    OUTPUT_DIR,
    ModelParams,
    get_logger,
)
from .fd_detection import EarlyWarningDetector, load_locked_thresholds
from .fd_evaluation import recall_at_fpr, window_metrics
from .fd_explainability import (
    build_tree_explainer,
    calculate_shap_profile,
    compute_explainability_signals,
)
from .fd_monitoring import compute_monitoring_signals
from .fd_preprocessing import make_windows, matured_cutoff
from .fd_training import predict_scores, train_model

logger = get_logger(__name__)

METHOD_B0 = "B0_static"
METHOD_B1 = "B1_periodic"
METHOD_B2 = "B2_psi_trigger"
METHOD_B3 = "B3_cbpe_style"
METHOD_B4 = "B4_shap_shift"
METHOD_B5 = "B5_prodem_style"
METHOD_P1 = "P1_aga_full"
METHOD_P1_NO_SHAP = "P1_noSHAP"
METHOD_P1_A = "P1_a_full_only"
METHOD_P1_B = "P1_b_drop_only"
METHOD_P1_C = "P1_c_recency_only"

BASELINE_METHODS: Tuple[str, ...] = (
    METHOD_B0,
    METHOD_B1,
    METHOD_B2,
    METHOD_B3,
    METHOD_B4,
    METHOD_B5,
    METHOD_P1,
    METHOD_P1_NO_SHAP,
    METHOD_P1_A,
    METHOD_P1_B,
    METHOD_P1_C,
)


def retrain_on_matured_data(
    df: pd.DataFrame,
    decision_day: int,
    delay: int,
    span_days: int = 28,
    seed: int = 1,
    features: Sequence[str] = MODEL_FEATURES,
    params: ModelParams = MODEL_PARAMS,
) -> Any:
    """Standard retraining on all features using the last ``span_days`` matured days."""
    cutoff = matured_cutoff(decision_day, delay)
    start_day = cutoff - span_days + 1
    mask = (df[COL_DAY] >= start_day) & (df[COL_DAY] <= cutoff)
    df_train = df[mask].reset_index(drop=True)
    if len(df_train) == 0:
        raise ValueError(f"No matured training data for days {start_day}..{cutoff}")
    return train_model(df_train, seed=seed, features=features, params=params)


class CBPEstimator:
    """NannyML-CBPE-style confidence-based performance estimator.

    Assumes P(Y=1|X) ~ p_hat(X) (no concept drift assumption).
    Estimates Recall@1%FPR from predicted probabilities.
    """

    def __init__(self, target_fpr: float = 0.01):
        self.target_fpr = target_fpr

    def estimate_recall(self, probs: np.ndarray) -> float:
        """Estimate recall based on predicted probabilities."""
        p = np.asarray(probs, dtype=np.float64)
        if len(p) == 0:
            return float("nan")
        # Under 1% FPR, the top 1% scored instances are predicted positive
        top_k = max(int(len(p) * self.target_fpr), 1)
        sorted_p = np.sort(p)[::-1]
        est_tp = float(np.sum(sorted_p[:top_k]))
        est_total_pos = float(np.sum(p))
        if est_total_pos <= 1e-6:
            return 0.0
        return float(np.clip(est_tp / est_total_pos, 0.0, 1.0))


class ProdemMetaModel:
    """PRODEM-style error-detection meta-model trained on matured data."""

    def __init__(self, seed: int = 42):
        self.clf = LogisticRegression(penalty="l2", C=1.0, random_state=seed, max_iter=200)
        self.is_fitted = False

    def fit(self, base_model: Any, df_matured: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES) -> None:
        """Train meta-model to predict whether base model makes an error."""
        y_true = df_matured[COL_FRAUD].to_numpy()
        p_pred = predict_scores(base_model, df_matured)
        # Binary prediction using top 1% threshold
        thr = float(np.quantile(p_pred, 0.99))
        y_pred = (p_pred > thr).astype(int)
        errors = (y_true != y_pred).astype(int)

        X = np.nan_to_num(df_matured[list(features)].to_numpy(), nan=-1.0)
        X_meta = np.column_stack([X, p_pred])
        self.clf.fit(X_meta, errors)
        self.is_fitted = True

    def predict_error_rate(self, base_model: Any, df_window: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES) -> float:
        """Predict expected error rate on current unlabeled window traffic."""
        if not self.is_fitted:
            return 0.0
        p_pred = predict_scores(base_model, df_window)
        X = np.nan_to_num(df_window[list(features)].to_numpy(), nan=-1.0)
        X_meta = np.column_stack([X, p_pred])
        error_probs = self.clf.predict_proba(X_meta)[:, 1]
        return float(np.mean(error_probs))


def run_benchmark_method(
    method: str,
    df: pd.DataFrame,
    delay: int = 7,
    onset_day: int = 105,
    seed: int = 1,
    base_model: Optional[Any] = None,
    explainer: Optional[Any] = None,
    detector: Optional[EarlyWarningDetector] = None,
    reference_df: Optional[pd.DataFrame] = None,
    reference_scores: Optional[np.ndarray] = None,
    reference_profile: Optional[np.ndarray] = None,
    window_cache: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute a single adaptation method / baseline across all test windows.

    Returns weekly performance metrics and execution summary.
    """
    from .fd_preprocessing import build_splits

    sp = build_splits(df, delay=delay)
    cfg = sp["config"]

    # Initial training on days 35..62 if not provided
    if base_model is None:
        base_model = train_model(df[sp["train"]], seed=seed)
    current_model = base_model

    needs_shap = method in (METHOD_B4, METHOD_P1, METHOD_P1_NO_SHAP, METHOD_P1_A, METHOD_P1_B, METHOD_P1_C)
    needs_mon = method in (METHOD_B2, METHOD_P1, METHOD_P1_NO_SHAP, METHOD_P1_A, METHOD_P1_B, METHOD_P1_C)

    if explainer is None and needs_shap:
        explainer = build_tree_explainer(base_model)

    test_df = df[sp["test"]].reset_index(drop=True)

    # Reference window days 70..90
    if reference_df is None:
        ref_mask = (test_df[COL_DAY] >= cfg.reference_start) & (test_df[COL_DAY] <= cfg.reference_end)
        reference_df = test_df[ref_mask].reset_index(drop=True)
        reference_scores = predict_scores(base_model, reference_df)
        if needs_shap:
            reference_profile = calculate_shap_profile(explainer, reference_df, seed=42)

    if detector is None:
        detector = EarlyWarningDetector()
    detector.reset()

    cbpe = CBPEstimator()
    prodem = ProdemMetaModel(seed=seed)

    windows = list(make_windows(cfg.test_start, cfg.test_end, 7))
    records = []
    retrain_count = 0
    t0_start = time.time()

    # Pre-drift baseline recall
    base_scores = predict_scores(base_model, test_df[test_df[COL_DAY] < onset_day])
    base_recall = recall_at_fpr(test_df[test_df[COL_DAY] < onset_day][COL_FRAUD].to_numpy(), base_scores)

    for i, win in enumerate(windows):
        w_start, w_end = win
        w_idx = w_start // 7
        wm = (test_df[COL_DAY] >= w_start) & (test_df[COL_DAY] <= w_end)
        w_df = test_df[wm].reset_index(drop=True)
        y_true = w_df[COL_FRAUD].to_numpy()

        is_base = (current_model is base_model)
        cached = window_cache.get(i) if (window_cache and is_base) else None

        if cached is not None:
            scores = cached["scores"]
            metrics = cached["metrics"]
            mon = cached.get("mon", {})
            exp = cached.get("exp", {"shap_shift": 0.0})
        else:
            scores = predict_scores(current_model, w_df)
            metrics = window_metrics(y_true, scores)
            mon = compute_monitoring_signals(reference_df, reference_scores, w_df, scores, MODEL_FEATURES) if needs_mon else {}
            exp = (
                window_cache[i]["exp"]
                if (window_cache and i in window_cache)
                else (
                    compute_explainability_signals(explainer, w_df, reference_profile, MODEL_FEATURES, max_samples=2000, seed=42)
                    if (needs_shap and explainer is not None)
                    else {"shap_shift": 0.0}
                )
            )

        retrain_triggered = False
        action_name = "NONE"
        candidate_chosen = "NONE"

        # Check adaptation trigger at end of window (only post-onset or if periodic)
        if w_start >= onset_day:
            decision_day = w_end + 1

            if method == METHOD_B0:
                pass  # Never retrain

            elif method == METHOD_B1:
                # Periodic: retrain every week
                retrain_triggered = True
                action_name = "PERIODIC_RETRAIN"
                current_model = retrain_on_matured_data(df, decision_day=decision_day, delay=delay, seed=seed)

            elif method == METHOD_B2:
                # Conventional PSI trigger: mean_psi >= 0.005 or max_psi >= 0.05
                if mon.get("mean_psi", 0.0) >= 0.005 or mon.get("max_psi", 0.0) >= 0.05:
                    retrain_triggered = True
                    action_name = "PSI_TRIGGERED_RETRAIN"
                    current_model = retrain_on_matured_data(df, decision_day=decision_day, delay=delay, seed=seed)

            elif method == METHOD_B3:
                # CBPE-style: estimated recall drops >= 20%
                est_rec = cbpe.estimate_recall(scores)
                if est_rec <= 0.80 * base_recall:
                    retrain_triggered = True
                    action_name = "CBPE_TRIGGERED_RETRAIN"
                    current_model = retrain_on_matured_data(df, decision_day=decision_day, delay=delay, seed=seed)

            elif method == METHOD_B4:
                # SHAP-shift trigger: shap_shift >= locked control threshold (0.01427)
                if exp.get("shap_shift", 0.0) >= 0.01427:
                    retrain_triggered = True
                    action_name = "SHAP_SHIFT_RETRAIN"
                    current_model = retrain_on_matured_data(df, decision_day=decision_day, delay=delay, seed=seed)

            elif method == METHOD_B5:
                # PRODEM-style error prediction
                m_cutoff = matured_cutoff(decision_day, delay)
                matured_slice = df[(df[COL_DAY] >= m_cutoff - 27) & (df[COL_DAY] <= m_cutoff)].reset_index(drop=True)
                prodem.fit(base_model, matured_slice)
                base_err = prodem.predict_error_rate(base_model, reference_df)
                pred_err = prodem.predict_error_rate(base_model, w_df)
                if pred_err > max(1.25 * base_err, 0.025):
                    retrain_triggered = True
                    action_name = "PRODEM_RETRAIN"
                    current_model = retrain_on_matured_data(df, decision_day=decision_day, delay=delay, seed=seed)

            elif method in (METHOD_P1, METHOD_P1_NO_SHAP, METHOD_P1_A, METHOD_P1_B, METHOD_P1_C):
                # EWS-guided AGA decision
                combined_signals = {}
                combined_signals.update(mon)
                combined_signals["shap_shift"] = exp.get("shap_shift", 0.0)

                dec_info = detector.evaluate_window(combined_signals, current_day=w_start)
                if dec_info["decision"] == "ADAPT":
                    retrain_triggered = True
                    action_name = "AGA_ADAPT"

                    forced = None
                    if method == METHOD_P1_A:
                        forced = CANDIDATE_FULL
                    elif method == METHOD_P1_B:
                        forced = CANDIDATE_DROP_D
                    elif method == METHOD_P1_C:
                        forced = CANDIDATE_RECENCY

                    sel_mode = "psi" if method == METHOD_P1_NO_SHAP else "shap"
                    aga_res = run_aga(
                        df=df,
                        decision_day=decision_day,
                        delay=delay,
                        deltas=exp,
                        selection_mode=sel_mode,
                        seed=seed,
                        forced_candidate=forced,
                    )
                    current_model = aga_res.selected_model
                    candidate_chosen = aga_res.winning_candidate

            if retrain_triggered:
                retrain_count += 1

        rec = {
            "week": w_idx,
            "day_start": w_start,
            "day_end": w_end,
            "method": method,
            "recall_at_1fpr": metrics["recall_at_1fpr"],
            "ap": metrics["ap"],
            "ap_over_prevalence": metrics["ap_over_prevalence"],
            "retrain_triggered": retrain_triggered,
            "action": action_name,
            "candidate_chosen": candidate_chosen,
        }
        records.append(rec)

    res_df = pd.DataFrame(records)
    elapsed = time.time() - t0_start

    post_df = res_df[res_df["week"] >= (onset_day // 7)]
    mean_post_rec = float(post_df["recall_at_1fpr"].mean())
    min_post_rec = float(post_df["recall_at_1fpr"].min())
    mean_post_ap = float(post_df["ap"].mean())
    mean_post_ap_over_prev = float(post_df["ap_over_prevalence"].mean())

    # True degradation definition (Spec Section 7): first window where recall falls < 80% of baseline
    degraded_windows = post_df[post_df["recall_at_1fpr"] < 0.80 * base_recall]
    if len(degraded_windows) == 0:
        degrade_day = None
        recovery_days = None  # No degradation occurred -> N/A
    else:
        degrade_day = int(degraded_windows["day_start"].iloc[0])
        # Look for recovery at or after degradation onset
        post_degrade_df = post_df[post_df["day_start"] >= degrade_day]
        recovered_df = post_degrade_df[post_degrade_df["recall_at_1fpr"] >= 0.90 * base_recall]
        if len(recovered_df) > 0:
            recov_day = int(recovered_df["day_start"].iloc[0])
            recovery_days = recov_day - degrade_day
        else:
            recovery_days = -1  # Never recovered

    # Lead time relative to true degradation day (Spec Section 13: negative if late)
    retrained_windows = post_df[post_df["retrain_triggered"]]
    first_retrain_day = int(retrained_windows["day_start"].iloc[0]) if len(retrained_windows) > 0 else None
    lead_time_days = None
    if degrade_day is not None and first_retrain_day is not None:
        lead_time_days = degrade_day - first_retrain_day

    summary = {
        "method": method,
        "base_recall": float(base_recall),
        "post_onset_mean_recall": mean_post_rec,
        "post_onset_min_recall": min_post_rec,
        "relative_retention_pct": float(mean_post_rec / base_recall * 100.0) if base_recall > 0 else 0.0,
        "post_onset_mean_ap": mean_post_ap,
        "post_onset_mean_ap_over_prevalence": mean_post_ap_over_prev,
        "retrain_count": retrain_count,
        "lead_time_days": lead_time_days,
        "recovery_time_days": recovery_days,
        "elapsed_seconds": float(elapsed),
    }
    return res_df, summary


def run_phase6_benchmark(
    delay: int = 7,
    onset_day: int = 105,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    scenarios: Sequence[str] = ("control", "S1", "S2", "S3"),
    methods: Sequence[str] = BASELINE_METHODS,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run full Phase 6 grid: Delay x Scenarios x Seeds across baselines and AGA."""
    from .fd_data_loader import load_handbook
    from .fd_drift_injection import inject_drift
    from .fd_feature_engineering import build_features
    from .fd_preprocessing import build_splits

    target_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "baselines"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Handbook data for Phase 6 benchmarking...")
    raw = load_handbook()

    all_summaries = []

    for sc in scenarios:
        for seed in seeds:
            logger.info("Executing benchmark for Scenario '%s', Seed %d...", sc, seed)
            injected = inject_drift(raw, sc, onset_day=onset_day, seed=seed)
            feat = build_features(injected, delay=delay)

            sp = build_splits(feat, delay=delay)
            cfg = sp["config"]
            base_model = train_model(feat[sp["train"]], seed=seed)
            test_df = feat[sp["test"]].reset_index(drop=True)

            ref_mask = (test_df[COL_DAY] >= cfg.reference_start) & (test_df[COL_DAY] <= cfg.reference_end)
            reference_df = test_df[ref_mask].reset_index(drop=True)
            reference_scores = predict_scores(base_model, reference_df)

            explainer = build_tree_explainer(base_model)
            reference_profile = calculate_shap_profile(explainer, reference_df, seed=42)

            # Precompute base model window cache for maximum efficiency
            window_cache = {}
            windows = list(make_windows(cfg.test_start, cfg.test_end, 7))
            for i, (w_start, w_end) in enumerate(windows):
                wm = (test_df[COL_DAY] >= w_start) & (test_df[COL_DAY] <= w_end)
                w_df = test_df[wm].reset_index(drop=True)
                y_true = w_df[COL_FRAUD].to_numpy()
                scrs = predict_scores(base_model, w_df)
                mtr = window_metrics(y_true, scrs)
                mn = compute_monitoring_signals(reference_df, reference_scores, w_df, scrs, MODEL_FEATURES)
                ex = compute_explainability_signals(explainer, w_df, reference_profile, MODEL_FEATURES, max_samples=2000, seed=42)
                window_cache[i] = {"scores": scrs, "metrics": mtr, "mon": mn, "exp": ex}

            for m in methods:
                _, summ = run_benchmark_method(
                    method=m,
                    df=feat,
                    delay=delay,
                    onset_day=onset_day,
                    seed=seed,
                    base_model=base_model,
                    explainer=explainer,
                    reference_df=reference_df,
                    reference_scores=reference_scores,
                    reference_profile=reference_profile,
                    window_cache=window_cache,
                )
                summ["scenario"] = sc
                summ["seed"] = seed
                summ["delay"] = delay
                all_summaries.append(summ)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(target_dir / f"benchmark_summary_d{delay}_all_seeds.csv", index=False)
    logger.info("Phase 6 benchmark completed (%d rows). Saved to %s", len(summary_df), target_dir)
    return summary_df


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run Phase 6 baselines and AGA benchmarks.")
    parser.add_argument("--delay", type=int, default=7, help="Label delay in days")
    parser.add_argument("--onset", type=int, default=105, help="Drift onset day")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--all-seeds", action="store_true", help="Run seeds 1..5")
    args = parser.parse_args()

    seeds = [1, 2, 3, 4, 5] if args.all_seeds else [args.seed]
    res_df = run_phase6_benchmark(delay=args.delay, onset_day=args.onset, seeds=seeds)
    print("\n=== Phase 6 Benchmark Summary ===")
    print(res_df.to_string(index=False))
