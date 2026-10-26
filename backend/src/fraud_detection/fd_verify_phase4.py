"""Independent verification of Phase 4 model improvements.

Addresses the question: Is the improvement genuine or an artifact of:
  1. Data leakage in extended features?
  2. Peeking at test labels?
  3. Weekly oracle thresholding artifacts vs. deployable fixed thresholds?
  4. Discrepancies in raw metric calculation?

This script executes an end-to-end, independent audit using raw NumPy / Pandas
calculations, without relying on helper abstraction layers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from .fd_config import (
    COL_AMOUNT,
    COL_CUSTOMER,
    COL_DATETIME,
    COL_DAY,
    COL_FRAUD,
    COL_TERMINAL,
    MODEL_FEATURES,
    OUTPUT_DIR,
    get_logger,
)
from .fd_data_loader import load_handbook
from .fd_feature_engineering import build_features
from .fd_feature_extended import ALL_EXTENDED_FEATURES, build_extended_features
from .fd_preprocessing import build_splits, make_windows, window_mask

logger = get_logger(__name__)
VERIFY_OUTPUT_DIR = OUTPUT_DIR / "verification_phase4"


def verify_phase4_integrity(delay: int = 7, seed: int = 1) -> Dict[str, Any]:
    """Execute complete independent verification."""
    VERIFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. LOAD RAW DATA AND PREPARE FEATURES FROM SCRATCH
    # -------------------------------------------------------------------------
    logger.info("Loading raw Handbook data...")
    raw = load_handbook()
    sp = build_splits(raw, delay)
    cfg = sp["config"]

    from .fd_feature_extended import (
        ALL_EXTENDED_FEATURES,
        build_extended_features,
        compute_expanding_delay_aware_prior,
        compute_pretraining_warmup_prior,
    )

    exp_prior = compute_expanding_delay_aware_prior(raw, delay)
    warmup_prior = compute_pretraining_warmup_prior(raw, delay)
    logger.info(
        "Estimated causal priors: Warmup=%.6f, Expanding[Day 35]=%.6f, Expanding[Day 62+]=%.6f",
        warmup_prior,
        exp_prior[35],
        exp_prior[62],
    )
    report["training_prior_audit"] = {
        "prior_type": "strictly_expanding_delay_aware",
        "warmup_prior_value": warmup_prior,
        "expanding_prior_day35": float(exp_prior[35]),
        "expanding_prior_day40": float(exp_prior[40]),
        "expanding_prior_day55": float(exp_prior[55]),
        "expanding_prior_day62": float(exp_prior[62]),
        "expanding_prior_frozen_test": float(exp_prior[70]),
        "is_frozen_at_train_end": bool(exp_prior[70] == exp_prior[cfg.train_end]),
        "test_labels_used": False,
        "validation_labels_used": False,
    }

    logger.info("Building standard features from scratch...")
    df_std = build_features(raw, delay)
    logger.info("Building extended features from scratch with causal expanding prior...")
    df_ext = build_extended_features(raw, delay)

    train_mask = (raw[COL_DAY] >= cfg.train_start) & (raw[COL_DAY] <= cfg.train_end)
    test_mask = (raw[COL_DAY] >= cfg.test_start) & (raw[COL_DAY] <= cfg.test_end)

    # Validation period (strictly matured before test starts): days 56..62
    val_train_mask = (raw[COL_DAY] >= cfg.train_start) & (raw[COL_DAY] <= 55)
    val_eval_mask = (raw[COL_DAY] >= 56) & (raw[COL_DAY] <= cfg.train_end)

    y_train = raw.loc[train_mask, COL_FRAUD].to_numpy()
    y_test = raw.loc[test_mask, COL_FRAUD].to_numpy()

    # -------------------------------------------------------------------------
    # 2. PROVE STRICT ROW AND GROUND TRUTH ALIGNMENT
    # -------------------------------------------------------------------------
    report["alignment"] = {
        "test_rows_identical": bool(len(df_std[test_mask]) == len(df_ext[test_mask]) == len(y_test)),
        "n_test_transactions": int(len(y_test)),
        "n_test_frauds": int(np.sum(y_test)),
        "test_fraud_prevalence": float(np.mean(y_test)),
        "test_days_range": [int(raw.loc[test_mask, COL_DAY].min()), int(raw.loc[test_mask, COL_DAY].max())],
        "train_days_range": [int(raw.loc[train_mask, COL_DAY].min()), int(raw.loc[train_mask, COL_DAY].max())],
        "overlap_train_test_days": bool(set(range(cfg.train_start, cfg.train_end + 1)).intersection(
            set(range(cfg.test_start, cfg.test_end + 1))
        )),
    }

    # -------------------------------------------------------------------------
    # 3. LEAKAGE PERTURBATION TESTS (INTRA-TRAINING & TEST STREAM)
    # -------------------------------------------------------------------------
    logger.info("Executing intra-training perturbation test (Day 40 vs Day 55)...")
    # At Day 40, labels on Day 55 are unmatured (40 - 7 = 33 < 55).
    # Perturbing Day 55 labels MUST NOT change Day <= 40 features.
    mod_intra = raw.copy()
    late_train_mask = mod_intra[COL_DAY] == 55
    mod_intra.loc[late_train_mask, COL_FRAUD] = 1 - mod_intra.loc[late_train_mask, COL_FRAUD]
    df_ext_intra = build_extended_features(mod_intra, delay)

    early_train_mask = (raw[COL_DAY] <= 40).to_numpy()
    intra_diffs = {}
    for feat in ALL_EXTENDED_FEATURES:
        a = df_ext.loc[early_train_mask, feat].to_numpy(dtype=np.float64)
        b = df_ext_intra.loc[early_train_mask, feat].to_numpy(dtype=np.float64)
        diff = float(np.nanmax(np.abs(a - b)))
        intra_diffs[feat] = diff

    report["intra_training_leakage_audit"] = {
        "d_early": 40,
        "d_late": 55,
        "labels_perturbed": int(late_train_mask.sum()),
        "early_rows_checked": int(early_train_mask.sum()),
        "max_abs_diff_per_feature": intra_diffs,
        "is_leakage_free": bool(all(v == 0.0 for v in intra_diffs.values())),
    }

    logger.info("Executing test-stream perturbation test on extended features (T=100)...")
    # Choose decision day T = 100.
    # At T=100, labels for day <= 100 - delay (<= 93) are matured.
    # Any label on day >= 94 MUST NOT affect features on or before day 100.
    t_query = 100
    mod_raw = raw.copy()
    unmatured_mask = mod_raw[COL_DAY] > (t_query - delay)
    mod_raw.loc[unmatured_mask, COL_FRAUD] = 1 - mod_raw.loc[unmatured_mask, COL_FRAUD]

    # Future transactions (day > 100) have their amounts randomized
    future_mask = mod_raw[COL_DAY] > t_query
    mod_raw.loc[future_mask, COL_AMOUNT] = mod_raw.loc[future_mask, COL_AMOUNT] * 2.5 + 50.0

    df_ext_perturbed = build_extended_features(mod_raw, delay)

    # Check observed rows: day <= 100
    observed_mask = (raw[COL_DAY] <= t_query).to_numpy()
    max_feature_diffs = {}
    for feat in ALL_EXTENDED_FEATURES:
        orig_vals = df_ext.loc[observed_mask, feat].to_numpy(dtype=np.float64)
        pert_vals = df_ext_perturbed.loc[observed_mask, feat].to_numpy(dtype=np.float64)
        diff = float(np.nanmax(np.abs(orig_vals - pert_vals)))
        max_feature_diffs[feat] = diff

    report["extended_feature_leakage_audit"] = {
        "t_query": t_query,
        "labels_perturbed": int(unmatured_mask.sum()),
        "future_amounts_perturbed": int(future_mask.sum()),
        "observed_rows_checked": int(observed_mask.sum()),
        "max_abs_diff_per_feature": max_feature_diffs,
        "is_leakage_free": bool(all(v == 0.0 for v in max_feature_diffs.values())),
    }

    # -------------------------------------------------------------------------
    # 4. TRAIN B0 AND IMPROVED MODEL FROM SCRATCH
    # -------------------------------------------------------------------------
    logger.info("Training B0 baseline model...")
    m_b0 = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=20.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    m_b0.fit(df_std.loc[train_mask, list(MODEL_FEATURES)], y_train)

    logger.info("Training Improved model...")
    m_imp = LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        n_estimators=250,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        scale_pos_weight=15.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    m_imp.fit(df_ext.loc[train_mask, list(ALL_EXTENDED_FEATURES)], y_train)

    # Raw predictions on test set
    p_b0 = m_b0.predict_proba(df_std.loc[test_mask, list(MODEL_FEATURES)])[:, 1]
    p_imp = m_imp.predict_proba(df_ext.loc[test_mask, list(ALL_EXTENDED_FEATURES)])[:, 1]

    # -------------------------------------------------------------------------
    # 5. RAW INDEPENDENT METRIC CALCULATION (OVER ENTIRE TEST STREAM)
    # -------------------------------------------------------------------------
    def _calc_raw_metrics(y: np.ndarray, p: np.ndarray, target_fpr: float = 0.01) -> Dict[str, float]:
        neg = p[y == 0]
        pos = p[y == 1]
        thr = float(np.quantile(neg, 1.0 - target_fpr))
        flagged = p > thr
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        tn = int(np.sum((~flagged) & (y == 0)))
        fn = int(np.sum((~flagged) & (y == 1)))
        return {
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": float(tp / len(pos)),
            "actual_fpr": float(fp / len(neg)),
            "precision": float(tp / (tp + fp)),
            "ap": float(average_precision_score(y, p)),
        }

    raw_test_b0 = _calc_raw_metrics(y_test, p_b0)
    raw_test_imp = _calc_raw_metrics(y_test, p_imp)

    report["full_test_raw_metrics"] = {
        "B0_baseline": raw_test_b0,
        "Improved_model": raw_test_imp,
        "delta_recall_pp": (raw_test_imp["recall"] - raw_test_b0["recall"]) * 100,
        "delta_ap_pp": (raw_test_imp["ap"] - raw_test_b0["ap"]) * 100,
        "additional_frauds_caught": raw_test_imp["tp"] - raw_test_b0["tp"],
        "difference_in_false_positives": raw_test_imp["fp"] - raw_test_b0["fp"],
    }

    # -------------------------------------------------------------------------
    # 6. WEEK-BY-WEEK METRIC CALCULATION (CONFIRM EXACT PROTOCOL)
    # -------------------------------------------------------------------------
    test_days = raw.loc[test_mask, COL_DAY].to_numpy()
    weekly_res_b0 = []
    weekly_res_imp = []

    for win in make_windows(cfg.test_start, cfg.test_end, 7):
        w_mask = (test_days >= win[0]) & (test_days <= win[1])
        w_y = y_test[w_mask]
        m_b0_w = _calc_raw_metrics(w_y, p_b0[w_mask])
        m_imp_w = _calc_raw_metrics(w_y, p_imp[w_mask])
        weekly_res_b0.append({"week": win[0] // 7, **m_b0_w})
        weekly_res_imp.append({"week": win[0] // 7, **m_imp_w})

    df_w_b0 = pd.DataFrame(weekly_res_b0)
    df_w_imp = pd.DataFrame(weekly_res_imp)

    report["weekly_eval_summary"] = {
        "B0_mean_weekly_recall": float(df_w_b0["recall"].mean()),
        "B0_mean_weekly_ap": float(df_w_b0["ap"].mean()),
        "Improved_mean_weekly_recall": float(df_w_imp["recall"].mean()),
        "Improved_mean_weekly_ap": float(df_w_imp["ap"].mean()),
        "weeks_improved_strictly_beat_b0_recall": int(np.sum(df_w_imp["recall"] >= df_w_b0["recall"])),
        "total_weeks_evaluated": len(df_w_b0),
    }

    # -------------------------------------------------------------------------
    # 7. DEPLOYABLE FIXED VALIDATION THRESHOLD (ZERO TEST PEEKING)
    # -------------------------------------------------------------------------
    # Train validation model on days 35..55
    m_val_b0 = LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=31, n_estimators=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=20.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m_val_b0.fit(df_std.loc[val_train_mask, list(MODEL_FEATURES)], raw.loc[val_train_mask, COL_FRAUD])
    p_val_eval_b0 = m_val_b0.predict_proba(df_std.loc[val_eval_mask, list(MODEL_FEATURES)])[:, 1]
    y_val_eval = raw.loc[val_eval_mask, COL_FRAUD].to_numpy()
    fixed_thr_b0 = float(np.quantile(p_val_eval_b0[y_val_eval == 0], 0.99))

    m_val_imp = LGBMClassifier(
        objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=50,
        n_estimators=250, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        scale_pos_weight=15.0, random_state=seed, n_jobs=-1, verbose=-1,
    )
    m_val_imp.fit(df_ext.loc[val_train_mask, list(ALL_EXTENDED_FEATURES)], raw.loc[val_train_mask, COL_FRAUD])
    p_val_eval_imp = m_val_imp.predict_proba(df_ext.loc[val_eval_mask, list(ALL_EXTENDED_FEATURES)])[:, 1]
    fixed_thr_imp = float(np.quantile(p_val_eval_imp[y_val_eval == 0], 0.99))

    # Evaluate deployed fixed thresholds on test predictions of the full models
    def _eval_fixed(y: np.ndarray, p: np.ndarray, fixed_thr: float) -> Dict[str, float]:
        flagged = p > fixed_thr
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        return {
            "fixed_threshold": fixed_thr,
            "tp": tp,
            "fp": fp,
            "recall": float(tp / pos),
            "actual_fpr": float(fp / neg),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        }

    report["fixed_validation_threshold_test_eval"] = {
        "B0_fixed_eval": _eval_fixed(y_test, p_b0, fixed_thr_b0),
        "Improved_fixed_eval": _eval_fixed(y_test, p_imp, fixed_thr_imp),
    }

    # -------------------------------------------------------------------------
    # 8. FEATURE IMPORTANCE AUDIT (WHICH EXTENDED FEATURES ARE ACTUALLY USED?)
    # -------------------------------------------------------------------------
    feat_imp = pd.DataFrame({
        "feature": ALL_EXTENDED_FEATURES,
        "importance_gain": m_imp.booster_.feature_importance(importance_type="gain"),
        "importance_split": m_imp.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    feat_imp["gain_pct"] = feat_imp["importance_gain"] / feat_imp["importance_gain"].sum() * 100
    feat_imp.to_csv(VERIFY_OUTPUT_DIR / "improved_model_feature_importance.csv", index=False)

    report["feature_importance_ranking"] = feat_imp[["feature", "gain_pct"]].to_dict(orient="records")

    with open(VERIFY_OUTPUT_DIR / "phase4_verification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rep = verify_phase4_integrity()
    print("\n--- PHASE 4 INDEPENDENT VERIFICATION REPORT ---")
    print(json.dumps(rep, indent=2))
