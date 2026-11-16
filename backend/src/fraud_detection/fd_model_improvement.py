"""Phase 4: Comparative model improvement and temporal validation (Exploratory Module).

NOTE: This module contains exploratory model comparison experiments outside the primary
benchmark protocol (Phases 5-7). The primary benchmark protocol uses LightGBM with MODEL_FEATURES
to isolate the contribution of delay-aware attribution and early-warning adaptation under
delayed feedback (Spec Sections 1-13).

Compares candidate algorithms (Logistic Regression, LightGBM default, LightGBM tuned,
XGBoost, CatBoost) across standard and extended feature sets using strict chronological validation:
  - Sub-train: days 35..55 (21 days)
  - Validation: days 56..62 (7 days, matured during the gap days 63..69)
  - Test: days 70..181 (weeks 10..25, untouched during tuning)

Evaluates over multiple seeds (seeds 1, 2, 3) and reports mean, std, Recall@1% FPR,
actual FPR, Precision, AP, and confusion matrix counts.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fd_config import (
    COL_DAY,
    COL_FRAUD,
    MODEL_DIR,
    MODEL_FEATURES,
    OUTPUT_DIR,
    TARGET_FPR,
    get_logger,
)
from .fd_evaluation import average_precision, evaluate_weekly, recall_at_fpr, window_metrics
from .fd_feature_engineering import load_or_build_features
from .fd_feature_extended import ALL_EXTENDED_FEATURES, load_or_build_extended_features
from .fd_preprocessing import build_splits, make_windows, window_mask

logger = get_logger(__name__)

IMPROVEMENT_OUTPUT_DIR = OUTPUT_DIR / "model_improvement"


def train_candidate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    features: Sequence[str],
) -> Any:
    """Train a specific model architecture with strict feature and seed controls."""
    X = X_train[list(features)]

    if model_name == "logistic_regression":
        from sklearn.impute import SimpleImputer
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1.0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            )),
        ])
        model.fit(X, y_train)

    elif model_name == "lightgbm_default":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
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
        model.fit(X, y_train)

    elif model_name == "lightgbm_tuned":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
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
        model.fit(X, y_train)

    elif model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=20.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X, y_train)

    elif model_name == "catboost":
        import catboost as cb
        cb_dir = OUTPUT_DIR / "catboost_info"
        cb_dir.mkdir(parents=True, exist_ok=True)
        model = cb.CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=-1,
            train_dir=str(cb_dir),
            verbose=0,
        )
        model.fit(X, y_train)

    else:
        raise ValueError(f"Unknown candidate model: {model_name}")

    model.fd_features_ = list(features)
    model.fd_name_ = model_name
    return model


def predict_model_scores(model: Any, X_df: pd.DataFrame) -> np.ndarray:
    """Predict calibrated or ranking probabilities."""
    X = X_df[model.fd_features_]
    return model.predict_proba(X)[:, 1]


def select_validation_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    target_fpr: float = TARGET_FPR,
) -> float:
    """Select fixed operational decision threshold using ONLY validation negatives (Days 56..62).

    Guarantees test labels are never accessed to pick or adjust thresholds.
    Formula: quantile(scores_val[y_val == 0], 1.0 - target_fpr).
    """
    y = np.asarray(y_val, dtype=int)
    p = np.asarray(scores_val, dtype=float)
    neg_scores = p[y == 0]
    if len(neg_scores) == 0:
        raise ValueError("No negative/legitimate transactions in validation slice to compute threshold.")
    return float(np.quantile(neg_scores, 1.0 - target_fpr))


def evaluate_fixed_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a locked, pre-determined decision threshold without test label peeking.

    Denominators strictly follow standard fraud detection definitions:
    - fixed_recall = TP / n_fraud (TP / (TP + FN))
    - fixed_actual_fpr = FP / n_genuine (FP / (FP + TN))
    - fixed_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    n_total = len(y)
    n_fraud = int(np.sum(y))
    n_genuine = n_total - n_fraud

    flagged = p > threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    tn = int(np.sum((~flagged) & (y == 0)))
    fn = int(np.sum((~flagged) & (y == 1)))

    recall = float(tp / n_fraud) if n_fraud > 0 else 0.0
    actual_fpr = float(fp / n_genuine) if n_genuine > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "fixed_threshold": float(threshold),
        "fixed_tp": tp,
        "fixed_fp": fp,
        "fixed_tn": tn,
        "fixed_fn": fn,
        "fixed_recall": recall,
        "fixed_actual_fpr": actual_fpr,
        "fixed_precision": precision,
        "fixed_flagged": int(tp + fp),
    }


def compute_detailed_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = TARGET_FPR,
) -> Dict[str, Any]:
    """Compute diagnostic oracle ranking evaluation metrics for a given evaluation slice.

    NOTE: This calculates oracle diagnostic metrics (such as Recall@1%FPR) where
    threshold = quantile(scores[y==0], 1 - target_fpr) is pegged directly to the
    given slice's negatives. In operational production, deployable thresholds MUST
    be locked from validation data using select_validation_threshold().
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    n_total = len(y)
    n_fraud = int(np.sum(y))
    n_genuine = n_total - n_fraud

    if n_fraud == 0 or n_genuine == 0:
        return {
            "n_tx": n_total,
            "n_fraud": n_fraud,
            "prevalence": float(n_fraud / n_total) if n_total > 0 else 0.0,
            "oracle_recall_at_1fpr": float("nan"),
            "oracle_actual_fpr": float("nan"),
            "oracle_precision_at_1fpr": float("nan"),
            "oracle_threshold": float("nan"),
            "oracle_tp": 0, "oracle_fp": 0, "oracle_tn": 0, "oracle_fn": 0, "oracle_flagged": 0,
            "ap": float("nan"),
            "ap_over_prevalence": float("nan"),
            # Legacy aliases for backward compatibility
            "recall_at_1fpr": float("nan"),
            "actual_fpr": float("nan"),
            "precision_at_1fpr": float("nan"),
            "threshold": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "flagged": 0,
        }

    neg_scores = p[y == 0]
    thr = float(np.quantile(neg_scores, 1.0 - target_fpr))

    flagged = p > thr
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    tn = int(np.sum((~flagged) & (y == 0)))
    fn = int(np.sum((~flagged) & (y == 1)))

    recall = float(tp / n_fraud) if n_fraud > 0 else 0.0
    actual_fpr = float(fp / n_genuine) if n_genuine > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    ap = average_precision(y, p)
    prev = float(n_fraud / n_total)

    return {
        "n_tx": n_total,
        "n_fraud": n_fraud,
        "prevalence": prev,
        # Explicitly named diagnostic oracle ranking metrics
        "oracle_threshold": thr,
        "oracle_tp": tp,
        "oracle_fp": fp,
        "oracle_tn": tn,
        "oracle_fn": fn,
        "oracle_recall_at_1fpr": recall,
        "oracle_actual_fpr": actual_fpr,
        "oracle_precision_at_1fpr": precision,
        "oracle_flagged": int(tp + fp),
        "ap": ap,
        "ap_over_prevalence": ap / prev if prev > 0 else float("nan"),
        # Preserved legacy keys for backward compatibility
        "recall_at_1fpr": recall,
        "actual_fpr": actual_fpr,
        "precision_at_1fpr": precision,
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "flagged": int(tp + fp),
    }


def run_temporal_validation(
    df: pd.DataFrame,
    feature_set: str = "standard",
    delay: int = 7,
    seeds: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Evaluate candidate models on days 56..62 after training on days 35..55."""
    features = MODEL_FEATURES if feature_set == "standard" else ALL_EXTENDED_FEATURES
    candidates = [
        "logistic_regression",
        "lightgbm_default",
        "lightgbm_tuned",
        "xgboost",
        "catboost",
    ]

    # Split for temporal validation: Sub-train (35..55), Val (56..62)
    subtrain_mask = (df[COL_DAY] >= 35) & (df[COL_DAY] <= 55)
    val_mask = (df[COL_DAY] >= 56) & (df[COL_DAY] <= 62)

    train_data = df[subtrain_mask]
    val_data = df[val_mask]
    y_train = train_data[COL_FRAUD].to_numpy()
    y_val = val_data[COL_FRAUD].to_numpy()

    logger.info("Validation experiment: %d subtrain rows, %d validation rows",
                len(train_data), len(val_data))

    records: List[Dict[str, Any]] = []
    for model_name in candidates:
        for seed in seeds:
            logger.info("Training %s (feature_set=%s, seed=%d)", model_name, feature_set, seed)
            model = train_candidate_model(model_name, train_data, y_train, seed, features)
            scores = predict_model_scores(model, val_data)
            metrics = compute_detailed_metrics(y_val, scores)
            rec = {
                "model": model_name,
                "feature_set": feature_set,
                "seed": seed,
                **metrics,
            }
            records.append(rec)

    res_df = pd.DataFrame(records)
    return res_df


def run_full_test_comparison(
    df_standard: pd.DataFrame,
    df_extended: pd.DataFrame,
    best_candidate_name: str,
    best_feature_set: str,
    delay: int = 7,
    seeds: Sequence[int] = (1, 2, 3),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train on full days 35..62 and evaluate across all 16 test weeks (days 70..181)."""
    sp_std = build_splits(df_standard, delay)
    sp_ext = build_splits(df_extended, delay)
    cfg = sp_std["config"]

    test_std = df_standard[sp_std["test"]].reset_index(drop=True)
    test_ext = df_extended[sp_ext["test"]].reset_index(drop=True)
    y_test = test_std[COL_FRAUD].to_numpy()

    # Compare Baseline LightGBM (standard features) vs Best Model
    comparisons = [
        ("B0_baseline_lgbm", "lightgbm_default", "standard", MODEL_FEATURES, df_standard, test_std),
        (f"Improved_{best_candidate_name}", best_candidate_name, best_feature_set,
         MODEL_FEATURES if best_feature_set == "standard" else ALL_EXTENDED_FEATURES,
         df_standard if best_feature_set == "standard" else df_extended,
         test_std if best_feature_set == "standard" else test_ext),
    ]

    weekly_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    for label, model_name, f_set, feat_cols, full_df, test_df in comparisons:
        sp = sp_std if f_set == "standard" else sp_ext
        train_df = full_df[sp["train"]]
        y_tr = train_df[COL_FRAUD].to_numpy()

        # Validation slice for deployable threshold selection: Days 35..55 subtrain, Days 56..62 val
        subtrain_m = (full_df[COL_DAY] >= 35) & (full_df[COL_DAY] <= 55)
        val_m = (full_df[COL_DAY] >= 56) & (full_df[COL_DAY] <= 62)
        subtrain_df = full_df[subtrain_m]
        val_df = full_df[val_m]
        y_sub = subtrain_df[COL_FRAUD].to_numpy()
        y_val = val_df[COL_FRAUD].to_numpy()

        for seed in seeds:
            logger.info("Evaluating %s (seed=%d): Stage 1 (Threshold Selection on Validation Days 56..62)", label, seed)
            val_model = train_candidate_model(model_name, subtrain_df, y_sub, seed, feat_cols)
            val_scores = predict_model_scores(val_model, val_df)
            fixed_thr = select_validation_threshold(y_val, val_scores)

            logger.info("Evaluating %s (seed=%d): Stage 2 (Final Model Trained on Days 35..62, Tested on Test Stream)", label, seed)
            full_model = train_candidate_model(model_name, train_df, y_tr, seed, feat_cols)
            test_scores = predict_model_scores(full_model, test_df)

            # Weekly evaluation
            w_df = evaluate_weekly(test_df, test_scores, cfg.test_start, cfg.test_end)
            w_df["model_label"] = label
            w_df["seed"] = seed
            weekly_records.append(w_df)

            # Deployable Fixed Validation Threshold metrics (zero test label peeking)
            fixed_m = evaluate_fixed_threshold(y_test, test_scores, fixed_thr)

            # Diagnostic Oracle ranking metrics (Recall@1%FPR, AP)
            oracle_m = compute_detailed_metrics(y_test, test_scores)

            summary_records.append({
                "model_label": label,
                "model_architecture": model_name,
                "feature_set": f_set,
                "seed": seed,
                "mean_weekly_recall": float(w_df["recall_at_1fpr"].mean()),
                "std_weekly_recall": float(w_df["recall_at_1fpr"].std()),
                "mean_weekly_ap": float(w_df["ap"].mean()),
                # Deployable Fixed Validation Threshold metrics (zero test label peeking):
                "fixed_threshold": fixed_m["fixed_threshold"],
                "fixed_tp": fixed_m["fixed_tp"],
                "fixed_fp": fixed_m["fixed_fp"],
                "fixed_tn": fixed_m["fixed_tn"],
                "fixed_fn": fixed_m["fixed_fn"],
                "fixed_recall": fixed_m["fixed_recall"],
                "fixed_actual_fpr": fixed_m["fixed_actual_fpr"],
                "fixed_precision": fixed_m["fixed_precision"],
                "fixed_flagged": fixed_m["fixed_flagged"],
                # Diagnostic Oracle metrics (Recall@1%FPR ranking baseline):
                "oracle_threshold": oracle_m["oracle_threshold"],
                "oracle_tp": oracle_m["oracle_tp"],
                "oracle_fp": oracle_m["oracle_fp"],
                "oracle_tn": oracle_m["oracle_tn"],
                "oracle_fn": oracle_m["oracle_fn"],
                "oracle_recall_at_1fpr": oracle_m["oracle_recall_at_1fpr"],
                "oracle_actual_fpr": oracle_m["oracle_actual_fpr"],
                "oracle_precision_at_1fpr": oracle_m["oracle_precision_at_1fpr"],
                "ap": oracle_m["ap"],
                "ap_over_prevalence": oracle_m["ap_over_prevalence"],
                "n_tx": oracle_m["n_tx"],
                "n_fraud": oracle_m["n_fraud"],
                "prevalence": oracle_m["prevalence"],
                # Legacy aliases for backward compatibility:
                "threshold": oracle_m["oracle_threshold"],
                "recall_at_1fpr": oracle_m["oracle_recall_at_1fpr"],
                "actual_fpr": oracle_m["oracle_actual_fpr"],
                "precision_at_1fpr": oracle_m["oracle_precision_at_1fpr"],
                "tp": oracle_m["oracle_tp"],
                "fp": oracle_m["oracle_fp"],
                "tn": oracle_m["oracle_tn"],
                "fn": oracle_m["oracle_fn"],
                "flagged": oracle_m["oracle_flagged"],
            })

    weekly_combined = pd.concat(weekly_records, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)
    return weekly_combined, summary_df


def execute_phase4_benchmark(delay: int = 7) -> Dict[str, Any]:
    """Execute complete Phase 4 benchmark and save artifacts to outputs/."""
    from .fd_data_loader import load_handbook

    IMPROVEMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_handbook()

    logger.info("Loading standard features...")
    df_std = load_or_build_features(raw, delay)
    from .fd_feature_extended import (
        ALL_EXTENDED_FEATURES,
        compute_expanding_delay_aware_prior,
        load_or_build_extended_features,
    )

    logger.info("Building/loading extended features with causal expanding delay-aware prior...")
    df_ext = load_or_build_extended_features(raw, delay)

    # 1. Validation experiments (standard features)
    logger.info("Starting validation benchmark with standard features...")
    val_std = run_temporal_validation(df_std, feature_set="standard", delay=delay)

    # 2. Validation experiments (extended features)
    logger.info("Starting validation benchmark with extended features...")
    val_ext = run_temporal_validation(df_ext, feature_set="extended", delay=delay)

    val_all = pd.concat([val_std, val_ext], ignore_index=True)
    val_all.to_csv(IMPROVEMENT_OUTPUT_DIR / "validation_benchmark_all_seeds.csv", index=False)

    # Group by model and feature_set to find the top performer on validation
    val_summary = (
        val_all.groupby(["model", "feature_set"])
        .agg({
            "recall_at_1fpr": ["mean", "std"],
            "ap": ["mean", "std"],
            "precision_at_1fpr": ["mean", "std"],
            "actual_fpr": ["mean", "std"],
        })
        .round(4)
    )
    val_summary.columns = [f"{c[0]}_{c[1]}" for c in val_summary.columns]
    val_summary = val_summary.sort_values("recall_at_1fpr_mean", ascending=False).reset_index()
    val_summary.to_csv(IMPROVEMENT_OUTPUT_DIR / "validation_summary_ranked.csv", index=False)

    best_row = val_summary.iloc[0]
    best_model = best_row["model"]
    best_fset = best_row["feature_set"]
    logger.info("Best validated model: %s with %s features (Recall@1%%FPR: %.4f)",
                best_model, best_fset, best_row["recall_at_1fpr_mean"])

    # 3. Final Test Evaluation across weeks 10..25
    logger.info("Running test period evaluation...")
    weekly_test, test_summary = run_full_test_comparison(
        df_std, df_ext, best_candidate_name=best_model, best_feature_set=best_fset, delay=delay
    )
    weekly_test.to_csv(IMPROVEMENT_OUTPUT_DIR / "test_weekly_comparison.csv", index=False)
    test_summary.to_csv(IMPROVEMENT_OUTPUT_DIR / "test_summary_comparison.csv", index=False)

    return {
        "validation_summary": val_summary.to_dict(orient="records"),
        "test_summary": test_summary.to_dict(orient="records"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = execute_phase4_benchmark()
    logger.info("Phase 4 benchmark complete!")
