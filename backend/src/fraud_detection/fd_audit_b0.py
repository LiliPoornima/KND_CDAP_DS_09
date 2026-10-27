"""Phase 3 audit of the B0 baseline (read-only with respect to the pipeline).

Empirically verifies, on the REAL Handbook data, that:
  1. freshly rebuilt features equal the cached features (no stale cache),
  2. features equal an independent brute-force computation on sampled rows,
  3. labels that have not matured at time t cannot influence features of rows seen up to t,
  4. train / gap / test are chronological and respect ``label_day <= t - delay``,
  5. no single feature is suspiciously predictive (leak smell test),
  6. B0 metrics are reproducible (two fresh runs vs. the saved CSV),
  7. what the "recall at 1% FPR" protocol really gives as actual FPR (oracle per-window
     threshold) versus a deployable fixed threshold chosen on a validation period.

Run: ``python -m backend.src.fraud_detection.fd_audit_b0``
Nothing here changes the B0 model, features or protocol.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .fd_config import (
    COL_AMOUNT, COL_CUSTOMER, COL_DATETIME, COL_DAY, COL_FRAUD, COL_TERMINAL,
    MODEL_FEATURES, OUTPUT_DIR, PROCESSED_DIR, TERMINAL_RISK_WINDOW, build_split_config,
    get_logger,
)
from .fd_evaluation import evaluate_weekly, recall_at_fpr
from .fd_feature_engineering import build_features
from .fd_preprocessing import build_splits, make_windows, matured_mask, window_mask
from .fd_training import predict_scores, train_model

logger = get_logger(__name__)
AUDIT_DIR = OUTPUT_DIR / "audit"


def check_cache_matches_fresh(raw: pd.DataFrame, fresh: pd.DataFrame, delay: int) -> Dict:
    """Compare cached feature parquet (if present) with freshly rebuilt features."""
    from .fd_feature_engineering import features_fingerprint

    path = PROCESSED_DIR / f"features_default_d{delay}_{features_fingerprint(raw, delay)}.parquet"
    if not path.is_file():
        return {"cache_present": False}
    cached = pd.read_parquet(path)
    diffs = {}
    for c in list(MODEL_FEATURES) + [COL_FRAUD, COL_DAY]:
        a, b = cached[c].to_numpy(dtype=np.float64), fresh[c].to_numpy(dtype=np.float64)
        diffs[c] = float(np.nanmax(np.abs(a - b))) if len(a) == len(b) else float("inf")
    return {"cache_present": True, "rows_equal": len(cached) == len(fresh),
            "max_abs_diff_per_feature": diffs,
            "cache_equals_fresh": bool(all(v == 0.0 for v in diffs.values()))}


def brute_force_check(df: pd.DataFrame, delay: int, n_rows: int = 300, seed: int = 0) -> Dict:
    """Recompute customer/terminal features row-by-row for sampled rows (slow, independent)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n_rows, replace=False)
    by_cust = {k: g for k, g in df.groupby(COL_CUSTOMER)[[COL_DATETIME, COL_AMOUNT]]}
    by_term = {k: g for k, g in df.groupby(COL_TERMINAL)[[COL_DAY, COL_FRAUD]]}
    worst = {"cust_cnt": 0.0, "cust_mean": 0.0, "t_vol": 0.0, "t_risk": 0.0}
    for i in idx:
        r = df.iloc[i]
        g = by_cust[r[COL_CUSTOMER]]
        for w, td in (("1D", pd.Timedelta("1D")), ("7D", pd.Timedelta("7D")),
                      ("30D", pd.Timedelta("30D"))):
            prev = g[(g[COL_DATETIME] >= r[COL_DATETIME] - td) & (g[COL_DATETIME] < r[COL_DATETIME])]
            worst["cust_cnt"] = max(worst["cust_cnt"], abs(r[f"c_cnt_{w}"] - len(prev)))
            if len(prev):
                worst["cust_mean"] = max(worst["cust_mean"],
                                         abs(r[f"c_mean_{w}"] - prev[COL_AMOUNT].mean()))
            elif not np.isnan(r[f"c_mean_{w}"]):
                worst["cust_mean"] = float("inf")
        D = int(r[COL_DAY])
        tg = by_term[r[COL_TERMINAL]]
        m = matured_mask(tg, D, delay) & (tg[COL_DAY] > D - delay - TERMINAL_RISK_WINDOW)
        sub = tg[m]
        worst["t_vol"] = max(worst["t_vol"], abs(r["t_vol"] - len(sub)))
        exp = sub[COL_FRAUD].sum() / len(sub) if len(sub) else 0.0
        worst["t_risk"] = max(worst["t_risk"], abs(r["t_risk"] - exp))
    return {"rows_checked": int(n_rows), "max_abs_error": worst,
            "passes": bool(worst["cust_cnt"] == 0 and worst["t_vol"] == 0
                           and worst["cust_mean"] < 1e-2 and worst["t_risk"] < 1e-6)}


def perturbation_leakage_check(raw: pd.DataFrame, delay: int, t: int) -> Dict:
    """Flip all labels NOT matured at time t; features of rows with day <= t must not change."""
    base = build_features(raw, delay)
    mod = raw.copy()
    future = ~matured_mask(mod, t, delay)
    mod.loc[future, COL_FRAUD] = 1 - mod.loc[future, COL_FRAUD]
    alt = build_features(mod, delay)
    seen = (base[COL_DAY] <= t).to_numpy()
    res = {"t": t, "delay": delay, "labels_flipped": int(future.sum()),
           "rows_checked": int(seen.sum())}
    for c in ("t_risk", "t_vol", "c_mean_30D", "c_cnt_30D", "amt_ratio"):
        a = base.loc[seen, c].to_numpy(dtype=np.float64)
        b = alt.loc[seen, c].to_numpy(dtype=np.float64)
        res[f"max_diff_{c}"] = float(np.nanmax(np.abs(a - b)))
    # sanity: features of LATER rows are expected to change (proves the test has power)
    later = (base[COL_DAY] > t + delay).to_numpy()
    res["power_check_later_rows_max_diff_t_risk"] = float(
        np.nanmax(np.abs(base.loc[later, "t_risk"].to_numpy(dtype=np.float64)
                         - alt.loc[later, "t_risk"].to_numpy(dtype=np.float64))))
    res["passes"] = bool(all(res[f"max_diff_{c}"] == 0.0 for c in
                             ("t_risk", "t_vol", "c_mean_30D", "c_cnt_30D", "amt_ratio"))
                         and res["power_check_later_rows_max_diff_t_risk"] > 0)
    return res


def chronology_check(df: pd.DataFrame) -> Dict:
    """Train/gap/test chronology and the delayed-label rule for every delay."""
    out = {"sorted_by_time": bool(df[COL_DATETIME].is_monotonic_increasing),
           "day_monotonic": bool(df[COL_DAY].is_monotonic_increasing), "delays": {}}
    for d in (7, 14, 28):
        sp = build_splits(df, d)
        cfg = sp["config"]
        tr_max = int(df.loc[sp["train"], COL_DAY].max())
        te_min = int(df.loc[sp["test"], COL_DAY].min())
        out["delays"][d] = {
            "train_days": [int(df.loc[sp["train"], COL_DAY].min()), tr_max],
            "test_days": [te_min, int(df.loc[sp["test"], COL_DAY].max())],
            "last_train_label_matured_at_day": tr_max + d,
            "test_starts_after_train_labels_matured": bool(te_min >= tr_max + d),
            "train_test_row_overlap": int((sp["train"] & sp["test"]).sum()),
        }
    return out


def single_feature_auc(test: pd.DataFrame) -> pd.DataFrame:
    """AUC of each raw feature on the test period (very high values would suggest a leak)."""
    from sklearn.metrics import roc_auc_score

    rows = []
    y = test[COL_FRAUD].to_numpy()
    for f in MODEL_FEATURES:
        x = test[f].to_numpy(dtype=np.float64)
        x = np.where(np.isnan(x), -1.0, x)
        auc = roc_auc_score(y, x)
        rows.append({"feature": f, "auc": auc, "auc_direction_free": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc_direction_free", ascending=False)


def fpr_protocol_check(df: pd.DataFrame, delay: int, seed: int) -> Dict[str, pd.DataFrame]:
    """Actual FPR of the oracle per-window threshold vs a deployable validation threshold.

    DIAGNOSTIC ONLY. The deployable variant trains a separate validation model on days
    35..55, picks the 99th percentile of NEGATIVE scores on days 56..62 (all matured before
    the test period) and applies that fixed threshold to the test weeks. The B0 model and
    protocol are not changed.
    """
    sp = build_splits(df, delay)
    cfg = sp["config"]
    model = train_model(df[sp["train"]], seed=seed)
    test = df[sp["test"]].reset_index(drop=True)
    s = predict_scores(model, test)
    y = test[COL_FRAUD].to_numpy()
    rows = []
    for win in make_windows(cfg.test_start, cfg.test_end, 7):
        m = window_mask(test, win).to_numpy()
        yy, ss = y[m], s[m]
        thr = np.quantile(ss[yy == 0], 0.99)
        flag = ss > thr
        rows.append({"week": win[0] // 7, "oracle_threshold": thr,
                     "actual_fpr": float(flag[yy == 0].mean()),
                     "recall": float(flag[yy == 1].mean()),
                     "precision": float(yy[flag].mean()), "flagged": int(flag.sum()),
                     "false_positives": int((flag & (yy == 0)).sum())})
    oracle = pd.DataFrame(rows)

    val_tr = df[(df[COL_DAY] >= cfg.train_start) & (df[COL_DAY] <= 55)]
    val_va = df[(df[COL_DAY] >= 56) & (df[COL_DAY] <= cfg.train_end)]
    vmodel = train_model(val_tr, seed=seed)
    thr = float(np.quantile(predict_scores(vmodel, val_va)[val_va[COL_FRAUD].to_numpy() == 0], 0.99))
    sv = predict_scores(vmodel, test)
    rows = []
    for win in make_windows(cfg.test_start, cfg.test_end, 7):
        m = window_mask(test, win).to_numpy()
        yy, ss = y[m], sv[m]
        flag = ss > thr
        rows.append({"week": win[0] // 7, "fixed_threshold": thr,
                     "actual_fpr": float(flag[yy == 0].mean()),
                     "recall": float(flag[yy == 1].mean()),
                     "precision": float(yy[flag].mean()) if flag.any() else float("nan"),
                     "flagged": int(flag.sum()), "false_positives": int((flag & (yy == 0)).sum())})
    return {"oracle_window_threshold_B0_model": oracle, "fixed_validation_threshold": pd.DataFrame(rows)}


def run_audit(delay: int = 7, seed: int = 1) -> Dict:
    """Run every audit check and save results under outputs/fraud_detection/audit/."""
    from .fd_data_loader import load_handbook

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_handbook()
    fresh = build_features(raw, delay)
    report: Dict = {"delay": delay, "seed": seed}

    report["cache_vs_fresh"] = check_cache_matches_fresh(raw, fresh, delay)
    report["brute_force"] = brute_force_check(fresh, delay)
    report["perturbation_leakage"] = [perturbation_leakage_check(raw, delay, t) for t in (90, 120)]
    report["chronology"] = chronology_check(fresh)

    sp = build_splits(fresh, delay)
    auc = single_feature_auc(fresh[sp["test"]])
    auc.to_csv(AUDIT_DIR / "single_feature_auc_test.csv", index=False)
    report["max_single_feature_auc"] = float(auc.auc_direction_free.max())

    # B0 reproducibility: two independent fresh runs vs the saved CSV
    runs = []
    for _ in range(2):
        m = train_model(fresh[sp["train"]], seed=seed)
        t = fresh[sp["test"]].reset_index(drop=True)
        runs.append(evaluate_weekly(t, predict_scores(m, t), sp["config"].test_start,
                                    sp["config"].test_end))
    saved_path = OUTPUT_DIR / "b0" / f"b0_weekly_d{delay}_seed{seed}.csv"
    rep = {"run1_vs_run2_max_abs_diff_recall":
           float(np.abs(runs[0].recall_at_1fpr - runs[1].recall_at_1fpr).max()),
           "mean_recall": float(runs[0].recall_at_1fpr.mean()),
           "mean_ap": float(runs[0].ap.mean()), "std_recall": float(runs[0].recall_at_1fpr.std())}
    if saved_path.is_file():
        saved = pd.read_csv(saved_path)
        rep["fresh_vs_saved_csv_max_abs_diff_recall"] = float(
            np.abs(runs[0].recall_at_1fpr.to_numpy() - saved.recall_at_1fpr.to_numpy()).max())
        rep["fresh_vs_saved_csv_max_abs_diff_ap"] = float(
            np.abs(runs[0].ap.to_numpy() - saved.ap.to_numpy()).max())
    report["b0_reproduction"] = rep

    prot = fpr_protocol_check(fresh, delay, seed)
    for k, v in prot.items():
        v.to_csv(AUDIT_DIR / f"fpr_protocol_{k}.csv", index=False)
    o, f = prot["oracle_window_threshold_B0_model"], prot["fixed_validation_threshold"]
    report["fpr_protocol"] = {
        "oracle_actual_fpr_mean": float(o.actual_fpr.mean()),
        "oracle_actual_fpr_min_max": [float(o.actual_fpr.min()), float(o.actual_fpr.max())],
        "oracle_recall_mean": float(o.recall.mean()),
        "fixed_threshold_value": float(f.fixed_threshold.iloc[0]),
        "fixed_actual_fpr_mean": float(f.actual_fpr.mean()),
        "fixed_actual_fpr_min_max": [float(f.actual_fpr.min()), float(f.actual_fpr.max())],
        "fixed_recall_mean": float(f.recall.mean()),
        "fixed_precision_mean": float(f.precision.mean()),
    }

    with open(AUDIT_DIR / f"audit_report_d{delay}_seed{seed}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run_audit(), indent=2, default=float))
