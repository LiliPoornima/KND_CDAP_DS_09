"""End-to-end Phase 1 pipeline: load -> validate -> features/target -> split -> tune ->
train RF + XGBoost -> evaluate -> SHAP -> reliability checks -> save artifacts.

Run from the repository root:

    python -m backend.src.financial_capability.fc_run_pipeline

Every number written to disk is computed in this run; nothing is hard-coded.
Use ``--quick`` for a fast smoke run with tiny grids (results then are NOT final).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

from .fc_config import (
    CAPABILITY_GROUPS,
    CLASS_NAMES,
    CV_FOLDS,
    DOWNLOAD_URL,
    FEATURE_NAMES,
    FEATURE_SPEC_BY_NAME,
    FORBIDDEN_MODEL_COLUMNS,
    LOW_MAX_SCORE,
    MEDIUM_MAX_SCORE,
    METADATA_FILE,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_FOREST_FILE,
    RANDOM_SEED,
    RAW_CSV_PATH,
    SEARCH_CONFIG,
    SELECTED_MODEL_FILE,
    SHAP_SAMPLE_SIZE,
    SOURCE_URL,
    STABILITY_SEEDS,
    STABILITY_TOP_K,
    TEST_SIZE,
    XGBOOST_FILE,
    get_logger,
)
from .fc_data_loader import load_cfpb_data
from .fc_evaluation import (
    confusion_frame,
    evaluate_model,
    majority_baseline,
    metrics_to_frame,
    per_class_frame,
    subgroup_metrics,
)
from .fc_explainability import FeatureExplainer, stability_summary, top_features
from .fc_feature_selection import (
    assert_no_forbidden_features,
    data_dictionary,
    excluded_columns_report,
    select_features_and_target,
)
from .fc_preprocessing import class_distribution, split_data
from .fc_train import (
    _make_rf,
    _make_xgb,
    build_pipeline,
    fit_pipeline,
    select_best_model,
    tune_random_forest,
    tune_xgboost,
)
from .fc_validation import validate_dataset

logger = get_logger(__name__)

QUICK_RF_GRID = {"n_estimators": [100], "max_depth": [10], "min_samples_leaf": [3],
                 "max_features": ["sqrt"]}
QUICK_XGB_GRID = {"n_estimators": [80], "max_depth": [3], "learning_rate": [0.08],
                  "subsample": [0.8], "colsample_bytree": [0.8]}


# --------------------------------------------------------------------------- helpers
def _clean(obj: Any) -> Any:
    """Make nested data JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    return obj


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Tiny markdown table writer (avoids an extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                         else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _factory(name: str):
    return _make_rf if name == "Random Forest" else _make_xgb


def _fit_family(name: str, params: Dict[str, Any], X, y, seed: int):
    return fit_pipeline(build_pipeline(_factory(name)(seed, **params)), X, y)


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- reliability
def explanation_stability(name: str, params: Dict[str, Any], X, y) -> Dict[str, Any]:
    """Re-split with several seeds, refit, and compare the top SHAP features and scores."""
    top_sets, scores = [], []
    for seed in STABILITY_SEEDS:
        Xtr, Xte, ytr, yte = split_data(X, y, seed=seed)
        model = _fit_family(name, params, Xtr, ytr, seed)
        metrics = evaluate_model(model, Xte, yte)
        sample = Xte.sample(n=min(300, len(Xte)), random_state=seed)
        imp = FeatureExplainer(model).explain_global(sample)
        top_sets.append(top_features(imp, STABILITY_TOP_K))
        scores.append({"seed": seed, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    summary = stability_summary(top_sets)
    summary["top_k"] = STABILITY_TOP_K
    summary["per_seed_scores"] = scores
    f1s = [s["macro_f1"] for s in scores]
    summary["macro_f1_mean"] = float(np.mean(f1s))
    summary["macro_f1_std"] = float(np.std(f1s))
    return summary


def feature_group_ablation(name: str, params: Dict[str, Any], Xtr, ytr, Xte, yte) -> pd.DataFrame:
    """Compare capability-only, background-only and all features (diagnostic, single split)."""
    capability = [f for f in FEATURE_NAMES if FEATURE_SPEC_BY_NAME[f].group in CAPABILITY_GROUPS]
    context = [f for f in FEATURE_NAMES if f not in capability]
    rows = []
    for label, feats in (("Capability questions only", capability),
                         ("Background (context) only", context),
                         ("All approved features", list(FEATURE_NAMES))):
        pipe = fit_pipeline(build_pipeline(_factory(name)(RANDOM_SEED, **params), feats),
                            Xtr[feats], ytr)
        m = evaluate_model(pipe, Xte[feats], yte)
        rows.append({"feature_set": label, "n_features": len(feats),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main run
def run(quick: bool = False, data_path: Optional[Path] = None,
        model_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run every Phase 1 modelling stage and save artifacts; return the metadata dict."""
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_path) if data_path else RAW_CSV_PATH

    _log("[1/8] Loading and validating the dataset")
    raw = load_cfpb_data(csv_path)
    report = validate_dataset(raw)
    _log(f"      rows={report.n_rows} columns={report.n_columns} duplicates={report.n_duplicate_rows} "
         f"missing_cells={report.n_missing_cells} special_target_codes={report.target_special_code_counts}")

    _log("[2/8] Selecting leakage-safe features and building the target")
    X, y = select_features_and_target(raw)
    assert_no_forbidden_features(X.columns)
    dist_all = class_distribution(y)
    data_dictionary().to_csv(output_dir / "data_dictionary.csv", index=False)
    excluded_columns_report(raw).to_csv(output_dir / "excluded_columns.csv", index=False)
    _log(f"      X={X.shape} class distribution={dist_all}")

    _log("[3/8] Stratified train/test split")
    Xtr, Xte, ytr, yte = split_data(X, y)
    dist_train, dist_test = class_distribution(ytr), class_distribution(yte)
    _log(f"      train={len(Xtr)} test={len(Xte)}")

    _log("[4/8] Cross-validated tuning on the training set (RF and XGBoost)")
    rf_grid = QUICK_RF_GRID if quick else SEARCH_CONFIG.rf_grid
    xgb_grid = QUICK_XGB_GRID if quick else SEARCH_CONFIG.xgb_grid
    rf_cv = tune_random_forest(Xtr, ytr, CV_FOLDS, RANDOM_SEED, rf_grid)
    xgb_cv = tune_xgboost(Xtr, ytr, CV_FOLDS, RANDOM_SEED, xgb_grid)
    pd.concat([rf_cv, xgb_cv]).assign(params=lambda d: d["params"].astype(str)).to_csv(
        output_dir / "cv_results.csv", index=False)
    best = {"Random Forest": rf_cv.iloc[0], "XGBoost": xgb_cv.iloc[0]}
    for n, r in best.items():
        _log(f"      {n}: best CV macro-F1={r['cv_macro_f1_mean']:.4f} "
             f"(+/-{r['cv_macro_f1_std']:.4f}) params={r['params']}")
    selected = select_best_model(rf_cv, xgb_cv)

    _log("[5/8] Fitting both models on the full training set and testing once")
    models = {n: _fit_family(n, dict(best[n]["params"]), Xtr, ytr, RANDOM_SEED) for n in best}
    results = {n: evaluate_model(m, Xte, yte) for n, m in models.items()}
    baseline = majority_baseline(Xtr, ytr, Xte, yte)
    metrics_to_frame(results).to_csv(output_dir / "test_metrics.csv", index=False)
    per_class_frame(results).to_csv(output_dir / "test_per_class_metrics.csv", index=False)
    for n, m in results.items():
        confusion_frame(m).to_csv(
            output_dir / f"confusion_matrix_{n.lower().replace(' ', '_')}.csv")
        _log(f"      {n}: accuracy={m['accuracy']:.4f} macro_F1={m['macro_f1']:.4f} "
             f"balanced_acc={m['balanced_accuracy']:.4f} ROC-AUC={m['roc_auc_ovr_macro']:.4f}")
    _log(f"      majority baseline accuracy={baseline['accuracy']:.4f}; selected (by CV) = {selected}")

    _log("[6/8] SHAP global explanation for the selected model")
    chosen_model = models[selected]
    sample = Xte.sample(n=min(SHAP_SAMPLE_SIZE, len(Xte)), random_state=RANDOM_SEED)
    importance = FeatureExplainer(chosen_model).explain_global(sample)
    importance.to_csv(output_dir / "shap_global_importance.csv", index=False)

    _log("[7/8] Reliability checks: subgroups, explanation stability, feature-group ablation")
    sub = subgroup_metrics(chosen_model, Xte, yte)
    sub.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    chosen_params = dict(best[selected]["params"])
    stability = explanation_stability(selected, chosen_params, X, y)
    ablation = feature_group_ablation(selected, chosen_params, Xtr, ytr, Xte, yte)
    ablation.to_csv(output_dir / "feature_group_ablation.csv", index=False)
    _log(f"      SHAP top-{STABILITY_TOP_K} mean Jaccard overlap across {len(STABILITY_SEEDS)} splits = "
         f"{stability['mean_pairwise_jaccard']:.3f}")

    _log("[8/8] Saving artifacts and metadata")
    joblib.dump(models["Random Forest"], model_dir / RANDOM_FOREST_FILE)
    joblib.dump(models["XGBoost"], model_dir / XGBOOST_FILE)
    joblib.dump(chosen_model, model_dir / SELECTED_MODEL_FILE)

    metadata = {
        "component": "05 - Explainable Financial Capability Assessment (Phase 1)",
        "training_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_run": quick,
        "random_seed": RANDOM_SEED,
        "package_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
        "dataset": {
            "name": "CFPB National Financial Well-Being Survey (NFWBS) 2016 public-use file",
            "file_name": csv_path.name,
            "source_page": SOURCE_URL,
            "direct_link": DOWNLOAD_URL,
            "file_sha256": _sha256(csv_path),
            "file_last_modified_utc": datetime.fromtimestamp(
                csv_path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            "validation": report.to_dict(),
        },
        "target": {
            "source_column": "FWBscore",
            "class_names": list(CLASS_NAMES),
            "class_to_int": {n: i for i, n in enumerate(CLASS_NAMES)},
            "band_rule": {"Low": f"score <= {LOW_MAX_SCORE}",
                          "Medium": f"{LOW_MAX_SCORE + 1} to {MEDIUM_MAX_SCORE}",
                          "High": f"score >= {MEDIUM_MAX_SCORE + 1}"},
            "band_rule_status": "Proposed 3-class grouping; cut points to be confirmed with supervisor.",
            "class_distribution_all": dist_all,
            "class_distribution_train": dist_train,
            "class_distribution_test": dist_test,
            "rows_dropped_invalid_target": int(len(raw) - len(X)),
        },
        "features": {
            "names": list(FEATURE_NAMES),
            "n_features": len(FEATURE_NAMES),
            "forbidden_columns": list(FORBIDDEN_MODEL_COLUMNS),
        },
        "split": {"test_size": TEST_SIZE, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                  "stratified": True, "cv_folds": CV_FOLDS},
        "tuning": {
            "rf_best": {"params": best["Random Forest"]["params"],
                        "cv_macro_f1_mean": best["Random Forest"]["cv_macro_f1_mean"],
                        "cv_macro_f1_std": best["Random Forest"]["cv_macro_f1_std"]},
            "xgb_best": {"params": best["XGBoost"]["params"],
                         "cv_macro_f1_mean": best["XGBoost"]["cv_macro_f1_mean"],
                         "cv_macro_f1_std": best["XGBoost"]["cv_macro_f1_std"]},
            "rf_top5": rf_cv.head(5).assign(params=lambda d: d["params"].astype(str)),
            "xgb_top5": xgb_cv.head(5).assign(params=lambda d: d["params"].astype(str)),
            "n_rf_candidates": int(len(rf_cv)), "n_xgb_candidates": int(len(xgb_cv)),
        },
        "selected_model": selected,
        "selection_rule": "Highest mean cross-validated macro-F1 on the training set (test set not used).",
        "test_metrics": results,
        "majority_baseline": baseline,
        "subgroup_metrics": sub,
        "explanation_stability": stability,
        "feature_group_ablation": ablation,
        "global_importance": importance,
        "limitations": [
            "US adults surveyed in 2016; validity for Sri Lankan users has not been established.",
            "Cross-sectional survey: associations only, no causal claims.",
            "Self-reported answers; capability rules are proposed and need expert review.",
            "Results are educational support, not financial advice or a diagnosis.",
        ],
    }
    (model_dir / METADATA_FILE).write_text(
        json.dumps(_clean(metadata), indent=2), encoding="utf-8")
    _write_summary(output_dir / "RESULTS_SUMMARY.md", metadata, results, rf_cv, xgb_cv,
                   baseline, sub, stability, ablation, importance)
    _log("Done. Artifacts in " + str(model_dir))
    return metadata


def _write_summary(path: Path, meta: Dict[str, Any], results, rf_cv, xgb_cv, baseline,
                   sub, stability, ablation, importance) -> None:
    """Human-readable report of the real run (all values come from this run)."""
    v = meta["dataset"]["validation"]
    t = meta["target"]
    parts = [
        "# Component 05 - Phase 1 results (generated by fc_run_pipeline.py)",
        f"Trained: {meta['training_date_utc']} | seed {meta['random_seed']} | quick run: {meta['quick_run']}",
        "", "## 1. Dataset (computed from the downloaded file)",
        f"- File: `{meta['dataset']['file_name']}` (sha256 `{meta['dataset']['file_sha256'][:16]}...`)",
        f"- Rows: {v['n_rows']} | Columns: {v['n_columns']} | Duplicate rows: {v['n_duplicate_rows']} "
        f"| Missing cells: {v['n_missing_cells']}",
        f"- FWBscore special codes (target unavailable, rows dropped): {v['target_special_code_counts']}",
        f"- Valid target rows: {v['n_valid_target_rows']} (score range {v['target_min']:.0f}-{v['target_max']:.0f})",
        f"- Out-of-codebook values in selected features: {v['out_of_codebook_values'] or 'none'}",
        "", "## 2. Target classes",
        f"- Rule: {t['band_rule']} ({t['band_rule_status']})",
        f"- All: {t['class_distribution_all']}",
        f"- Train: {t['class_distribution_train']}",
        f"- Test: {t['class_distribution_test']}",
        "", "## 3. Cross-validation (training set only)",
        "### Random Forest (top 5)", _md_table(rf_cv.head(5).assign(params=lambda d: d["params"].astype(str))),
        "### XGBoost (top 5)", _md_table(xgb_cv.head(5).assign(params=lambda d: d["params"].astype(str))),
        f"**Selected model (highest CV macro-F1): {meta['selected_model']}**",
        "", "## 4. Held-out test results",
        _md_table(metrics_to_frame(results)),
        f"Majority-class baseline: accuracy {baseline['accuracy']:.4f}, macro-F1 {baseline['macro_f1']:.4f}",
        "", "### Per-class metrics", _md_table(per_class_frame(results)),
    ]
    for n, m in results.items():
        parts += [f"### Confusion matrix - {n} (rows actual, columns predicted)",
                  _md_table(confusion_frame(m).reset_index().rename(columns={"index": ""}), "{:.0f}")]
    parts += [
        "", "## 5. Subgroup performance (selected model; differences only, not a fairness claim)",
        _md_table(sub),
        "", "## 6. Explanation stability (selected model)",
        f"- Mean pairwise Jaccard overlap of the top-{stability['top_k']} SHAP features across "
        f"{stability['n_splits']} re-splits: {stability['mean_pairwise_jaccard']:.3f}",
        f"- Macro-F1 across re-splits: {stability['macro_f1_mean']:.4f} +/- {stability['macro_f1_std']:.4f}",
        "", "## 7. Feature-group ablation (diagnostic, single split)", _md_table(ablation),
        "", "## 8. Top global SHAP features (selected model)",
        _md_table(importance[["feature", "label", "group", "mean_abs_shap"]].head(12)),
        "", "## Limitations", *[f"- {s}" for s in meta["limitations"]],
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train Component 05 Phase 1 models.")
    parser.add_argument("--quick", action="store_true", help="tiny grids for a smoke run")
    parser.add_argument("--data", type=Path, default=None, help="path to NFWBS_PUF_2016_data.csv")
    args = parser.parse_args(argv)
    try:
        run(quick=args.quick, data_path=args.data)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

