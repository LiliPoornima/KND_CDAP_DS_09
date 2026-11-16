"""Experiment grid runner, metrics reporting, and visualization (Spec Section 13).

Provides end-to-end evaluation across:
- Delays d in {7, 14, 28}
- Scenarios in {control, S1, S2, S3}
- Seeds in {1..5}
- Baselines and Adaptation methods (B0..B5, P1, P1-noSHAP, P1-a, P1-b, P1-c)

Reports:
- Recall@1%FPR, AP, AP / prevalence
- True degradation window, first alarm window, first adapt window
- Lead Time to First Alarm (t_degrade - t_first_alarm)
- Adaptation Lag (t_first_adapt - t_degrade)
- Degradation-relative first-ADAPT metric (t_degrade - t_first_adapt)
- False alarm count in no-degradation windows
- Retrain count
- Recovery time (days until recall returns to >= 90% of baseline after degradation)
- Computational wall-clock time (seconds)

Saves results CSVs and publication-quality PNG figures under outputs/fraud_detection/.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fd_adaptation import run_aga
from .fd_baselines import (
    BASELINE_METHODS,
    CBPEstimator,
    METHOD_B0,
    METHOD_B1,
    METHOD_B2,
    METHOD_B3,
    METHOD_B4,
    METHOD_B5,
    METHOD_P1,
    METHOD_P1_A,
    METHOD_P1_B,
    METHOD_P1_C,
    METHOD_P1_NO_SHAP,
    ProdemMetaModel,
    retrain_on_matured_data,
    run_benchmark_method,
)
from .fd_config import (
    COL_DAY,
    COL_FRAUD,
    MODEL_FEATURES,
    OUTPUT_DIR,
    get_logger,
)
from .fd_data_loader import load_handbook
from .fd_detection import EarlyWarningDetector
from .fd_drift_injection import inject_drift
from .fd_evaluation import recall_at_fpr, window_metrics
from .fd_explainability import (
    build_tree_explainer,
    calculate_shap_profile,
    compute_explainability_signals,
)
from .fd_feature_engineering import build_features
from .fd_monitoring import compute_monitoring_signals
from .fd_preprocessing import build_splits, make_windows, matured_cutoff
from .fd_training import predict_scores, train_model

logger = get_logger(__name__)

FIGURES_DIR = OUTPUT_DIR / "figures"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"


def run_single_experiment_stream(
    raw_df: pd.DataFrame,
    scenario: str = "S1",
    delay: int = 7,
    seed: int = 1,
    methods: Sequence[str] = BASELINE_METHODS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all methods on a single (scenario, delay, seed) stream.

    Returns:
    - long_df: window-by-window detailed records (metrics, signals, actions)
    - summary_df: aggregated stream metrics with separated alarm & adapt timings
    """
    sp_cfg = build_splits(pd.DataFrame({COL_DAY: [0, 182]}), delay=delay)["config"]
    onset_day = sp_cfg.drift_onset

    injected = inject_drift(raw_df, scenario=scenario, onset_day=onset_day, seed=seed)
    feat_df = build_features(injected, delay=delay)

    sp = build_splits(feat_df, delay=delay)
    cfg = sp["config"]

    base_model = train_model(feat_df[sp["train"]], seed=seed)
    test_df = feat_df[sp["test"]].reset_index(drop=True)

    ref_mask = (test_df[COL_DAY] >= cfg.reference_start) & (test_df[COL_DAY] <= cfg.reference_end)
    reference_df = test_df[ref_mask].reset_index(drop=True)
    reference_scores = predict_scores(base_model, reference_df)

    explainer = build_tree_explainer(base_model)
    reference_profile = calculate_shap_profile(explainer, reference_df, seed=42)

    # Precompute base model window cache
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

    all_window_records = []
    summaries = []

    for m in methods:
        res_df, summ = run_benchmark_method(
            method=m,
            df=feat_df,
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
        res_df["scenario"] = scenario
        res_df["seed"] = seed
        res_df["delay"] = delay
        all_window_records.append(res_df)

        # Compute separate alarm and adapt timings
        post_df = res_df[res_df["week"] >= (onset_day // 7)]
        base_recall = summ["base_recall"]
        degraded_windows = post_df[post_df["recall_at_1fpr"] < 0.80 * base_recall]

        if len(degraded_windows) > 0:
            degrade_day = int(degraded_windows["day_start"].iloc[0])
        else:
            degrade_day = None

        retrained_windows = post_df[post_df["retrain_triggered"]]
        if len(retrained_windows) > 0:
            first_adapt_day = int(retrained_windows["day_start"].iloc[0])
        else:
            first_adapt_day = None

        # Determine first alarm day (for EWS methods, where decision was MONITOR or ADAPT)
        # For non-EWS triggers, first alarm coincides with first retrain
        first_alarm_day = first_adapt_day
        if m in (METHOD_P1, METHOD_P1_NO_SHAP, METHOD_P1_A, METHOD_P1_B, METHOD_P1_C):
            # Evaluate EWS signals directly to identify MONITOR alarm day
            det = EarlyWarningDetector()
            det.reset()
            for w_idx_iter, (ws_iter, _) in enumerate(windows):
                if ws_iter >= onset_day:
                    cached_w = window_cache[w_idx_iter]
                    sig_combined = dict(cached_w["mon"])
                    sig_combined["shap_shift"] = cached_w["exp"].get("shap_shift", 0.0)
                    dec_res = det.evaluate_window(sig_combined, current_day=ws_iter)
                    if dec_res["decision"] in ("MONITOR", "ADAPT"):
                        first_alarm_day = ws_iter
                        break

        # Calculate metrics with clear sign conventions
        lead_time_to_alarm = (degrade_day - first_alarm_day) if (degrade_day is not None and first_alarm_day is not None) else None
        adaptation_lag = (first_adapt_day - degrade_day) if (degrade_day is not None and first_adapt_day is not None) else None
        degrade_rel_first_adapt = (degrade_day - first_adapt_day) if (degrade_day is not None and first_adapt_day is not None) else None

        # False alarm count in no-degradation windows
        if degrade_day is None:
            false_alarms = summ["retrain_count"]
        else:
            pre_degrade_retrains = post_df[(post_df["day_start"] < degrade_day) & post_df["retrain_triggered"]]
            false_alarms = len(pre_degrade_retrains)

        summ["scenario"] = scenario
        summ["seed"] = seed
        summ["delay"] = delay
        summ["true_degrade_day"] = degrade_day
        summ["first_alarm_day"] = first_alarm_day
        summ["first_adapt_day"] = first_adapt_day
        summ["lead_time_to_first_alarm_days"] = lead_time_to_alarm
        summ["adaptation_lag_days"] = adaptation_lag
        summ["degrade_rel_first_adapt_days"] = degrade_rel_first_adapt
        summ["false_alarms"] = false_alarms
        summaries.append(summ)

    long_df = pd.concat(all_window_records, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    return long_df, summary_df


def generate_experiment_figures(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """Generate publication-ready diagnostic charts per Spec Section 13."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        METHOD_B0: "#7f7f7f",       # Gray (Static)
        METHOD_B1: "#ff7f0e",       # Orange (Periodic)
        METHOD_B2: "#2ca02c",       # Green (PSI)
        METHOD_B3: "#8c564b",       # Brown (CBPE)
        METHOD_B4: "#9467bd",       # Purple (SHAP Shift)
        METHOD_B5: "#e377c2",       # Pink (PRODEM)
        METHOD_P1: "#1f77b4",       # Blue (AGA Full)
        METHOD_P1_NO_SHAP: "#17becf", # Cyan (noSHAP)
        METHOD_P1_B: "#d62728",     # Red (DROP_D)
        METHOD_P1_C: "#bcbd22",     # Olive (RECENCY)
    }

    # 1. Recall Trajectories across Time for Key Scenarios
    for sc in long_df["scenario"].unique():
        for d in long_df["delay"].unique():
            sc_long = long_df[(long_df["scenario"] == sc) & (long_df["delay"] == d)]
            if len(sc_long) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            key_methods = [METHOD_B0, METHOD_B1, METHOD_B2, METHOD_B4, METHOD_P1, METHOD_P1_B]

            for m in key_methods:
                m_sub = sc_long[sc_long["method"] == m]
                if len(m_sub) == 0:
                    continue
                # Average across seeds
                grouped = m_sub.groupby("day_start")["recall_at_1fpr"].agg(["mean", "std"]).reset_index()
                ax.plot(
                    grouped["day_start"],
                    grouped["mean"],
                    label=m,
                    color=colors.get(m, "#333333"),
                    linewidth=2.2 if m == METHOD_P1 else 1.5,
                    linestyle="--" if "B0" in m else "-",
                )

            onset = 105 if d == 7 else 112
            ax.axvline(onset, color="black", linestyle=":", linewidth=1.8, label=f"Drift Onset (Day {onset})")
            ax.set_title(f"Recall@1%FPR Over Time — Scenario: {sc.upper()} (Delay {d})", fontsize=13, fontweight="bold")
            ax.set_xlabel("Transaction Day", fontsize=11)
            ax.set_ylabel("Recall @ 1% FPR", fontsize=11)
            ax.set_ylim(0.3, 0.9)
            ax.legend(loc="lower left" if sc in ("S1", "S2") else "best", frameon=True, fontsize=9)
            plt.tight_layout()

            plot_path = output_dir / f"recall_trajectory_{sc}_d{d}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            generated_plots.append(plot_path)

    # 2. Retraining Count Comparison Bars (Highlighting False Alarms in S3 & Control)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    d7_summ = summary_df[summary_df["delay"] == summary_df["delay"].min()]
    scenarios_order = ["control", "S1", "S2", "S3"]
    methods_to_plot = [METHOD_B0, METHOD_B1, METHOD_B2, METHOD_B4, METHOD_B5, METHOD_P1]

    bar_width = 0.12
    x_indices = np.arange(len(scenarios_order))

    for i, m in enumerate(methods_to_plot):
        m_means = []
        for sc in scenarios_order:
            val = d7_summ[(d7_summ["scenario"] == sc) & (d7_summ["method"] == m)]["retrain_count"].mean()
            m_means.append(val if not math.isnan(val) else 0.0)
        ax.bar(
            x_indices + (i - len(methods_to_plot) / 2) * bar_width + bar_width / 2,
            m_means,
            width=bar_width,
            label=m,
            color=colors.get(m, "#333"),
        )

    ax.set_title("Retraining Events by Scenario (Delay 7) — Note False Alarms in S3", fontsize=13, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([sc.upper() for sc in scenarios_order], fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Number of Retrains", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    retrain_plot_path = output_dir / "retrain_counts_by_scenario.png"
    fig.savefig(retrain_plot_path)
    plt.close(fig)
    generated_plots.append(retrain_plot_path)

    # 3. Delay Sensitivity (Relative Retention % across Delays 7, 14, 28)
    delays_avail = sorted(summary_df["delay"].unique())
    if len(delays_avail) > 1:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for m in [METHOD_B0, METHOD_B1, METHOD_P1, METHOD_P1_C]:
            means_per_delay = []
            for d in delays_avail:
                val = summary_df[(summary_df["delay"] == d) & (summary_df["scenario"] == "S1") & (summary_df["method"] == m)]["relative_retention_pct"].mean()
                means_per_delay.append(val)
            ax.plot(delays_avail, means_per_delay, marker="o", label=m, color=colors.get(m, "#333"), linewidth=2.0)

        ax.set_title("Label Delay Sensitivity in S1 Concept Drift", fontsize=13, fontweight="bold")
        ax.set_xlabel("Label Delay d (Days)", fontsize=11)
        ax.set_ylabel("Post-Onset Retention (%)", fontsize=11)
        ax.set_xticks(delays_avail)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()

        sens_plot_path = output_dir / "delay_sensitivity_s1.png"
        fig.savefig(sens_plot_path)
        plt.close(fig)
        generated_plots.append(sens_plot_path)

    return generated_plots


def run_full_experiment_grid(
    delays: Sequence[int] = (7, 14, 28),
    scenarios: Sequence[str] = ("control", "S1", "S2", "S3"),
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    methods: Sequence[str] = BASELINE_METHODS,
    generate_plots: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Phase 7 experiment grid across Delays, Scenarios, and Seeds."""
    out_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Handbook data for Full Experiment Grid...")
    raw = load_handbook()

    all_long_dfs = []
    all_summary_dfs = []

    for d in delays:
        for sc in scenarios:
            for sd in seeds:
                logger.info("Running Experiment Stream: Delay %d, Scenario '%s', Seed %d...", d, sc, sd)
                long_df, summ_df = run_single_experiment_stream(
                    raw_df=raw,
                    scenario=sc,
                    delay=d,
                    seed=sd,
                    methods=methods,
                )
                all_long_dfs.append(long_df)
                all_summary_dfs.append(summ_df)

    combined_long = pd.concat(all_long_dfs, ignore_index=True)
    combined_summary = pd.concat(all_summary_dfs, ignore_index=True)

    combined_long.to_csv(out_dir / "grid_results_long.csv", index=False)
    combined_summary.to_csv(out_dir / "grid_results_summary.csv", index=False)
    logger.info("Saved combined grid results (%d summary rows) to %s", len(combined_summary), out_dir)

    if generate_plots:
        plots = generate_experiment_figures(combined_long, combined_summary, output_dir=FIGURES_DIR)
        logger.info("Generated %d diagnostic plots under %s", len(plots), FIGURES_DIR)

    return combined_long, combined_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run Component 04 Full Experiment Grid (Spec Section 13)")
    parser.add_argument("--delay", type=int, default=7, help="Label delay in days (7, 14, 28)")
    parser.add_argument("--scenario", type=str, default="S1", help="Drift scenario (control, S1, S2, S3, or all)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (1..5, or -1 for all)")
    parser.add_argument("--all-grid", action="store_true", help="Run full 3-delay x 4-scenario x 5-seed grid")
    parser.add_argument("--all-seeds", action="store_true", help="Run seeds 1..5 for selected delay & scenario")
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG figure generation")
    args = parser.parse_args()

    if args.all_grid:
        delays = [7, 14, 28]
        scenarios = ["control", "S1", "S2", "S3"]
        seeds = [1, 2, 3, 4, 5]
    else:
        delays = [args.delay]
        scenarios = ["control", "S1", "S2", "S3"] if args.scenario.lower() == "all" else [args.scenario]
        seeds = [1, 2, 3, 4, 5] if (args.all_seeds or args.seed == -1) else [args.seed]

    long_res, summ_res = run_full_experiment_grid(
        delays=delays,
        scenarios=scenarios,
        seeds=seeds,
        generate_plots=not args.no_plots,
    )
    print("\n=== Experiment Grid Execution Summary ===")
    print(summ_res[["scenario", "delay", "seed", "method", "post_onset_mean_recall", "retrain_count", "lead_time_to_first_alarm_days", "adaptation_lag_days"]].head(25).to_string(index=False))
