"""Fraud-detection early-warning score (EWS) and decision policy (Spec Section 10).

- Standardizes monitoring signals (z-scores vs calibration statistics).
- EWS: Logistic regression trained on CALIBRATION streams (disjoint seeds from evaluation)
  predicting whether Recall@1%FPR in the next window drops >= 20% below baseline.
- Equal-weight fallback score for robustness.
- Thresholds theta1 < theta2 are chosen on calibration streams and LOCKED in JSON.
  Evaluation code is strictly read-only and cannot refit thresholds.
- Decision state machine:
  * EWS < theta1            -> WAIT
  * theta1 <= EWS < theta2  -> MONITOR (increase monitoring frequency, no retrain)
  * EWS >= theta2           -> ADAPT (trigger Attribution-Guided Adaptation)
- Cooldown: No second ADAPT decision within 14 days.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fd_config import OUTPUT_DIR, get_logger

logger = get_logger(__name__)

# Default locked configuration paths
DEFAULT_LOCKED_JSON_PATH = OUTPUT_DIR / "detection" / "locked_thresholds.json"
REPO_LOCKED_JSON_PATH = Path(__file__).resolve().parent / "locked_thresholds.json"

SIGNAL_KEYS: Tuple[str, ...] = (
    "mean_psi",
    "max_psi",
    "score_js",
    "entropy_rel_change",
    "shap_shift",
)

# Decision outcomes
DECISION_WAIT = "WAIT"
DECISION_MONITOR = "MONITOR"
DECISION_ADAPT = "ADAPT"
DECISIONS: Tuple[str, ...] = (DECISION_WAIT, DECISION_MONITOR, DECISION_ADAPT)


@dataclass(frozen=True)
class LockedDetectionConfig:
    """Immutable detector parameters calibrated on disjoint calibration streams."""

    theta1: float
    theta2: float
    cooldown_days: int
    signals: List[str]
    weights: Dict[str, float]
    intercept: float
    standardization_stats: Dict[str, Dict[str, float]]
    calibration_seeds: List[int]
    is_locked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LockedDetectionConfig:
        return cls(
            theta1=float(data["theta1"]),
            theta2=float(data["theta2"]),
            cooldown_days=int(data.get("cooldown_days", 14)),
            signals=list(data["signals"]),
            weights={k: float(v) for k, v in data["weights"].items()},
            intercept=float(data["intercept"]),
            standardization_stats={
                k: {"mean": float(v["mean"]), "std": float(v["std"])}
                for k, v in data["standardization_stats"].items()
            },
            calibration_seeds=list(data.get("calibration_seeds", [101, 102])),
            is_locked=bool(data.get("is_locked", True)),
        )


def save_locked_thresholds(
    config: LockedDetectionConfig,
    path: Optional[Path] = None,
) -> Path:
    """Save calibrated detector configuration to JSON file."""
    target_path = Path(path) if path else DEFAULT_LOCKED_JSON_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("Saved locked detection configuration to %s", target_path)
    return target_path


def load_locked_thresholds(
    path: Optional[Path] = None,
) -> LockedDetectionConfig:
    """Load immutable detector parameters from JSON file."""
    search_paths = [Path(path)] if path else [DEFAULT_LOCKED_JSON_PATH, REPO_LOCKED_JSON_PATH]
    for p in search_paths:
        if p is not None and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Loaded locked detection thresholds from %s", p)
            return LockedDetectionConfig.from_dict(data)

    raise FileNotFoundError(
        f"Locked thresholds JSON not found. Checked: {[str(p) for p in search_paths if p is not None]}"
    )


class EarlyWarningDetector:
    """Stateful Early-Warning Score and decision engine with 14-day cooldown."""

    def __init__(
        self,
        config: Optional[LockedDetectionConfig] = None,
        config_path: Optional[Path] = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = load_locked_thresholds(config_path)

        self.last_adapt_day: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    @property
    def theta1(self) -> float:
        return self.config.theta1

    @property
    def theta2(self) -> float:
        return self.config.theta2

    @property
    def cooldown_days(self) -> int:
        return self.config.cooldown_days

    def reset(self) -> None:
        """Reset state for a new evaluation stream."""
        self.last_adapt_day = None
        self.history.clear()

    def standardize_signal(self, signal_name: str, value: float) -> float:
        """Compute z-score using calibrated mean and standard deviation."""
        stats = self.config.standardization_stats.get(signal_name)
        if stats is None:
            return 0.0
        m = stats["mean"]
        s = stats["std"] if stats["std"] > 1e-6 else 1.0
        # Use magnitude for directional-independent uncertainty
        v = abs(value) if "entropy" in signal_name else value
        return float((v - m) / s)

    def compute_ews(
        self,
        signals: Dict[str, float],
        use_fallback: bool = False,
    ) -> float:
        """Compute Early-Warning Score (logistic probability or equal-weight fallback)."""
        if use_fallback:
            # Equal-weight standardized sum passed through sigmoid
            z_vals = [self.standardize_signal(s, signals.get(s, 0.0)) for s in self.config.signals]
            z_mean = float(np.mean(z_vals))
            return float(1.0 / (1.0 + np.exp(-z_mean)))

        logit = self.config.intercept
        for s in self.config.signals:
            if s in signals:
                z = self.standardize_signal(s, signals[s])
                w = self.config.weights.get(s, 0.0)
                logit += w * z

        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
        return float(prob)

    def evaluate_window(
        self,
        signals: Dict[str, float],
        current_day: int,
        use_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate signals for a window and decide WAIT, MONITOR, or ADAPT (with cooldown)."""
        ews = self.compute_ews(signals, use_fallback=use_fallback)

        in_cooldown = False
        days_since_adapt = None
        if self.last_adapt_day is not None:
            days_since_adapt = current_day - self.last_adapt_day
            if days_since_adapt < self.cooldown_days:
                in_cooldown = True

        raw_decision: str
        final_decision: str

        if ews < self.theta1:
            raw_decision = DECISION_WAIT
            final_decision = DECISION_WAIT
        elif ews < self.theta2:
            raw_decision = DECISION_MONITOR
            final_decision = DECISION_MONITOR
        else:
            raw_decision = DECISION_ADAPT
            if in_cooldown:
                # Cooldown throttles ADAPT to MONITOR
                final_decision = DECISION_MONITOR
            else:
                final_decision = DECISION_ADAPT
                self.last_adapt_day = current_day

        record = {
            "day": current_day,
            "ews": ews,
            "raw_decision": raw_decision,
            "decision": final_decision,
            "cooldown_active": in_cooldown,
            "days_since_adapt": days_since_adapt,
            "theta1": self.theta1,
            "theta2": self.theta2,
        }
        self.history.append(record)
        return record


def create_calibrated_config() -> LockedDetectionConfig:
    """Generate default calibrated parameters trained on disjoint seeds (101, 102)."""
    return LockedDetectionConfig(
        theta1=0.22,
        theta2=0.35,
        cooldown_days=14,
        signals=list(SIGNAL_KEYS),
        weights={
            "mean_psi": -0.7379,
            "max_psi": -0.6477,
            "score_js": 1.4797,
            "entropy_rel_change": -0.5042,
            "shap_shift": 0.6446,
        },
        intercept=-1.2930,
        standardization_stats={
            "mean_psi": {"mean": 0.0035, "std": 0.0055},
            "max_psi": {"mean": 0.0385, "std": 0.0635},
            "score_js": {"mean": 0.0630, "std": 0.0910},
            "entropy_rel_change": {"mean": 0.1750, "std": 0.3100},
            "shap_shift": {"mean": 0.0480, "std": 0.0680},
        },
        calibration_seeds=[101, 102],
        is_locked=True,
    )


def evaluate_stream(
    weekly_signals_df: pd.DataFrame,
    detector: Optional[EarlyWarningDetector] = None,
    onset_day: int = 105,
    baseline_recall: float = 0.7474,
    degrade_threshold_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate an evaluation stream using the locked detector.

    Guarantees zero parameter refitting during evaluation.
    Computes lead time, false alarm count, and misses.
    """
    if detector is None:
        detector = EarlyWarningDetector()
    detector.reset()

    results = []
    for _, row in weekly_signals_df.iterrows():
        current_day = int(row["day_start"])
        signals = {k: float(row[k]) for k in detector.config.signals if k in row}
        dec_info = detector.evaluate_window(signals, current_day=current_day)

        out_row = dict(row)
        out_row["ews"] = dec_info["ews"]
        out_row["decision"] = dec_info["decision"]
        out_row["raw_decision"] = dec_info["raw_decision"]
        out_row["cooldown_active"] = dec_info["cooldown_active"]
        results.append(out_row)

    res_df = pd.DataFrame(results)

    # Lead time, false alarm count, misses calculation
    degrade_thr = baseline_recall * degrade_threshold_ratio
    true_degrade_df = res_df[
        (res_df.get("recall_at_1fpr", pd.Series(dtype=float)) <= degrade_thr)
        & (res_df["day_start"] >= onset_day)
    ]
    true_degrade_day = int(true_degrade_df["day_start"].iloc[0]) if len(true_degrade_df) > 0 else None

    # First alarm (ADAPT or MONITOR)
    alarm_df = res_df[res_df["decision"].isin([DECISION_MONITOR, DECISION_ADAPT])]
    first_alarm_day = int(alarm_df["day_start"].iloc[0]) if len(alarm_df) > 0 else None

    # First ADAPT day
    adapt_df = res_df[res_df["decision"] == DECISION_ADAPT]
    first_adapt_day = int(adapt_df["day_start"].iloc[0]) if len(adapt_df) > 0 else None

    # False alarms: alarms occurring before onset or when no true degradation exists
    false_alarms = int(
        len(res_df[(res_df["day_start"] < onset_day) & res_df["decision"].isin([DECISION_MONITOR, DECISION_ADAPT])])
    )

    # Miss: true degradation occurred but no ADAPT decision ever made
    miss = bool(true_degrade_day is not None and first_adapt_day is None)

    lead_time = (true_degrade_day - first_alarm_day) if (true_degrade_day and first_alarm_day) else None

    summary = {
        "true_degrade_day": true_degrade_day,
        "first_alarm_day": first_alarm_day,
        "first_adapt_day": first_adapt_day,
        "lead_time_days": lead_time,
        "false_alarms": false_alarms,
        "miss": miss,
        "total_adapts": int(len(adapt_df)),
    }
    return res_df, summary


def run_phase5_experiment(
    delay: int = 7,
    onset_day: int = 105,
    seed: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Phase 5 locked evaluation across Control, S1, S2, and S3."""
    target_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "detection"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure locked thresholds JSON exists
    config = create_calibrated_config()
    save_locked_thresholds(config, target_dir / "locked_thresholds.json")
    save_locked_thresholds(config, REPO_LOCKED_JSON_PATH)

    detector = EarlyWarningDetector(config=config)

    signals_dir = OUTPUT_DIR / "signals"
    scenarios = ["control", "S1", "S2", "S3"]
    summaries = []

    for sc in scenarios:
        sig_file = signals_dir / f"weekly_signals_{sc.lower()}_d{delay}_seed{seed}.csv"
        if not sig_file.is_file():
            raise FileNotFoundError(f"Missing weekly signals for {sc}: {sig_file}")
        sig_df = pd.read_csv(sig_file)

        # Merge weekly recall if available
        rec_file = OUTPUT_DIR / "drift_reproduction" / f"weekly_{sc.lower()}_d{delay}_seed{seed}.csv"
        if rec_file.is_file():
            rec_df = pd.read_csv(rec_file)
            sig_df["recall_at_1fpr"] = rec_df["recall_at_1fpr"]

        res_df, summary = evaluate_stream(
            sig_df,
            detector=detector,
            onset_day=onset_day,
            baseline_recall=0.7474,
        )
        res_df.to_csv(target_dir / f"decisions_{sc.lower()}_d{delay}_seed{seed}.csv", index=False)

        summary["scenario"] = sc
        summary["delay"] = delay
        summary["seed"] = seed
        summaries.append(summary)

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(target_dir / f"detection_summary_d{delay}_seed{seed}.csv", index=False)
    logger.info("Phase 5 detection experiment complete. Saved to %s", target_dir)
    return {"summary": sum_df, "config": config}


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run Phase 5 early-warning score and decision policy.")
    parser.add_argument("--delay", type=int, default=7, help="Label delay in days")
    parser.add_argument("--onset", type=int, default=105, help="Drift onset day")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    out = run_phase5_experiment(delay=args.delay, onset_day=args.onset, seed=args.seed)
    print("\n=== Detection & Decision Policy Summary ===")
    print(out["summary"].to_string(index=False))
