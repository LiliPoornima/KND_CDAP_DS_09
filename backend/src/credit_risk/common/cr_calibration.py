"""Calibration metrics and reliability-bin summaries."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def cr_calibration_summary(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Calculate Brier score, log loss, ECE, and reliability-bin values.

    This function evaluates supplied probabilities; it does not fit or
    calibrate a model. Callers are responsible for passing predictions from
    a validation or held-out test partition rather than training data.
    ``strategy`` may be ``"uniform"`` or ``"quantile"``.
    """

    if (
        not isinstance(n_bins, int)
        or isinstance(n_bins, bool)
        or n_bins < 2
    ):
        raise ValueError("n_bins must be an integer of at least 2.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    try:
        truth = np.asarray(list(y_true), dtype=float)
        probability = np.asarray(list(y_probability), dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Targets and probabilities must be numeric iterables.") from exc

    if truth.ndim != 1 or probability.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional.")
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("Targets and probabilities must have equal non-zero length.")
    if not np.isfinite(truth).all() or not np.isfinite(probability).all():
        raise ValueError("Targets and probabilities must be finite.")
    if not set(np.unique(truth)).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only binary labels: 0 and 1.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if strategy == "uniform":
        bin_ids = pd.cut(
            probability,
            bins=np.linspace(0.0, 1.0, n_bins + 1),
            labels=False,
            include_lowest=True,
        )
        expected_bins = list(range(n_bins))
    elif np.unique(probability).size < 2:
        bin_ids = np.zeros(probability.size, dtype=int)
        expected_bins = [0]
    else:
        bin_ids = pd.qcut(
            probability,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        expected_bins = sorted(int(value) for value in pd.Series(bin_ids).dropna().unique())

    frame = pd.DataFrame(
        {
            "bin": np.asarray(bin_ids, dtype=float),
            "y_true": truth,
            "y_probability": probability,
        }
    )
    binned = (
        frame.groupby("bin", observed=True)
        .agg(
            count=("y_true", "size"),
            mean_predicted_probability=("y_probability", "mean"),
            observed_rate=("y_true", "mean"),
        )
        .reindex(expected_bins)
    )
    binned.index.name = "bin"
    binned["count"] = binned["count"].fillna(0).astype(int)
    binned["absolute_calibration_gap"] = (
        binned["mean_predicted_probability"] - binned["observed_rate"]
    ).abs()
    occupied = binned["count"] > 0
    ece = float(
        (
            binned.loc[occupied, "count"]
            / truth.size
            * binned.loc[occupied, "absolute_calibration_gap"]
        ).sum()
    )
    clipped = np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    return {
        "brier_score": float(np.mean((probability - truth) ** 2)),
        "log_loss": float(
            -np.mean(
                truth * np.log(clipped)
                + (1.0 - truth) * np.log1p(-clipped)
            )
        ),
        "expected_calibration_error": ece,
        "calibration_bins": binned.reset_index(),
        "n_bins": int(n_bins),
        "binning_strategy": strategy,
    }
