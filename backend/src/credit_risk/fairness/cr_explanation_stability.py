"""Quantify global feature-attribution stability across repeated fits."""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def cr_compare_explanation_stability(
    attribution_runs: list[pd.DataFrame],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare feature attributions from repeated fits on the same rows.

    Each DataFrame must contain signed per-row feature attributions and use
    the same row index and feature columns. Global importance is mean
    absolute attribution. Pairwise results report Spearman rank correlation,
    top-k Jaccard overlap, and cosine similarity of the signed attribution
    vectors. Attributions describe the fitted models; they are not causal.
    """

    if not isinstance(attribution_runs, list) or len(attribution_runs) < 2:
        raise ValueError("At least two attribution runs are required.")
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer.")

    reference = attribution_runs[0]
    if not isinstance(reference, pd.DataFrame) or reference.empty:
        raise TypeError("Each attribution run must be a non-empty DataFrame.")
    if not reference.columns.is_unique:
        raise ValueError("Attribution feature columns must be unique.")

    feature_names = list(reference.columns)
    row_index = reference.index
    run_importances: list[pd.Series] = []

    for run_number, values in enumerate(attribution_runs, start=1):
        if not isinstance(values, pd.DataFrame) or values.empty:
            raise TypeError(
                f"Attribution run {run_number} must be a non-empty DataFrame."
            )
        if not values.columns.is_unique:
            raise ValueError(
                f"Attribution run {run_number} has duplicate feature columns."
            )
        if list(values.columns) != feature_names or not values.index.equals(row_index):
            raise ValueError(
                "All attribution runs must use identical feature columns "
                "in the same order and identical row indexes."
            )
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Attributions must contain only finite values.")
        run_importances.append(
            pd.Series(np.abs(numeric).mean(axis=0), index=feature_names)
        )

    importance_rows = []
    for run_number, importance in enumerate(run_importances, start=1):
        ranks = importance.rank(method="average", ascending=False)
        for feature in feature_names:
            importance_rows.append(
                {
                    "run": run_number,
                    "feature": feature,
                    "mean_absolute_attribution": float(importance[feature]),
                    "rank": float(ranks[feature]),
                }
            )
    importance_table = pd.DataFrame(importance_rows)

    effective_k = min(top_k, len(feature_names))
    pairwise_rows = []
    for left, right in combinations(range(len(attribution_runs)), 2):
        left_importance = run_importances[left]
        right_importance = run_importances[right]
        left_ranks = left_importance.rank(method="average", ascending=False)
        right_ranks = right_importance.rank(method="average", ascending=False)
        rank_correlation = left_ranks.corr(right_ranks, method="pearson")

        left_top = set(
            left_importance.nlargest(effective_k, keep="first").index
        )
        right_top = set(
            right_importance.nlargest(effective_k, keep="first").index
        )
        union = left_top | right_top

        left_vector = attribution_runs[left].to_numpy(dtype=float).ravel()
        right_vector = attribution_runs[right].to_numpy(dtype=float).ravel()
        denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
        cosine = (
            float(np.dot(left_vector, right_vector) / denominator)
            if denominator > 0
            else np.nan
        )

        pairwise_rows.append(
            {
                "run_a": left + 1,
                "run_b": right + 1,
                "spearman_rank_correlation": (
                    float(rank_correlation)
                    if pd.notna(rank_correlation)
                    else np.nan
                ),
                "top_k": effective_k,
                "top_k_jaccard_overlap": (
                    len(left_top & right_top) / len(union) if union else np.nan
                ),
                "attribution_cosine_similarity": cosine,
            }
        )

    pairwise = pd.DataFrame(pairwise_rows)
    summary = {
        "runs_compared": len(attribution_runs),
        "features_compared": len(feature_names),
        "top_k": effective_k,
        "mean_spearman_rank_correlation": float(
            pairwise["spearman_rank_correlation"].mean()
        ),
        "mean_top_k_jaccard_overlap": float(
            pairwise["top_k_jaccard_overlap"].mean()
        ),
        "mean_attribution_cosine_similarity": float(
            pairwise["attribution_cosine_similarity"].mean()
        ),
    }
    return {
        "summary": summary,
        "feature_importance_by_run": importance_table,
        "pairwise_stability": pairwise,
    }
