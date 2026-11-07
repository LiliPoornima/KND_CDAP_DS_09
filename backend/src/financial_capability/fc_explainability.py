"""SHAP explanations (global and individual) for the trained pipeline.

SHAP describes how the MODEL behaves. It does not prove why a real person's
finances are what they are, and a large SHAP value is not automatically a
capability weakness (capability gaps come from fc_capability_rules.py).

One-hot columns are summed back to their original survey question so the
explanation is expressed per question, in the same order as FEATURE_NAMES.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from .fc_config import CLASS_NAMES, FEATURE_NAMES, FEATURE_SPEC_BY_NAME, NOMINAL_FEATURES


def _to_3d(values: Any) -> np.ndarray:
    """Normalise SHAP output to shape (n_samples, n_transformed_features, n_classes)."""
    if isinstance(values, list):  # older SHAP: one array per class
        return np.stack(values, axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 2:  # single output: add a class axis
        arr = arr[:, :, None]
    return arr


def original_feature_of(transformed_name: str) -> str:
    """Map e.g. 'cat__EMPLOY_2' -> 'EMPLOY' and 'num__SAVEHABIT' -> 'SAVEHABIT'."""
    name = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if transformed_name.startswith("cat__"):
        for nominal in sorted(NOMINAL_FEATURES, key=len, reverse=True):
            if name == nominal or name.startswith(nominal + "_"):
                return nominal
    return name


class FeatureExplainer:
    """SHAP TreeExplainer wrapper that reports per-question contributions."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.transformed_names: List[str] = list(self.prep.get_feature_names_out())
        self.origin: List[str] = [original_feature_of(n) for n in self.transformed_names]
        self.feature_order: List[str] = [f for f in FEATURE_NAMES if f in set(self.origin)]
        self.explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------ core
    def shap_by_feature(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values grouped per original question: (n_samples, n_features, n_classes).

        Feature axis order == ``self.feature_order`` (aligned with FEATURE_NAMES).
        """
        matrix = self.prep.transform(X)
        raw = _to_3d(self.explainer.shap_values(matrix, check_additivity=False))
        if raw.shape[1] != len(self.transformed_names):
            raise ValueError("SHAP feature count does not match transformed columns.")
        out = np.zeros((raw.shape[0], len(self.feature_order), raw.shape[2]))
        index = {f: i for i, f in enumerate(self.feature_order)}
        for col, origin in enumerate(self.origin):
            out[:, index[origin], :] += raw[:, col, :]
        return out

    # ------------------------------------------------------------------ global
    def explain_global(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per question, overall and per class, sorted by overall importance."""
        values = self.shap_by_feature(X_sample)
        abs_mean = np.abs(values).mean(axis=0)  # (features, classes)
        frame = pd.DataFrame(abs_mean, columns=[f"mean_abs_shap_{c}" for c in CLASS_NAMES],
                             index=self.feature_order)
        frame.insert(0, "mean_abs_shap", abs_mean.mean(axis=1))
        frame.insert(0, "label", [FEATURE_SPEC_BY_NAME[f].label for f in frame.index])
        frame.insert(1, "group", [FEATURE_SPEC_BY_NAME[f].group for f in frame.index])
        frame.index.name = "feature"
        return frame.sort_values("mean_abs_shap", ascending=False).reset_index()

    # ------------------------------------------------------------------ local
    def explain_individual(self, one_row: pd.DataFrame,
                           class_index: Optional[int] = None) -> Dict[str, Any]:
        """Per-question SHAP for one assessment, for the predicted (or chosen) class."""
        if len(one_row) != 1:
            raise ValueError("explain_individual expects exactly one row.")
        proba = self.pipeline.predict_proba(one_row)[0]
        if class_index is None:
            class_index = int(np.argmax(proba))
        values = self.shap_by_feature(one_row)[0, :, class_index]
        rows = []
        for feat, val in zip(self.feature_order, values):
            spec = FEATURE_SPEC_BY_NAME[feat]
            raw_value = one_row.iloc[0][feat]
            shown = "Not answered" if pd.isna(raw_value) else spec.options.get(
                int(raw_value) if float(raw_value).is_integer() else raw_value, str(raw_value))
            rows.append({
                "feature": feat, "label": spec.label, "answer": shown,
                "shap_value": float(val),
                "effect": "pushes toward" if val > 0 else ("pushes away from" if val < 0 else "neutral"),
            })
        table = (pd.DataFrame(rows)
                 .assign(abs_value=lambda d: d["shap_value"].abs())
                 .sort_values("abs_value", ascending=False)
                 .drop(columns="abs_value").reset_index(drop=True))
        return {
            "class_index": class_index,
            "class_name": CLASS_NAMES[class_index],
            "probability": float(proba[class_index]),
            "table": table,
            "base_value": float(np.atleast_1d(self.explainer.expected_value)[class_index]),
        }


def explain_global(pipeline, X_sample: pd.DataFrame,
                   explainer: Optional[FeatureExplainer] = None) -> pd.DataFrame:
    """Global importance table (see ``FeatureExplainer.explain_global``)."""
    return (explainer or FeatureExplainer(pipeline)).explain_global(X_sample)


def explain_individual(pipeline, one_row: pd.DataFrame, class_index: Optional[int] = None,
                       explainer: Optional[FeatureExplainer] = None) -> Dict[str, Any]:
    """Local explanation for one row (see ``FeatureExplainer.explain_individual``)."""
    return (explainer or FeatureExplainer(pipeline)).explain_individual(one_row, class_index)


def describe_local_explanation(result: Dict[str, Any], top_n: int = 5) -> List[str]:
    """Plain-language sentences that match the sign of each SHAP value."""
    cls = result["class_name"]
    sentences = []
    for _, r in result["table"].head(top_n).iterrows():
        if r["shap_value"] == 0:
            continue
        verb = "pushed the model toward" if r["shap_value"] > 0 else "pushed the model away from"
        sentences.append(
            f"**{r['label']}** (answer: {r['answer']}) {verb} '{cls}' ({r['shap_value']:+.3f})."
        )
    return sentences


def top_features(importance: pd.DataFrame, k: int) -> List[str]:
    """Names of the k most important questions from a global importance table."""
    return list(importance.sort_values("mean_abs_shap", ascending=False)["feature"].head(k))


def stability_summary(top_sets: List[List[str]]) -> Dict[str, Any]:
    """Mean pairwise Jaccard overlap of top-k feature sets plus per-feature frequency."""
    sets = [set(t) for t in top_sets]
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            overlaps.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    counts: Dict[str, int] = {}
    for s in sets:
        for f in s:
            counts[f] = counts.get(f, 0) + 1
    return {
        "n_splits": len(sets),
        "mean_pairwise_jaccard": float(np.mean(overlaps)) if overlaps else 1.0,
        "feature_frequency_in_top_k": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }

