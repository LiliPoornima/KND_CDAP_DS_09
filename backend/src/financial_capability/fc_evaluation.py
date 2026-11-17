"""Model evaluation: overall, per-class and subgroup metrics (all computed, none hard-coded)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .fc_config import CLASS_NAMES, MIN_SUBGROUP_SIZE


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Return accuracy, macro metrics, per-class metrics, confusion matrix and ROC-AUC.

    Macro averages treat the three classes equally, so a model cannot hide a
    weak Medium class behind a strong High class. One-vs-rest macro ROC-AUC is
    appropriate here because the models output class probabilities.
    """
    labels = list(range(len(CLASS_NAMES)))
    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))
    proba = model.predict_proba(X_test)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc_ovr_macro": float(roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=labels)),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES),
    }


def majority_baseline(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Always-predict-the-biggest-class baseline, to show the models really learn something."""
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    pred = dummy.predict(X_test)
    return {
        "strategy": "most_frequent",
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def metrics_to_frame(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """One summary row per model from ``evaluate_model`` outputs."""
    keep = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "weighted_f1", "roc_auc_ovr_macro", "log_loss"]
    return pd.DataFrame([{"model": name, **{k: m[k] for k in keep}}
                         for name, m in results.items()])


def per_class_frame(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Long table of per-class precision/recall/F1/support for every model."""
    rows: List[dict] = []
    for name, m in results.items():
        for cls, vals in m["per_class"].items():
            rows.append({"model": name, "class": cls, **vals})
    return pd.DataFrame(rows)


def confusion_frame(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = actual, columns = predicted)."""
    labels = metrics["confusion_matrix_labels"]
    return pd.DataFrame(metrics["confusion_matrix"],
                        index=[f"Actual {c}" for c in labels],
                        columns=[f"Predicted {c}" for c in labels])


def income_band(ppincimp: pd.Series) -> pd.Series:
    """Group the 9 income codes into 3 bands: <$40k, $40k-$74,999, $75k+."""
    return pd.cut(ppincimp, bins=[0, 3, 6, 9],
                  labels=["Under $40,000", "$40,000-$74,999", "$75,000 or more"])


def age_band(agecat: pd.Series) -> pd.Series:
    """Group the 8 age codes into 4 bands."""
    return pd.cut(agecat, bins=[0, 2, 4, 6, 8],
                  labels=["18-34", "35-54", "55-69", "70+"])


def subgroup_metrics(model, X_test: pd.DataFrame, y_test: pd.Series,
                     min_size: int = MIN_SUBGROUP_SIZE) -> pd.DataFrame:
    """Accuracy and macro-F1 per age band and income band (groups below ``min_size`` skipped).

    This reports differences; it does not prove the model is fair.
    """
    pred = pd.Series(model.predict(X_test), index=X_test.index)
    y = pd.Series(np.asarray(y_test), index=X_test.index)
    rows = []
    for family, groups in (("Age", age_band(X_test["agecat"])),
                           ("Income", income_band(X_test["PPINCIMP"]))):
        for g in groups.cat.categories:
            mask = (groups == g).to_numpy()
            n = int(mask.sum())
            if n < min_size:
                rows.append({"family": family, "group": str(g), "n": n,
                             "accuracy": np.nan, "macro_f1": np.nan,
                             "note": f"skipped: fewer than {min_size} respondents"})
                continue
            rows.append({
                "family": family, "group": str(g), "n": n,
                "accuracy": float(accuracy_score(y[mask], pred[mask])),
                "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", zero_division=0)),
                "note": "",
            })
    return pd.DataFrame(rows)

