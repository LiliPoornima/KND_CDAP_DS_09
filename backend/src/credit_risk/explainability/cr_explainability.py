"""Generate local SHAP attributions and global importance summaries."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def cr_explain_model(
    model: Any,
    X: pd.DataFrame,
    *,
    sample_size: int = 100,
    background_size: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Explain a fitted binary classifier with SHAP values.

    Tree and linear estimators inside a scikit-learn Pipeline use their
    matching SHAP explainer. Other estimators use SHAP's model-agnostic
    explainer, which may be slower. The returned attribution DataFrame uses
    the positive class (label 1) and sampled input row indexes.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain rows and predictor columns.")
    if not X.columns.is_unique:
        raise ValueError("X must have unique feature names.")
    if not isinstance(model, Pipeline):
        if X.isna().any().any():
            raise ValueError("X must not contain missing predictor values.")
        if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
            raise ValueError(
                "Standalone estimators require numeric predictors for SHAP analysis."
            )
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("X must contain only finite predictor values.")
    for name, value in (("sample_size", sample_size), ("background_size", background_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability. Install it with `pip install shap`."
        ) from exc

    sampled = X.sample(
        n=min(sample_size, len(X)),
        random_state=random_state,
    )
    background = X.sample(
        n=min(background_size, len(X)),
        random_state=random_state + 1,
    )

    if isinstance(model, Pipeline):
        if not hasattr(model, "classes_") or 1 not in model.classes_:
            raise ValueError("model must be a fitted binary classifier with class 1.")
        estimator = model.steps[-1][1]
        if len(model.steps) > 1:
            transformer = model[:-1]
            transformed_sample = transformer.transform(sampled)
            transformed_background = transformer.transform(background)
            try:
                explanation_feature_names = list(
                    transformer.get_feature_names_out(X.columns)
                )
            except (AttributeError, ValueError):
                explanation_feature_names = list(X.columns)
        else:
            transformed_sample = sampled.to_numpy()
            transformed_background = background.to_numpy()
            explanation_feature_names = list(X.columns)
    else:
        if not hasattr(model, "classes_") or 1 not in model.classes_:
            raise ValueError("model must be a fitted binary classifier with class 1.")
        estimator = model
        transformed_sample = sampled.to_numpy()
        transformed_background = background.to_numpy()
        explanation_feature_names = list(X.columns)

    for transformed in (transformed_sample, transformed_background):
        values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(
                "Preprocessing must produce finite numeric values before SHAP analysis."
            )

    if isinstance(estimator, RandomForestClassifier):
        if hasattr(transformed_sample, "toarray"):
            transformed_sample = transformed_sample.toarray()
        if hasattr(transformed_background, "toarray"):
            transformed_background = transformed_background.toarray()
        explainer = shap.TreeExplainer(estimator, data=transformed_background)
        shap_result = explainer(transformed_sample)
    elif isinstance(estimator, LogisticRegression):
        if hasattr(transformed_sample, "toarray"):
            transformed_sample = transformed_sample.toarray()
        if hasattr(transformed_background, "toarray"):
            transformed_background = transformed_background.toarray()
        explainer = shap.LinearExplainer(estimator, transformed_background)
        shap_result = explainer(transformed_sample)
    else:
        explanation_feature_names = list(X.columns)
        if isinstance(model, Pipeline):
            prediction_function = model.predict_proba
            background_for_model = background
            rows_for_model = sampled
        else:
            prediction_function = model.predict_proba
            background_for_model = background
            rows_for_model = sampled
        explainer = shap.Explainer(
            prediction_function,
            background_for_model,
            algorithm="permutation",
        )
        shap_result = explainer(rows_for_model)

    raw_values = getattr(shap_result, "values", shap_result)
    class_positions = list(model.classes_).index(1)
    if isinstance(raw_values, list):
        if class_positions >= len(raw_values):
            raise ValueError("SHAP did not return values for class 1.")
        values = np.asarray(raw_values[class_positions])
    else:
        values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, class_positions]
    if values.ndim != 2 or values.shape[1] != len(explanation_feature_names):
        raise ValueError(
            "SHAP returned an unexpected shape; check model class and feature transforms."
        )

    attributions = pd.DataFrame(
        values,
        index=sampled.index,
        columns=explanation_feature_names,
    )
    global_importance = (
        attributions.abs()
        .mean(axis=0)
        .rename("mean_absolute_shap")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
    )

    return {
        "attributions": attributions,
        "global_importance": global_importance,
        "feature_names": explanation_feature_names,
        "sampled_features": sampled,
        "explainer": explainer,
    }
