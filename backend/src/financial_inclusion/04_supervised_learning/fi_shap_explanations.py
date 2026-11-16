
"""SHAP explanations for the saved Component 02 Logistic Regression model.

SHAP values explain model decision scores (logits), not probabilities
or causal reasons for financial exclusion.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap

from fi_supervised_inference import (
    CLUSTER_LABELS,
    FEATURES,
    load_model,
    validate_inputs,
    get_transformed_features,
)


def _normalise_shap_values(
    explanation: shap.Explanation,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SHAP values and base values to explicit class dimensions."""
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    # Current SHAP multiclass convention: samples x features x classes.
    if values.shape != (n_samples, n_features, n_classes):
        # Accommodate APIs that return one (samples x features) array per class.
        if (
            isinstance(explanation.values, list)
            and len(explanation.values) == n_classes
        ):
            values = np.stack(explanation.values, axis=-1)

    if values.shape != (n_samples, n_features, n_classes):
        raise ValueError(
            "Unexpected SHAP values shape. "
            f"Expected {(n_samples, n_features, n_classes)}, "
            f"received {values.shape}."
        )

    if base.ndim == 1 and base.shape == (n_classes,):
        base = np.broadcast_to(base, (n_samples, n_classes))
    elif base.shape != (n_samples, n_classes):
        raise ValueError(
            "Unexpected SHAP base-values shape: "
            f"{base.shape}; expected {(n_samples, n_classes)}."
        )

    if not np.isfinite(values).all() or not np.isfinite(base).all():
        raise ValueError("SHAP output contains non-finite values.")

    return values, base


def _encoded_feature_groups(
    pipeline: Any,
    encoded_feature_names: list[str],
) -> dict[str, list[int]]:
    """Map transformed feature positions to their original predictors."""
    preprocessor = pipeline.named_steps["preprocessing"]
    groups: dict[str, list[int]] = {feature: [] for feature in FEATURES}

    categorical_transformer = preprocessor.named_transformers_["categorical"]
    encoder = categorical_transformer.named_steps["encoder"]

    categorical_columns = list(
        preprocessor.transformers_[0][2]
    )
    categorical_slice = preprocessor.output_indices_["categorical"]
    categorical_start = categorical_slice.start

    position = categorical_start
    for column, categories in zip(
        categorical_columns, encoder.categories_
    ):
        count = len(categories)
        groups[column] = list(range(position, position + count))
        position += count

    numeric_slice = preprocessor.output_indices_["numeric"]
    numeric_columns = list(preprocessor.transformers_[1][2])

    for offset, column in enumerate(numeric_columns):
        groups[column] = [numeric_slice.start + offset]

    all_positions = sorted(
        position
        for positions in groups.values()
        for position in positions
    )

    if all_positions != list(range(len(encoded_feature_names))):
        raise ValueError(
            "Encoded feature mapping is incomplete or contains duplicates."
        )

    return groups


def explain_predictions(
    inputs: pd.DataFrame | dict,
    background: pd.DataFrame,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Calculate multiclass SHAP explanations for the saved classifier.

    Parameters
    ----------
    inputs:
        One or more rows with the seven original predictors.
    background:
        Reference rows selected from the training partition.
    pipeline:
        Optional already-loaded saved pipeline.

    Returns
    -------
    Dictionary containing raw encoded-feature SHAP values, grouped
    contributions, global importance, prediction probabilities and
    numerical additivity checks.
    """
    if pipeline is None:
        pipeline = load_model()

    input_frame = validate_inputs(inputs, pipeline)
    background_frame = validate_inputs(background, pipeline)

    if len(background_frame) < 2:
        raise ValueError("Use at least two reference rows.")

    encoded_inputs = get_transformed_features(input_frame, pipeline)
    encoded_background = get_transformed_features(background_frame, pipeline)

    classifier = pipeline.named_steps["model"]
    class_ids = [int(value) for value in classifier.classes_]

    # Explain the fitted classifier after the existing fitted preprocessing.
    # LinearExplainer output is in the classifier's decision-score space.
    explainer = shap.LinearExplainer(
        classifier,
        encoded_background.to_numpy(),
    )
    explanation = explainer(encoded_inputs.to_numpy())

    values, base_values = _normalise_shap_values(
        explanation,
        n_samples=len(input_frame),
        n_features=encoded_inputs.shape[1],
        n_classes=len(class_ids),
    )

    decision_scores = np.asarray(
        classifier.decision_function(encoded_inputs.to_numpy())
    )

    if decision_scores.ndim == 1:
        decision_scores = decision_scores[:, None]

    reconstructed_scores = base_values + values.sum(axis=1)
    additivity_errors = np.abs(reconstructed_scores - decision_scores)
    max_additivity_error = float(additivity_errors.max())

    if not np.allclose(
        reconstructed_scores, decision_scores, rtol=1e-4, atol=1e-4
    ):
        raise ValueError(
            "SHAP additivity check failed. Maximum absolute error: "
            f"{max_additivity_error}"
        )

    groups = _encoded_feature_groups(
        pipeline, encoded_inputs.columns.tolist()
    )

    # Group encoded one-hot contributions by original predictor.
    # Signed contributions are summed within each original predictor.
    grouped_values = np.zeros(
        (len(input_frame), len(FEATURES), len(class_ids)),
        dtype=float,
    )

    for feature_index, feature in enumerate(FEATURES):
        positions = groups[feature]
        grouped_values[:, feature_index, :] = values[:, positions, :].sum(
            axis=1
        )

    grouped_local = []
    for row_index in range(len(input_frame)):
        predicted_class = int(
            pipeline.predict(input_frame.iloc[[row_index]])[0]
        )
        class_position = class_ids.index(predicted_class)

        for feature_index, feature in enumerate(FEATURES):
            grouped_local.append(
                {
                    "row_index": row_index,
                    "predicted_cluster_id": predicted_class,
                    "predicted_cluster_label": CLUSTER_LABELS[predicted_class],
                    "explained_class_id": predicted_class,
                    "feature": feature,
                    "shap_decision_score_contribution": float(
                        grouped_values[
                            row_index, feature_index, class_position
                        ]
                    ),
                }
            )

    # Global importance: mean absolute grouped contribution, by class.
    global_rows = []
    for class_position, class_id in enumerate(class_ids):
        for feature_index, feature in enumerate(FEATURES):
            global_rows.append(
                {
                    "cluster_id": class_id,
                    "cluster_label": CLUSTER_LABELS[class_id],
                    "feature": feature,
                    "mean_absolute_shap_decision_score": float(
                        np.abs(
                            grouped_values[:, feature_index, class_position]
                        ).mean()
                    ),
                }
            )

    predictions = pipeline.predict(input_frame)
    probabilities = pipeline.predict_proba(input_frame)

    probability_rows = []
    for row_index, predicted_class in enumerate(predictions):
        class_position = class_ids.index(int(predicted_class))
        probability_rows.append(
            {
                "row_index": row_index,
                "predicted_cluster_id": int(predicted_class),
                "predicted_cluster_label": CLUSTER_LABELS[
                    int(predicted_class)
                ],
                "predicted_cluster_probability": float(
                    probabilities[row_index, class_position]
                ),
            }
        )

    return {
        "class_ids": class_ids,
        "encoded_feature_names": encoded_inputs.columns.tolist(),
        "encoded_feature_groups": groups,
        "raw_shap_values": values,
        "base_values": base_values,
        "grouped_shap_values": grouped_values,
        "local_explanations": pd.DataFrame(grouped_local),
        "global_importance": pd.DataFrame(global_rows),
        "predictions": pd.DataFrame(probability_rows),
        "max_additivity_error": max_additivity_error,
        "explained_output": "Logistic Regression decision scores (logits)",
    }
