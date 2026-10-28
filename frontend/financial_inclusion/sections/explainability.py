"""Stakeholder-facing view of saved classifier evaluation and SHAP results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .. import data_loader as loader
from ..constants import CLUSTER_IDS, CLUSTER_LABELS


FEATURE_LABELS = {
    "female": "Sex",
    "educ": "Education",
    "inc_q": "Income quintile",
    "emp_in": "Employment",
    "urbanicity": "Urbanicity",
    "regionwb": "World Bank region",
    "age": "Age",
}
def _metrics_from_confusion(matrix: list[list[int]]) -> dict[str, float]:
    """Recalculate aggregate metrics from the saved 5 x 5 confusion matrix."""
    values = np.asarray(matrix, dtype=float)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    true_positive = np.diag(values)
    recall = np.divide(true_positive, support, out=np.zeros(5), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(5), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5),
        where=(precision + recall) > 0,
    )
    total = support.sum()
    return {
        "Accuracy": float(true_positive.sum() / total),
        "Balanced Accuracy": float(recall.mean()),
        "Macro-F1": float(f1.mean()),
        "Weighted-F1": float(np.average(f1, weights=support)),
    }


def _evaluation_discrepancies(metrics: pd.DataFrame, matrices: dict) -> list[str]:
    issues = []
    for model, matrix in matrices.items():
        if model not in set(metrics["Model"]):
            issues.append(f"{model}: confusion matrix has no matching metric row.")
            continue
        saved = metrics.loc[metrics["Model"] == model].iloc[0]
        calculated = _metrics_from_confusion(matrix)
        for metric, expected in calculated.items():
            if not np.isclose(float(saved[metric]), expected, atol=5.1e-5, rtol=0):
                issues.append(
                    f"{model} {metric}: saved summary and confusion matrix differ."
                )
    return issues


def _class_report_table(report: str) -> pd.DataFrame:
    rows = []
    for line in report.splitlines():
        parts = line.split()
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        cluster_id = int(parts[0])
        if cluster_id not in CLUSTER_LABELS:
            continue
        try:
            precision, recall, f1, support = map(float, parts[1:])
        except ValueError:
            continue
        rows.append(
            {
                "Profile": CLUSTER_LABELS[cluster_id],
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "Support": int(support),
            }
        )
    return pd.DataFrame(rows)


def _report_matrix_discrepancies(report: str, matrix: list[list[int]]) -> list[str]:
    """Check rounded per-class report values against the saved matrix."""
    table = _class_report_table(report)
    values = np.asarray(matrix, dtype=float)
    if len(table) != len(CLUSTER_IDS):
        return ["Classification report does not cover all five profile classes."]
    issues = []
    for cluster_id, (_, row) in enumerate(table.iterrows()):
        support = values[cluster_id].sum()
        predicted = values[:, cluster_id].sum()
        tp = values[cluster_id, cluster_id]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (
            int(row.Support) != int(support)
            or not np.isclose(row.Precision, precision, atol=0.0051, rtol=0)
            or not np.isclose(row.Recall, recall, atol=0.0051, rtol=0)
            or not np.isclose(row["F1-score"], f1, atol=0.0051, rtol=0)
        ):
            issues.append(f"{CLUSTER_LABELS[cluster_id]}: report and confusion matrix differ.")
    return issues


def _shap_artifact_checks(
    global_imp: pd.DataFrame,
    local: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: dict,
) -> tuple[dict[str, bool | None], dict[str, object]]:
    """Combine observed CSV integrity checks with the saved numerical audit."""
    expected_features = set(FEATURE_LABELS)
    global_classes = set(pd.to_numeric(global_imp["cluster_id"], errors="coerce").dropna().astype(int))
    local_classes = set(pd.to_numeric(local["explained_class_id"], errors="coerce").dropna().astype(int))
    local_per_sample = local.groupby("row_index").size()
    global_values = pd.to_numeric(
        global_imp["mean_absolute_shap_decision_score"], errors="coerce"
    ).to_numpy(dtype=float)
    local_values = pd.to_numeric(
        local["shap_decision_score_contribution"], errors="coerce"
    ).to_numpy(dtype=float)
    finite = bool(np.isfinite(global_values).all() and np.isfinite(local_values).all())
    local_feature_coverage = bool(
        set(local["feature"]) == expected_features
        and local.groupby("row_index")["feature"].nunique().eq(len(expected_features)).all()
    )
    predictions_unique = not predictions["row_index"].duplicated().any()
    pred_ids = predictions.drop_duplicates("row_index").set_index("row_index")["predicted_cluster_id"]
    local_pred_ids = local.groupby("row_index")["predicted_cluster_id"].nunique()
    sample_alignment = bool(
        predictions_unique
        and local_pred_ids.eq(1).all()
        and set(local_pred_ids.index) == set(pred_ids.index)
        and all(
            int(local.loc[local["row_index"] == row_id, "predicted_cluster_id"].iloc[0])
            == int(pred_ids.loc[row_id])
            for row_id in pred_ids.index
        )
        and (local["predicted_cluster_id"] == local["explained_class_id"]).all()
    )
    shape_check = bool(
        len(global_imp) == len(CLUSTER_IDS) * len(expected_features)
        and set(global_imp["feature"]) == expected_features
        and global_imp.groupby("cluster_id").size().eq(len(expected_features)).all()
        and local_feature_coverage
        and len(predictions) > 0
        and len(local) == len(predictions) * len(expected_features)
        and local_per_sample.eq(len(expected_features)).all()
        and global_classes == set(CLUSTER_IDS)
        and local_classes.issubset(set(CLUSTER_IDS))
        and sample_alignment
    )

    saved_checks = validation.get("checks_passed", {})
    if not isinstance(saved_checks, dict):
        saved_checks = {}
    try:
        additivity_error = float(validation["max_additivity_error"])
        additivity_value_valid = np.isfinite(additivity_error) and additivity_error <= 1e-4
    except (KeyError, TypeError, ValueError):
        additivity_error = None
        additivity_value_valid = False
    try:
        grouped_error = float(validation["max_grouped_sum_error"])
        grouped_value_valid = np.isfinite(grouped_error) and grouped_error <= 1e-8
    except (KeyError, TypeError, ValueError):
        grouped_error = None
        grouped_value_valid = False
    checks: dict[str, bool | None] = {
        "shapes": shape_check and saved_checks.get("shapes") is True,
        "finite_shap_values": finite and saved_checks.get("finite_shap_values") is True,
        "additivity": saved_checks.get("additivity") is True and additivity_value_valid,
        "grouped_contribution_conservation": saved_checks.get("grouped_contribution_conservation") is True and grouped_value_valid,
        "all_five_classes": (
            saved_checks.get("all_five_classes") is True
            and validation.get("class_ids") == list(CLUSTER_IDS)
            and global_classes == set(CLUSTER_IDS)
        ),
    }
    details = {
        "Observed global rows": len(global_imp),
        "Observed local rows": len(local),
        "Observed prediction rows": len(predictions),
        "Observed classes": len(global_classes),
        "Observed grouped features": len(expected_features),
        "Local sample/class alignment": sample_alignment,
        "Maximum additivity error": additivity_error,
        "Maximum grouped-sum error": grouped_error,
    }
    return checks, details


def _show_validation(
    global_imp: pd.DataFrame,
    local: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: dict,
) -> None:
    st.subheader("SHAP Validation")
    checks, details = _shap_artifact_checks(global_imp, local, predictions, validation)
    all_passed = all(value is True for value in checks.values())
    if all_passed:
        st.success(
            "The saved SHAP diagnostics and loaded-table consistency checks passed. "
            "These numerical checks do not assess predictive accuracy, fairness, "
            "or generalisation."
        )
    else:
        failed = [name.replace("_", " ") for name, value in checks.items() if value is not True]
        st.warning(
            "SHAP outputs are available, but validation is incomplete or failed for: "
            + ", ".join(failed)
            + ". Review the technical details before relying on these explanations."
        )
    st.warning(
        "Evaluation uses the existing random validation split. Generalization to unseen countries "
        "has not been established; country-held-out validation is a future evaluation requirement."
    )

    cols = st.columns(4)
    cols[0].metric("Validation status", "Passed" if all_passed else "Needs review")
    cols[1].metric("Background rows", str(validation.get("background_rows", "Unavailable")))
    cols[2].metric("Explained rows", str(validation.get("explained_rows", len(predictions))))
    cols[3].metric("Classes checked", "5" if checks["all_five_classes"] else "Unverified")

    with st.expander("Technical validation details"):
        display = pd.DataFrame(
            {
                "Check": [
                    "Saved shape diagnostics and table consistency",
                    "Finite SHAP values",
                    "Saved additivity check to Logistic Regression decision scores",
                    "Saved grouped-contribution conservation check",
                    "Saved global importance covers five classes",
                ],
                "Result": [
                    "Passed" if value is True else "Failed or unavailable"
                    for value in checks.values()
                ],
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
        st.dataframe(
            pd.DataFrame(
                {
                    "Diagnostic": list(details),
                    "Value": [str(value) for value in details.values()],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            validation.get(
                "split_provenance_note",
                "Split provenance note is unavailable in the saved validation metadata.",
            )
        )


def render(df: pd.DataFrame) -> None:
    metrics = None
    reports = {}
    matrices = {}
    try:
        metrics = loader.prepare_supervised_model_comparison(
            loader.load_supervised_metrics()
        )
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved model comparison is unavailable or inconsistent: {exc}")

    if metrics is not None:
        st.subheader("Saved Model Evaluation")
        issues = []
        try:
            matrices = loader.load_supervised_confusion_matrices()
            issues = _evaluation_discrepancies(metrics, matrices)
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Confusion-matrix cross-check is unavailable: {exc}")
        if issues:
            st.warning("Evaluation artifacts need review: " + " ".join(issues))

        st.dataframe(metrics.style.format({column: "{:.3f}" for column in loader.SUPERVISED_METRIC_COLUMNS}), hide_index=True, width="stretch")
        st.caption(
            "Performance is modest and varies by metric. XGBoost has the highest saved accuracy; "
            "Logistic Regression was selected for SHAP based on interpretability and its class-balanced "
            "evaluation. No model is best on every metric."
        )
        st.caption(
            "Accuracy is overall correctness; balanced accuracy averages recall across classes; "
            "macro-F1 averages class F1 equally; weighted-F1 weights each class F1 by its support."
        )
        st.markdown("**Model explained below: Logistic Regression**")

        try:
            reports = loader.load_supervised_classification_reports()
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Class-level report is unavailable: {exc}")
        lr_report = reports.get("Logistic Regression")
        lr_matrix = matrices.get("Logistic Regression")
        if lr_report and lr_matrix:
            report_issues = _report_matrix_discrepancies(lr_report, lr_matrix)
            if report_issues:
                st.warning("Class-level report cross-check needs review: " + " ".join(report_issues))
        with st.expander("Logistic Regression class-level results"):
            report = lr_report
            if report:
                class_table = _class_report_table(report)
                if len(class_table) == len(CLUSTER_IDS):
                    st.dataframe(class_table.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}), hide_index=True, width="stretch")
                    lowest_recall = class_table.loc[class_table["Recall"].idxmin()]
                    highest_recall = class_table.loc[class_table["Recall"].idxmax()]
                    st.caption(
                        "Profile recall varies: it is lowest for "
                        f"{lowest_recall['Profile']} ({lowest_recall['Recall']:.2f}) and "
                        f"highest for {highest_recall['Profile']} ({highest_recall['Recall']:.2f})."
                    )
                else:
                    st.warning("The saved report does not contain results for all five profile classes.")
            else:
                st.warning("The saved Logistic Regression classification report is unavailable.")
            matrix = lr_matrix
            if matrix is not None:
                labels = [CLUSTER_LABELS[cluster_id] for cluster_id in CLUSTER_IDS]
                st.markdown("**Confusion matrix** · rows are actual profiles; columns are predicted profiles.")
                st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), width="stretch")
            else:
                st.warning("The saved Logistic Regression confusion matrix is unavailable.")

    try:
        global_imp = loader.load_shap_global_importance()
        local = loader.load_shap_local_explanations()
        predictions = loader.load_shap_predictions()
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved SHAP explanation tables could not be validated: {exc}")
        return
    try:
        validation = loader.load_shap_validation_summary()
    except (OSError, ValueError, KeyError) as exc:
        validation = {}
        st.warning(f"Saved SHAP validation metadata is unavailable or invalid: {exc}")

    st.subheader("Global SHAP Importance")
    st.caption("The saved global results aggregate one-hot contributions by the seven original predictors.")
    cls = st.selectbox(
        "Profile class explained",
        CLUSTER_IDS,
        format_func=lambda cluster_id: CLUSTER_LABELS[cluster_id],
        key="shap_global_class",
    )
    shown = global_imp.loc[global_imp.cluster_id == cls].copy()
    if shown.empty:
        st.warning(f"No saved global SHAP values are available for {CLUSTER_LABELS[cls]}.")
    else:
        shown["display_feature"] = shown["feature"].map(FEATURE_LABELS).fillna(shown["feature"])
        shown = shown.sort_values("mean_absolute_shap_decision_score", ascending=True)
        fig = px.bar(
            shown,
            x="mean_absolute_shap_decision_score",
            y="display_feature",
            orientation="h",
            text="mean_absolute_shap_decision_score",
            hover_data={"feature": True, "mean_absolute_shap_decision_score": ":.4f", "display_feature": False},
            labels={
                "mean_absolute_shap_decision_score": "Mean absolute SHAP contribution to class decision score",
                "display_feature": "Predictor",
                "feature": "Saved feature name",
            },
            title=f"Global feature contributions · {CLUSTER_LABELS[cls]}",
        )
        fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(height=390, margin=dict(l=10, r=70, t=55, b=35), showlegend=False)
        st.plotly_chart(fig, config={"responsive": True})
        st.markdown("### Global Importance Findings")
        st.write(
            "Mean absolute SHAP values summarize the average magnitude of each predictor's "
            "contribution to this class's decision score across the saved explained sample. "
            "Larger values indicate greater average contribution magnitude."
        )
        leaders = shown.nlargest(3, "mean_absolute_shap_decision_score")
        leader_text = ", ".join(
            f"{FEATURE_LABELS.get(row.feature, row.feature)} "
            f"({row.mean_absolute_shap_decision_score:.3f})"
            for row in leaders.itertuples(index=False)
        )
        if not leaders.empty:
            st.markdown(f"**Leading predictors in this saved sample:** {leader_text}.")
            st.write(
                "The classifier relies more heavily on these predictors when distinguishing "
                f"{CLUSTER_LABELS[cls]} in the saved sample."
            )
        st.caption(
            "Global absolute importance has no direction. It does not establish causality, "
            "explain why exclusion occurs, or show that changing a predictor changes a person's circumstances."
        )
    st.subheader("Local SHAP Explanation")
    valid_rows = sorted(set(local.row_index).intersection(set(predictions.row_index)))
    if not valid_rows:
        st.warning("No saved local explanations align with the saved validation predictions.")
    else:
        row_id = st.selectbox(
            "Saved validation example",
            valid_rows,
            format_func=lambda value: f"Example {valid_rows.index(value) + 1}",
            key="shap_local_sample",
        )
        matched_prediction = predictions.loc[predictions.row_index == row_id]
        detail = local.loc[local.row_index == row_id].copy()
        if len(matched_prediction) != 1 or detail.empty:
            st.warning("The selected saved example has missing or duplicate explanation rows.")
        else:
            pred = matched_prediction.iloc[0]
            predicted_class = int(pred.predicted_cluster_id)
            explained_classes = set(detail.explained_class_id.astype(int))
            if explained_classes != {predicted_class} or not (detail.predicted_cluster_id.astype(int) == predicted_class).all():
                st.warning("The saved local explanation does not match the selected example's predicted profile.")
            else:
                st.markdown(f"**Predicted saved profile:** {CLUSTER_LABELS[predicted_class]}")
                st.caption(
                    f"Explained class: {CLUSTER_LABELS[predicted_class]}. "
                    f"The displayed {float(pred.predicted_cluster_probability):.1%} is the model's probability for this derived profile label, not a probability of inclusion or exclusion."
                )
                detail["display_feature"] = detail["feature"].map(FEATURE_LABELS).fillna(detail["feature"])
                detail = detail.sort_values("shap_decision_score_contribution")
                detail["direction"] = np.where(detail.shap_decision_score_contribution >= 0, "Raises decision score", "Lowers decision score")
                fig_local = px.bar(
                    detail,
                    x="shap_decision_score_contribution",
                    y="display_feature",
                    color="direction",
                    color_discrete_map={"Raises decision score": "#168C8C", "Lowers decision score": "#D97757"},
                    orientation="h",
                    hover_data={"feature": True, "shap_decision_score_contribution": ":.4f", "display_feature": False},
                    labels={
                        "shap_decision_score_contribution": "Contribution to class decision score (logit scale)",
                        "display_feature": "Predictor",
                        "feature": "Saved feature name",
                        "direction": "Direction",
                    },
                    title=f"Saved example · contributions to {CLUSTER_LABELS[predicted_class]}",
                )
                fig_local.add_vline(x=0, line_dash="dash", line_color="#475569")
                fig_local.update_layout(height=390, margin=dict(l=10, r=25, t=55, b=35), legend_title_text="Effect on decision score")
                st.plotly_chart(fig_local, config={"responsive": True})
                st.markdown("### Prediction Findings")
                positive = detail.loc[
                    detail["shap_decision_score_contribution"] > 0
                ].nlargest(3, "shap_decision_score_contribution")
                negative = detail.loc[
                    detail["shap_decision_score_contribution"] < 0
                ].nsmallest(3, "shap_decision_score_contribution")

                def contribution_list(rows: pd.DataFrame) -> str:
                    return ", ".join(
                        f"{FEATURE_LABELS.get(item.feature, item.feature)} "
                        f"({item.shap_decision_score_contribution:+.3f})"
                        for item in rows.itertuples(index=False)
                    )

                if not positive.empty:
                    st.markdown(
                        "**Strongest positive contributions:** "
                        f"{contribution_list(positive)}. These raised the decision score for "
                        f"{CLUSTER_LABELS[predicted_class]}."
                    )
                else:
                    st.caption("No positive feature contributions were present in this saved explanation.")
                if not negative.empty:
                    st.markdown(
                        "**Strongest negative contributions:** "
                        f"{contribution_list(negative)}. These lowered the decision score for "
                        f"{CLUSTER_LABELS[predicted_class]}."
                    )
                else:
                    st.caption("No negative feature contributions were present in this saved explanation.")
                st.caption(
                    "This is a saved validation example, not a live individual assessment. "
                    "The contributions describe classifier behavior, not causes."
                )

    _show_validation(global_imp, local, predictions, validation)
