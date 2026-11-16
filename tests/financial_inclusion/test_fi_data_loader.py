"""Tests for financial-inclusion data loading and dashboard analytics."""

from pathlib import Path
import sys

import pandas as pd
import pytest

from frontend.financial_inclusion import data_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMENSION_MODULE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "financial_inclusion"
    / "02_dimension_creation"
)
CLUSTERING_MODULE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "financial_inclusion"
    / "03_clustering"
)
SUPERVISED_MODULE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "src"
    / "financial_inclusion"
    / "04_supervised_learning"
)
for module_dir in (
    DIMENSION_MODULE_DIR,
    CLUSTERING_MODULE_DIR,
    SUPERVISED_MODULE_DIR,
):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from fi_individual_profiling import profile_individual
from fi_individual_score_calculation import calculate_dimension_scores
from fi_supervised_inference import predict_profiles


def sample_community_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "economy": ["A", "A", "B", "B"],
            "cluster_id": [0, 1, 1, 2],
            "group": ["urban", "rural", "urban", "urban"],
            "score_connectivity": [1.0, 0.0, 0.5, None],
            "score_financial_access": [1.0, 0.0, 1.0, 0.0],
            "score_digital_financial_usage": [0.5, 0.0, 0.5, 1.0],
            "score_digital_safety": [1.0, 0.0, 0.5, 1.0],
            "score_financial_resilience": [0.5, 0.5, 1.0, 0.0],
        }
    )


def test_community_analysis_matches_filtered_direct_calculations():
    frame = sample_community_data()
    result = data_loader.calculate_community_analysis(
        frame,
        "A",
        filter_column="group",
        filter_value="urban",
    )
    selected = frame.loc[
        (frame["economy"] == "A") & (frame["group"] == "urban")
    ]
    reference = frame.loc[frame["group"] == "urban"]

    assert result["respondents"] == len(selected) == 1
    assert result["reference_respondents"] == len(reference) == 3
    for column in data_loader.DIMENSION_COLUMNS:
        row = result["comparison"].loc[
            result["comparison"]["column"] == column
        ].iloc[0]
        assert row["economy_mean"] == pytest.approx(selected[column].mean())
        assert row["reference_mean"] == pytest.approx(reference[column].mean())
        assert row["difference"] == pytest.approx(
            selected[column].mean() - reference[column].mean()
        )


def test_community_reference_can_exclude_economy_and_filter_saved_profile():
    frame = sample_community_data()
    result = data_loader.calculate_community_analysis(
        frame,
        "A",
        cluster_id=1,
        exclude_selected_economy=True,
    )

    assert result["respondents"] == 1
    assert result["reference_respondents"] == 1
    assert result["cluster_id"] == 1
    assert result["exclude_selected_economy"] is True
    assert result["comparison"].iloc[0]["reference_mean"] == pytest.approx(0.5)


def test_community_excluded_reference_requires_other_respondents():
    frame = sample_community_data().loc[lambda rows: rows["economy"] == "A"]
    with pytest.raises(ValueError, match="excluding the selected economy"):
        data_loader.calculate_community_analysis(
            frame,
            "A",
            exclude_selected_economy=True,
        )


def test_community_cluster_counts_and_percentages_use_selected_economy():
    result = data_loader.calculate_community_analysis(
        sample_community_data(), "A"
    )
    distribution = result["cluster_distribution"]

    assert distribution["respondents"].tolist() == [1, 1, 0, 0, 0]
    assert distribution["percent_of_economy"].tolist() == pytest.approx(
        [50.0, 50.0, 0.0, 0.0, 0.0]
    )


def test_community_analysis_rejects_empty_and_unknown_selections():
    frame = sample_community_data()
    with pytest.raises(ValueError, match="dataset cannot be empty"):
        data_loader.calculate_community_analysis(frame.iloc[0:0], "A")
    with pytest.raises(ValueError, match="No respondents"):
        data_loader.calculate_community_analysis(frame, "Missing")
    with pytest.raises(ValueError, match="No respondents match"):
        data_loader.calculate_community_analysis(
            frame, "A", filter_column="group", filter_value="missing"
        )
    with pytest.raises(ValueError, match="Unknown filter column"):
        data_loader.calculate_community_analysis(
            frame, "A", filter_column="unknown", filter_value="x"
        )


def test_missing_dimension_values_are_reported_and_excluded_from_means():
    result = data_loader.calculate_community_analysis(
        sample_community_data(), "B"
    )
    connectivity = result["comparison"].loc[
        result["comparison"]["column"] == "score_connectivity"
    ].iloc[0]

    assert connectivity["economy_valid_respondents"] == 1
    assert connectivity["economy_missing_respondents"] == 1
    assert connectivity["economy_mean"] == pytest.approx(0.5)


def test_recommendations_are_ranked_and_contain_structured_hypotheses():
    comparison = pd.DataFrame(
        {
            "dimension": ["Connectivity", "Financial access"],
            "difference": [-0.25, 0.10],
        }
    )

    recommendations = data_loader.build_community_recommendations(comparison)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation["dimension"] == "Connectivity"
    assert recommendation["difference"] == pytest.approx(-0.25)
    assert recommendation["observed_gap"]
    assert recommendation["action"]
    assert recommendation["stakeholder"]
    assert recommendation["monitor"]
    assert "hypothesis" in recommendation["limitation"]


def test_profile_action_display_extracts_fields_instead_of_stringifying():
    actions = [
        {
            "dimension": "Connectivity",
            "reason": "Observed gap",
            "action": "Explore access barriers.",
            "stakeholder": "Local partners",
            "monitor": "Access indicator",
            "unused": "not displayed",
        }
    ]

    displayed = data_loader.format_suggested_actions(actions)

    assert displayed.columns.tolist() == [
        "dimension",
        "reason",
        "action",
        "stakeholder",
        "monitor",
    ]
    assert displayed.iloc[0].to_dict() == {
        "dimension": "Connectivity",
        "reason": "Observed gap",
        "action": "Explore access barriers.",
        "stakeholder": "Local partners",
        "monitor": "Access indicator",
    }


def test_individual_profile_matches_scoring_pipeline_and_blocks_missing_resilience():
    complete_indicators = {
        "con1": 1,
        "internet_use": 1,
        "account": 1,
        "anydigpayment": 0,
        "con18": 2,
        "saved": 1,
        "fin24a": 2,
        "fin24b": 3,
    }
    expected_scores = calculate_dimension_scores(complete_indicators)
    profile = profile_individual(complete_indicators)

    assert profile["status"] == "success"
    assert profile["dimension_scores"] == expected_scores["dimension_scores"]
    assert profile["profile"]["cluster_id"] in data_loader.CLUSTER_IDS
    assert all(
        {"dimension", "reason", "action", "stakeholder", "monitor"}
        <= action.keys()
        for action in profile["explanation"]["suggested_actions"]
    )

    incomplete_indicators = {**complete_indicators, "fin24b": None}
    incomplete = profile_individual(incomplete_indicators)

    assert incomplete["status"] == "incomplete"
    assert "score_financial_resilience" in incomplete["unavailable_dimensions"]


def test_saved_classifier_inference_returns_all_class_probabilities():
    source = data_loader.load_final_dataset()
    predictor_columns = [
        "female",
        "educ",
        "inc_q",
        "emp_in",
        "urbanicity",
        "regionwb",
        "age",
    ]
    inputs = source[predictor_columns].iloc[0].to_dict()

    prediction = predict_profiles(inputs).iloc[0]
    probabilities = [
        prediction[f"probability_cluster_{cluster_id}"]
        for cluster_id in data_loader.CLUSTER_IDS
    ]

    assert int(prediction["predicted_cluster_id"]) in data_loader.CLUSTER_IDS
    assert prediction["predicted_cluster_label"]
    assert 0 <= prediction["predicted_cluster_probability"] <= 1
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)


def test_saved_shap_outputs_and_validation_metadata_load():
    global_importance = data_loader.load_shap_global_importance()
    local_explanations = data_loader.load_shap_local_explanations()
    predictions = data_loader.load_shap_predictions()
    summary = data_loader.load_shap_validation_summary()

    assert len(global_importance) == 35
    assert len(local_explanations) == 700
    assert len(predictions) == 100
    assert summary["checks_passed"]
    assert all(summary["checks_passed"].values())
    assert "causal" in summary["interpretation_limit"]


def test_supervised_model_comparison_contains_all_saved_models_and_metrics():
    metrics = data_loader.prepare_supervised_model_comparison(
        data_loader.load_supervised_metrics()
    )

    assert metrics["Model"].tolist() == list(
        data_loader.SUPERVISED_MODEL_ORDER
    )
    assert set(data_loader.SUPERVISED_METRIC_COLUMNS).issubset(metrics.columns)
    assert metrics.set_index("Model").loc[
        "XGBoost", "Accuracy"
    ] == pytest.approx(0.4728785063)
    assert metrics.set_index("Model").loc[
        "Logistic Regression", "Balanced Accuracy"
    ] == pytest.approx(0.3782702355)


def test_supervised_comparison_rejects_missing_models_or_metric_fields():
    metrics = data_loader.load_supervised_metrics()

    with pytest.raises(ValueError, match="missing expected models"):
        data_loader.prepare_supervised_model_comparison(
            metrics.loc[metrics["Model"] != "XGBoost"]
        )
    with pytest.raises(ValueError, match="missing columns"):
        data_loader.prepare_supervised_model_comparison(
            metrics.drop(columns=["Macro-F1"])
        )


def test_final_clustering_evidence_matches_saved_final_solution():
    evidence = data_loader.load_clustering_evidence()

    assert evidence["candidate"] == "5D_k5"
    assert evidence["algorithm"] == "KMeans"
    assert evidence["clusters"] == 5
    assert evidence["silhouette_score"] == pytest.approx(0.6312848397)
    assert evidence["davies_bouldin_score"] == pytest.approx(0.8045740069)
    assert evidence["calinski_harabasz_score"] == pytest.approx(78662.3799)
    assert evidence["seed_runs"] == 10
    assert evidence["mean_pairwise_ari"] == pytest.approx(1.0)
    assert evidence["min_pairwise_ari"] == pytest.approx(1.0)
    assert evidence["kmeans_gmm_ari"] == pytest.approx(1.0)
    assert evidence["metadata_validated"] is True
    assert evidence["legacy_summary_conflict"] is True


def test_saved_supervised_reports_and_confusion_matrices_parse_by_model():
    reports = data_loader.load_supervised_classification_reports()
    matrices = data_loader.load_supervised_confusion_matrices()

    assert set(reports) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert set(matrices) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert all(
        len(matrix) == 5 and all(len(row) == 5 for row in matrix)
        for matrix in matrices.values()
    )
    assert "precision" in reports["Logistic Regression"].lower()
