import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.src.credit_risk.common.cr_calibration import (
    cr_group_calibration_summary,
)
from backend.src.credit_risk.experiments.cr_ablation import (
    cr_run_validation_selected_feature_ablation,
)
from backend.src.credit_risk.experiments import run_inspection
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_bootstrap_paired_group_fairness_difference,
)


def test_validation_selected_ablation_saves_test_subgroup_metrics(tmp_path):
    rng = np.random.default_rng(7)
    n = 240
    feature = rng.normal(size=n)
    target = (feature + rng.normal(size=n) > 0.1).astype(int)
    frame = pd.DataFrame({"feature": feature, "removed": rng.normal(size=n), "y": target})
    train, validation, test = frame.iloc[:120], frame.iloc[120:180], frame.iloc[180:]
    groups = pd.DataFrame(
        {"cohort": ["A", "B"] * (len(test) // 2)},
        index=test.index,
    )

    def trainer(data, *, target_column):
        model = LogisticRegression(random_state=42)
        predictors = data.drop(columns=[target_column])
        model.fit(predictors, data[target_column])
        return model

    output = tmp_path / "subgroups.csv"
    calibration_output = tmp_path / "subgroup_calibration.csv"
    bins_output = tmp_path / "subgroup_calibration_bins.csv"
    paired_output = tmp_path / "paired_comparisons.csv"
    paired_fairness_output = tmp_path / "paired_fairness_comparisons.csv"
    result = cr_run_validation_selected_feature_ablation(
        train,
        validation,
        test,
        {"secondary": ["removed"]},
        trainer=trainer,
        model_name="Logistic Regression",
        target_column="y",
        thresholds=[0.4, 0.5, 0.6],
        test_groups=groups,
        subgroup_output_path=output,
        subgroup_calibration_output_path=calibration_output,
        subgroup_calibration_bins_output_path=bins_output,
        paired_comparison_output_path=paired_output,
        paired_fairness_comparison_output_path=paired_fairness_output,
        paired_bootstrap_replicates=100,
    )

    saved = pd.read_csv(output)
    calibration = pd.read_csv(calibration_output)
    bins = pd.read_csv(bins_output)
    paired = pd.read_csv(paired_output)
    paired_fairness = pd.read_csv(paired_fairness_output)
    assert len(result) == 2
    assert set(saved["Scenario"]) == {"full", "without_secondary"}
    assert set(saved["group"]) == {"A", "B"}
    assert {"selection_rate", "true_positive_rate", "false_positive_rate"}.issubset(
        saved.columns
    )
    assert saved["sample_sufficient"].all()
    assert set(calibration["Scenario"]) == {"full", "without_secondary"}
    assert {"brier_score", "log_loss", "expected_calibration_error"}.issubset(
        calibration.columns
    )
    assert {"mean_predicted_probability", "observed_rate", "count"}.issubset(
        bins.columns
    )
    assert len(paired) == 9
    assert paired["comparison"].eq("without_secondary_minus_full").all()
    assert paired["model_a"].eq("full").all()
    assert paired["model_b"].eq("without_secondary").all()
    assert paired["valid_replicates"].eq(100).all()
    assert len(paired_fairness) == 4
    assert paired_fairness["comparison"].eq("without_secondary_minus_full").all()
    assert paired_fairness["metric"].nunique() == 4


def test_paired_group_fairness_bootstrap_is_zero_for_identical_predictions():
    truth = np.tile(np.array([0, 1] * 20), 2)
    groups = np.repeat(["A", "B"], 40)
    predictions = ((np.arange(len(truth)) % 3) == 0).astype(int)

    result = cr_bootstrap_paired_group_fairness_difference(
        truth,
        predictions,
        predictions,
        groups,
        n_bootstrap=100,
        random_state=9,
    )

    assert len(result) == 4
    assert np.allclose(result["difference_b_minus_a"], 0.0)
    assert np.allclose(result["lower"], 0.0)
    assert np.allclose(result["upper"], 0.0)
    assert result["valid_replicates"].eq(100).all()


def test_group_calibration_keeps_small_groups_with_support_flags():
    truth = [0, 1, 0, 1, 0, 1]
    probability = [0.1, 0.8, 0.4, 0.7, 0.2, 0.6]
    result = cr_group_calibration_summary(
        truth,
        probability,
        ["small"] * 6,
        attribute_name="cohort",
        min_group_size=8,
    )

    row = result["group_metrics"].iloc[0]
    assert row["n"] == 6
    assert not row["sample_sufficient"]
    assert row["positive_support"] == 3
    assert row["negative_support"] == 3
    assert not row["calibration_reportable"]
    assert result["calibration_bins"]["count"].sum() == 6


def test_training_subsample_explanation_stability_uses_fixed_eval_rows(
    tmp_path,
    monkeypatch,
):
    rng = np.random.default_rng(11)
    feature = rng.normal(size=120)
    target = (feature + rng.normal(size=120) > 0).astype(int)
    train = pd.DataFrame({"feature": feature, "other": rng.normal(size=120), "y": target})
    evaluation = pd.DataFrame(
        {"feature": rng.normal(size=24), "other": rng.normal(size=24)},
        index=range(500, 524),
    )
    seen_indexes = []

    def fake_explain(model, X, **kwargs):
        seen_indexes.append(tuple(X.index))
        attributions = pd.DataFrame(
            {"feature": X["feature"], "other": X["other"] * 0.2},
            index=X.index,
        )
        importance = (
            attributions.abs().mean()
            .rename("mean_absolute_shap")
            .sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index()
        )
        return {"attributions": attributions, "global_importance": importance}

    monkeypatch.setattr(run_inspection, "cr_explain_model", fake_explain)
    result = run_inspection.cr_run_training_subsample_explanation_stability(
        Pipeline([("classifier", LogisticRegression(random_state=42))]),
        train,
        evaluation,
        tmp_path,
        target_column="y",
        training_fraction=0.7,
        resample_states=(3, 4, 5),
        sample_size=12,
        background_size=10,
    )

    assert len(set(seen_indexes)) == 1
    assert result["summary"].iloc[0]["runs_compared"] == 3
    assert (tmp_path / "training_subsample_stability_pairwise.csv").is_file()
    assert (tmp_path / "training_subsample_feature_ranks.csv").is_file()
