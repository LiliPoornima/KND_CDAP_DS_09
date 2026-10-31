import numpy as np

from backend.src.credit_risk.common.cr_evaluation import (
    cr_bootstrap_paired_binary_performance,
)


def test_paired_bootstrap_returns_zero_interval_for_identical_models():
    truth = [0, 0, 0, 0, 1, 1, 1, 1]
    probability = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    result = cr_bootstrap_paired_binary_performance(
        iter(truth),
        probability,
        probability,
        threshold_a=0.5,
        threshold_b=0.5,
        model_a_name="LR",
        model_b_name="RF",
        n_bootstrap=100,
    )

    assert len(result) == 9
    assert np.allclose(result["difference_b_minus_a"], 0.0)
    assert np.allclose(result["lower"], 0.0)
    assert np.allclose(result["upper"], 0.0)
    assert result["valid_replicates"].eq(100).all()
    assert result["bootstrap_method"].eq("paired_stratified_by_target_class").all()


def test_paired_bootstrap_difference_is_model_b_minus_model_a():
    truth = [0, 0, 0, 0, 1, 1, 1, 1]
    probability_a = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    probability_b = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]

    result = cr_bootstrap_paired_binary_performance(
        truth,
        probability_a,
        probability_b,
        threshold_a=0.5,
        threshold_b=0.5,
        n_bootstrap=100,
    )

    brier = result.set_index("metric").loc["brier_score"]
    assert brier["estimate_b"] > brier["estimate_a"]
    assert brier["difference_b_minus_a"] == brier["estimate_b"] - brier["estimate_a"]
