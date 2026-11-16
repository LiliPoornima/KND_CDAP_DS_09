"""Unit tests for fd_monitoring.py and fd_explainability.py on toy data (Spec Section 9 & 15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_config import MODEL_FEATURES
from backend.src.fraud_detection.fd_explainability import (
    calculate_per_feature_delta,
    calculate_shap_profile,
    calculate_shap_shift,
)
from backend.src.fraud_detection.fd_monitoring import (
    calculate_binary_entropy,
    calculate_entropy_relative_change,
    calculate_feature_psi,
    calculate_features_psi,
    calculate_score_js,
    compute_monitoring_signals,
)


# --------------------------------------------------------------------------- PSI Tests
def test_psi_identical_distribution_is_zero():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=10.0, scale=2.0, size=5000)
    psi = calculate_feature_psi(data, data)
    assert pytest.approx(0.0, abs=1e-5) == psi


def test_psi_known_shift():
    # Large shift in distribution must produce high PSI (> 0.25 is standard large drift threshold)
    ref = np.linspace(0.0, 10.0, 1000)
    shifted = np.linspace(8.0, 18.0, 1000)
    psi = calculate_feature_psi(ref, shifted)
    assert psi > 0.5


def test_psi_handles_nan_and_outliers():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # Window with NaNs and values outside reference range
    win = np.array([np.nan, -5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])
    psi = calculate_feature_psi(ref, win)
    assert np.isfinite(psi)
    assert psi >= 0.0


def test_calculate_features_psi_mean_and_max():
    ref_df = pd.DataFrame({
        "feat_a": np.arange(100, dtype=float),
        "feat_b": np.ones(100),
    })
    win_df = pd.DataFrame({
        "feat_a": np.arange(100, dtype=float) + 50.0,  # shifted
        "feat_b": np.ones(100),                         # unchanged
    })
    res = calculate_features_psi(ref_df, win_df, features=["feat_a", "feat_b"])
    assert "mean_psi" in res
    assert "max_psi" in res
    assert res["psi_feat_a"] > res["psi_feat_b"]
    assert pytest.approx(0.0, abs=1e-5) == res["psi_feat_b"]


# --------------------------------------------------------------------------- Score JS Tests
def test_score_js_identical_scores():
    scores = np.array([0.05, 0.12, 0.25, 0.50, 0.85, 0.92])
    js = calculate_score_js(scores, scores)
    assert pytest.approx(0.0, abs=1e-5) == js


def test_score_js_divergent_scores():
    # Low risk vs high risk scores
    ref_scores = np.full(500, 0.01)
    win_scores = np.full(500, 0.99)
    js = calculate_score_js(ref_scores, win_scores)
    assert js > 0.5


# --------------------------------------------------------------------------- Entropy Tests
def test_binary_entropy_known_values():
    # Maximum uncertainty: p = 0.5 -> H(p) = ln(2) ~ 0.6931
    max_h = calculate_binary_entropy(np.array([0.5]))
    assert pytest.approx(np.log(2.0), abs=1e-4) == max_h

    # Extreme certainty: p near 0 or 1 -> H(p) near 0
    min_h = calculate_binary_entropy(np.array([1e-6, 1.0 - 1e-6]))
    assert min_h < 0.001


def test_entropy_relative_change():
    ref_scores = np.array([0.1, 0.1, 0.1])
    win_scores = np.array([0.5, 0.5, 0.5])  # higher uncertainty
    rel_change = calculate_entropy_relative_change(ref_scores, win_scores)
    assert rel_change > 0.0


# --------------------------------------------------------------------------- Explainability Tests
class MockExplainerList:
    """Simulates SHAP TreeExplainer returning list of [class_0, class_1]."""
    def shap_values(self, X):
        n_rows, n_cols = X.shape
        c0 = -np.ones((n_rows, n_cols))
        c1 = np.arange(1, n_cols + 1, dtype=float) * np.ones((n_rows, 1))
        return [c0, c1]


class MockExplainerNdarray:
    """Simulates SHAP TreeExplainer returning 3D array (N, n_features, 2)."""
    def shap_values(self, X):
        n_rows, n_cols = X.shape
        arr = np.zeros((n_rows, n_cols, 2))
        arr[:, :, 1] = np.arange(1, n_cols + 1, dtype=float)
        return arr


def test_shap_profile_normalization_and_formats():
    features = ["f1", "f2", "f3"]
    df = pd.DataFrame(np.random.randn(50, 3), columns=features)

    # Test list format
    exp_list = MockExplainerList()
    prof_list = calculate_shap_profile(exp_list, df, features=features)
    assert pytest.approx(1.0, abs=1e-6) == np.sum(prof_list)
    assert prof_list[2] > prof_list[1] > prof_list[0]  # f3 > f2 > f1

    # Test 3D ndarray format
    exp_arr = MockExplainerNdarray()
    prof_arr = calculate_shap_profile(exp_arr, df, features=features)
    assert pytest.approx(1.0, abs=1e-6) == np.sum(prof_arr)
    pd.testing.assert_series_equal(pd.Series(prof_list), pd.Series(prof_arr))


def test_shap_shift_and_deltas():
    prof_ref = np.array([0.5, 0.3, 0.2])
    prof_win = np.array([0.2, 0.3, 0.5])

    # Identical shift
    assert pytest.approx(0.0, abs=1e-5) == calculate_shap_shift(prof_ref, prof_ref)

    # Shifted
    shift = calculate_shap_shift(prof_win, prof_ref)
    assert shift > 0.1

    # Deltas
    deltas = calculate_per_feature_delta(prof_win, prof_ref, features=["f1", "f2", "f3"])
    assert pytest.approx(-0.3, abs=1e-5) == deltas["delta_f1"]
    assert pytest.approx(0.0, abs=1e-5) == deltas["delta_f2"]
    assert pytest.approx(0.3, abs=1e-5) == deltas["delta_f3"]
