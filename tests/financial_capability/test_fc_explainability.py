"""SHAP tests: explanations work and feature names / order line up with the model inputs."""

import numpy as np
import pytest

from backend.src.financial_capability.fc_config import CLASS_NAMES, FEATURE_NAMES
from backend.src.financial_capability.fc_explainability import (
    FeatureExplainer,
    describe_local_explanation,
    explain_global,
    explain_individual,
    original_feature_of,
    stability_summary,
    top_features,
)


@pytest.fixture(scope="module", params=["rf_model", "xgb_model"])
def explainer(request):
    return FeatureExplainer(request.getfixturevalue(request.param))


def test_original_feature_mapping():
    assert original_feature_of("num__SAVEHABIT") == "SAVEHABIT"
    assert original_feature_of("cat__EMPLOY_2.0") == "EMPLOY"
    assert original_feature_of("cat__HOUSING_1.0") == "HOUSING"


def test_shap_feature_order_matches_model_inputs(explainer, splits):
    _, Xte, _, _ = splits
    assert explainer.feature_order == list(FEATURE_NAMES)
    values = explainer.shap_by_feature(Xte.head(20))
    assert values.shape == (20, len(FEATURE_NAMES), len(CLASS_NAMES))
    assert np.isfinite(values).all()


def test_global_importance_covers_every_feature(explainer, splits):
    _, Xte, _, _ = splits
    imp = explainer.explain_global(Xte.head(60))
    assert set(imp["feature"]) == set(FEATURE_NAMES)
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    assert (imp["mean_abs_shap"] >= 0).all()
    assert len(top_features(imp, 5)) == 5


def test_individual_explanation_is_aligned_and_for_predicted_class(explainer, splits):
    _, Xte, _, _ = splits
    row = Xte.head(1)
    res = explainer.explain_individual(row)
    table = res["table"]
    assert set(table["feature"]) == set(FEATURE_NAMES)
    expected = int(np.argmax(explainer.pipeline.predict_proba(row)[0]))
    assert res["class_index"] == expected and res["class_name"] == CLASS_NAMES[expected]
    # signs in the table agree with the wording of the generated sentences
    for sentence, (_, r) in zip(describe_local_explanation(res, top_n=3), table.head(3).iterrows()):
        assert ("toward" in sentence) == (r["shap_value"] > 0)


def test_individual_explanation_rejects_multiple_rows(explainer, splits):
    _, Xte, _, _ = splits
    with pytest.raises(ValueError):
        explainer.explain_individual(Xte.head(2))


def test_module_level_helpers(rf_model, splits):
    _, Xte, _, _ = splits
    assert len(explain_global(rf_model, Xte.head(30))) == len(FEATURE_NAMES)
    assert "table" in explain_individual(rf_model, Xte.head(1))


def test_stability_summary_identical_sets_give_overlap_one():
    s = stability_summary([["a", "b"], ["a", "b"], ["a", "b"]])
    assert s["mean_pairwise_jaccard"] == 1.0
    s2 = stability_summary([["a", "b"], ["c", "d"]])
    assert s2["mean_pairwise_jaccard"] == 0.0

