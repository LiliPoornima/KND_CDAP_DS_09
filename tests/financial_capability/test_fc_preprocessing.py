"""Preprocessing tests: special codes, target mapping, split, and fit-on-train-only."""

import numpy as np
import pandas as pd
import pytest

from backend.src.financial_capability.fc_config import FEATURE_NAMES, NOMINAL_FEATURES
from backend.src.financial_capability.fc_feature_selection import (
    build_target,
    clean_special_codes,
    score_to_class,
)
from backend.src.financial_capability.fc_preprocessing import (
    build_preprocessor,
    class_distribution,
    split_data,
)


@pytest.mark.parametrize("score,label", [
    (0, "Low"), (49, "Low"), (50, "Medium"), (57, "Medium"), (58, "High"), (100, "High"),
])
def test_score_band_boundaries(score, label):
    assert score_to_class(score) == label


def test_special_target_codes_become_missing(raw_df):
    y = build_target(raw_df)
    assert y.iloc[:3].isna().all()
    assert y.iloc[3:].notna().all()
    assert set(y.dropna().unique()) <= {0.0, 1.0, 2.0}


def test_special_codes_are_not_treated_as_real_values(raw_df):
    X = clean_special_codes(raw_df)
    assert (X["SAVEHABIT"].dropna() >= 1).all()          # -1 removed
    assert X["SAVINGSRANGES"].dropna().max() <= 7        # 98/99 removed
    assert X["SAVEHABIT"].isna().sum() > 0
    assert X["SAVINGSRANGES"].isna().sum() > 0


def test_automated_saving_code_7_recoded_to_zero():
    row = pd.DataFrame([{n: 1 for n in FEATURE_NAMES}])
    row["AUTOMATED_1"] = 7
    row["AUTOMATED_2"] = 7
    row["EMPLOY"] = 99
    out = clean_special_codes(row)
    assert out.loc[0, "AUTOMATED_1"] == 0 and out.loc[0, "AUTOMATED_2"] == 0
    assert np.isnan(out.loc[0, "EMPLOY"])


def test_split_is_stratified_disjoint_and_reproducible(xy):
    X, y = xy
    a = split_data(X, y)
    b = split_data(X, y)
    Xtr, Xte, ytr, yte = a
    assert not set(Xtr.index) & set(Xte.index)
    assert len(Xtr) + len(Xte) == len(X)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    for cls in (0, 1, 2):  # class shares are preserved
        assert abs((ytr == cls).mean() - (yte == cls).mean()) < 0.02


def test_preprocessor_is_fitted_on_training_data_only(splits):
    Xtr, Xte, _, _ = splits
    prep = build_preprocessor().fit(Xtr)
    imputer = prep.named_transformers_["num"].named_steps["impute"]
    expected = Xtr[list(imputer.feature_names_in_)].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected, equal_nan=True)
    # Transforming test data must not change the fitted statistics.
    before = imputer.statistics_.copy()
    prep.transform(Xte)
    assert np.allclose(imputer.statistics_, before)


def test_preprocessor_handles_missing_and_unseen_categories(splits):
    Xtr, _, _, _ = splits
    prep = build_preprocessor().fit(Xtr)
    odd = Xtr.head(3).copy()
    odd.iloc[0] = np.nan                       # everything missing
    odd.loc[odd.index[1], "EMPLOY"] = 77       # unseen nominal code
    out = prep.transform(odd)
    assert not np.isnan(out).any()


def test_nominal_features_are_one_hot_encoded(splits):
    Xtr, _, _, _ = splits
    prep = build_preprocessor().fit(Xtr)
    names = prep.get_feature_names_out()
    for nominal in NOMINAL_FEATURES:
        assert any(n.startswith(f"cat__{nominal}_") for n in names)


def test_class_distribution_sums_to_total(xy):
    _, y = xy
    dist = class_distribution(y)
    assert sum(v["count"] for v in dist.values()) == len(y)
    assert set(dist) == {"Low", "Medium", "High"}

