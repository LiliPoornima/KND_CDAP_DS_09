"""Shared fixtures for Component 05 tests (small synthetic data; no network, no real file needed)."""

import numpy as np
import pandas as pd
import pytest

from backend.src.financial_capability.fc_config import (
    FEATURE_SPECS,
    FWB_SCALE_ITEMS,
)
from backend.src.financial_capability.fc_feature_selection import select_features_and_target
from backend.src.financial_capability.fc_preprocessing import split_data
from backend.src.financial_capability.fc_train import train_random_forest, train_xgboost


def make_raw_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the real column names, codebook-valid codes and a learnable target."""
    rng = np.random.default_rng(seed)
    data = {}
    for spec in FEATURE_SPECS:
        valid = list(spec.options)
        if spec.recode:  # automated-saving questions: raw 0 / 1 / 7
            valid = [0, 1, 7]
        data[spec.name] = rng.choice(valid, size=n)
    df = pd.DataFrame(data)
    for item in FWB_SCALE_ITEMS:
        df[item] = rng.integers(1, 6, size=n)
    # Score depends on a few answers so the models have something to learn.
    score = (30 + 4 * df["SAVEHABIT"] + 3 * df["PPINCIMP"] + 2 * df["GOALCONF"]
             + rng.normal(0, 4, size=n)).round().clip(0, 100).astype(int)
    df["FWBscore"] = score
    # Sprinkle special codes (refused) the way the real file does.
    df.loc[df.sample(frac=0.02, random_state=1).index, "SAVEHABIT"] = -1
    df.loc[df.sample(frac=0.05, random_state=2).index, "SAVINGSRANGES"] = 99
    df.loc[df.index[:3], "FWBscore"] = [-4, -1, -1]
    df["PUF_ID"] = np.arange(n)
    df["finalwt"] = 1.0
    return df


@pytest.fixture(scope="session")
def raw_df():
    return make_raw_frame()


@pytest.fixture(scope="session")
def xy(raw_df):
    return select_features_and_target(raw_df)


@pytest.fixture(scope="session")
def splits(xy):
    X, y = xy
    return split_data(X, y)  # Xtr, Xte, ytr, yte


@pytest.fixture(scope="session")
def rf_model(splits):
    Xtr, _, ytr, _ = splits
    return train_random_forest(Xtr, ytr, {"n_estimators": 40, "max_depth": 6})


@pytest.fixture(scope="session")
def xgb_model(splits):
    Xtr, _, ytr, _ = splits
    return train_xgboost(Xtr, ytr, {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1})


@pytest.fixture()
def sample_response():
    """A complete, valid response (cleaned codes) used by rules / prediction / what-if tests."""
    return {
        "SAVEHABIT": 4, "SAVINGSRANGES": 4, "AUTOMATED_1": 1, "AUTOMATED_2": 0, "FS1_7": 3,
        "MANAGE1_1": 5, "MANAGE1_2": 4, "MANAGE1_3": 5, "MANAGE1_4": 4, "PROPPLAN_1": 4,
        "PROPPLAN_2": 4, "FS1_6": 4, "ACT1_1": 5, "SUBKNOWL1": 5, "LMscore": 3, "FS1_3": 3,
        "FS2_3": 2, "FINGOALS": 1, "PROPPLAN_3": 4, "PROPPLAN_4": 3, "GOALCONF": 3, "ACT1_2": 4,
        "SCFHORIZON": 3, "FS1_1": 4, "agecat": 4, "PPINCIMP": 6, "PPEDUC": 3, "EMPLOY": 2,
        "HOUSING": 1, "PPHHSIZE": 3, "PPMARIT": 1,
    }

