"""Tests for the leakage helper, windows and splits."""

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_config import build_split_config
from backend.src.fraud_detection.fd_preprocessing import (
    build_splits, make_windows, matured_cutoff, matured_mask,
)


def _df():
    return pd.DataFrame({"TX_TIME_DAYS": np.arange(0, 183)})


@pytest.mark.parametrize("d", [7, 14, 28])
def test_matured_mask_never_includes_future_labels(d):
    df = _df()
    rng = np.random.default_rng(0)
    for t in rng.integers(40, 182, size=50):
        m = matured_mask(df, int(t), d)
        assert df.loc[m, "TX_TIME_DAYS"].max() <= t - d
        assert df.loc[~m, "TX_TIME_DAYS"].min() > t - d
        assert matured_cutoff(int(t), d) == t - d


def test_matured_boundary_is_inclusive():
    df = _df()
    assert matured_mask(df, 100, 7)[93]
    assert not matured_mask(df, 100, 7)[94]


def test_windows():
    w7 = make_windows(70, 181, 7)
    assert len(w7) == 16 and w7[0] == (70, 76) and w7[-1] == (175, 181)
    w3 = make_windows(70, 181, 3)
    assert w3[0] == (70, 72) and all(hi - lo == 2 for lo, hi in w3)


def test_split_d7_matches_spec():
    c = build_split_config(7)
    assert (c.train_start, c.train_end) == (35, 62)
    assert (c.gap_start, c.gap_end) == (63, 69)
    assert (c.test_start, c.test_end) == (70, 181)
    assert (c.reference_start, c.reference_end) == (70, 90)
    assert c.drift_onset == 105


@pytest.mark.parametrize("d,start", [(14, 77), (28, 91)])
def test_split_other_delays(d, start):
    c = build_split_config(d)
    assert c.test_start == start and c.drift_onset == 112
    assert c.reference_end == start + 20
    assert c.gap_end == start - 1


def test_build_splits_masks_disjoint():
    s = build_splits(_df(), 7)
    assert not (s["train"] & s["gap"]).any()
    assert not (s["gap"] & s["test"]).any()
    assert s["train"].sum() == 28 and s["reference"].sum() == 21
