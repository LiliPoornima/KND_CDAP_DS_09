"""Tests for drift injection generators and invariants (Spec Section 8 and 15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.src.fraud_detection.fd_config import (
    COL_AMOUNT,
    COL_CUSTOMER,
    COL_DATETIME,
    COL_DAY,
    COL_FRAUD,
    COL_ID,
    COL_SCENARIO,
    COL_SECONDS,
    COL_TERMINAL,
    SCENARIO_CONTROL,
    SCENARIO_S1,
    SCENARIO_S2,
    SCENARIO_S3,
)
from backend.src.fraud_detection.fd_drift_injection import (
    get_scenario_generator,
    inject_control,
    inject_drift,
    inject_s1_concept,
    inject_s2_mixed,
    inject_s3_virtual_covariate,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create a realistic miniature DataFrame spanning days 90..120."""
    records = []
    base_time = pd.Timestamp("2018-04-01")
    tx_id = 1
    # 20 customers, 10 terminals, days 90 to 120
    for day in range(90, 121):
        for cust in range(1, 21):
            # 2 transactions per customer per day
            for hour in (10, 16):
                dt = base_time + pd.Timedelta(days=day, hours=hour)
                amt = 50.0 + (cust * 7.5) + (day % 5) * 20.0
                # base fraud rate ~1%
                is_fraud = 1 if (cust == 1 and day % 10 == 0 and hour == 10) else 0
                scenario = 2 if is_fraud else 0
                records.append({
                    COL_ID: tx_id,
                    COL_DATETIME: dt,
                    COL_CUSTOMER: cust,
                    COL_TERMINAL: (cust % 10) + 1,
                    COL_AMOUNT: amt,
                    COL_SECONDS: int((dt - base_time).total_seconds()),
                    COL_DAY: day,
                    COL_FRAUD: is_fraud,
                    COL_SCENARIO: scenario,
                })
                tx_id += 1
    df = pd.DataFrame(records)
    return df.sort_values([COL_DATETIME, COL_ID]).reset_index(drop=True)


# --------------------------------------------------------------------------- Invariant: Non-mutation
def test_all_generators_never_mutate_input(sample_raw_df: pd.DataFrame):
    raw_copy = sample_raw_df.copy(deep=True)
    onset = 105

    for gen in (inject_control, inject_s1_concept, inject_s2_mixed, inject_s3_virtual_covariate):
        out = gen(sample_raw_df, onset_day=onset, seed=42)
        assert out is not sample_raw_df
        pd.testing.assert_frame_equal(sample_raw_df, raw_copy)


# --------------------------------------------------------------------------- Control Invariants
def test_control_is_exact_identity(sample_raw_df: pd.DataFrame):
    out = inject_control(sample_raw_df, onset_day=105, seed=1)
    pd.testing.assert_frame_equal(out, sample_raw_df)


# --------------------------------------------------------------------------- S1 Concept Invariants
def test_s1_concept_changes_no_input_columns(sample_raw_df: pd.DataFrame):
    onset = 105
    out = inject_s1_concept(sample_raw_df, onset_day=onset, seed=1)

    # Input columns must be 100% identical
    input_cols = [
        COL_ID, COL_DATETIME, COL_CUSTOMER, COL_TERMINAL,
        COL_AMOUNT, COL_SECONDS, COL_DAY,
    ]
    for col in input_cols:
        pd.testing.assert_series_equal(out[col], sample_raw_df[col], check_names=True)

    # Pre-onset labels must be untouched
    pre_mask = sample_raw_df[COL_DAY] < onset
    pd.testing.assert_series_equal(
        out.loc[pre_mask, COL_FRAUD],
        sample_raw_df.loc[pre_mask, COL_FRAUD],
    )
    pd.testing.assert_series_equal(
        out.loc[pre_mask, COL_SCENARIO],
        sample_raw_df.loc[pre_mask, COL_SCENARIO],
    )

    # Post-onset rows with amount > 180 and originally 0 must become fraud=1, scenario=1
    post_high = (sample_raw_df[COL_DAY] >= onset) & (sample_raw_df[COL_AMOUNT] > 180.0)
    assert (out.loc[post_high, COL_FRAUD] == 1).all()
    assert (out.loc[post_high, COL_SCENARIO] == 1).all()


# --------------------------------------------------------------------------- S2 Mixed Invariants
def test_s2_mixed_compromises_post_onset_only(sample_raw_df: pd.DataFrame):
    onset = 105
    out = inject_s2_mixed(sample_raw_df, onset_day=onset, seed=1)

    # Pre-onset period must be completely identical
    pre_mask = sample_raw_df[COL_DAY] < onset
    pd.testing.assert_frame_equal(out[pre_mask], sample_raw_df[pre_mask])

    # In post-onset, amounts and frauds should change for compromised transactions
    post_mask = sample_raw_df[COL_DAY] >= onset
    assert out.loc[post_mask, COL_FRAUD].sum() > sample_raw_df.loc[post_mask, COL_FRAUD].sum()
    assert (out.loc[post_mask, COL_AMOUNT] >= sample_raw_df.loc[post_mask, COL_AMOUNT]).all()


# --------------------------------------------------------------------------- S3 Virtual Invariants
def test_s3_virtual_covariate_changes_no_labels(sample_raw_df: pd.DataFrame):
    onset = 105
    out = inject_s3_virtual_covariate(sample_raw_df, onset_day=onset, seed=1)

    # Label invariance: fraud count and scenario mix must be exactly untouched
    assert out[COL_FRAUD].sum() == sample_raw_df[COL_FRAUD].sum()
    assert (
        out[COL_SCENARIO].value_counts().to_dict()
        == sample_raw_df[COL_SCENARIO].value_counts().to_dict()
    )

    # Pre-onset rows should have unmodified timestamps
    pre_orig = sample_raw_df[sample_raw_df[COL_DAY] < onset].sort_values(COL_ID).reset_index(drop=True)
    pre_out = out[out[COL_ID].isin(pre_orig[COL_ID])].sort_values(COL_ID).reset_index(drop=True)
    pd.testing.assert_series_equal(pre_out[COL_DATETIME], pre_orig[COL_DATETIME])

    # Recomputed TX_TIME_DAYS must match TX_DATETIME
    recomputed_days = (out[COL_DATETIME] - pd.Timestamp("2018-04-01")).dt.days
    pd.testing.assert_series_equal(out[COL_DAY], recomputed_days, check_names=False)


# --------------------------------------------------------------------------- Dispatcher & Seeds
def test_dispatcher_and_generator_map(sample_raw_df: pd.DataFrame):
    out_ctrl = inject_drift(sample_raw_df, SCENARIO_CONTROL, onset_day=105, seed=1)
    pd.testing.assert_frame_equal(out_ctrl, sample_raw_df)

    out_s1 = inject_drift(sample_raw_df, "s1", onset_day=105, seed=1)
    assert out_s1.loc[out_s1[COL_AMOUNT] > 180, COL_FRAUD].sum() > 0

    with pytest.raises(ValueError, match="Unknown drift scenario"):
        get_scenario_generator("non_existent_scenario")


def test_seed_determinism(sample_raw_df: pd.DataFrame):
    # Same seed -> exact same output
    out1 = inject_s2_mixed(sample_raw_df, onset_day=105, seed=123)
    out2 = inject_s2_mixed(sample_raw_df, onset_day=105, seed=123)
    pd.testing.assert_frame_equal(out1, out2)

    # Different seed -> different customer selection in S2
    out3 = inject_s2_mixed(sample_raw_df, onset_day=105, seed=456)
    assert not out1.equals(out3)


# --------------------------------------------------------------------------- Spec Section 15 Regression Targets
def test_reproduction_summary_satisfies_spec_targets():
    """Verify that the generated reproduction summary satisfies Section 15 acceptance criteria."""
    from backend.src.fraud_detection.fd_config import OUTPUT_DIR

    summary_file = OUTPUT_DIR / "drift_reproduction" / "reproduction_summary_d7_seed1.csv"
    if not summary_file.is_file():
        pytest.skip("Reproduction summary CSV not yet generated.")

    df = pd.read_csv(summary_file)
    rec_by_sc = df.set_index("scenario")

    # Spec Section 15:
    # tolerance: +/- 0.06 absolute on recall, +/- 10 percentage points on relative drop
    # Control: recall ~0.75 before and after (change within about 5%)
    ctrl = rec_by_sc.loc["control"]
    assert pytest.approx(0.75, abs=0.06) == ctrl["recall_pre_onset"]
    assert pytest.approx(0.75, abs=0.06) == ctrl["recall_post_onset"]
    assert abs(ctrl["relative_change_pct"]) <= 5.0

    # S1: ~0.75 -> 0.53 (~ -29%)
    s1 = rec_by_sc.loc["S1"]
    assert pytest.approx(0.53, abs=0.06) == s1["recall_post_onset"]
    assert pytest.approx(-29.0, abs=10.0) == s1["relative_change_pct"]

    # S2: ~0.75 -> 0.46 (~ -38%)
    s2 = rec_by_sc.loc["S2"]
    assert pytest.approx(0.46, abs=0.06) == s2["recall_post_onset"]
    assert pytest.approx(-38.0, abs=10.0) == s2["relative_change_pct"]

    # S3: ~0.75 -> 0.735 (~ -2%)
    s3 = rec_by_sc.loc["S3"]
    assert pytest.approx(0.735, abs=0.06) == s3["recall_post_onset"]
    assert pytest.approx(-2.0, abs=10.0) == s3["relative_change_pct"]
