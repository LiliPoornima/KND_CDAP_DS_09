import numpy as np

from backend.src.credit_risk.common.cr_calibration import (
    cr_bootstrap_group_calibration,
    cr_group_calibration_summary,
)


def test_group_calibration_bootstrap_matches_point_estimates_and_support_rules():
    truth = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    probability = [0.0, 0.1, 0.2, 0.6, 0.8, 1.0, 0.2, 0.3, 0.1, 0.4, 0.8]
    groups = ["supported"] * 6 + ["limited"] * 5

    result = cr_bootstrap_group_calibration(
        truth,
        probability,
        groups,
        attribute_name="cohort",
        n_bootstrap=30,
        random_state=17,
        min_group_size=6,
        min_class_count=3,
    )

    supported = result[result["group"] == "supported"]
    expected = cr_group_calibration_summary(
        truth[:6], probability[:6], groups[:6], attribute_name="cohort"
    )["group_metrics"].iloc[0]
    assert len(supported) == 4
    assert supported["calibration_reportable"].all()
    assert supported["valid_replicates"].eq(30).all()
    for metric in supported["metric"]:
        row = supported[supported["metric"] == metric].iloc[0]
        assert np.isclose(row["estimate"], expected[metric])
        assert row["lower"] <= row["estimate"] <= row["upper"]

    limited = result[result["group"] == "limited"]
    assert not limited["calibration_reportable"].any()
    assert limited["valid_replicates"].eq(0).all()
    assert limited[["lower", "upper"]].isna().all().all()


def test_group_calibration_bootstrap_rejects_invalid_probability():
    try:
        cr_bootstrap_group_calibration(
            [0, 1],
            [0.2, 1.1],
            ["A", "A"],
            n_bootstrap=5,
        )
    except ValueError as error:
        assert "Probabilities" in str(error)
    else:
        raise AssertionError("Out-of-range probability should be rejected.")
