import pandas as pd
import pytest

from frontend.financial_inclusion.constants import CLUSTER_LABELS
from frontend.financial_inclusion.sections.gaps import (
    build_gap_comparison,
    gap_insights,
    region_dimension_summary,
)
from frontend.financial_inclusion.sections.subgroups import (
    AGE_BANDS,
    VERIFIED_REGION_CATEGORIES,
    demographic_profile_distribution,
)


def test_gap_comparison_uses_selected_minus_pooled_and_reports_valid_counts():
    selected = pd.DataFrame(
        {
            "score_connectivity": [1.0],
            "score_financial_access": [0.0],
            "score_digital_financial_usage": [0.5],
            "score_digital_safety": [0.0],
            "score_financial_resilience": [0.5],
        }
    )
    pooled = pd.DataFrame(
        {
            "score_connectivity": [1.0, 0.0],
            "score_financial_access": [0.0, 1.0],
            "score_digital_financial_usage": [0.5, 0.5],
            "score_digital_safety": [0.0, None],
            "score_financial_resilience": [0.5, 0.5],
        }
    )

    comparison = build_gap_comparison(selected, pooled)
    connectivity = comparison.loc[
        comparison["Dimension"] == "Connectivity"
    ].iloc[0]
    financial_access = comparison.loc[
        comparison["Dimension"] == "Financial access"
    ].iloc[0]

    assert connectivity["Difference in mean score"] == pytest.approx(0.5)
    assert connectivity["Selected valid respondents"] == 1
    assert connectivity["Pooled valid respondents"] == 2
    assert comparison.loc[
        comparison["Dimension"] == "Digital safety",
        "Pooled valid respondents",
    ].iloc[0] == 1
    assert financial_access["Difference in mean score"] == pytest.approx(-0.5)

    insights = gap_insights(comparison)
    assert any("higher observed mean" in insight and "Connectivity" in insight for insight in insights)
    assert any("lower observed mean" in insight and "Financial access" in insight for insight in insights)


def test_gap_insights_handles_equal_and_unavailable_means():
    comparison = pd.DataFrame(
        {
            "Dimension": ["Connectivity", "Financial access"],
            "Difference in mean score": [0.0, 0.0],
        }
    )
    assert "equal across all dimensions" in gap_insights(comparison)[0]

    comparison["Difference in mean score"] = float("nan")
    assert gap_insights(comparison) == []


def test_demographic_age_bands_have_documented_boundaries_and_category_denominators():
    frame = pd.DataFrame(
        {
            "age": [15, 24, 25, 65, 100, None, 14],
            "cluster_id": [0, 1, 1, 2, 4, 0, 3],
        }
    )

    summary, excluded = demographic_profile_distribution(frame, "age")

    assert excluded == 2
    assert set(summary["Category"]) == {"15–24", "25–34", "65+"}
    assert tuple(AGE_BANDS) == ("15–24", "25–34", "35–44", "45–54", "55–64", "65+")
    totals = summary.groupby("Category")["Share within category (%)"].sum()
    assert totals.to_dict() == pytest.approx({category: 100.0 for category in totals.index})
    assert summary.loc[summary["Category"] == "15–24", "Category respondents"].eq(2).all()
    assert summary.loc[summary["cluster_id"] == 0, "Profile"].iloc[0] == CLUSTER_LABELS[0]


def test_demographics_exclude_unverified_region_codes_and_missing_values():
    verified_region = VERIFIED_REGION_CATEGORIES[0]
    frame = pd.DataFrame(
        {
            "regionwb": [verified_region, verified_region, "unknown code", None],
            "cluster_id": [0, 1, 2, 3],
        }
    )

    summary, excluded = demographic_profile_distribution(frame, "regionwb")

    assert excluded == 2
    assert summary["Category"].unique().tolist() == [verified_region]
    assert summary["Category respondents"].eq(2).all()
    assert summary["Share within category (%)"].sum() == pytest.approx(100.0)
    assert summary.loc[summary["cluster_id"] == 1, "Profile"].iloc[0] == CLUSTER_LABELS[1]


def test_demographics_reject_unvalidated_cluster_assignments():
    frame = pd.DataFrame({"age": [30], "cluster_id": [7]})

    with pytest.raises(ValueError, match="valid saved profile labels"):
        demographic_profile_distribution(frame, "age")


def test_region_overview_retains_only_verified_region_categories():
    region = VERIFIED_REGION_CATEGORIES[0]
    frame = pd.DataFrame(
        {
            "regionwb": [region, "unknown code"],
            "score_connectivity": [0.25, 1.0],
            "score_financial_access": [0.50, 1.0],
            "score_digital_financial_usage": [0.75, 1.0],
            "score_digital_safety": [0.25, 1.0],
            "score_financial_resilience": [0.50, 1.0],
        }
    )

    summary = region_dimension_summary(frame)

    assert summary["World Bank region"].tolist() == [region]
    assert summary.loc[0, "Respondents"] == 1
    assert summary.loc[0, "Connectivity"] == pytest.approx(0.25)
