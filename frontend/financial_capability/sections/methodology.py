"""Methodology & Limitations section."""

from __future__ import annotations

import streamlit as st

from backend.src.financial_capability.fc_config import (
    DOWNLOAD_URL,
    FWB_SCALE_ITEMS,
    LOW_MAX_SCORE,
    MEDIUM_MAX_SCORE,
    SOURCE_URL,
)
from frontend.financial_capability.data_loader import get_metadata


def render() -> None:
    st.subheader("Methodology & Limitations")
    meta = get_metadata()

    st.markdown("### Data source")
    st.markdown(f"CFPB National Financial Well-Being Survey (NFWBS) 2016 public-use file, with the "
                f"official codebook and user guide. Page: {SOURCE_URL} | file: {DOWNLOAD_URL}")

    st.markdown("### Target")
    st.markdown(
        f"The target is the CFPB Financial Well-Being score (`FWBscore`, 0-100) grouped into three "
        f"classes: **Low** (score <= {LOW_MAX_SCORE}: Very Low + Low + Medium Low), **Medium** "
        f"({LOW_MAX_SCORE + 1}-{MEDIUM_MAX_SCORE}: Medium High) and **High** (>= {MEDIUM_MAX_SCORE + 1}: "
        "High + Very High). This 3-class grouping is a proposed research choice, not the original "
        "CFPB category system; the cut points must be confirmed with the supervisor. Rows whose "
        "score is a special code (-4 / -1) are removed.")
    if meta:
        st.caption(f"Class counts in this run: {meta['target']['class_distribution_all']}")

    st.markdown("### Leakage prevention")
    st.markdown(
        f"The ten items summed into the score ({', '.join(FWB_SCALE_ITEMS)}), `FWBscore` and any "
        "target-derived field are never model inputs; an automated test fails if one enters the "
        "features. Outcome-like hardship/stress questions and sensitive demographics (gender, "
        "ethnicity) are also excluded. The model therefore answers a *capability -> well-being* "
        "question, so its accuracy is moderate rather than inflated.")

    st.markdown("### Pipeline")
    st.markdown(
        "- Special codes (Refused, Don't know, Prefer not to say) are converted to missing using "
        "the codebook; they are not treated as real values.\n"
        "- A scikit-learn `Pipeline` (median/mode imputation, one-hot for nominal answers, model) is "
        "fitted on training data only, also inside every cross-validation fold.\n"
        "- Stratified 80/20 split with a fixed seed; hyper-parameters are chosen by 5-fold CV on "
        "the training set (macro-F1); the test set is used once for the final comparison.\n"
        "- Random Forest (balanced class weights) and XGBoost (balanced sample weights) use the "
        "same features and split. No resampling (e.g. SMOTE) is used.\n"
        "- The fitted pipeline is saved locally and loaded by this page; the page never retrains.")

    st.markdown("### Explanations, rules and what-if")
    st.markdown(
        "- **SHAP** explains the model's behaviour (global and individual). It does not prove "
        "causation or a person's real ability.\n"
        "- **Capability gaps** are found by separate, readable rules on the person's own answers "
        "(codebook meaning), not from SHAP. Thresholds are proposed and need expert review.\n"
        "- **Learning categories** are labels only (Saving, Budgeting, Basic Financial Knowledge, "
        "Financial Goal Planning). There are no lessons, quizzes or progress tracking in Phase 1.\n"
        "- **What-if** re-runs the same saved model with changed answers. It is a model simulation, "
        "not causal evidence.")

    st.markdown("### Limitations")
    for item in (meta["limitations"] if meta else [
        "US adults surveyed in 2016; validity for Sri Lankan users has not been established.",
        "Cross-sectional survey: associations only, no causal claims.",
        "Self-reported answers; capability rules are proposed and need expert review.",
        "Results are educational support, not financial advice or a diagnosis.",
    ]):
        st.markdown(f"- {item}")
    st.markdown(
        "- Ground-truth labels for 'best lesson' do not exist in this dataset, so category "
        "suggestions cannot be validated here; user evaluation would be needed.\n"
        "- The Medium class (a narrow score band) is the hardest to predict; see the per-class "
        "metrics and confusion matrices.")

