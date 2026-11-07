"""Financial-capability configuration: paths, target definition, feature dictionary.

Single source of truth for Component 05 (Phase 1). Nothing here imports other
components. Every feature below was checked against the official CFPB
National Financial Well-Being Survey (NFWBS) 2016 public-use codebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------- paths
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = REPO_ROOT / "data" / "financial_capability"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RAW_FILENAME = "NFWBS_PUF_2016_data.csv"
RAW_CSV_PATH: Path = RAW_DIR / RAW_FILENAME
MODEL_DIR: Path = REPO_ROOT / "models" / "financial_capability"
OUTPUT_DIR: Path = REPO_ROOT / "outputs" / "financial_capability"

SOURCE_URL = "https://www.consumerfinance.gov/data-research/financial-well-being-survey-data/"
DOWNLOAD_URL = "https://www.consumerfinance.gov/documents/5614/NFWBS_PUF_2016_data.csv"

# Names of the saved artifacts.
RANDOM_FOREST_FILE = "fc_random_forest.joblib"
XGBOOST_FILE = "fc_xgboost.joblib"
SELECTED_MODEL_FILE = "fc_selected_model.joblib"
METADATA_FILE = "fc_model_metadata.json"

# --------------------------------------------------------------------------- experiment
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# --------------------------------------------------------------------------- target
TARGET_SOURCE = "FWBscore"
TARGET_SPECIAL_CODES: Tuple[int, ...] = (-4, -1)  # -4 not written to dataset, -1 refused
CLASS_NAMES: Tuple[str, ...] = ("Low", "Medium", "High")
CLASS_TO_INT: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Proposed 3-class grouping (research document, section 8, step 3):
#   Low    = Very Low + Low + Medium Low   (score <= 49)
#   Medium = Medium High                   (50 - 57)
#   High   = High + Very High              (score >= 58)
# NOTE: the supplied user guide / codebook do not list the six score bands.
# The cut points come from the CFPB financial well-being scale bands as found
# on CFPB web material and MUST be confirmed with the supervisor. They are
# kept here (and only here) so they can be changed in one place.
LOW_MAX_SCORE: int = 49
MEDIUM_MAX_SCORE: int = 57

# --------------------------------------------------------------------------- leakage
# The ten survey items that are summed into FWBscore (codebook Table 1).
FWB_SCALE_ITEMS: Tuple[str, ...] = (
    "FWB1_1", "FWB1_2", "FWB1_3", "FWB1_4", "FWB1_5", "FWB1_6",
    "FWB2_1", "FWB2_2", "FWB2_3", "FWB2_4",
)

# Fields that carry the target or are built from it: never a model input.
TARGET_DERIVED_FIELDS: Tuple[str, ...] = (
    "FWBscore", "target", "target_class", "target_band",
)

# Outcome-like variables: they measure financial hardship/stress, i.e. the
# same concept as the target, not a teachable capability. Excluded to keep the
# model a "capability -> well-being" model (documented design choice).
OUTCOME_ADJACENT_FIELDS: Tuple[str, ...] = (
    "ENDSMEET", "COVERCOSTS", "ABSORBSHOCK", "DISTRESS", "SWB_1", "SWB_2", "SWB_3",
    "MATHARDSHIP_1", "MATHARDSHIP_2", "MATHARDSHIP_3",
    "MATHARDSHIP_4", "MATHARDSHIP_5", "MATHARDSHIP_6",
    "COLLECT", "REJECTED_1", "REJECTED_2", "BORROW_1", "BORROW_2",
    "OUTLOOK_1", "OUTLOOK_2", "HOUSESAT", "HEALTH",
)

# Identifiers, survey weights and sampling-frame fields.
ADMIN_FIELDS: Tuple[str, ...] = ("PUF_ID", "sample", "finalwt", "fpl", "PCTLT200FPL")

# Sensitive demographics, kept out of the model and out of all rule triggers.
SENSITIVE_FIELDS: Tuple[str, ...] = ("PPGENDER", "PPETHM")

FORBIDDEN_MODEL_COLUMNS: Tuple[str, ...] = (
    FWB_SCALE_ITEMS + TARGET_DERIVED_FIELDS + OUTCOME_ADJACENT_FIELDS
    + ADMIN_FIELDS + SENSITIVE_FIELDS
)

# --------------------------------------------------------------------------- feature groups
GROUP_SAVING = "Saving"
GROUP_BUDGETING = "Budgeting"
GROUP_KNOWLEDGE = "Basic Financial Knowledge"
GROUP_GOALS = "Financial Goal Planning"
GROUP_CONTEXT = "Background (context only)"
CAPABILITY_GROUPS: Tuple[str, ...] = (GROUP_SAVING, GROUP_BUDGETING, GROUP_KNOWLEDGE, GROUP_GOALS)

# Learning-category labels (Phase 1 = labels only, no lesson content).
LEARNING_CATEGORY: Dict[str, str] = {
    GROUP_SAVING: "Saving",
    GROUP_BUDGETING: "Budgeting",
    GROUP_KNOWLEDGE: "Basic Financial Knowledge",
    GROUP_GOALS: "Financial Goal Planning",
}

KIND_ORDINAL = "ordinal"
KIND_BINARY = "binary"
KIND_NOMINAL = "nominal"


@dataclass(frozen=True)
class FeatureSpec:
    """One approved model input with its codebook meaning.

    ``options`` maps each VALID (cleaned) value to its response label.
    Raw codes listed in ``missing_codes`` become NaN; ``recode`` maps raw
    codes to cleaned codes (used for the automated-saving questions).
    """

    name: str
    label: str
    group: str
    kind: str
    options: Dict[int, str]
    question: str
    missing_codes: Tuple[int, ...] = (-1,)
    recode: Optional[Dict[int, int]] = None
    note: str = ""


_LIKERT_AGREE5 = {1: "Strongly disagree", 2: "Disagree", 3: "Neither agree nor disagree",
                  4: "Agree", 5: "Strongly agree"}
_FREQ5 = {1: "Not applicable or never", 2: "Seldom", 3: "Sometimes", 4: "Often", 5: "Always"}
_SKILL5 = {1: "Not at all", 2: "Very little", 3: "Somewhat", 4: "Very well", 5: "Completely"}
_YES_NO = {0: "No", 1: "Yes"}

FEATURE_SPECS: Tuple[FeatureSpec, ...] = (
    # ---- Saving
    FeatureSpec("SAVEHABIT", "Saving habit", GROUP_SAVING, KIND_ORDINAL,
                {1: "Strongly disagree", 2: "Disagree", 3: "Disagree slightly",
                 4: "Agree slightly", 5: "Agree", 6: "Strongly agree"},
                "Putting money into savings is a habit for me"),
    FeatureSpec("SAVINGSRANGES", "Savings balance", GROUP_SAVING, KIND_ORDINAL,
                {1: "$0", 2: "$1-99", 3: "$100-999", 4: "$1,000-4,999",
                 5: "$5,000-19,999", 6: "$20,000-74,999", 7: "$75,000 or more"},
                "How much money do you have in savings today?",
                missing_codes=(-1, 98, 99),
                note="98 = I don't know and 99 = Prefer not to say are treated as missing."),
    FeatureSpec("AUTOMATED_1", "Automatic saving: retirement account", GROUP_SAVING, KIND_BINARY,
                _YES_NO, "Do you save automatically into a retirement savings account?",
                recode={0: 0, 1: 1, 7: 0},
                note="Raw 7 (I do not have this type of account) is recoded to 0 = not automated."),
    FeatureSpec("AUTOMATED_2", "Automatic saving: other savings account", GROUP_SAVING, KIND_BINARY,
                _YES_NO, "Do you save automatically into a non-retirement savings account?",
                recode={0: 0, 1: 1, 7: 0},
                note="Raw 7 (I do not have this type of account) is recoded to 0 = not automated."),
    FeatureSpec("FS1_7", "Skill: making myself save", GROUP_SAVING, KIND_ORDINAL, _SKILL5,
                "I know how to make myself save"),
    # ---- Budgeting / money management
    FeatureSpec("MANAGE1_1", "Paid all bills on time", GROUP_BUDGETING, KIND_ORDINAL, _FREQ5,
                "In the past 12 months, how often have you paid all your bills on time?"),
    FeatureSpec("MANAGE1_2", "Stayed within budget", GROUP_BUDGETING, KIND_ORDINAL, _FREQ5,
                "In the past 12 months, how often have you stayed within your budget or spending plan?"),
    FeatureSpec("MANAGE1_3", "Paid credit card in full", GROUP_BUDGETING, KIND_ORDINAL, _FREQ5,
                "In the past 12 months, how often have you paid off your credit card balance in full each month?"),
    FeatureSpec("MANAGE1_4", "Checked statements for errors", GROUP_BUDGETING, KIND_ORDINAL, _FREQ5,
                "In the past 12 months, how often have you checked your statements, bills and receipts for errors?"),
    FeatureSpec("PROPPLAN_1", "Consults budget", GROUP_BUDGETING, KIND_ORDINAL, _LIKERT_AGREE5,
                "I consult my budget to see how much money I have left"),
    FeatureSpec("PROPPLAN_2", "Plans to stick to budget", GROUP_BUDGETING, KIND_ORDINAL, _LIKERT_AGREE5,
                "I actively consider the steps I need to take to stick to my budget"),
    FeatureSpec("FS1_6", "Skill: avoiding overspending", GROUP_BUDGETING, KIND_ORDINAL, _SKILL5,
                "I know how to keep myself from spending too much"),
    FeatureSpec("ACT1_1", "Follows through on commitments to others", GROUP_BUDGETING, KIND_ORDINAL,
                _SKILL5, "I follow through on my financial commitments to others"),
    # ---- Basic financial knowledge
    FeatureSpec("SUBKNOWL1", "Self-rated financial knowledge", GROUP_KNOWLEDGE, KIND_ORDINAL,
                {1: "1 - Very low", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7 - Very high"},
                "How would you assess your overall financial knowledge?"),
    FeatureSpec("LMscore", "Basic financial knowledge quiz (0-3 correct)", GROUP_KNOWLEDGE,
                KIND_ORDINAL, {0: "0 correct", 1: "1 correct", 2: "2 correct", 3: "3 correct"},
                "Number of the three basic questions (compound interest, inflation, "
                "stocks vs mutual funds) answered correctly",
                missing_codes=()),
    FeatureSpec("FS1_3", "Skill: complex financial decisions", GROUP_KNOWLEDGE, KIND_ORDINAL, _SKILL5,
                "I know how to make complex financial decisions"),
    FeatureSpec("FS2_3", "Struggles to understand financial information", GROUP_KNOWLEDGE,
                KIND_ORDINAL, {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"},
                "I struggle to understand financial information"),
    # ---- Financial goal planning
    FeatureSpec("FINGOALS", "Has a current or recent financial goal", GROUP_GOALS, KIND_BINARY,
                _YES_NO, "Do you have a current or recent financial goal?"),
    FeatureSpec("PROPPLAN_3", "Sets financial goals", GROUP_GOALS, KIND_ORDINAL, _LIKERT_AGREE5,
                "I set financial goals for what I want to achieve with my money"),
    FeatureSpec("PROPPLAN_4", "Has a clear action plan", GROUP_GOALS, KIND_ORDINAL, _LIKERT_AGREE5,
                "I prepare a clear plan of action with detailed steps to achieve my financial goals"),
    FeatureSpec("GOALCONF", "Confidence in achieving goals", GROUP_GOALS, KIND_ORDINAL,
                {1: "Not at all confident", 2: "Not very confident", 3: "Somewhat confident",
                 4: "Very confident"},
                "Confidence in own ability to achieve financial goals"),
    FeatureSpec("ACT1_2", "Follows through on own goals", GROUP_GOALS, KIND_ORDINAL, _SKILL5,
                "I follow through on financial goals I set for myself"),
    FeatureSpec("SCFHORIZON", "Financial planning time horizon", GROUP_GOALS, KIND_ORDINAL,
                {1: "The next few months", 2: "The next year", 3: "The next few years",
                 4: "The next 5 to 10 years", 5: "Longer than 10 years"},
                "Which time period is most important when planning your family's saving and spending?"),
    FeatureSpec("FS1_1", "Skill: following through on intentions", GROUP_GOALS, KIND_ORDINAL, _SKILL5,
                "I know how to get myself to follow through on my financial intentions"),
    # ---- Background context (not capability triggers)
    FeatureSpec("agecat", "Age group", GROUP_CONTEXT, KIND_ORDINAL,
                {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-61",
                 6: "62-69", 7: "70-74", 8: "75+"},
                "Age group", missing_codes=()),
    FeatureSpec("PPINCIMP", "Household income", GROUP_CONTEXT, KIND_ORDINAL,
                {1: "Less than $20,000", 2: "$20,000 to $29,999", 3: "$30,000 to $39,999",
                 4: "$40,000 to $49,999", 5: "$50,000 to $59,999", 6: "$60,000 to $74,999",
                 7: "$75,000 to $99,999", 8: "$100,000 to $149,999", 9: "$150,000 or more"},
                "Household income", missing_codes=()),
    FeatureSpec("PPEDUC", "Education", GROUP_CONTEXT, KIND_ORDINAL,
                {1: "Less than high school", 2: "High school degree/GED",
                 3: "Some college/Associate", 4: "Bachelor's degree",
                 5: "Graduate/professional degree"},
                "Highest degree received", missing_codes=()),
    FeatureSpec("EMPLOY", "Employment status", GROUP_CONTEXT, KIND_NOMINAL,
                {1: "Self-employed", 2: "Work full-time for an employer or the military",
                 3: "Work part-time for an employer or the military", 4: "Homemaker",
                 5: "Full-time student", 6: "Permanently sick, disabled or unable to work",
                 7: "Unemployed or temporarily laid off", 8: "Retired"},
                "Primary or only employment status", missing_codes=(-1, 99)),
    FeatureSpec("HOUSING", "Housing situation", GROUP_CONTEXT, KIND_NOMINAL,
                {1: "I own my home", 2: "I rent", 3: "I do not currently own or rent"},
                "Which best describes your housing situation?"),
    FeatureSpec("PPHHSIZE", "Household size", GROUP_CONTEXT, KIND_ORDINAL,
                {1: "1", 2: "2", 3: "3", 4: "4", 5: "5+"}, "Household size", missing_codes=()),
    FeatureSpec("PPMARIT", "Marital status", GROUP_CONTEXT, KIND_NOMINAL,
                {1: "Married", 2: "Widowed", 3: "Divorced/Separated", 4: "Never married",
                 5: "Living with partner"},
                "Marital status", missing_codes=()),
)

FEATURE_SPEC_BY_NAME: Dict[str, FeatureSpec] = {s.name: s for s in FEATURE_SPECS}
FEATURE_NAMES: Tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)
NOMINAL_FEATURES: Tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS if s.kind == KIND_NOMINAL)
NUMERIC_FEATURES: Tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS if s.kind != KIND_NOMINAL)

# Columns the CSV must contain for Component 05 to run.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    (TARGET_SOURCE,) + FWB_SCALE_ITEMS + FEATURE_NAMES
)

# --------------------------------------------------------------------------- model grids
@dataclass(frozen=True)
class SearchConfig:
    """Small, documented hyper-parameter grids tuned with CV on training data only."""

    rf_grid: Dict[str, list] = field(default_factory=lambda: {
        "n_estimators": [300],
        "max_depth": [None, 10, 16],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.5],
    })
    xgb_grid: Dict[str, list] = field(default_factory=lambda: {
        "n_estimators": [200, 400, 600],
        "max_depth": [1, 2, 3],
        "learning_rate": [0.02, 0.05, 0.08],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    })


SEARCH_CONFIG = SearchConfig()

# SHAP / reliability settings
SHAP_SAMPLE_SIZE: int = 500
STABILITY_SEEDS: Tuple[int, ...] = (11, 22, 33, 44, 55)
STABILITY_TOP_K: int = 8
MIN_SUBGROUP_SIZE: int = 50


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configuration is left to the entry point)."""
    return logging.getLogger(name)
