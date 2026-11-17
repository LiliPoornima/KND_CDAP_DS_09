"""Leakage-safe feature selection and target definition."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .fc_config import (
    CLASS_NAMES,
    CLASS_TO_INT,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FORBIDDEN_MODEL_COLUMNS,
    LOW_MAX_SCORE,
    MEDIUM_MAX_SCORE,
    TARGET_SOURCE,
    TARGET_SPECIAL_CODES,
)
from .fc_data_loader import FinancialCapabilityDataError


class LeakageError(ValueError):
    """Raised when a forbidden (target-leaking / outcome-like) field would enter X."""


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    """Fail loudly if any column is a score item, target copy or other forbidden field.

    Why: the 10 FWB items are summed into the target. Feeding them to the model
    would let it re-learn the scoring formula and report fake accuracy.
    """
    bad = sorted(set(columns) & set(FORBIDDEN_MODEL_COLUMNS))
    if bad:
        raise LeakageError(f"Forbidden fields in model inputs: {bad}")


def score_to_class(score: float) -> Optional[str]:
    """Map an FWB score to Low / Medium / High (None if the score is missing)."""
    if score is None or pd.isna(score):
        return None
    if score <= LOW_MAX_SCORE:
        return CLASS_NAMES[0]
    if score <= MEDIUM_MAX_SCORE:
        return CLASS_NAMES[1]
    return CLASS_NAMES[2]


def build_target(df: pd.DataFrame) -> pd.Series:
    """Integer class target (0 Low, 1 Medium, 2 High); NaN where the score is a special code."""
    if TARGET_SOURCE not in df.columns:
        raise FinancialCapabilityDataError(f"Target column {TARGET_SOURCE} not found.")
    score = pd.to_numeric(df[TARGET_SOURCE], errors="coerce")
    score = score.mask(score.isin(TARGET_SPECIAL_CODES))
    labels = score.map(score_to_class)
    return labels.map(CLASS_TO_INT).astype("float").rename("target")


def clean_special_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the approved feature columns with codebook special codes handled.

    Rules (per codebook): -1 (Refused), 98/99 (savings: don't know / prefer not
    to say) and 99 (employment: refused) become NaN; the "no such account"
    code 7 in the automated-saving questions becomes 0; any value outside the
    documented options also becomes NaN. This is a fixed recoding (no fitting),
    so it cannot leak information between train and test.
    """
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise FinancialCapabilityDataError(f"Missing feature columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for spec in FEATURE_SPECS:
        col = pd.to_numeric(df[spec.name], errors="coerce").astype(float)
        if spec.recode:
            col = col.map(lambda v, m=spec.recode: m.get(v, v) if not pd.isna(v) else v)
        if spec.missing_codes:
            col = col.mask(col.isin(spec.missing_codes))
        col = col.where(col.isin(list(spec.options)) | col.isna())
        out[spec.name] = col.astype(float)
    return out[list(FEATURE_NAMES)]


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y): approved cleaned features and the 3-class target.

    Rows whose target is a special code (-4 / -1) are dropped because no label
    exists for them. X contains only the approved features in a fixed order.
    """
    assert_no_forbidden_features(FEATURE_NAMES)
    y_all = build_target(df)
    keep = y_all.notna()
    X = clean_special_codes(df.loc[keep])
    y = y_all.loc[keep].astype(int)
    assert_no_forbidden_features(X.columns)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def data_dictionary() -> pd.DataFrame:
    """Data dictionary of the approved features (name, meaning, coding, handling)."""
    rows = []
    for s in FEATURE_SPECS:
        rows.append({
            "variable": s.name,
            "label": s.label,
            "group": s.group,
            "type": s.kind,
            "codebook_question": s.question,
            "valid_values": "; ".join(f"{k} = {v}" for k, v in s.options.items()),
            "treated_as_missing": ", ".join(str(c) for c in s.missing_codes) or "-",
            "recoding_note": s.note or "-",
        })
    return pd.DataFrame(rows)


def excluded_columns_report(df: pd.DataFrame) -> pd.DataFrame:
    """Every CSV column that is NOT a model input, with the reason."""
    from .fc_config import (  # local import keeps the top of the module small
        ADMIN_FIELDS, FWB_SCALE_ITEMS, OUTCOME_ADJACENT_FIELDS, SENSITIVE_FIELDS,
        TARGET_DERIVED_FIELDS,
    )
    reasons = {}
    for c in FWB_SCALE_ITEMS:
        reasons[c] = "Leakage: one of the 10 items summed into the FWB score (target)"
    for c in TARGET_DERIVED_FIELDS:
        reasons[c] = "Leakage: target or target-derived field"
    for c in OUTCOME_ADJACENT_FIELDS:
        reasons[c] = "Outcome-like hardship / stress / well-being measure (not a capability input)"
    for c in ADMIN_FIELDS:
        reasons[c] = "Identifier, survey weight or sampling-frame field"
    for c in SENSITIVE_FIELDS:
        reasons[c] = "Sensitive demographic attribute kept out of the model"
    rows = []
    for col in df.columns:
        if col in FEATURE_NAMES:
            continue
        rows.append({"column": col, "reason": reasons.get(
            col, "Not selected: not part of the approved capability/context feature set")})
    return pd.DataFrame(rows)

