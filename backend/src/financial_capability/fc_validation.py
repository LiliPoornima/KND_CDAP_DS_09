"""Dataset validation: honest, computed data-quality report (nothing hard-coded)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import pandas as pd

from .fc_config import (
    FEATURE_SPECS,
    FWB_SCALE_ITEMS,
    REQUIRED_COLUMNS,
    TARGET_SOURCE,
    TARGET_SPECIAL_CODES,
)
from .fc_data_loader import FinancialCapabilityDataError


@dataclass
class ValidationReport:
    """Facts measured from the downloaded file."""

    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    n_missing_cells: int
    missing_required_columns: List[str] = field(default_factory=list)
    target_special_code_counts: Dict[str, int] = field(default_factory=dict)
    n_valid_target_rows: int = 0
    target_min: float = float("nan")
    target_max: float = float("nan")
    out_of_codebook_values: Dict[str, int] = field(default_factory=dict)
    refused_or_missing_code_counts: Dict[str, int] = field(default_factory=dict)
    dtype_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required_columns and self.n_valid_target_rows > 0

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataset(
    df: pd.DataFrame, required_columns: Sequence[str] = REQUIRED_COLUMNS
) -> ValidationReport:
    """Validate the raw dataframe; raise on structural problems, report the rest."""
    if df is None or df.empty:
        raise FinancialCapabilityDataError("Dataset is empty.")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise FinancialCapabilityDataError(f"Missing required columns: {missing}")

    target = pd.to_numeric(df[TARGET_SOURCE], errors="coerce")
    special = {str(code): int((target == code).sum()) for code in TARGET_SPECIAL_CODES}
    valid_target = target[~target.isin(TARGET_SPECIAL_CODES) & target.notna()]
    if valid_target.empty:
        raise FinancialCapabilityDataError("No valid target values found in FWBscore.")

    out_of_codebook: Dict[str, int] = {}
    special_counts: Dict[str, int] = {}
    for spec in FEATURE_SPECS:
        col = pd.to_numeric(df[spec.name], errors="coerce")
        known = set(spec.options) | set(spec.missing_codes) | set(spec.recode or {})
        n_unknown = int((~col.isin(known) & col.notna()).sum())
        if n_unknown:
            out_of_codebook[spec.name] = n_unknown
        n_special = int(col.isin(spec.missing_codes).sum())
        if n_special:
            special_counts[spec.name] = n_special

    report = ValidationReport(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        n_duplicate_rows=int(df.duplicated().sum()),
        n_missing_cells=int(df.isna().sum().sum()),
        target_special_code_counts=special,
        n_valid_target_rows=int(len(valid_target)),
        target_min=float(valid_target.min()),
        target_max=float(valid_target.max()),
        out_of_codebook_values=out_of_codebook,
        refused_or_missing_code_counts=special_counts,
        dtype_counts={str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
    )
    if report.n_duplicate_rows:
        report.warnings.append(f"{report.n_duplicate_rows} duplicate rows found.")
    if out_of_codebook:
        report.warnings.append(f"Values outside the codebook in: {sorted(out_of_codebook)}")
    if report.target_min < 0:
        report.warnings.append("Negative target values remain after removing special codes.")
    absent_items = [c for c in FWB_SCALE_ITEMS if c not in df.columns]
    if absent_items:
        report.warnings.append(f"Score items not in file (cannot be excluded by name): {absent_items}")
    return report
