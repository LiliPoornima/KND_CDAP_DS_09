"""Loading the CFPB NFWBS 2016 public-use file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .fc_config import DOWNLOAD_URL, RAW_CSV_PATH, SOURCE_URL, get_logger

logger = get_logger(__name__)


class FinancialCapabilityDataError(ValueError):
    """Raised when the dataset is missing, empty or does not match the codebook."""


def load_cfpb_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV exactly as published (no cleaning at this stage).

    Why: loading is kept separate from cleaning so the raw numbers (row count,
    special codes) can be reported honestly in the data-quality step.
    """
    csv_path = Path(path) if path is not None else RAW_CSV_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download NFWBS_PUF_2016_data.csv from "
            f"{SOURCE_URL} (direct link: {DOWNLOAD_URL}) and place it in that folder."
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise FinancialCapabilityDataError(f"Dataset at {csv_path} is empty.")
    logger.info("Loaded %s rows x %s columns from %s", len(df), df.shape[1], csv_path)
    return df

