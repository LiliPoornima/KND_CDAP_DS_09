"""Load individual tables from the Home Credit competition archive."""

from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd


HOME_CREDIT_TABLES = frozenset(
    {
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv",
        "sample_submission.csv",
        "HomeCredit_columns_description.csv",
    }
)


def cr_load_home_credit_table(
    archive_path: str | Path,
    table_name: str = "application_train.csv",
    chunksize: int | None = None,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read one named CSV from the Home Credit ZIP without extracting it.

    ``chunksize`` can be supplied for the large history tables so callers
    can aggregate records without loading the entire table into memory.
    The archive is never modified and only the selected table is opened.
    """

    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"Home Credit archive not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(
            f"Expected a ZIP archive, but received a directory: {path}"
        )
    if table_name not in HOME_CREDIT_TABLES:
        raise ValueError(
            f"Unsupported Home Credit table '{table_name}'. "
            f"Choose one of: {sorted(HOME_CREDIT_TABLES)}"
        )
    if chunksize is not None and (
        not isinstance(chunksize, int)
        or isinstance(chunksize, bool)
        or chunksize < 1
    ):
        raise ValueError("chunksize must be a positive integer or None.")
    if usecols is not None and (
        not usecols
        or any(not isinstance(column, str) or not column for column in usecols)
        or len(usecols) != len(set(usecols))
    ):
        raise ValueError("usecols must be None or unique non-empty column names.")

    try:
        with ZipFile(path) as archive:
            matches = [
                item.filename
                for item in archive.infolist()
                if PurePosixPath(item.filename).name == table_name
                and not item.is_dir()
            ]
            if not matches:
                raise FileNotFoundError(
                    f"'{table_name}' was not found in {path}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Archive contains more than one '{table_name}' table."
                )

            member_name = matches[0]
            if chunksize is not None:
                return _iter_home_credit_csv(
                    path,
                    member_name,
                    chunksize,
                    usecols,
                )

            with archive.open(member_name) as csv_stream:
                try:
                    result = pd.read_csv(csv_stream, usecols=usecols)
                except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
                    raise ValueError(
                        f"Unable to read '{table_name}' from {path}: {exc}"
                    ) from exc
    except BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {path}") from exc

    if result.empty:
        raise ValueError(f"'{table_name}' contains no data rows.")

    return result


def _iter_home_credit_csv(
    archive_path: Path,
    member_name: str,
    chunksize: int,
    usecols: Sequence[str] | None,
) -> Iterator[pd.DataFrame]:
    """Yield chunks while keeping the ZIP and member streams open."""

    try:
        with ZipFile(archive_path) as archive:
            with archive.open(member_name) as csv_stream:
                try:
                    yield from pd.read_csv(
                        csv_stream,
                        chunksize=chunksize,
                        usecols=usecols,
                    )
                except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
                    raise ValueError(
                        f"Unable to read '{member_name}' from "
                        f"{archive_path}: {exc}"
                    ) from exc
    except BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {archive_path}") from exc
