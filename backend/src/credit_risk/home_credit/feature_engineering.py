"""Application-level features for the Home Credit research dataset."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.src.credit_risk.home_credit.data_loader import (
    cr_load_home_credit_table,
)


def cr_create_home_credit_features(
    df: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Create ratios and history summaries from documented application fields.

    ``DAYS_EMPLOYED == 365243`` is the dataset's sentinel for an unavailable
    employment duration, so its derived value is set to missing and later
    handled by the train-fitted preprocessing step. ``AGE_YEARS`` is created
    for subgroup auditing and is excluded from model predictors by default.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if not df.columns.is_unique:
        raise ValueError("DataFrame must have unique column names.")

    result = df.copy()

    if "DAYS_BIRTH" in result.columns:
        result["AGE_YEARS"] = -pd.to_numeric(
            result["DAYS_BIRTH"],
            errors="coerce",
        ) / 365.25
    elif strict:
        raise ValueError("Missing required Home Credit column: DAYS_BIRTH")

    if "DAYS_EMPLOYED" in result.columns:
        result["DAYS_EMPLOYED"] = pd.to_numeric(
            result["DAYS_EMPLOYED"],
            errors="coerce",
        ).replace(365243, np.nan)
        result["YEARS_EMPLOYED"] = -result["DAYS_EMPLOYED"] / 365.25
    elif strict:
        raise ValueError("Missing required Home Credit column: DAYS_EMPLOYED")

    ratio_definitions = {
        "CREDIT_TO_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "ANNUITY_TO_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "CREDIT_TO_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
        "ANNUITY_TO_CREDIT_RATIO": ("AMT_ANNUITY", "AMT_CREDIT"),
    }
    for feature, (numerator, denominator) in ratio_definitions.items():
        present = numerator in result.columns and denominator in result.columns
        if not present:
            if strict:
                missing = [
                    column
                    for column in (numerator, denominator)
                    if column not in result.columns
                ]
                raise ValueError(f"Missing required Home Credit columns: {missing}")
            continue

        numerator_values = pd.to_numeric(result[numerator], errors="coerce")
        denominator_values = pd.to_numeric(result[denominator], errors="coerce")
        valid_denominator = denominator_values.where(denominator_values > 0)
        result[feature] = numerator_values / valid_denominator

    external_score_columns = [
        column
        for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if column in result.columns
    ]
    if external_score_columns:
        result["MEAN_EXTERNAL_SCORE"] = result[
            external_score_columns
        ].mean(axis=1, skipna=True)
    elif strict:
        raise ValueError("No documented EXT_SOURCE score columns were found.")

    return result


def _combine_grouped_sums(
    current: pd.DataFrame | None,
    partial: pd.DataFrame,
) -> pd.DataFrame:
    return partial if current is None else current.add(partial, fill_value=0)


def _combine_grouped_maxima(
    current: pd.DataFrame | None,
    partial: pd.DataFrame,
) -> pd.DataFrame:
    return partial if current is None else current.combine(partial, np.fmax)


def _aggregate_home_credit_table(
    archive_path: str | Path,
    table_name: str,
    usecols: Sequence[str],
    prepare_chunk: Callable[[pd.DataFrame], pd.DataFrame],
    sum_columns: Sequence[str],
    max_columns: Sequence[str] = (),
    *,
    chunksize: int,
) -> pd.DataFrame:
    """Aggregate a table by applicant while keeping only per-applicant totals."""

    sums: pd.DataFrame | None = None
    maxima: pd.DataFrame | None = None
    for raw_chunk in cr_load_home_credit_table(
        archive_path,
        table_name=table_name,
        chunksize=chunksize,
        usecols=usecols,
    ):
        features = prepare_chunk(raw_chunk)
        if features.empty:
            continue
        grouped = features.groupby("SK_ID_CURR", sort=False)
        if sum_columns:
            partial_sums = grouped[list(sum_columns)].sum()
            sums = _combine_grouped_sums(sums, partial_sums)
        if max_columns:
            partial_maxima = grouped[list(max_columns)].max()
            maxima = _combine_grouped_maxima(maxima, partial_maxima)

    if sums is None and maxima is None:
        columns = list(sum_columns) + list(max_columns)
        return pd.DataFrame(
            columns=columns,
            index=pd.Index([], name="SK_ID_CURR"),
        )
    if sums is None:
        result = maxima
    elif maxima is None:
        result = sums
    else:
        result = sums.join(maxima, how="outer")
    result.index.name = "SK_ID_CURR"
    return result


def cr_aggregate_home_credit_histories(
    archive_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build application-time-safe aggregates from linked history tables.

    Relative-time fields are filtered to information available by the current
    application date: previous decisions and bureau credits must be strictly
    earlier than the application; monthly balances and installments must be
    dated at or before it. Installment payments recorded after the application
    are excluded. Only per-applicant aggregates are retained in memory.
    """

    if (
        not isinstance(chunksize, int)
        or isinstance(chunksize, bool)
        or chunksize < 1
    ):
        raise ValueError("chunksize must be a positive integer.")

    bureau_mapping_parts: list[pd.DataFrame] = []

    def prepare_bureau(chunk: pd.DataFrame) -> pd.DataFrame:
        days_credit = pd.to_numeric(chunk["DAYS_CREDIT"], errors="coerce")
        days_update = pd.to_numeric(chunk["DAYS_CREDIT_UPDATE"], errors="coerce")
        valid = chunk.loc[
            days_credit.lt(0) & days_update.le(0),
            [
                "SK_ID_CURR",
                "SK_ID_BUREAU",
                "DAYS_CREDIT",
                "CREDIT_ACTIVE",
                "CREDIT_DAY_OVERDUE",
                "CNT_CREDIT_PROLONG",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE",
            ],
        ].copy()
        if valid.empty:
            return valid.assign(
                BUREAU_PRIOR_CREDIT_COUNT=pd.Series(dtype=float)
            )
        bureau_mapping_parts.append(
            valid[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
        )
        overdue_days = pd.to_numeric(
            valid["CREDIT_DAY_OVERDUE"], errors="coerce"
        ).fillna(0).clip(lower=0)
        valid["BUREAU_PRIOR_CREDIT_COUNT"] = 1
        valid["BUREAU_ACTIVE_CREDIT_COUNT"] = valid["CREDIT_ACTIVE"].eq("Active").astype(int)
        valid["BUREAU_CLOSED_CREDIT_COUNT"] = valid["CREDIT_ACTIVE"].eq("Closed").astype(int)
        valid["BUREAU_OVERDUE_CREDIT_COUNT"] = overdue_days.gt(0).astype(int)
        valid["BUREAU_OVERDUE_DAYS_SUM"] = overdue_days
        valid["BUREAU_OVERDUE_DAYS_MAX"] = overdue_days
        valid["BUREAU_CREDIT_AGE_DAYS_SUM"] = -days_credit.loc[valid.index]
        valid["BUREAU_CREDIT_AMOUNT_SUM"] = pd.to_numeric(
            valid["AMT_CREDIT_SUM"], errors="coerce"
        ).fillna(0)
        valid["BUREAU_DEBT_AMOUNT_SUM"] = pd.to_numeric(
            valid["AMT_CREDIT_SUM_DEBT"], errors="coerce"
        ).fillna(0)
        valid["BUREAU_OVERDUE_AMOUNT_SUM"] = pd.to_numeric(
            valid["AMT_CREDIT_SUM_OVERDUE"], errors="coerce"
        ).fillna(0)
        valid["BUREAU_PROLONGED_CREDIT_COUNT"] = pd.to_numeric(
            valid["CNT_CREDIT_PROLONG"], errors="coerce"
        ).fillna(0).gt(0).astype(int)
        return valid

    bureau_features = _aggregate_home_credit_table(
        archive_path,
        "bureau.csv",
        [
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "DAYS_CREDIT",
            "DAYS_CREDIT_UPDATE",
            "CREDIT_ACTIVE",
            "CREDIT_DAY_OVERDUE",
            "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
        ],
        prepare_bureau,
        [
            "BUREAU_PRIOR_CREDIT_COUNT",
            "BUREAU_ACTIVE_CREDIT_COUNT",
            "BUREAU_CLOSED_CREDIT_COUNT",
            "BUREAU_OVERDUE_CREDIT_COUNT",
            "BUREAU_OVERDUE_DAYS_SUM",
            "BUREAU_CREDIT_AGE_DAYS_SUM",
            "BUREAU_CREDIT_AMOUNT_SUM",
            "BUREAU_DEBT_AMOUNT_SUM",
            "BUREAU_OVERDUE_AMOUNT_SUM",
            "BUREAU_PROLONGED_CREDIT_COUNT",
        ],
        ["BUREAU_OVERDUE_DAYS_MAX"],
        chunksize=chunksize,
    )
    if bureau_mapping_parts:
        bureau_mapping = (
            pd.concat(bureau_mapping_parts, ignore_index=True)
            .drop_duplicates("SK_ID_BUREAU")
            .set_index("SK_ID_BUREAU")["SK_ID_CURR"]
        )
    else:
        bureau_mapping = pd.Series(dtype="int64", name="SK_ID_CURR")

    def prepare_bureau_balance(chunk: pd.DataFrame) -> pd.DataFrame:
        months = pd.to_numeric(chunk["MONTHS_BALANCE"], errors="coerce")
        valid = chunk.loc[months.le(0)].copy()
        if valid.empty:
            return valid.assign(BUREAU_BALANCE_MONTH_COUNT=pd.Series(dtype=float))
        valid["SK_ID_CURR"] = valid["SK_ID_BUREAU"].map(bureau_mapping)
        valid = valid.loc[valid["SK_ID_CURR"].notna()].copy()
        if valid.empty:
            return valid.assign(BUREAU_BALANCE_MONTH_COUNT=pd.Series(dtype=float))
        status = valid["STATUS"].astype("string")
        valid["BUREAU_BALANCE_MONTH_COUNT"] = 1
        valid["BUREAU_BALANCE_DPD_MONTH_COUNT"] = status.isin(
            ["1", "2", "3", "4", "5"]
        ).astype(int)
        valid["BUREAU_BALANCE_SEVERE_DPD_MONTH_COUNT"] = status.isin(
            ["3", "4", "5"]
        ).astype(int)
        valid["BUREAU_BALANCE_STATUS_5_COUNT"] = status.eq("5").astype(int)
        valid["BUREAU_BALANCE_CLOSED_MONTH_COUNT"] = status.eq("C").astype(int)
        valid["BUREAU_BALANCE_UNKNOWN_MONTH_COUNT"] = status.eq("X").astype(int)
        valid["BUREAU_BALANCE_MONTH_AGE_SUM"] = -months.loc[valid.index]
        return valid

    bureau_balance_features = _aggregate_home_credit_table(
        archive_path,
        "bureau_balance.csv",
        ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        prepare_bureau_balance,
        [
            "BUREAU_BALANCE_MONTH_COUNT",
            "BUREAU_BALANCE_DPD_MONTH_COUNT",
            "BUREAU_BALANCE_SEVERE_DPD_MONTH_COUNT",
            "BUREAU_BALANCE_STATUS_5_COUNT",
            "BUREAU_BALANCE_CLOSED_MONTH_COUNT",
            "BUREAU_BALANCE_UNKNOWN_MONTH_COUNT",
            "BUREAU_BALANCE_MONTH_AGE_SUM",
        ],
        chunksize=chunksize,
    )

    def prepare_previous_application(chunk: pd.DataFrame) -> pd.DataFrame:
        decision_days = pd.to_numeric(chunk["DAYS_DECISION"], errors="coerce")
        valid = chunk.loc[decision_days.lt(0)].copy()
        if valid.empty:
            return valid.assign(PREVIOUS_APPLICATION_COUNT=pd.Series(dtype=float))
        valid["PREVIOUS_APPLICATION_COUNT"] = 1
        valid["PREVIOUS_APPROVED_COUNT"] = valid["NAME_CONTRACT_STATUS"].eq("Approved").astype(int)
        valid["PREVIOUS_REFUSED_COUNT"] = valid["NAME_CONTRACT_STATUS"].eq("Refused").astype(int)
        valid["PREVIOUS_CANCELED_COUNT"] = valid["NAME_CONTRACT_STATUS"].eq("Canceled").astype(int)
        valid["PREVIOUS_DECISION_AGE_DAYS_SUM"] = -decision_days.loc[valid.index]
        for source, feature in (
            ("AMT_APPLICATION", "PREVIOUS_APPLICATION_AMOUNT_SUM"),
            ("AMT_CREDIT", "PREVIOUS_CREDIT_AMOUNT_SUM"),
        ):
            valid[feature] = pd.to_numeric(valid[source], errors="coerce").fillna(0)
        return valid

    previous_features = _aggregate_home_credit_table(
        archive_path,
        "previous_application.csv",
        [
            "SK_ID_CURR",
            "DAYS_DECISION",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
        ],
        prepare_previous_application,
        [
            "PREVIOUS_APPLICATION_COUNT",
            "PREVIOUS_APPROVED_COUNT",
            "PREVIOUS_REFUSED_COUNT",
            "PREVIOUS_CANCELED_COUNT",
            "PREVIOUS_DECISION_AGE_DAYS_SUM",
            "PREVIOUS_APPLICATION_AMOUNT_SUM",
            "PREVIOUS_CREDIT_AMOUNT_SUM",
        ],
        chunksize=chunksize,
    )

    def prepare_installments(chunk: pd.DataFrame) -> pd.DataFrame:
        due_days = pd.to_numeric(chunk["DAYS_INSTALMENT"], errors="coerce")
        entry_days = pd.to_numeric(chunk["DAYS_ENTRY_PAYMENT"], errors="coerce")
        known_by_application = due_days.le(0) & (entry_days.isna() | entry_days.le(0))
        valid = chunk.loc[known_by_application].copy()
        if valid.empty:
            return valid.assign(INSTALLMENT_DUE_COUNT=pd.Series(dtype=float))
        due = due_days.loc[valid.index]
        entry = entry_days.loc[valid.index]
        delay_days = (entry - due).where(entry.notna(), 0).clip(lower=0)
        valid["INSTALLMENT_DUE_COUNT"] = 1
        valid["INSTALLMENT_LATE_COUNT"] = delay_days.gt(0).astype(int)
        valid["INSTALLMENT_UNPAID_COUNT"] = entry.isna().astype(int)
        valid["INSTALLMENT_LATE_DAYS_SUM"] = delay_days
        valid["INSTALLMENT_LATE_DAYS_MAX"] = delay_days
        valid["INSTALLMENT_PAYMENT_AMOUNT_SUM"] = pd.to_numeric(
            valid["AMT_PAYMENT"], errors="coerce"
        ).where(entry.notna(), 0).fillna(0)
        valid["INSTALLMENT_AMOUNT_SUM"] = pd.to_numeric(
            valid["AMT_INSTALMENT"], errors="coerce"
        ).fillna(0)
        return valid

    installment_features = _aggregate_home_credit_table(
        archive_path,
        "installments_payments.csv",
        [
            "SK_ID_CURR",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
        prepare_installments,
        [
            "INSTALLMENT_DUE_COUNT",
            "INSTALLMENT_LATE_COUNT",
            "INSTALLMENT_UNPAID_COUNT",
            "INSTALLMENT_LATE_DAYS_SUM",
            "INSTALLMENT_PAYMENT_AMOUNT_SUM",
            "INSTALLMENT_AMOUNT_SUM",
        ],
        ["INSTALLMENT_LATE_DAYS_MAX"],
        chunksize=chunksize,
    )

    def prepare_pos_cash(chunk: pd.DataFrame) -> pd.DataFrame:
        months = pd.to_numeric(chunk["MONTHS_BALANCE"], errors="coerce")
        valid = chunk.loc[months.le(0)].copy()
        if valid.empty:
            return valid.assign(POS_CASH_MONTH_COUNT=pd.Series(dtype=float))
        valid["POS_CASH_MONTH_COUNT"] = 1
        valid["POS_CASH_DPD_SUM"] = pd.to_numeric(
            valid["SK_DPD"], errors="coerce"
        ).fillna(0).clip(lower=0)
        valid["POS_CASH_DPD_MAX"] = valid["POS_CASH_DPD_SUM"]
        valid["POS_CASH_DPD_DEF_SUM"] = pd.to_numeric(
            valid["SK_DPD_DEF"], errors="coerce"
        ).fillna(0).clip(lower=0)
        valid["POS_CASH_DPD_DEF_MAX"] = valid["POS_CASH_DPD_DEF_SUM"]
        valid["POS_CASH_ACTIVE_MONTH_COUNT"] = valid[
            "NAME_CONTRACT_STATUS"
        ].eq("Active").astype(int)
        valid["POS_CASH_COMPLETED_MONTH_COUNT"] = valid[
            "NAME_CONTRACT_STATUS"
        ].eq("Completed").astype(int)
        valid["POS_CASH_MONTH_AGE_SUM"] = -months.loc[valid.index]
        valid["POS_CASH_FUTURE_INSTALLMENTS_SUM"] = pd.to_numeric(
            valid["CNT_INSTALMENT_FUTURE"], errors="coerce"
        ).fillna(0)
        return valid

    pos_features = _aggregate_home_credit_table(
        archive_path,
        "POS_CASH_balance.csv",
        [
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "NAME_CONTRACT_STATUS",
            "SK_DPD",
            "SK_DPD_DEF",
            "CNT_INSTALMENT_FUTURE",
        ],
        prepare_pos_cash,
        [
            "POS_CASH_MONTH_COUNT",
            "POS_CASH_DPD_SUM",
            "POS_CASH_DPD_DEF_SUM",
            "POS_CASH_ACTIVE_MONTH_COUNT",
            "POS_CASH_COMPLETED_MONTH_COUNT",
            "POS_CASH_MONTH_AGE_SUM",
            "POS_CASH_FUTURE_INSTALLMENTS_SUM",
        ],
        ["POS_CASH_DPD_MAX", "POS_CASH_DPD_DEF_MAX"],
        chunksize=chunksize,
    )

    def prepare_credit_card(chunk: pd.DataFrame) -> pd.DataFrame:
        months = pd.to_numeric(chunk["MONTHS_BALANCE"], errors="coerce")
        valid = chunk.loc[months.le(0)].copy()
        if valid.empty:
            return valid.assign(CREDIT_CARD_MONTH_COUNT=pd.Series(dtype=float))
        valid["CREDIT_CARD_MONTH_COUNT"] = 1
        valid["CREDIT_CARD_DPD_SUM"] = pd.to_numeric(
            valid["SK_DPD"], errors="coerce"
        ).fillna(0).clip(lower=0)
        valid["CREDIT_CARD_DPD_MAX"] = valid["CREDIT_CARD_DPD_SUM"]
        valid["CREDIT_CARD_DPD_DEF_SUM"] = pd.to_numeric(
            valid["SK_DPD_DEF"], errors="coerce"
        ).fillna(0).clip(lower=0)
        valid["CREDIT_CARD_DPD_DEF_MAX"] = valid["CREDIT_CARD_DPD_DEF_SUM"]
        for source, feature in (
            ("AMT_BALANCE", "CREDIT_CARD_BALANCE_SUM"),
            ("AMT_CREDIT_LIMIT_ACTUAL", "CREDIT_CARD_LIMIT_SUM"),
            ("AMT_PAYMENT_TOTAL_CURRENT", "CREDIT_CARD_PAYMENT_SUM"),
            ("AMT_DRAWINGS_CURRENT", "CREDIT_CARD_DRAWINGS_SUM"),
        ):
            valid[feature] = pd.to_numeric(valid[source], errors="coerce").fillna(0)
        valid["CREDIT_CARD_MONTH_AGE_SUM"] = -months.loc[valid.index]
        return valid

    credit_card_features = _aggregate_home_credit_table(
        archive_path,
        "credit_card_balance.csv",
        [
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "AMT_DRAWINGS_CURRENT",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
        prepare_credit_card,
        [
            "CREDIT_CARD_MONTH_COUNT",
            "CREDIT_CARD_DPD_SUM",
            "CREDIT_CARD_DPD_DEF_SUM",
            "CREDIT_CARD_BALANCE_SUM",
            "CREDIT_CARD_LIMIT_SUM",
            "CREDIT_CARD_PAYMENT_SUM",
            "CREDIT_CARD_DRAWINGS_SUM",
            "CREDIT_CARD_MONTH_AGE_SUM",
        ],
        ["CREDIT_CARD_DPD_MAX", "CREDIT_CARD_DPD_DEF_MAX"],
        chunksize=chunksize,
    )

    table_summaries = [
        bureau_features,
        bureau_balance_features,
        previous_features,
        installment_features,
        pos_features,
        credit_card_features,
    ]
    combined = pd.concat(table_summaries, axis=1, join="outer").fillna(0)

    def safe_ratio(
        numerator: str,
        denominator: str,
        output_name: str,
    ) -> None:
        combined[output_name] = (
            combined[numerator]
            / combined[denominator].where(combined[denominator] > 0)
        ).fillna(0)

    safe_ratio(
        "BUREAU_OVERDUE_CREDIT_COUNT",
        "BUREAU_PRIOR_CREDIT_COUNT",
        "BUREAU_OVERDUE_CREDIT_RATE",
    )
    safe_ratio(
        "BUREAU_DEBT_AMOUNT_SUM",
        "BUREAU_CREDIT_AMOUNT_SUM",
        "BUREAU_DEBT_TO_CREDIT_RATIO",
    )
    safe_ratio(
        "BUREAU_CREDIT_AGE_DAYS_SUM",
        "BUREAU_PRIOR_CREDIT_COUNT",
        "BUREAU_MEAN_CREDIT_AGE_DAYS",
    )
    safe_ratio(
        "BUREAU_BALANCE_DPD_MONTH_COUNT",
        "BUREAU_BALANCE_MONTH_COUNT",
        "BUREAU_BALANCE_DPD_MONTH_RATE",
    )
    safe_ratio(
        "BUREAU_BALANCE_SEVERE_DPD_MONTH_COUNT",
        "BUREAU_BALANCE_MONTH_COUNT",
        "BUREAU_BALANCE_SEVERE_DPD_MONTH_RATE",
    )
    safe_ratio(
        "BUREAU_BALANCE_MONTH_AGE_SUM",
        "BUREAU_BALANCE_MONTH_COUNT",
        "BUREAU_BALANCE_MEAN_MONTH_AGE",
    )
    safe_ratio(
        "PREVIOUS_APPROVED_COUNT",
        "PREVIOUS_APPLICATION_COUNT",
        "PREVIOUS_APPROVAL_RATE",
    )
    safe_ratio(
        "PREVIOUS_REFUSED_COUNT",
        "PREVIOUS_APPLICATION_COUNT",
        "PREVIOUS_REFUSAL_RATE",
    )
    safe_ratio(
        "PREVIOUS_CANCELED_COUNT",
        "PREVIOUS_APPLICATION_COUNT",
        "PREVIOUS_CANCELLATION_RATE",
    )
    safe_ratio(
        "PREVIOUS_DECISION_AGE_DAYS_SUM",
        "PREVIOUS_APPLICATION_COUNT",
        "PREVIOUS_MEAN_DECISION_AGE_DAYS",
    )
    safe_ratio(
        "PREVIOUS_CREDIT_AMOUNT_SUM",
        "PREVIOUS_APPLICATION_AMOUNT_SUM",
        "PREVIOUS_CREDIT_TO_APPLICATION_AMOUNT_RATIO",
    )
    safe_ratio(
        "INSTALLMENT_LATE_COUNT",
        "INSTALLMENT_DUE_COUNT",
        "INSTALLMENT_LATE_RATE",
    )
    safe_ratio(
        "INSTALLMENT_UNPAID_COUNT",
        "INSTALLMENT_DUE_COUNT",
        "INSTALLMENT_UNPAID_RATE",
    )
    safe_ratio(
        "INSTALLMENT_LATE_DAYS_SUM",
        "INSTALLMENT_DUE_COUNT",
        "INSTALLMENT_MEAN_LATE_DAYS",
    )
    safe_ratio(
        "INSTALLMENT_PAYMENT_AMOUNT_SUM",
        "INSTALLMENT_AMOUNT_SUM",
        "INSTALLMENT_PAYMENT_COVERAGE_RATIO",
    )
    safe_ratio(
        "POS_CASH_DPD_SUM",
        "POS_CASH_MONTH_COUNT",
        "POS_CASH_MEAN_DPD",
    )
    safe_ratio(
        "POS_CASH_DPD_DEF_SUM",
        "POS_CASH_MONTH_COUNT",
        "POS_CASH_MEAN_DPD_DEF",
    )
    safe_ratio(
        "POS_CASH_MONTH_AGE_SUM",
        "POS_CASH_MONTH_COUNT",
        "POS_CASH_MEAN_MONTH_AGE",
    )
    safe_ratio(
        "POS_CASH_ACTIVE_MONTH_COUNT",
        "POS_CASH_MONTH_COUNT",
        "POS_CASH_ACTIVE_MONTH_RATE",
    )
    safe_ratio(
        "POS_CASH_COMPLETED_MONTH_COUNT",
        "POS_CASH_MONTH_COUNT",
        "POS_CASH_COMPLETED_MONTH_RATE",
    )
    safe_ratio(
        "CREDIT_CARD_DPD_SUM",
        "CREDIT_CARD_MONTH_COUNT",
        "CREDIT_CARD_MEAN_DPD",
    )
    safe_ratio(
        "CREDIT_CARD_DPD_DEF_SUM",
        "CREDIT_CARD_MONTH_COUNT",
        "CREDIT_CARD_MEAN_DPD_DEF",
    )
    safe_ratio(
        "CREDIT_CARD_BALANCE_SUM",
        "CREDIT_CARD_LIMIT_SUM",
        "CREDIT_CARD_UTILIZATION_RATIO",
    )
    safe_ratio(
        "CREDIT_CARD_MONTH_AGE_SUM",
        "CREDIT_CARD_MONTH_COUNT",
        "CREDIT_CARD_MEAN_MONTH_AGE",
    )
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    combined.index.name = "SK_ID_CURR"
    if not combined.columns.is_unique:
        raise ValueError("Home Credit history feature names must be unique.")
    return combined.reset_index()
