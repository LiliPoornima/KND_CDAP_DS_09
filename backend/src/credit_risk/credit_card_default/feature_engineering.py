"""Behavioural feature construction for the Taiwan credit-card dataset."""

import numpy as np
import pandas as pd


PAYMENT_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAYMENT_AMOUNT_COLUMNS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
BILL_AMOUNT_COLUMNS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]


def cr_create_credit_card_features(
    df: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Add documented six-month repayment-behaviour summaries.

    Positive ``PAY_*`` codes denote a delayed payment in the dataset's
    documentation. The ratio feature averages only months with a positive
    statement balance; rows without such a month remain missing for
    train-fitted imputation by the experiment preparation function.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if not df.columns.is_unique:
        raise ValueError("DataFrame must have unique column names.")

    required = PAYMENT_STATUS_COLUMNS + PAYMENT_AMOUNT_COLUMNS + BILL_AMOUNT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if strict and missing:
        raise ValueError(f"Missing required credit-history columns: {missing}")

    available_status = [c for c in PAYMENT_STATUS_COLUMNS if c in df.columns]
    available_payments = [c for c in PAYMENT_AMOUNT_COLUMNS if c in df.columns]
    available_bills = [c for c in BILL_AMOUNT_COLUMNS if c in df.columns]
    result = df.copy()

    for column in available_status + available_payments + available_bills:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"Column '{column}' must contain numeric values.")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Column '{column}' must contain finite values.")

    if available_status:
        status = result[available_status]
        result["BehavioralDelinquencyMonths"] = (status > 0).sum(axis=1)
        result["MaximumReportedDelayLevel"] = status.clip(lower=0).max(axis=1)

    if available_payments:
        result["MonthsWithPositivePayment"] = (
            result[available_payments] > 0
        ).sum(axis=1)

    paired_months = [
        (payment, bill)
        for payment, bill in zip(PAYMENT_AMOUNT_COLUMNS, BILL_AMOUNT_COLUMNS)
        if payment in result.columns and bill in result.columns
    ]
    if paired_months:
        monthly_ratios = []
        for payment_column, bill_column in paired_months:
            bill = result[bill_column]
            positive_bill = bill.where(bill > 0)
            monthly_ratios.append(result[payment_column] / positive_bill)
        ratio_frame = pd.concat(monthly_ratios, axis=1)
        result["MeanPositiveBillPaymentRatio"] = ratio_frame.mean(
            axis=1,
            skipna=True,
        )

    return result
