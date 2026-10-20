# Financial Inclusion Dimension Scoring Policy

## 1. Purpose

This document records the provisional scoring policy for Component 02 of the Financial Inclusion research project. It documents the dimension indicators, score construction rules, missing-data handling, country coverage, and limitations.

The scores are provisional research measures. Valid score ranges do not, by themselves, establish construct validity or justify country rankings.

## 2. Dataset Overview

- Dataset: World Bank Global Findex 2025 individual microdata
- Respondents: 144,090
- Economies: 140
- Respondent identifier: `wpid_random`
- Original source columns are preserved in the dimension dataset.
- No rows are dropped and missing values are not imputed during dimension creation.

## 3. Dimension Scoring Rules

### 3.1 Connectivity

**Components**
- `dim_connect_phone`: Derived from `con1`; code 1 = 1 and code 2 = 0.
- `dim_connect_internet`: Preserves the existing binary `internet_use` indicator.

**Score**

`score_connectivity` is the unweighted mean of phone access and internet use.

Both components are available for all 140 economies. The score has 0% overall missingness.

**Limitation:** Phone access and internet use measure related but distinct aspects of connectivity. Equal weighting is a provisional methodological choice.

### 3.2 Financial Access

**Main indicator:** `dim_access_account`, derived from the existing binary `account` indicator.

**Score**

`score_financial_access` equals `dim_access_account`.

The score has 0% overall missingness and data coverage across all 140 economies.

The separate `dim_access_financial_institution` indicator is retained for comparison but is not added to the score, avoiding automatic double weighting of related account measures.

### 3.3 Digital Financial Usage

**Main indicator:** `dim_usage_any_digital_payment`, derived from `anydigpayment`.

**Score**

`score_digital_financial_usage` equals the available binary digital-payment indicator.

Missing values remain missing; they are not converted to zero.

Overall score missingness is 28.55%. Data are available in 98 of 140 economies, while 42 economies have at least 80% missingness.

**Limitation:** Country comparisons must account for uneven coverage and the conditional applicability of survey questions.

### 3.4 Digital Safety

**Protection indicator:** `dim_safety_phone_protection`, derived from `con18`.

**Separate risk indicator:** `risk_scam_exposure`, derived from `con21`.

`score_digital_safety` currently represents phone security protection only. Scam exposure is kept separate because exposure to scams is not the same construct as having security protection.

Overall missingness for the safety score is 55.38%. Data are available in 75 economies, while 65 economies have at least 80% missingness.

**Limitation:** The current score does not represent every aspect of digital safety. It should not be interpreted as a comprehensive digital-safety measure or compared across all economies without considering coverage.

### 3.5 Financial Resilience

**Components**
- `dim_resilience_saved`: Existing binary saving indicator.
- `dim_resilience_emergency_ease`: Provisional normalized mapping of `fin24a`: code 1 = 0, code 2 = 0.5, code 3 = 1.
- `dim_resilience_income_buffer`: Provisional normalized mapping of `fin24b`: code 1 = 0, code 2 = 1/3, code 3 = 2/3, code 4 = 1.

Special response codes are not included in these normalized components.

**Score**

`score_financial_resilience` is the unweighted mean of all three components. The score is calculated only when all three components are available.

Overall score missingness is 39.15%. Data are available in 98 economies, while 42 economies have at least 80% missingness.

**Limitation:** The mappings assume ordered responses can be represented on a normalized scale. Equal spacing between response categories has not been established and must be treated as a methodological assumption.

## 4. Country Coverage Summary

| Dimension | Economies with data | Economies with at least 80% missingness | Overall score missingness |
|---|---:|---:|---:|
| Connectivity | 140 | 0 | 0.00% |
| Financial Access | 140 | 0 | 0.00% |
| Digital Financial Usage | 98 | 42 | 28.55% |
| Digital Safety | 75 | 65 | 55.38% |
| Financial Resilience | 98 | 42 | 39.15% |

Connectivity and Financial Access are the strongest starting points for broad cross-economy profiling based on current coverage. The other dimensions remain useful for supplementary analyses where data coverage is adequate.

## 5. Response-Code and Data-Quality Decisions

- Source response codes are preserved in the original columns.
- Missing values are not imputed.
- `con13` is not selected as a primary indicator because approximately 98.8% of responses are missing, despite its codebook definition and skip logic being verified.
- `con24` remains unresolved because 298 observations contain codes 3 or 4 that conflict with the currently documented expected codes. These values must not be recoded by assumption.
- Dimension validation confirms that all five scores fall within the expected 0–1 range. This validates the numerical range, not the full conceptual validity of the measures.

## 6. Provisional Analysis Policy

1. Use Connectivity and Financial Access as the core dimensions for broad country-level profiling.
2. Use Digital Financial Usage and Financial Resilience for supplementary analysis with explicit coverage checks.
3. Treat Digital Safety as a separate supplementary protection/risk analysis.
4. Report valid sample sizes and missingness alongside all country-level results.
5. Do not treat missing scores as zero.
6. Do not produce an overall combined index or final country ranking until the weighting policy, construct validity, and comparability limitations have been reviewed.

## 7. Status

**Status: Provisional — subject to methodological review.**

This document records the current implementation and its limitations. It does not constitute final approval of the scoring methodology or the primary analytical model.