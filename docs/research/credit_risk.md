# Credit Risk Research: Component 1 Completion Plan

## Research scope

The study will evaluate fairness-aware and explainable credit-risk methods using the public datasets available to the project. Its broader motivation is global financial inclusion, but its empirical claims will be limited to the populations represented by the datasets. The datasets do not establish worldwide representativeness, and the Give Me Some Credit competition documentation does not identify a country for its sample.

The three candidate datasets have different target definitions and observation structures. Give Me Some Credit predicts serious delinquency over two years; Default of Credit Card Clients predicts next-month credit-card payment default for Taiwan customers; Home Credit defines its target using payment difficulties on early installments of a sampled loan and supplies related prior-history tables. Each dataset will therefore have its own preprocessing, split, model evaluation, and results. Metrics will not be pooled or interpreted as if the labels and populations were identical.

None of the inspected datasets contains mobile-wallet, telecom, utility-payment, or other direct alternative-data feeds. The planned feature work should be described as repayment-behaviour features or alternative-data-inspired proxies. The available files also do not establish a direct thin-file label, so claims about thin-file borrowers require careful qualification.

## Existing work confirmed

- Give Me Some Credit has raw and processed CSVs, a train/validation/test split, training-fitted imputation, feature engineering, Logistic Regression and Random Forest baselines, model evaluation, and threshold analysis.
- The existing model results and threshold summaries are historical evidence; they are not treated as completed fairness, calibration, explanation, or robustness analysis.
- The default-credit-card and Home Credit package modules now contain dataset-specific loading, preprocessing, feature engineering, training, and experiment workflows. These implementations have not yet produced model results.
- Common calibration and evaluation helpers, fairness metrics, explanation and stability helpers, and a feature-ablation runner are implemented. The `run_inspection.py` entry point remains a placeholder and still needs orchestration or removal from the documented workflow.
- The user's raw dataset archives are outside the repository. They will be read in place; large or licensed raw data will not be copied into Git.

## Dataset audit observations

- **Give Me Some Credit:** the local training file has 150,000 rows and 12 columns, including 10 original predictors, the target, and an exported row index. The target contains 10,026 positive outcomes (6.68%). `MonthlyIncome` has 29,731 missing values and `NumberOfDependents` has 3,924. No sex, gender, race, or ethnicity field is present. Age and dependents are available for limited subgroup analysis, with careful interpretation.
- **Home Credit:** the archive's `application_train.csv` has 307,511 rows and 122 columns, with 282,686 target-zero and 24,825 target-one applications (8.07%). It contains `CODE_GENDER` and `DAYS_BIRTH`; the `XNA` gender category has only four records and cannot support a reliable standalone subgroup conclusion. Sixty-seven columns contain missing values (9,152,465 missing cells overall). The archive also has linked history tables; they require application-time-safe aggregation before model use.
- **Default of Credit Card Clients:** the supplied archive contains the original Excel dataset. UCI documents 30,000 Taiwan cardholders and a next-month default target. It includes sex, age, marital status, education, six months of payment history, statement amounts, and payments. The Python runtime available in this session lacks `xlrd`; `xlrd` has been added to project dependencies for the loader.
- The geographic origin of the Give Me Some Credit sample and the applicants in this Home Credit competition archive is not established by the files inspected. Taiwan is the only confirmed geography among the three sources. These data support multi-dataset experiments, not a claim of worldwide population coverage.

## Execution plan

### 1. Dataset audit and protocol

Record each source, licence or competition terms, observation unit, target definition and horizon, available features, missingness, class balance, possible leakage, and protected/group attributes. Review all three data dictionaries before defining transformations. Confirm whether Home Credit's linked history can be aggregated using information available at the application time.

**Deliverable:** a reproducible audit summary and a dataset-specific inclusion/exclusion note.

### 2. Dataset-specific preparation

Reuse the existing `credit_risk` package. Complete the existing dataset modules rather than duplicating loaders or creating a parallel project structure. Keep target labels and group attributes separate from predictors as appropriate. Fit imputers, encoders, scalers, and any resampling only on training data. Use the established Give Me Some Credit split as the primary benchmark and define independent splits for the secondary datasets.

**Deliverable:** leakage-aware train/validation/test data for each dataset and documented feature definitions.

### 3. Baseline and feature-group experiments

Retain Logistic Regression and Random Forest as baselines. Build only features supported by the dictionaries and available history. Compare conventional and behaviour-derived feature groups with matched splits. Do not describe engineered repayment-history features as genuine mobile, telecom, utility, or wallet data.

**Deliverable:** model comparison and controlled feature-ablation tables.

### 4. Predictive performance and calibration

Select classification thresholds on validation data, then freeze them for test evaluation. Report ranking metrics and threshold-based metrics, plus Brier score and reliability-bin summaries. Treat the Home Credit, Taiwan, and Give Me Some Credit results separately because their labels and populations differ.

**Deliverable:** per-dataset model metrics, threshold decisions, and calibration summaries.

### 5. Fairness audit

Use documented group attributes only where available and suitable. Report group counts, outcome rates, TPR/FPR, and relevant differences with uncertainty where feasible. If a dataset lacks suitable group labels or subgroup support, report that limitation rather than infer protected traits from proxies.

**Deliverable:** dataset-specific subgroup results and an explicit coverage/limitations table.

### 6. Explainability and stability

Generate global and local explanations for selected models. Repeat fits under predefined seeds or resamples and quantify explanation stability with rank similarity and top-k overlap. Keep explanations descriptive and avoid causal interpretations.

**Deliverable:** explanation plots/tables and stability summaries.

### 7. Reproducible reporting

Save outputs under the existing `outputs/credit_risk/` structure. Record seeds, package versions, feature sets, split definitions, and model settings. Update Component 1's proposal/report only after results are actually generated, separating findings from planned analyses.

**Deliverable:** reproducible artifacts and an evidence-aligned final report.

## Implementation progress

- **Started:** expanded this document into the Component 1 plan and recorded the dataset audit findings above.
- **Implemented:** `cr_evaluate_model()` accepts an explicit decision threshold while preserving the model's previous `predict()` policy when omitted. It also reports Brier score, log loss, expected calibration error, and reliability-bin values.
- **Implemented:** the Home Credit loader reads one table from the supplied ZIP without extracting it and can stream large tables in chunks. The labeled application table was inspected in 50,000-row chunks; counts and subgroup support above come from that audit.
- **Implemented:** the credit-card dataset experiment module can read the supplied `.xls` directly or from its ZIP, using the `xlrd` dependency.
- **Implemented:** reusable group fairness metrics, SHAP explanation summaries, and repeated-fit explanation stability comparison are now in their existing credit-risk modules.
- **Implemented:** the Taiwan dataset has a ZIP/Excel reader, canonical target-name handling, a stratified 70/10/20 split, training-only median imputation, audit-group separation, behaviour-summary features, and baseline experiment orchestration.
- **Implemented:** Home Credit has an application-only split, application-ratio feature engineering, a mixed-type train-fitted preprocessing/model pipeline, validation threshold selection, held-out evaluation, and optional test subgroup reporting.
- **Implemented and inspected on the archive:** 75 applicant-level history features are now aggregated in chunks from bureau, bureau-balance, prior-application, installment, POS-cash, and credit-card tables. Relative dates exclude future information. The aggregate table has 353,895 unique IDs; 305,311 of the 307,511 labeled applicants match it, leaving 2,200 without matched history. The inspected aggregate had no missing or non-finite values, and the constructed rate features ranged from 0 to 1.
- **Implemented:** a reusable feature-group ablation runner is available for the numeric baseline models.
- **Completed in this stage:** created a workspace-local `.venv`, installed `requirements.txt`, corrected the Taiwan workbook's embedded-header handling, and ran stratified train/validation/test baselines for all three datasets. The original historical Give Me Some Credit files were preserved; new artifacts are under each dataset's `completion_run/` output folder.
- **Preliminary held-out results:**

  | Dataset | Model | Validation-selected threshold | Test F1 | Test ROC-AUC | Test PR-AUC (average precision) | Test Brier |
  |---|---|---:|---:|---:|---:|---:|
  | Give Me Some Credit | Logistic Regression | 0.50 | 0.2834 | 0.8007 | 0.3210 | 0.1808 |
  | Give Me Some Credit | Random Forest | 0.40 | 0.4102 | 0.8439 | 0.3462 | 0.0594 |
  | Taiwan card default | Logistic Regression | 0.50 | 0.5072 | 0.7507 | 0.5061 | 0.1922 |
  | Taiwan card default | Random Forest | 0.45 | 0.5316 | 0.7568 | 0.5187 | 0.1541 |
  | Home Credit | Logistic Regression | 0.50 | 0.2766 | 0.7695 | 0.2578 | 0.1933 |
  | Home Credit | Random Forest | 0.25 | 0.3081 | 0.7614 | 0.2371 | 0.0750 |

  These results come from independent splits and different target definitions; they are not pooled or used to claim that one population is intrinsically riskier. Thresholds maximize validation F1 over the configured grid and are research choices, not lending-policy recommendations. The Home Credit Logistic Regression was changed to the `newton-cholesky` solver after the `lbfgs` fit failed to converge at 3,000 iterations; the final fit converged in five iterations at the same split.
- **Completed in this stage:** calibration-bin and subgroup fairness CSVs were saved for each dataset. A 1,000-replicate within-group bootstrap now reports 95% percentile intervals for group rates and max-minus-min fairness gaps. For the Home Credit Random Forest, the age-bin selection-rate gap was 0.442 (95% CI 0.421–0.463) and the age-bin TPR gap was 0.676 (0.579–0.753). The gender selection-rate gap was 0.057 (0.051–0.064). These intervals condition on the fitted model, selected threshold, and observed group sizes; they do not include model-refit uncertainty.
- **Completed in this stage:** Give Me Some Credit Random Forest ablation at the fixed validation-selected threshold 0.40 yielded test F1 0.4102 for all features and 0.4074 without the two engineered delinquency aggregates. This small difference is descriptive; it is not evidence of a causal effect.
- **Completed in this stage:** SHAP global importance and sampled local attributions were saved for all three Random Forests. Three-seed explanation stability was completed for Give Me Some Credit (mean rank correlation 0.997, top-10 Jaccard 1.000, attribution cosine similarity 0.989) and Taiwan (0.918, 0.818, and 0.979). Home Credit's three-seed explanation stability is now completed and summarized below.
- **Completed in this stage:** Taiwan Random Forest feature-group ablation compared the full 23-feature set with repayment-status history removed (test F1 0.4547; ROC-AUC 0.7082), monthly amounts removed (F1 0.4573; ROC-AUC 0.6987), and two derived payment summaries removed (F1 0.5272; ROC-AUC 0.7550). The full model scored F1 0.5316 and ROC-AUC 0.7568. Each scenario selected its threshold on validation data; test results are descriptive, not causal.
- **Completed in this stage:** Home Credit Random Forest ablation compared all 199 predictors with the 75 aggregated prior-history features removed. Test F1 decreased from 0.3081 to 0.2910, ROC-AUC from 0.7614 to 0.7450, and PR-AUC from 0.2371 to 0.2224; both scenarios selected threshold 0.25 on validation data. The result describes this split and model only.
- **Completed in this stage:** Home Credit three-seed Random Forest explanation stability used the same 100 held-out rows and background sample for each fit. Across three pairwise comparisons, mean feature-rank correlation was 0.9909, top-10 Jaccard overlap 0.7677, and signed-attribution cosine similarity 0.9317 across 391 transformed features. Explanations are descriptive, not causal.
- **Implemented:** the Component 1 Streamlit page at `frontend/pages/1_Credit_Risk.py` displays saved model metrics, calibration, subgroup metrics and bootstrap intervals, global SHAP importance, explanation stability, and feature ablations for each dataset. It reads summary outputs only and does not load the large prepared split CSVs. The pushed app navigation already registers this page as **Credit Risk**; shared app and sidebar files were left unchanged.
- **Not yet completed:** performance uncertainty intervals, the subgroup-support coverage review, run manifests with package versions, and the evidence-based final research report. No test suite was run; the actual analysis runner completed successfully and `git diff --check` reported no whitespace errors.

## Ownership and change boundary

Implementation changes are limited to the user's `backend/src/credit_risk/` package, its credit-risk tests when test coverage is requested or needed, credit-risk data/output folders, and credit-risk research documentation. Do not modify other members' domain packages or files. Keep raw dataset archives outside Git unless the user confirms redistribution rights and a specific need to add data.

## Next implementation items

1. Add performance uncertainty intervals and finish the subgroup-support coverage review.
2. Record package versions, seeds, and source/split details in run manifests, then prepare the final Component 1 report from generated evidence.

The Taiwan and Home Credit ablation/stability stage can be reproduced from the saved split files with `python -m backend.src.credit_risk.experiments.run_inspection --run-extended-analysis --output-root outputs/credit_risk`. It writes the ablation tables under each dataset's `completion_run/metrics/` directory and Home Credit stability tables under `completion_run/explainability/`. The Streamlit results page reads the saved outputs and is available from the app's Credit Risk navigation entry.
