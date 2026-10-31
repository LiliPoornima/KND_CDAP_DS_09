# Component 1: Credit Risk Research Report

**Status:** Completed analysis of the three supplied public datasets  
**Branch:** `feature/c1-data-loader`  
**Interpretation:** Dataset-specific research evidence; not a production lending model or a worldwide validation study.

## Executive summary

This study evaluates Logistic Regression and Random Forest baselines on three datasets with different populations and default definitions: Give Me Some Credit, Taiwan Default of Credit Card Clients, and Home Credit Default Risk. The datasets were split and evaluated separately. Thresholds were chosen on validation data and frozen for the held-out test results.

Random Forest had higher test F1 and ROC-AUC on Give Me Some Credit and Taiwan. On Home Credit, Random Forest had higher F1 and lower Brier score and expected calibration error, while Logistic Regression had higher ROC-AUC and PR-AUC. These differences are descriptive for these splits; the confidence intervals below are not paired tests of model differences.

The public datasets do not contain genuine mobile-wallet, telecom, or utility-payment records, and they do not identify thin-file borrowers. Engineered repayment-history features are proxies for behavior available in these datasets. Taiwan is the only geography established by the supplied dataset documentation. The findings therefore do not establish global or thin-file performance.

## Research questions and findings

### RQ1 — Do behavior-derived features add predictive value?

Feature ablations show dataset-specific patterns. Removing Home Credit's 75 aggregated prior-history predictors reduced Random Forest F1 from 0.3081 to 0.2910 and ROC-AUC from 0.7614 to 0.7450. Removing Taiwan repayment-status history or monthly amount history also reduced F1 and ROC-AUC. In Give Me Some Credit, removing the two engineered delinquency aggregates caused only a small F1 decrease (0.4102 to 0.4074). These comparisons are descriptive: feature removals can change the fitted model, and they do not identify causal effects.

### RQ2 — Do error rates and selection rates vary across observed groups?

The supported-group audit found differences across age, dependents, gender, sex, education, and marital-status categories where those fields exist. For the Home Credit Random Forest, the largest reported gaps were age-band selection rate (0.442; 95% CI 0.421–0.463) and true-positive rate (0.676; 95% CI 0.579–0.753). Its two supported gender categories had a selection-rate gap of 0.057 (95% CI 0.051–0.064).

This is a partial audit, not a comprehensive fairness determination. Give Me Some Credit has no sex/gender, race, or ethnicity fields; none of the three datasets has race or ethnicity. Several Taiwan categories and Home Credit's `XNA` gender category have too little support for stable rate comparisons. The additional ablation audit below compares subgroup rates across the feature scenarios, but it does not establish causal effects or provide uncertainty intervals for differences between scenarios.

#### Subgroup outcomes across feature ablations

For each predefined ablation, the model was refitted on the same training split, its threshold was selected on validation data, and rates were calculated once on the held-out test split. Groups below the project support rules remain in the saved tables with their support flags; gap values are calculated only across eligible groups for that metric. These comparisons are descriptive and their gaps do not have scenario-to-scenario confidence intervals.

| Dataset / audit attribute | Full-model gap | Feature removal | Gap after removal | Reading |
|---|---:|---|---:|---|
| Give Me Some Credit / age selection rate | 0.139 | delinquency aggregates | 0.146 | Slightly higher |
| Give Me Some Credit / age TPR | 0.226 | delinquency aggregates | 0.272 | Higher |
| Taiwan / age selection rate | 0.094 | repayment-status history | 0.166 | Higher |
| Taiwan / education selection rate | 0.212 | repayment-status history | 0.286 | Higher; only support-eligible groups contribute |
| Taiwan / age selection rate | 0.094 | monthly amount history | 0.037 | Lower |
| Taiwan / marital-status TPR | 0.066 | monthly amount history | 0.253 | Higher; only eligible groups contribute |
| Home Credit / age selection rate | 0.442 | prior repayment history | 0.457 | Slightly higher; large in both scenarios |
| Home Credit / age TPR | 0.676 | prior repayment history | 0.689 | Slightly higher; large in both scenarios |

The direction and size of the gap changes depend on both dataset and audit attribute. For example, removing Taiwan monthly amount history reduced the age selection gap but increased the marital-status TPR gap. This highlights a performance and subgroup trade-off for these particular samples; it does not show that any feature group causes unfairness. Full per-group rates and support counts are saved as `feature_ablation_subgroup_metrics.csv` beside each dataset's ablation results and are shown in the dashboard's Feature ablation tab.

### RQ3 — Are predicted probabilities calibrated?

Calibration was assessed with Brier score, log loss, expected calibration error (ECE), and reliability bins. Random Forest had lower Brier and ECE than Logistic Regression on all three test sets. ECE used 10 equal-width probability bins. No post-hoc calibration method was fitted, and the reported metrics do not by themselves establish that probabilities are calibrated for future populations.

#### Calibration within observed groups

The saved test probabilities were also evaluated within each observed audit group for both baseline models and for each Random Forest feature-ablation scenario. The table shows the unweighted mean of the group ECE values across reportable age groups; age groups are reportable only when they meet the project rules of at least 30 rows and at least 5 events and 5 non-events. Other attributes and every group, including limited-support groups, remain in the detailed output with support flags. Each subgroup ECE uses the same 10 equal-width bins as the overall analysis.

| Dataset | Reportable age groups | Logistic Regression mean subgroup ECE | Random Forest mean subgroup ECE |
|---|---:|---:|---:|
| Give Me Some Credit | 6/6 | 0.380 | 0.057 |
| Taiwan | 5/6 | 0.226 | 0.103 |
| Home Credit | 6/6 | 0.329 | 0.084 |

Among the Random Forest scenarios, the unweighted mean age-group ECE was 0.057 for full Give Me Some Credit and 0.055 without the delinquency aggregates; 0.103 for full Taiwan, 0.121 without repayment-status history, 0.182 without monthly amount history, and 0.105 without the derived summaries; and 0.084 for full Home Credit versus 0.087 without prior repayment history. The Taiwan monthly-amount removal had the largest increase in this age-group summary. For the two baseline models, per-group 95% calibration intervals were estimated by resampling within each observed group and target class, holding the 10 equal-width ECE bins fixed. These intervals condition on the fitted models, probabilities, and observed group outcome counts; they do not include model-refit or population-shift uncertainty. Feature-ablation subgroup calibration remains descriptive without intervals, and the baseline intervals have no multiple-comparison adjustment. ECE depends on the chosen bins, while Brier score and log loss also reflect factors beyond calibration. Calibration-in-the-large is saved as mean predicted probability minus observed outcome rate, so positive values indicate average overprediction in that subgroup.

For the 18–24 age group, baseline ECE estimates and within-group 95% intervals were:

| Dataset | Logistic Regression ECE (95% interval) | Random Forest ECE (95% interval) |
|---|---:|---:|
| Give Me Some Credit | 0.491 (0.482–0.501) | 0.071 (0.054–0.110) |
| Taiwan | 0.251 (0.235–0.268) | 0.129 (0.115–0.152) |
| Home Credit | 0.450 (0.443–0.458) | 0.140 (0.135–0.145) |

This is a descriptive per-group comparison; overlapping or non-overlapping intervals are not a formal paired model test. Taiwan's age 65+ group does not meet the minimum support rule and has no interval. Other group results and support flags are retained in the saved tables.

Per-group metrics and reliability bins are saved as `subgroup_calibration_metrics.csv` and `subgroup_calibration_bins.csv`; baseline intervals are in `subgroup_calibration_confidence_intervals.csv`. Feature-scenario versions use the `feature_ablation_subgroup_calibration` filename and currently have no intervals. These outputs are available in the Credit Risk page's Performance & calibration and Feature ablation tabs.

### RQ4 — Are model explanations stable across fits?

Three-seed Random Forest explanation comparisons had mean feature-rank correlation / top-10 Jaccard overlap / attribution cosine similarity of 0.997 / 1.000 / 0.989 for Give Me Some Credit, 0.918 / 0.818 / 0.979 for Taiwan, and 0.991 / 0.768 / 0.932 for Home Credit. A second check refit Random Forests on five different stratified 70% samples of the training partition, holding the estimator seed, preprocessing, 100 held-out explanation rows, and 100-row training-only SHAP background fixed.

| Dataset | Pairwise fits | Mean feature-rank correlation | Mean top-10 Jaccard overlap | Mean attribution cosine similarity |
|---|---:|---:|---:|---:|
| Give Me Some Credit | 10 | 0.986 | 0.891 | 0.948 |
| Taiwan | 10 | 0.952 | 0.873 | 0.933 |
| Home Credit | 10 | 0.988 | 0.855 | 0.918 |

The rankings and signed attribution patterns were similar across these resamples, while top-10 overlap was lower than the rank correlations, especially for Home Credit. For Home Credit, preprocessing was fit once on the full training partition and held fixed so category encoding and feature names stayed identical; this isolates classifier sensitivity to row resampling and does not include preprocessing-fit uncertainty. These checks do not establish stability across new populations, time periods, or datasets. SHAP values describe model associations and are not causal explanations.

### RQ5 — What happens when feature groups are removed?

The controlled ablations are summarized below. Thresholds were selected on validation data for each scenario, then applied to the test split. This makes the reported changes descriptive comparisons, not causal effects.

| Dataset | Random Forest feature scenario | Predictors | Test threshold | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Give Me Some Credit | Full | 14 | 0.40 | 0.4102 | 0.8439 | 0.3462 |
| Give Me Some Credit | Without two delinquency aggregates | 12 | 0.40 | 0.4074 | 0.8422 | 0.3418 |
| Give Me Some Credit | Without total open-credit accounts | 13 | 0.40 | 0.4109 | 0.8440 | 0.3473 |
| Give Me Some Credit | Without dependents flag | 13 | 0.40 | 0.4141 | 0.8457 | 0.3480 |
| Taiwan | Full | 23 | 0.45 | 0.5316 | 0.7568 | 0.5187 |
| Taiwan | Without repayment-status history | 15 | 0.35 | 0.4547 | 0.7082 | 0.4265 |
| Taiwan | Without monthly amount history | 10 | 0.30 | 0.4573 | 0.6987 | 0.4151 |
| Taiwan | Without two derived summaries | 21 | 0.45 | 0.5272 | 0.7551 | 0.5168 |
| Home Credit | Full | 199 | 0.25 | 0.3081 | 0.7614 | 0.2371 |
| Home Credit | Without 75 prior-history features | 124 | 0.25 | 0.2910 | 0.7450 | 0.2224 |

Paired bootstrap intervals compare each reduced-feature scenario with its full-feature model on the same outcome-stratified test-row resamples. Differences below are ablation minus full-model F1; thresholds were selected separately on validation data and held fixed during resampling.

| Dataset | Removed feature group | F1 difference | Paired 95% bootstrap interval |
|---|---|---:|---:|
| Give Me Some Credit | Delinquency aggregates | −0.003 | −0.010 to +0.004 |
| Give Me Some Credit | Open-account aggregate | +0.001 | −0.007 to +0.008 |
| Give Me Some Credit | Dependents indicator | +0.004 | −0.003 to +0.010 |
| Taiwan | Repayment-status history | −0.077 | −0.093 to −0.058 |
| Taiwan | Monthly amount history | −0.074 | −0.092 to −0.056 |
| Taiwan | Derived payment summaries | −0.004 | −0.013 to +0.004 |
| Home Credit | Prior repayment history | −0.017 | −0.024 to −0.011 |

On these splits, all Give Me Some Credit and derived-summary F1 intervals include zero. Removing Taiwan repayment-status or monthly-amount history, and removing Home Credit prior-history features, reduced F1 in the paired test-row comparisons. Removing Taiwan monthly-amount history also increased Brier score by 0.058 and ECE by 0.075; removing Home Credit prior history increased each by about 0.002. These are conditional, metric-specific comparisons with no multiple-comparison adjustment; they describe the saved test splits and do not establish causal feature effects or performance on future populations.

### RQ6 — What trade-offs appear across performance, calibration, fairness, and explanation stability?

Home Credit shows a metric trade-off: Random Forest had higher F1 (0.3081 vs. 0.2766) and lower Brier (0.0750 vs. 0.1933) and ECE (0.0791 vs. 0.3249), while Logistic Regression had higher ROC-AUC (0.7695 vs. 0.7614) and PR-AUC (0.2578 vs. 0.2371). Group-rate gaps remain material for some observed groups. Explanation ranks were broadly stable across training-row resamples, although top-10 overlap was lower than rank correlation. No single model is best on every measure, and the analysis does not select a lending policy.

## Data and evaluation protocol

| Dataset | Observations and target | Geography supported by the supplied source | Split and predictors |
|---|---|---|---|
| Give Me Some Credit | 150,000 training rows; 10,026 positive labels (6.68%); serious delinquency/financial distress within two years | Not identified in the supplied competition files | Stratified 105,000 / 15,000 / 30,000 train/validation/test; 14 final predictors |
| Taiwan Default of Credit Card Clients | 30,000 cardholders; next-month default target | Taiwan | Stratified 21,000 / 3,000 / 6,000; 23 predictors after four behavior-summary features |
| Home Credit Default Risk | 307,511 labeled applications; 24,825 positive labels (8.07%); payment difficulty on the sampled loan | Not established by the supplied competition files | Stratified 215,257 / 30,751 / 61,503; 199 predictors, including 75 application-time-safe history aggregates |

Give Me Some Credit has missing `MonthlyIncome` (29,731 rows) and `NumberOfDependents` (3,924 rows). The Home Credit application table has substantial missingness; preprocessing was fit within the training pipeline. Home Credit prior-history tables were aggregated in chunks with relative-date checks to exclude information after the application time; 305,311 of 307,511 labeled applicants matched the aggregated history, leaving 2,200 without a match. The Taiwan source reports no missing values. Source and split hashes, row counts, class counts, feature counts, software versions, and experiment settings are recorded in each dataset's `completion_run/run_manifest.json`.

All models use independent dataset-specific splits. The threshold search selected a validation F1 threshold and froze it for the test evaluation. Performance confidence intervals use 1,000 outcome-stratified test-row bootstrap replicates (seed 42) and condition on the fitted model, selected threshold, and observed test class counts. They do not include refitting, threshold-selection, or population-shift uncertainty. Fairness intervals use 1,000 within-group bootstrap replicates. Project reporting eligibility is at least 30 rows per group and at least 5 positive and negative cases for TPR/FPR; this is a reporting rule, not a guarantee of statistical or legal sufficiency.

## Held-out model comparison

| Dataset | Model | Threshold | F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Give Me Some Credit | Logistic Regression | 0.50 | 0.2834 | 0.8007 | 0.3210 | 0.1808 | 0.3481 |
| Give Me Some Credit | Random Forest | 0.40 | 0.4102 | 0.8439 | 0.3462 | 0.0594 | 0.0525 |
| Taiwan | Logistic Regression | 0.50 | 0.5072 | 0.7507 | 0.5061 | 0.1922 | 0.2153 |
| Taiwan | Random Forest | 0.45 | 0.5316 | 0.7568 | 0.5187 | 0.1541 | 0.1013 |
| Home Credit | Logistic Regression | 0.50 | 0.2766 | 0.7695 | 0.2578 | 0.1933 | 0.3249 |
| Home Credit | Random Forest | 0.25 | 0.3081 | 0.7614 | 0.2371 | 0.0750 | 0.0791 |

The Random Forest F1 95% intervals are 0.394–0.427 for Give Me Some Credit, 0.511–0.552 for Taiwan, and 0.300–0.316 for Home Credit. Paired intervals below estimate the Random Forest minus Logistic Regression metric difference using the same outcome-stratified test-row resamples for both models. Positive differences favor Random Forest for F1, ROC-AUC, and PR-AUC; negative differences favor it for Brier score and ECE, where lower values are better.

| Dataset | Metric | Difference (RF − LR) | Paired 95% bootstrap interval |
|---|---|---:|---:|
| Give Me Some Credit | F1 | +0.127 | +0.112 to +0.142 |
| Give Me Some Credit | ROC-AUC | +0.043 | +0.034 to +0.052 |
| Give Me Some Credit | PR-AUC | +0.025 | +0.010 to +0.042 |
| Give Me Some Credit | Brier score | −0.121 | −0.123 to −0.120 |
| Give Me Some Credit | ECE | −0.296 | −0.297 to −0.294 |
| Taiwan | F1 | +0.024 | +0.010 to +0.039 |
| Taiwan | ROC-AUC | +0.006 | −0.005 to +0.017 |
| Taiwan | PR-AUC | +0.013 | −0.009 to +0.033 |
| Taiwan | Brier score | −0.038 | −0.041 to −0.035 |
| Taiwan | ECE | −0.114 | −0.119 to −0.108 |
| Home Credit | F1 | +0.031 | +0.025 to +0.038 |
| Home Credit | ROC-AUC | −0.008 | −0.012 to −0.004 |
| Home Credit | PR-AUC | −0.021 | −0.029 to −0.012 |
| Home Credit | Brier score | −0.118 | −0.120 to −0.117 |
| Home Credit | ECE | −0.246 | −0.247 to −0.245 |

Each interval uses 1,000 paired bootstrap replicates (seed 42), stratified within the observed test outcome classes, with both validation-selected thresholds frozen. These intervals condition on the fitted models and observed class counts; they do not include refitting, threshold-selection, or population-shift uncertainty. No multiplicity adjustment is applied across metrics or datasets, so treat the intervals as metric-specific evidence rather than a single confirmatory test.

## Subgroup support and selected Random Forest gaps

The table reports max-minus-min gaps among eligible groups. Taiwan education gaps are based on only the eligible codes and omit categories with insufficient support. Rates are rounded to three decimals.

| Dataset / attribute | Eligible groups | Selection-rate gap (95% CI) | TPR gap (95% CI) | FPR gap (95% CI) |
|---|---:|---:|---:|---:|
| Give Me Some Credit / age | 6/6 | 0.139 (0.128–0.170) | 0.226 (0.155–0.317) | 0.097 (0.086–0.131) |
| Give Me Some Credit / dependents | 3/3 | 0.039 (0.030–0.047) | 0.050 (0.013–0.118) | 0.026 (0.018–0.033) |
| Taiwan / sex | 2/2 | 0.022 (0.003–0.046) | 0.009 (0.001–0.067) | 0.020 (0.002–0.042) |
| Taiwan / age | 5/6 | 0.094 (0.061–0.147) | 0.139 (0.085–0.298) | 0.058 (0.034–0.114) |
| Taiwan / marital status | 3/4 | 0.027 (0.007–0.140) | 0.066 (0.013–0.360) | 0.062 (0.007–0.183) |
| Taiwan / education | 4/7 selection; 3/7 TPR; 4/7 FPR | 0.212 (0.132–0.282) | 0.093 (0.046–0.620) | 0.117 (0.058–0.186) |
| Home Credit / gender | 2 identified categories | 0.057 (0.051–0.064) | 0.065 (0.039–0.096) | 0.045 (0.038–0.051) |
| Home Credit / age | 6/6 | 0.442 (0.421–0.463) | 0.676 (0.579–0.753) | 0.406 (0.383–0.428) |

Limited-support categories include Taiwan age 65+ (n=19), education codes 0, 4, and 6 (n=2, 26, and 9), education code 5 with only three positive cases, and marital-status code 0 (n=7). Home Credit's `XNA` gender category has only two test records and is not compared. Group labels and categories reflect the supplied datasets; they should not be equated with a complete protected-class audit.

## Global and local explanations

The largest mean absolute SHAP features for the Random Forest models were:

| Dataset | Top features (mean absolute SHAP, descending) |
|---|---|
| Give Me Some Credit | `RevolvingUtilizationOfUnsecuredLines` (0.0363), `TotalDelinquencyCount` (0.0292), `age` (0.0157), `DebtRatio` (0.0156), `SevereDelinquencyCount` (0.0130) |
| Taiwan | `PAY_0` (0.0373), `MaximumReportedDelayLevel` (0.0309), `BehavioralDelinquencyMonths` (0.0286), `LIMIT_BAL` (0.0281), `BILL_AMT1` (0.0161) |
| Home Credit | `MEAN_EXTERNAL_SCORE` (0.0204), `EXT_SOURCE_2` (0.0106), `EXT_SOURCE_3` (0.0093), `CREDIT_TO_GOODS_RATIO` (0.0056), `EXT_SOURCE_1` (0.0055) |

Sampled per-row SHAP contributions are saved under each dataset's `completion_run` explanation outputs and are available in the Credit Risk page. A contribution indicates how a feature moves that fitted model's output relative to its SHAP baseline for that example; it is not a causal effect or an individual credit recommendation.

## Answers and scope limits

- The models can rank risk signals in these three historical datasets, with performance differences across datasets and metrics.
- Behavior-history features improved some held-out metrics in the tested ablations, particularly Home Credit and Taiwan; the evidence does not validate mobile-wallet, telecom, utility, or thin-file data.
- Subgroup gaps were assessed for the full baselines and predefined Random Forest ablations on the same held-out rows. Only observed attributes were available, sparse groups remain limited, and the ablation gap comparisons have no scenario-to-scenario uncertainty intervals.
- Overall and group-level calibration were measured, but no post-hoc calibration was applied. Subgroup calibration summaries have no confidence intervals, and future-population calibration is unknown.
- Random Forest explanation stability was checked across three estimator seeds and five stratified training-row resamples. The resampling analysis holds the estimator seed and preprocessing fixed; it does not measure new split, time-period, population, or model-class variability.
- Results are not pooled across datasets and do not support worldwide coverage, causal claims, a lending-policy recommendation, or production deployment.

## Sources and reproducibility

- Give Me Some Credit target and competition context: [Kaggle competition data page](https://www.kaggle.com/c/GiveMeSomeCredit/data).
- Taiwan dataset population, target, and dataset details: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients%2Bdataset).
- Home Credit competition and source tables: [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
- Model comparisons, bootstrap intervals, subgroup fairness and calibration summaries, feature-ablation results, SHAP tables, three-seed stability outputs, five-resample explanation-stability outputs, and run manifests are stored in `outputs/credit_risk/` under each dataset's `completion_run/` folder. The source archives remain outside the repository.
