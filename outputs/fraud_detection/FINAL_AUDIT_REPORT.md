# Component 04 Fraud Detection: Audit and Consistency Report

**Repository Branch:** `feature/component-04-Early-warning-data-preparation`  
**Specification:** [`backend/src/fraud_detection/COMPONENT_04_SPEC.md`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/COMPONENT_04_SPEC.md) (READ-ONLY)  
**Verification Date:** 2026-10-25  
**Evaluation Status:** Planned benchmark evaluation completed; audit corrections applied  
**Unit Test Coverage:** 101 passed unit tests in [`tests/fraud_detection/`](file:///d:/project/KND_CDAP_DS_09/tests/fraud_detection/)  

---

## 1. Executive Summary

This audit report documents the empirical evaluation and code consistency checks for Component 04 ("Explainable Early Warning and Attribution-Guided Adaptation for Delayed-Label Fraud Detection"). The audit was conducted in two sequential stages:
1. **Stage A (Forensic Code & Metric Audit):** Systematic verification of the full 660-run experimental grid ([`grid_results_summary.csv`](file:///d:/project/KND_CDAP_DS_09/outputs/fraud_detection/experiments/grid_results_summary.csv)), temporal non-leakage constraints, timing conventions, ablation dynamics, and baseline implementation behaviors.
2. **Stage B (Targeted Corrections & Consistency Pass):** Implementation of code corrections (calibrated CBPE trigger), addition of 8 focused unit tests (increasing test coverage from 93 to 101 tests), synchronization of documentation tables with exact empirical values, integration of the Sparkov Credit Card Dataset fallback evaluation (Spec §14/§17), and calibrated reporting of all findings.

---

## 2. Audit Findings & Corrective Resolutions Matrix

| # | Audit Item | Forensic Finding | Severity | Corrective Resolution | Status |
| :-: | :--- | :--- | :-: | :--- | :-: |
| **1** | **Factorial Grid Completeness** | All $3 \times 4 \times 5 \times 11 = 660$ runs present in `grid_results_summary.csv`. Zero missing values, zero infinite values, zero pipeline crashes. | Informational | Grid integrity checked and preserved without re-running. Pre-audit artifacts backed up to `_audit_backup/`. | **Checked** |
| **2** | **Statistical Consistency** | Summary markdown Section 4 table contained draft prototype figures for Delay $d=14$ ($S_1$ $P_1$: 0.5694 / 94.97% / 1.8 retrains) diverging from CSV (0.4375 / 72.97% / 0 retrains). | High | Updated [`RESULTS_SUMMARY.md`](file:///d:/project/KND_CDAP_DS_09/outputs/fraud_detection/RESULTS_SUMMARY.md) Section 4 table with exact CSV values across all delays and seeds. | **Resolved** |
| **3** | **Degradation Protocol Sensitivity** | Strict 1-window rule ($\text{Recall} < 0.80 \times \text{Base}$) flags single-window transient drops in low-prevalence streams ($S_1$ $d=28$ window 10 has 49 frauds). | Medium | Conducted 2-consecutive-window persistence analysis. Added unit test `test_two_consecutive_window_noise_suppression` demonstrating transient noise suppression. | **Resolved** |
| **4** | **Terminal Lookback Confound ($d=28$)** | Pre-drift base recall drops from 0.748 ($d=7$) to 0.599 ($d=14$) to 0.354 ($d=28$). Audit revealed lookback $[D-d-28, D-d]$ has zero matured labels before day 0, collapsing $t_{\text{risk}}$ importance by 97%. | Medium | Formally documented as a feature lookback boundary confound in Section 4 and Section 7 of `RESULTS_SUMMARY.md`. | **Resolved** |
| **5** | **Delay Scaling Dynamics** | In $S_1$, $P_1$ adapts under $d=7$ (3.6 retrains) but stays conservative under $d=14, 28$ (0 retrains). In $S_2$, $P_1$ adapts across all delays (5 retrains at $d=7$, 4 at $d=14$, 2 at $d=28$). | Low | Clarified that pure concept drift requires matured labels for detection, whereas mixed drift triggers label-free covariate/score signals. | **Resolved** |
| **6** | **Timing Sign & Discretization** | Code measures lead time as $t_{\text{degrade}} - t_{\text{alarm}}$ (positive = early). Recomputed from `grid_results_summary.csv` for $S_2$ ($d=7\text{d}$): degrade day was 119 across all 5 seeds; alarm day was 105 in Seed 1 ($+14\text{d}$) and 112 in Seeds 2–5 ($+7\text{d}$), yielding mean $+8.4\text{d} \pm 3.1\text{d}$ subject to 7-day discretization. | Medium | Updated summary table to $+8.4\text{d} \pm 3.1\text{d}$ and documented weekly window boundary discretization artifact. Added 3 timing convention unit tests. | **Resolved** |
| **7** | **$B_3$ Baseline CBPE Calibration** | Comparing uncalibrated CBPE estimate (~0.256 due to `scale_pos_weight=20`) against ground-truth recall (0.598) caused continuous retraining (11/11 weeks). | High | Modified [`fd_baselines.py`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/fd_baselines.py) to calibrate CBPE against its own pre-drift reference estimate (`base_est_rec`). Added unit test `test_calibrated_cbpe_trigger_avoids_continuous_retrain`. | **Resolved** |
| **8** | **TreeSHAP Profile Shift Ablation** | In 58 of 60 streams, candidate model selection was identical between $P_1$ and $P_1\text{-noSHAP}$ (`DROP_D` was chosen 0 times). $P_1\text{-b}$ `DROP_D` degraded recall to 75.3% retention. | Medium | Reported as a confirmed null result for candidate model selection; documented SHAP as an interpretability/diagnostic tool rather than feature pruner. | **Resolved** |
| **9** | **Real-Data Check & Sparkov Fallback** | Phase 9 Status: **IEEE-CIS check not performed (data unavailable); Sparkov fallback only**. Real-data validation was NOT performed because `train_transaction.csv` was unavailable. Sparkov synthetic fallback evaluated. | High | Evaluated Sparkov synthetic dataset (555,719 rows, 194 days). Degradation verdict: `False` (mean recall 82.0%). EWS issued 100% `WAIT` decisions. Crucially noted that 0 false alarms is not evidence of robustness because no degradation exists. | **Resolved** |
| **10** | **Exploratory Scope Boundary** | Exploratory module `fd_model_improvement.py` was not explicitly labeled as outside the benchmark grid. | Low | Added prominent docstring header in [`fd_model_improvement.py`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/fd_model_improvement.py) demarcating exploratory vs. benchmark code. | **Resolved** |
| **11** | **Dependencies Completeness** | `xgboost` and `catboost` imported in exploratory script but missing from `requirements-fd.txt`. | Low | Added `xgboost` and `catboost` to [`requirements-fd.txt`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/requirements-fd.txt). | **Resolved** |

---

## 3. Code Modifications (Stage B)

### 3.1 Calibrated $B_3$ CBPE Trigger & Dual Recovery Metrics
In [`backend/src/fraud_detection/fd_baselines.py`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/fd_baselines.py):
- **Pre-Drift CBPE Baseline:** Calculated CBPE's estimate on pre-drift baseline predictions (`base_est_rec = cbpe.estimate_recall(base_scores)`).
- **Calibrated Trigger Comparison:** Replaced raw ground-truth comparison with calibrated threshold:
  ```python
  # Calibrated comparison against CBPE's own baseline estimate
  cbpe_thresh = 0.80 * base_est_rec if base_est_rec > 0 else 0.80 * base_recall
  if est_rec <= cbpe_thresh:
      retrain_triggered = True
  ```
- **Dual Recovery Time Metrics:** Recorded both degradation MTTR ($t_{\text{recov}} - t_{\text{degrade}}$) and post-adaptation turnaround ($t_{\text{recov}} - t_{\text{first\_adapt}}$), resolving ambiguity between outage duration and model response lag.

### 3.2 Packaging & Documentation Clarifications
- Added `xgboost` and `catboost` to [`backend/src/fraud_detection/requirements-fd.txt`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/requirements-fd.txt).
- Updated [`backend/src/fraud_detection/fd_model_improvement.py`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/fd_model_improvement.py) with an explicit architectural note clarifying its exploratory status outside the primary LightGBM benchmark.

---

## 4. Test Suite Augmentation & Verification

Eight focused unit tests were implemented in [`tests/fraud_detection/test_fd_experiments.py`](file:///d:/project/KND_CDAP_DS_09/tests/fraud_detection/test_fd_experiments.py):
1. `test_recovery_time_no_degradation_returns_none`: Verifies `degrade_day` and `recovery_time_days` are `None` when recall remains above $80\%$ baseline throughout.
2. `test_recovery_time_degradation_with_recovery`: Verifies MTTR calculation ($t_{\text{recov}} - t_{\text{degrade}} = 14\text{d}$) and post-adaptation recovery time ($t_{\text{recov}} - t_{\text{adapt}} = 14\text{d}$).
3. `test_recovery_time_degradation_without_recovery`: Verifies `recovery_time_days == -1` when recall degrades and never recovers prior to stream end.
4. `test_timing_conventions_lead_time_positive`: Confirms sign convention $t_{\text{degrade}} - t_{\text{alarm}} > 0$ for early alarms ($+14\text{d}$).
5. `test_timing_conventions_adaptation_lag_positive`: Confirms sign convention $t_{\text{adapt}} - t_{\text{degrade}} > 0$ for adaptation lag ($+7\text{d}$).
6. `test_timing_conventions_edge_case_no_alarm_or_adapt`: Verifies `None` return when alarms or adaptations do not occur.
7. `test_two_consecutive_window_noise_suppression`: Verifies that a 2-consecutive-window degradation rule suppresses single-window stochastic noise dips.
8. `test_calibrated_cbpe_trigger_avoids_continuous_retrain`: Verifies that comparing `est_rec <= 0.80 * base_est_rec` avoids spurious triggers when score distributions are stable.

### Execution Log Verification:
```text
pytest tests/fraud_detection -q
101 passed, 4 warnings in 22.59s
```
**Total Test Count:** 101 passed tests, 0 failures, 0 errors.

---

## 5. Empirical Stream Evaluation (IEEE-CIS Status & Sparkov Fallback)

### 5.1 IEEE-CIS Protocol Check (Spec §14)
- **Phase 9 Status:** **IEEE-CIS check not performed (data unavailable); Sparkov fallback only**
- **Execution Status:** `DATA_MISSING`, Verdict: `DATA_NOT_FOUND`.
- **Honesty Disclosure:** Real-data validation was **NOT performed** because `train_transaction.csv` was unavailable (requires manual user authentication and Kaggle competition rules acceptance). As specified in Spec §3.2 and §14, IEEE-CIS cannot serve as real-degradation validation in this project.

### 5.2 Sparkov Dataset Fallback Evaluation (Spec §14 & §17)
- **Dataset Classification:** **SYNTHETIC DATA** (NOT real data, and NOT part of the original specification).
- **Origin & Acquisition:** Acquired as a practical fallback from Kaggle ([`kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection)) as provided by the user. The file [`data/fraud_detection/raw/sparkov/fraudTest.csv`](file:///d:/project/KND_CDAP_DS_09/data/fraud_detection/raw/sparkov/fraudTest.csv) (150.3 MB, 555,719 rows, 2,145 frauds, $0.386\%$ fraud prevalence, spanning 194 days) was generated using the open-source Sparkov synthetic credit card transaction simulator.
- **Protocol:** Trained base model on days $< 42$ (119,597 transactions, 466 frauds). Evaluated across 10 rolling 14-day windows spanning days 49..188.
- **Performance Findings:**
  - Frauds per window: Window 0: 141; Window 1: 186; Window 2: 130; Window 3: 171; Window 4: 232; Window 5: 126; Window 6: 129; Window 7: 154; Window 8: 171; Window 9: 87.
  - Recall@1% FPR ranged from $0.7092$ (baseline window 0) to $0.9154$ (mean $0.8198$).
  - Relative retention to first window stayed between $100.0\%$ and $129.1\%$.
  - Zero windows dropped below $80\%$ of baseline.
  - **Degradation Verdict:** **`False`** (contains **no real performance degradation**).
- **Label-Free Detector Findings:**
  - Feature Mean PSI: $0.00036$ to $0.00078$ (below control threshold $0.000845$).
  - Score JS Divergence: $0.0060$ to $0.0153$.
  - Calibrated EWS: $0.140$ to $0.174$ (below locked alert threshold $\theta_1 = 0.22$).
  - **Detector Decision:** 10/10 windows issued **`WAIT`** (0 false alarms, 0 retrains).
- **Critical Caveat on False Alarms:** **"0 false alarms" is NOT evidence of robustness because no degradation exists in this synthetic stream.** On a stationary synthetic stream without drift, a detector staying quiet does not prove it can handle real-world distribution shifts or maintain adequate sensitivity under complex non-stationary environments.

---

## 6. Honest Scientific Assessment of Claims (Spec §17)

| Claim | Initial Hypothesis | Audit Finding & Calibrated Verdict |
| :--- | :--- | :--- |
| **(a) False Alarm Suppression on Virtual Shift** | $P_1$ eliminates false alarms on $S_3$ where baselines fail. | **Supported on synthetic $S_3$ stream.** $B_1, B_2, B_3$ trigger 11.0 retrains; $P_1$ triggers 0.0 retrains across all seeds. |
| **(b) Early-Warning Lead Time on Mixed Drift** | $P_1$ detects drift prior to ground-truth degradation on $S_2$. | **Supported on synthetic $S_2$ stream ($d=7\text{d}$).** Recomputed from `grid_results_summary.csv`: degrade day was 119 across all 5 seeds, alarm was day 105 in Seed 1 ($+14\text{d}$) and day 112 in Seeds 2–5 ($+7\text{d}$), yielding mean $+8.4\text{d} \pm 3.1\text{d}$, subject to 7-day evaluation window discretization. |
| **(c) Retrain Count vs. Periodic Retraining ($B_1$)** | $P_1$ outperforms periodic retraining under drift. | **Partially supported.** $P_1$ reduces retrain count ($3.6$ in $S_1$, $5.0$ in $S_2$ vs. 11.0 in $B_1$), but **does NOT beat periodic retraining ($B_1$) on post-onset recall** (in $S_1$: $B_1$ 0.7849 vs. $P_1$ 0.6940; in $S_2$: $B_1$ 0.7719 vs. $P_1$ 0.7326). $P_1$'s value is maintenance reduction on benign shifts, not higher recall under drift. |
| **(d) TreeSHAP Profile Shift Contribution** | TreeSHAP profile shift steers candidate model selection. | **Null Result.** In 58 of 60 streams, candidate model selection was identical between $P_1$ and $P_1\text{-noSHAP}$ (`DROP_D` was chosen 0 times). TreeSHAP profile shift showed no measurable benefit for candidate model selection in this benchmark. |
| **(e) Feature Pruning via SHAP ($P_1\text{-b}$)** | Dropping top-$k$ drifted features restores recall. | **Refuted.** Pruning drifted features degrades recall ($75.3\%$ retention vs. $92.7\%$ for full retraining), as drifted features contain key fraud patterns. |
| **(f) Delay Scaling ($d=7, 14, 28$)** | Adaptation performance is invariant to label delay. | **Calibrated.** Increasing delay to 28 days truncates lookback history, reducing base recall from 0.748 to 0.354, and delays EWS detection in pure concept drift (0 retrains). |
| **(g) CBPE Drift Estimation Baseline ($B_3$)** | Standard CBPE accurately detects degradation without labels. | **Defect Identified & Corrected.** Probability distortions under heavy class weighting cause continuous alarms unless calibrated against CBPE's own baseline estimate. |
| **(h) Real-Data Degradation** | Framework validated on real IEEE-CIS drift. | **Disclaimed per Spec §14.** Real-data validation was NOT performed (IEEE-CIS data unavailable). Sparkov synthetic fallback stream contained no real performance degradation. |
| **(i) Computational Efficiency** | AGA is faster than periodic retraining. | **Calibrated.** AGA reduces total retrain events by $55\%$–$67\%$, but each adaptation event trains and evaluates 3 candidates on holdout sets, resulting in comparable total wall-clock runtime per retrain. |

---

## 7. What is NOT Validated

To maintain scientific integrity and prevent overclaiming, the following aspects were **NOT validated** in this study:
1. **Real Data:** No evaluation on genuine, unanonymized real-world banking telemetry or unperturbed real-world concept drift was performed. Both the Fraud Detection Handbook and the Sparkov fallback are synthetic datasets.
2. **Human-Reviewed Alerts:** No operational simulation of alert triage capacity, investigator review fatigue, alert queue backlog, or financial costs of false alarms under production volumes was conducted.
3. **Production Deployment:** No live streaming infrastructure, distributed latency testing, online asynchronous inference, concurrent drift handling, or database lock performance was tested.
4. **Significance Tests:** Formal statistical significance testing (e.g., paired t-tests or Wilcoxon signed-rank tests across evaluation streams) was not conducted; all reported values represent empirical sample means and sample standard deviations across 5 random seeds.

---

## 8. Final Verification Checklist

- [x] All 101 Component 04 unit tests passing (`pytest tests/fraud_detection -q`).
- [x] Full factorial grid (660 runs) completed and checked in `grid_results_summary.csv`.
- [x] Both timing conventions (`Lead Time to First Alarm` vs. `Adaptation Lag` vs. `MTTR`) separated and reported.
- [x] Fixed PSI ablation interface ensuring feature PSI is passed and logged.
- [x] IEEE-CIS status accurately reported as not performed (data unavailable).
- [x] Sparkov fallback evaluated and clearly labeled as synthetic data without degradation.
- [x] Diagnostic visualization plots saved under `outputs/fraud_detection/figures/`.
- [x] Negative ablation findings, null results, and "What is NOT Validated" limitations fully documented without overclaiming.
- [x] Repository scope boundaries and non-leakage temporal constraints tested by unit assertions.
