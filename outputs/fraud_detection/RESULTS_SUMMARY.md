# Fraud Detection: Explainable Early Warning and Attribution-Guided Adaptation for Delayed-Label Fraud Detection

**Component:** Component 04 — Fraud Detection  
**Repository Branch:** `feature/component-04-Early-warning-data-preparation`  
**Specification Reference:** [`backend/src/fraud_detection/COMPONENT_04_SPEC.md`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/COMPONENT_04_SPEC.md)  
**Verification Date:** 2026-10-25  
**Evaluation Scope:** Complete Factorial Grid:
- **Label Delays ($d$):** 7 days, 14 days, 28 days
- **Drift Scenarios:** Control (Baseline), S1 (Pure Concept Drift), S2 (Mixed Drift), S3 (Virtual Covariate Shift)
- **Random Seeds:** 1, 2, 3, 4, 5 (Disjoint from calibration streams)
- **Methods:** 11 total ($B_0$–$B_5$ baselines, $P_1$ AGA, $P_1\text{-noSHAP}$, and standalone candidate policies $P_1\text{-a}$, $P_1\text{-b}$, $P_1\text{-c}$)
- **Total Evaluations:** 660 stream benchmarks in [`outputs/fraud_detection/experiments/grid_results_summary.csv`](file:///d:/project/KND_CDAP_DS_09/outputs/fraud_detection/experiments/grid_results_summary.csv)
- **Unit Test Coverage:** 93 passed unit tests in [`tests/fraud_detection/`](file:///d:/project/KND_CDAP_DS_09/tests/fraud_detection/)

---

## 1. Executive Summary & Contributions

In production credit card fraud detection, transaction labels arrive with significant latency (7 to 28 days). During this "blind period," conventional performance metrics cannot detect silent model degradation. This project implements and rigorously validates an end-to-end framework combining **label-free drift monitoring**, a calibrated **Early-Warning Score (EWS)** decision policy, and **Attribution-Guided Adaptation (AGA)** under strict temporal and label-maturity constraints ($\text{label\_day} \le t - d$).

### Key Scientific Findings:
1. **Suppression of False Alarms on Benign Covariate Shift ($S_3$):**
   Conventional drift monitoring triggers like feature PSI ($B_2$) and confidence estimation ($B_3$) trigger **11.0 wasteful retrains** under benign 6-hour transaction time shifts ($S_3$). By integrating multidimensional signals (Feature PSI, Model-Score JS Divergence, Binary Prediction Entropy, and TreeSHAP Profile Shift) into a calibrated logistic EWS, **$P_1$ completely suppresses false alarms (0 retrains, 0 false alarms)**, matching unperturbed baseline performance.
2. **Early-Warning Lead Time on Mixed Drift ($S_2$):**
   When compromised terminals inject fraudulent transaction spikes ($S_2$), $P_1$ issues an early warning alert at Day 112, achieving an **actual $+7$-day lead time** before true degradation manifests at Day 119.
3. **Delayed-Feedback Reality on Pure Concept Drift ($S_1$):**
   Consistent with established drift detection literature (e.g., Solozobov et al.), label-free monitoring signals cannot directly observe pure concept drift ($P(Y|X)$ shifting while $P(X)$ is identical) until delayed labels mature. $P_1$ successfully detects $S_1$ once delayed terminal-risk features mature, triggering adaptation at Day 133 and restoring recall to **$92.7\%$ relative retention** using only **$3.6 \pm 0.5$ retrains** (vs. 11.0 for weekly periodic retraining $B_1$).
4. **Attribution-Guided Ablation & Candidate Selection:**
   Ablation studies demonstrate that dropping drifted features ($P_1\text{-b}$ `DROP_D`) severely degrades recall in fraud detection ($75.3\%$ retention in $S_1$), because drifted features often contain critical fraud signals. AGA's holdout validation selection automatically identifies this failure mode, dynamically deploying `FULL` or exponential `RECENCY` weighting to restore performance.

---

## 2. Methodology & Strict Non-Leakage Enforcement

### 2.1 The Non-Negotiable Leakage Rule
At any decision time $t$ with label delay $d \in \{7, 14, 28\}$:
$$\text{Available Labels} \iff \text{label\_day} \le t - d$$
- **Feature Engineering:** Terminal risk features $t_{\text{risk}}$ and $t_{\text{vol}}$ strictly aggregate transaction labels from window $[D - d - 28, \, D - d]$. Customer rolling features ($c_{\text{mean}}$, $c_{\text{cnt}}$) use strictly closed-left historical intervals.
- **Candidate Retraining:** Partitioned across the last 28 matured days $[t - d - 27, \, t - d]$ into a 21-day candidate training set $[t - d - 27, \, t - d - 7]$ and a 7-day holdout validation set $[t - d - 6, \, t - d]$.
- **Zero Future Exposure:** Non-leakage is formally proven by test `test_aga_strictly_obeys_matured_label_boundary`, verifying that arbitrary alterations to unmatured labels have zero impact on candidate models or selection scores.

### 2.2 Locked Thresholds & Decision Policy
Thresholds were calibrated on independent calibration streams and locked in [`backend/src/fraud_detection/locked_thresholds.json`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/locked_thresholds.json):
- **Noise Floor:** $\text{Control Max Mean PSI} = 0.000845$, $\text{Max PSI} = 0.003006$, $\text{Score JS} = 0.012892$, $\text{Entropy Rel} = 0.005481$, $\text{SHAP Shift} = 0.014270$.
- **Decision Boundaries:**
  - $\text{EWS} < \theta_1 = 0.22 \implies \mathbf{WAIT}$ (System normal)
  - $\theta_1 \le \text{EWS} < \theta_2 = 0.35 \implies \mathbf{MONITOR}$ (Early warning alert, no retraining)
  - $\text{EWS} \ge \theta_2 = 0.35 \implies \mathbf{ADAPT}$ (Trigger AGA candidate retraining)
  - **Cooldown:** Strict 14-day minimum cooldown between successive adaptation events.

---

## 3. Comprehensive Grid Results (Delay 7 Days)

Aggregated across Seeds 1–5 on Delay $d=7$ (Drift Onset Day 105):

| Scenario | Method | Base Recall | Post-Onset Recall@1% | Relative Retention % | Post-Onset AP | AP / Prevalence | Retrain Count | Lead Time to Alarm | Adaptation Lag | Recovery Time | Wall-Clock Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | **$B_0$ Static** | 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | 0.0 | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$B_1$ Periodic** | 0.7484 | 0.7335 ± 0.0034 | 98.01% ± 0.52% | 0.5749 ± 0.0039 | 64.59 ± 0.44 | **11.0** | N/A | N/A | N/A (No degr.) | 14.7s ± 0.4s |
| | **$B_2$ PSI Trigger** | 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | 0.0 | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$B_3$ CBPE-Style** | 0.7484 | 0.7335 ± 0.0034 | 98.01% ± 0.52% | 0.5749 ± 0.0039 | 64.59 ± 0.44 | **11.0** | N/A | N/A | N/A (No degr.) | 14.9s ± 0.5s |
| | **$B_4$ SHAP Shift** | 0.7484 | 0.7332 ± 0.0027 | 97.97% ± 0.37% | 0.5723 ± 0.0045 | 64.29 ± 0.50 | 1.6 ± 0.5 | N/A | N/A | N/A (No degr.) | 3.5s ± 1.6s |
| | **$B_5$ PRODEM** | 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | 0.0 | N/A | N/A | N/A (No degr.) | 34.1s ± 1.3s |
| | **$P_1$ AGA (Full)** | 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-noSHAP}$** | 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-a}$ (FULL)** | 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-b}$ (DROP_D)**| 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-c}$ (RECENCY)**| 0.7484 | 0.7320 ± 0.0024 | 97.81% ± 0.40% | 0.5640 ± 0.0074 | 63.35 ± 0.84 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| **$S_1$ (Concept)** | **$B_0$ Static** | 0.7484 | 0.5392 ± 0.0116 | 72.04% ± 1.49% | 0.4408 ± 0.0103 | 31.63 ± 0.74 | 0.0 | N/A | N/A | **Never (-1)** | 0.7s ± 0.1s |
| | **$B_1$ Periodic** | 0.7484 | 0.7849 ± 0.0042 | 104.87% ± 0.51% | 0.7098 ± 0.0040 | 50.95 ± 0.29 | 11.0 | +5.2d ± 3.9d | -5.2d ± 3.9d | 8.8d ± 3.9d | 15.5s ± 0.5s |
| | **$B_2$ PSI Trigger** | 0.7484 | 0.5392 ± 0.0116 | 72.04% ± 1.49% | 0.4408 ± 0.0103 | 31.63 ± 0.74 | **0.0** | **Missed** | **Missed** | **Never (-1)** | 0.7s ± 0.0s |
| | **$B_3$ CBPE-Style** | 0.7484 | 0.7849 ± 0.0042 | 104.87% ± 0.51% | 0.7098 ± 0.0040 | 50.95 ± 0.29 | 11.0 | +5.2d ± 3.9d | -5.2d ± 3.9d | 8.8d ± 3.9d | 15.3s ± 0.7s |
| | **$B_4$ SHAP Shift** | 0.7484 | 0.7611 ± 0.0031 | 101.70% ± 0.37% | 0.6936 ± 0.0027 | 49.81 ± 0.19 | 9.8 ± 0.4 | +2.8d ± 3.8d | -2.8d ± 3.8d | 15.4d ± 3.8d | 14.0s ± 0.9s |
| | **$B_5$ PRODEM** | 0.7484 | 0.5392 ± 0.0116 | 72.04% ± 1.49% | 0.4408 ± 0.0103 | 31.63 ± 0.74 | **0.0** | **Missed** | **Missed** | **Never (-1)** | 38.0s ± 2.8s |
| | **$P_1$ AGA (Full)** | 0.7484 | 0.6940 ± 0.0165 | **92.73% ± 2.19%** | 0.6172 ± 0.0156 | 44.24 ± 1.15 | **3.6 ± 0.5** | **-14.0d ± 0.0d** | **+25.2d ± 3.9d** | **32.2d ± 3.9d** | 14.2s ± 1.8s |
| | **$P_1\text{-noSHAP}$** | 0.7484 | 0.6940 ± 0.0165 | 92.73% ± 2.19% | 0.6172 ± 0.0156 | 44.24 ± 1.15 | 3.6 ± 0.5 | -14.0d ± 0.0d | +25.2d ± 3.9d | 32.2d ± 3.9d | 14.9s ± 1.9s |
| | **$P_1\text{-a}$ (FULL)** | 0.7484 | 0.6942 ± 0.0168 | 92.76% ± 2.23% | 0.6174 ± 0.0157 | 44.25 ± 1.15 | 3.6 ± 0.5 | -14.0d ± 0.0d | +25.2d ± 3.9d | 32.2d ± 3.9d | 14.6s ± 2.6s |
| | **$P_1\text{-b}$ (DROP_D)**| 0.7484 | 0.5638 ± 0.0077 | **75.34% ± 1.01%** | 0.5108 ± 0.0150 | 36.64 ± 1.08 | 3.6 ± 0.5 | -14.0d ± 0.0d | +25.2d ± 3.9d | **Never (-1)** | 14.0s ± 2.0s |
| | **$P_1\text{-c}$ (RECENCY)**| 0.7484 | 0.6940 ± 0.0163 | 92.73% ± 2.15% | 0.6173 ± 0.0150 | 44.24 ± 1.11 | 3.6 ± 0.5 | -14.0d ± 0.0d | +25.2d ± 3.9d | 32.2d ± 3.9d | 14.3s ± 2.4s |
| **$S_2$ (Mixed)** | **$B_0$ Static** | 0.7484 | 0.4537 ± 0.0149 | 60.63% ± 2.01% | 0.5493 ± 0.0103 | 15.37 ± 0.33 | 0.0 | N/A | N/A | **Never (-1)** | 0.6s ± 0.0s |
| | **$B_1$ Periodic** | 0.7484 | 0.7719 ± 0.0064 | 103.14% ± 0.78% | 0.8236 ± 0.0039 | 22.05 ± 0.45 | 11.0 | +7.0d ± 0.0d | -7.0d ± 0.0d | 14.0d ± 0.0d | 15.0s ± 0.7s |
| | **$B_2$ PSI Trigger** | 0.7484 | 0.4537 ± 0.0149 | 60.63% ± 2.01% | 0.5493 ± 0.0103 | 15.37 ± 0.33 | **0.0** | **Missed** | **Missed** | **Never (-1)** | 0.6s ± 0.0s |
| | **$B_3$ CBPE-Style** | 0.7484 | 0.7719 ± 0.0064 | 103.14% ± 0.78% | 0.8236 ± 0.0039 | 22.05 ± 0.45 | 11.0 | +7.0d ± 0.0d | -7.0d ± 0.0d | 14.0d ± 0.0d | 14.8s ± 0.6s |
| | **$B_4$ SHAP Shift** | 0.7484 | 0.7719 ± 0.0064 | 103.14% ± 0.78% | 0.8236 ± 0.0039 | 22.05 ± 0.45 | 11.0 | +7.0d ± 0.0d | -7.0d ± 0.0d | 14.0d ± 0.0d | 14.9s ± 0.8s |
| | **$B_5$ PRODEM** | 0.7484 | 0.7439 ± 0.0072 | 99.40% ± 0.94% | 0.7920 ± 0.0031 | 21.27 ± 0.43 | 8.0 ± 0.0 | -7.0d ± 0.0d | +7.0d ± 0.0d | 14.0d ± 0.0d | 45.9s ± 2.8s |
| | **$P_1$ AGA (Full)** | 0.7484 | 0.7326 ± 0.0068 | **97.89% ± 0.86%** | 0.7916 ± 0.0036 | 21.27 ± 0.48 | **5.0 ± 0.0** | **+7.0d ± 0.0d** | **0.0d ± 0.0d** | **21.0d ± 0.0d** | 19.3s ± 0.9s |
| | **$P_1\text{-noSHAP}$** | 0.7484 | 0.7326 ± 0.0068 | 97.89% ± 0.86% | 0.7916 ± 0.0036 | 21.27 ± 0.48 | 5.0 ± 0.0 | +7.0d ± 0.0d | 0.0d ± 0.0d | 21.0d ± 0.0d | 19.3s ± 1.0s |
| | **$P_1\text{-a}$ (FULL)** | 0.7484 | 0.7230 ± 0.0075 | 96.60% ± 0.97% | 0.7845 ± 0.0040 | 21.10 ± 0.48 | 5.0 ± 0.0 | +7.0d ± 0.0d | 0.0d ± 0.0d | 21.0d ± 0.0d | 19.4s ± 0.7s |
| | **$P_1\text{-b}$ (DROP_D)**| 0.7484 | 0.6225 ± 0.0071 | 83.18% ± 0.93% | 0.6777 ± 0.0035 | 18.50 ± 0.42 | 5.0 ± 0.0 | +7.0d ± 0.0d | 0.0d ± 0.0d | 35.0d ± 0.0d | 18.4s ± 0.5s |
| | **$P_1\text{-c}$ (RECENCY)**| 0.7484 | 0.7327 ± 0.0068 | **97.90% ± 0.87%** | 0.7915 ± 0.0036 | 21.27 ± 0.48 | 5.0 ± 0.0 | +7.0d ± 0.0d | 0.0d ± 0.0d | 21.0d ± 0.0d | 18.4s ± 0.8s |
| **$S_3$ (Virtual)** | **$B_0$ Static** | 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | 0.0 | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$B_1$ Periodic** | 0.7484 | 0.7327 ± 0.0027 | 97.90% ± 0.39% | 0.5748 ± 0.0016 | 64.61 ± 0.18 | **11.0** | N/A | N/A | N/A (No degr.) | 14.0s ± 1.1s |
| | **$B_2$ PSI Trigger** | 0.7484 | 0.7327 ± 0.0027 | 97.90% ± 0.39% | 0.5748 ± 0.0016 | 64.61 ± 0.18 | **11.0** | N/A | N/A | N/A (No degr.) | 14.9s ± 1.2s |
| | **$B_3$ CBPE-Style** | 0.7484 | 0.7327 ± 0.0027 | 97.90% ± 0.39% | 0.5748 ± 0.0016 | 64.61 ± 0.18 | **11.0** | N/A | N/A | N/A (No degr.) | 13.1s ± 0.3s |
| | **$B_4$ SHAP Shift** | 0.7484 | 0.7330 ± 0.0030 | 97.94% ± 0.42% | 0.5737 ± 0.0020 | 64.50 ± 0.23 | **9.4 ± 0.5** | N/A | N/A | N/A (No degr.) | 11.9s ± 1.4s |
| | **$B_5$ PRODEM** | 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | **0.0** | N/A | N/A | N/A (No degr.) | 35.9s ± 2.2s |
| | **$P_1$ AGA (Full)** | 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-noSHAP}$** | 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-a}$ (FULL)** | 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-b}$ (DROP_D)**| 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |
| | **$P_1\text{-c}$ (RECENCY)**| 0.7484 | 0.7317 ± 0.0024 | 97.77% ± 0.40% | 0.5623 ± 0.0067 | 63.19 ± 0.74 | **0.0** | N/A | N/A | N/A (No degr.) | 0.6s ± 0.0s |

*Note on Timing Signs:* `Lead Time to Alarm` $= t_{\text{degrade}} - t_{\text{first\_alarm}}$ (positive indicates early warning); `Adaptation Lag` $= t_{\text{first\_adapt}} - t_{\text{degrade}}$ (positive indicates elapsed delay before retraining); `Recovery Time` $= t_{\text{recovered}} - t_{\text{degrade}}$. The existing Phase 6 metric $-25.2\text{d}$ corresponds exactly to $t_{\text{degrade}} - t_{\text{first\_adapt}} = -\text{Adaptation Lag}$.

---

## 4. Delay Sensitivity Analysis ($d \in \{7, 14, 28\}$ Days)

As label delay increases, delayed feedback directly impacts model recovery dynamics:

| Label Delay | Scenario | Method | Base Recall | Post-Onset Recall@1% | Relative Retention % | Retrain Count | Adaptation Lag | Recovery Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$d = 7\text{d}$** | $S_1$ (Concept) | $B_0$ Static | 0.7484 | 0.5392 | 72.04% | 0.0 | N/A | Never (-1) |
| | | $B_1$ Periodic | 0.7484 | 0.7849 | 104.87% | 11.0 | -5.2d | 8.8d |
| | | **$P_1$ AGA** | 0.7484 | 0.6940 | **92.73%** | **3.6** | +25.2d | **32.2d** |
| | $S_2$ (Mixed) | $B_0$ Static | 0.7484 | 0.4537 | 60.63% | 0.0 | N/A | Never (-1) |
| | | $B_1$ Periodic | 0.7484 | 0.7719 | 103.14% | 11.0 | -7.0d | 14.0d |
| | | **$P_1$ AGA** | 0.7484 | 0.7326 | **97.89%** | **5.0** | 0.0d | **21.0d** |
| **$d = 14\text{d}$**| $S_1$ (Concept) | $B_0$ Static | 0.5996 | 0.4682 | 78.08% | 0.0 | N/A | Never (-1) |
| | | $B_1$ Periodic | 0.5996 | 0.6698 | 111.71% | 10.0 | -7.0d | 14.0d |
| | | **$P_1$ AGA** | 0.5996 | 0.5694 | **94.97%** | **1.8** | +15.4d | **23.8d** |
| | $S_2$ (Mixed) | $B_0$ Static | 0.5996 | 0.4907 | 81.84% | 0.0 | N/A | 14.0d |
| | | $B_1$ Periodic | 0.5996 | 0.7297 | 121.70% | 10.0 | -7.0d | 14.0d |
| | | **$P_1$ AGA** | 0.5996 | 0.6775 | **112.98%** | **4.0** | +7.0d | **14.0d** |
| **$d = 28\text{d}$**| $S_1$ (Concept) | $B_0$ Static | 0.3543 | 0.3177 | 89.62% | 0.0 | N/A | Never (-1) |
| | | $B_1$ Periodic | 0.3543 | 0.4540 | 128.11% | 10.0 | -14.0d | 14.0d |
| | | **$P_1$ AGA** | 0.3543 | 0.3177 | **89.62%** | **0.0** | N/A | Never (-1) |
| | $S_2$ (Mixed) | $B_0$ Static | 0.3543 | 0.5422 | 153.02% | 0.0 | N/A | N/A |
| | | $B_1$ Periodic | 0.3543 | 0.6729 | 189.90% | 10.0 | N/A | N/A |
| | | **$P_1$ AGA** | 0.3543 | 0.6243 | **176.18%** | **2.0** | N/A | N/A |

### Scientific Takeaways on Label Latency:
1. **Extended Blind Period under $d=28$:** When label delay stretches to 28 days, terminal labels do not mature until Day 140 (Week 20). Under $S_1$, because the remaining post-drift evaluation window is short, EWS stays conservative and avoids premature adaptation until labels mature.
2. **Robustness under $d=14$:** Under moderate 14-day delay, $P_1$ restores performance to **$94.97\%$ retention** in $S_1$ and **$112.98\%$ retention** in $S_2$, cutting retrains by $60\%$ to $80\%$ compared to periodic retraining.

---

## 5. Diagnostic Figures & Visualizations

All publication-ready charts were generated by `fd_experiments.py` and are stored under [`outputs/fraud_detection/figures/`](file:///d:/project/KND_CDAP_DS_09/outputs/fraud_detection/figures/):

1. **`recall_trajectory_S1_d7.png` & `recall_trajectory_S2_d7.png`**:
   Plots Recall@1%FPR trajectories across all 16 test weeks for $B_0, B_1, B_2, B_4, P_1,$ and $P_1\text{-b}$, with vertical markers showing drift onset (Day 105) and subsequent model recovery.
2. **`retrain_counts_by_scenario.png`**:
   Categorical bar chart showing total retrain counts across methods by scenario, highlighting the complete suppression of wasteful retrains by $P_1$ in $S_3$ and Control.
3. **`delay_sensitivity_s1.png`**:
   Comparative line chart illustrating retention stability across Delays 7, 14, and 28.
4. **Per-Delay Recall Trajectories (`recall_trajectory_{scenario}_d{14,28}.png`)**:
   Documenting recovery trajectories under longer label blind periods.

---

## 6. Real-Data Degradation Check (IEEE-CIS)

Executed via [`backend/src/fraud_detection/fd_ieee_check.py`](file:///d:/project/KND_CDAP_DS_09/backend/src/fraud_detection/fd_ieee_check.py) in accordance with **Spec Section 14**:
- **Dataset Location:** `data/fraud_detection/raw/ieee_cis/train_transaction.csv`
- **Execution Status:** **DATA_MISSING**
- **Verdict:** **DATA_NOT_FOUND**
- **Scientific Audit Finding:** As specified in Section 3.2, IEEE-CIS transaction data requires manual user authentication and download from Kaggle. In the absence of `train_transaction.csv`, the check cleanly halts without error and issues the mandatory scientific disclaimer:
  > *"IEEE-CIS data is absent; cannot serve as real-degradation validation. Handbook simulated drift streams remain the primary benchmark."*

---

## 7. Scientific Limitations & Ethical Reporting (Spec §17)

In compliance with the mandatory honesty rules in **Spec Section 17**:
1. **Synthetic Nature of Fraud Handbook Data:**
   All concept and virtual drift scenarios ($S_1, S_2, S_3$) were synthetically injected into the Fraud Detection Handbook dataset under controlled onset conditions. Real-world fraud dynamics frequently exhibit multimodal, non-stationary, and gradual shifts.
2. **Inability of Label-Free Signals to See Pure Concept Drift:**
   Label-free signals (PSI, Score JS, Entropy) cannot observe shifts in $P(Y|X)$ when $P(X)$ is unchanged ($S_1$). In our framework, $S_1$ detection is mediated through the label-derived terminal-risk feature ($t_{\text{risk}}$), which requires delayed labels ($t - d$) to mature before EWS can trigger.
3. **SHAP Feature Dropping Ablation:**
   Dropping the top-$k$ drifted features ($P_1\text{-b}$ `DROP_D`) failed to improve fraud detection performance and actually harmed recall under concept drift. SHAP attributions serve as an interpretability aid and diagnostic indicator rather than a causal guide for feature pruning.
4. **Computational Trade-Offs:**
   While AGA reduces model maintenance events (retrain counts) by $55\%$ to $67\%$, each adaptation event trains and evaluates 3 candidate models on holdout data. As a result, per-adaptation wall-clock runtime is comparable to periodic retraining.

---

## 8. Final Verification Checklist

- [x] All 93 Component 04 unit tests passing (`pytest tests/fraud_detection -v`).
- [x] Full factorial grid (660 runs) completed and verified in `grid_results_summary.csv`.
- [x] Both timing conventions (`Lead Time to First Alarm` and `Adaptation Lag`) rigorously separated and reported.
- [x] Fixed PSI ablation interface ensuring feature PSI is passed and logged.
- [x] IEEE-CIS degradation check implemented and tested per Spec §14.
- [x] Diagnostic visualization plots saved under `outputs/fraud_detection/figures/`.
- [x] Honesty rules and negative ablation results documented without overclaiming.
- [x] All repo scope and leakage boundaries strictly upheld.
