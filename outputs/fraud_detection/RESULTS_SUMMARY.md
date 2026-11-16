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
- **Unit Test Coverage:** 101 passed unit tests in [`tests/fraud_detection/`](file:///d:/project/KND_CDAP_DS_09/tests/fraud_detection/)

---

## 1. Executive Summary & Contributions

In production credit card fraud detection, transaction labels arrive with significant latency (7 to 28 days). During this "blind period," conventional performance metrics cannot detect silent model degradation. This project implements and evaluates a framework combining **label-free drift monitoring**, a calibrated **Early-Warning Score (EWS)** decision policy, and **Attribution-Guided Adaptation (AGA)** under temporal label-maturity constraints ($\text{label\_day} \le t - d$).

### Key Empirical Findings:
1. **Suppression of False Alarms on Benign Covariate Shift ($S_3$):**
   Conventional drift monitoring triggers like feature PSI ($B_2$) and confidence estimation ($B_3$) trigger **11.0 wasteful retrains** under benign 6-hour transaction time shifts ($S_3$). By integrating multidimensional signals (Feature PSI, Model-Score JS Divergence, Binary Prediction Entropy, and TreeSHAP Profile Shift) into a calibrated logistic EWS, **$P_1$ suppresses false alarms (0 retrains, 0 false alarms)**, matching unperturbed baseline performance.
2. **Early-Warning Lead Time on Mixed Drift ($S_2$):**
   On Scenario $S_2$ (Mixed Drift) under delay $d=7$ days, $P_1$ achieves a positive lead time across all 5 evaluation seeds of **$+8.4\text{d} \pm 3.1\text{d}$** (recomputed directly from `grid_results_summary.csv`: Seed 1: degrade day 119, alarm day 105 $\to +14.0\text{d}$; Seeds 2–5: degrade day 119, alarm day 112 $\to +7.0\text{d}$). This metric is subject to 7-day evaluation window discretization error.
3. **Retrain Count Reduction vs. Periodic Retraining ($B_1$):**
   $P_1$ substantially reduces retrain count ($3.6 \pm 0.5$ in $S_1$, $5.0 \pm 0.0$ in $S_2$ vs. 11.0 for weekly periodic retraining $B_1$), but **$P_1$ does NOT beat periodic retraining ($B_1$) on post-onset recall**. When matured labels are available, periodic retraining consistently maintains higher raw recall (in $S_1$: $B_1$ 0.7849 vs. $P_1$ 0.6940; in $S_2$: $B_1$ 0.7719 vs. $P_1$ 0.7326). $P_1$'s primary benefit is maintenance reduction on benign shifts ($S_3$ and Control), not higher recall under active drift.
4. **TreeSHAP Candidate Selection Null Result ($P_1$ vs. $P_1\text{-noSHAP}$):**
   In 58 of 60 evaluation streams, candidate model selection was identical between $P_1$ and $P_1\text{-noSHAP}$, and `DROP_D` was chosen 0 times by $P_1$. **TreeSHAP profile shift showed no measurable benefit for candidate model selection in this benchmark.** Dropping drifted features ($P_1\text{-b}$ `DROP_D`) severely degraded recall ($75.3\%$ retention in $S_1$), confirming that drifted features contain essential fraud patterns.
5. **Delayed-Feedback Reality on Pure Concept Drift ($S_1$) & $d=28$ Limitations:**
   Label-free signals cannot observe pure concept drift ($P(Y|X)$ shifting while $P(X)$ is identical) without matured labels. In our framework, $S_1$ detection is mediated through the label-derived terminal-risk feature ($t_{\text{risk}}$). Under $d=28$, feature lookback truncation during days 35..55 collapses $t_{\text{risk}}$ feature importance by $97\%$, reducing base recall from $0.748$ ($d=7$) to $0.354$ ($d=28$), and delayed label arrival prevents EWS from reaching the adaptation threshold (0 retrains).

---

## 2. Methodology & Non-Leakage Enforcement

### 2.1 The Non-Negotiable Leakage Rule
At any decision time $t$ with label delay $d \in \{7, 14, 28\}$:
$$\text{Available Labels} \iff \text{label\_day} \le t - d$$
- **Feature Engineering:** Terminal risk features $t_{\text{risk}}$ and $t_{\text{vol}}$ aggregate transaction labels strictly from window $[D - d - 28, \, D - d]$. Customer rolling features ($c_{\text{mean}}$, $c_{\text{cnt}}$) use strictly closed-left historical intervals.
- **Candidate Retraining:** Partitioned across the last 28 matured days $[t - d - 27, \, t - d]$ into a 21-day candidate training set $[t - d - 27, \, t - d - 7]$ and a 7-day holdout validation set $[t - d - 6, \, t - d]$.
- **Zero Future Exposure:** Non-leakage is tested by unit test `test_aga_strictly_obeys_matured_label_boundary`, verifying that arbitrary alterations to unmatured labels have zero impact on candidate models or selection scores.

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
| | **$P_1$ AGA (Full)** | 0.7484 | 0.7326 ± 0.0068 | **97.89% ± 0.86%** | 0.7916 ± 0.0036 | 21.27 ± 0.48 | **5.0 ± 0.0** | **+8.4d ± 3.1d** | **0.0d ± 0.0d** | **21.0d ± 0.0d** | 19.3s ± 0.9s |
| | **$P_1\text{-noSHAP}$** | 0.7484 | 0.7326 ± 0.0068 | 97.89% ± 0.86% | 0.7916 ± 0.0036 | 21.27 ± 0.48 | 5.0 ± 0.0 | +8.4d ± 3.1d | 0.0d ± 0.0d | 21.0d ± 0.0d | 19.3s ± 1.0s |
| | **$P_1\text{-a}$ (FULL)** | 0.7484 | 0.7230 ± 0.0075 | 96.60% ± 0.97% | 0.7845 ± 0.0040 | 21.10 ± 0.48 | 5.0 ± 0.0 | +8.4d ± 3.1d | 0.0d ± 0.0d | 21.0d ± 0.0d | 19.4s ± 0.7s |
| | **$P_1\text{-b}$ (DROP_D)**| 0.7484 | 0.6225 ± 0.0071 | 83.18% ± 0.93% | 0.6777 ± 0.0035 | 18.50 ± 0.42 | 5.0 ± 0.0 | +8.4d ± 3.1d | 0.0d ± 0.0d | 35.0d ± 0.0d | 18.4s ± 0.5s |
| | **$P_1\text{-c}$ (RECENCY)**| 0.7484 | 0.7327 ± 0.0068 | **97.90% ± 0.87%** | 0.7915 ± 0.0036 | 21.27 ± 0.48 | 5.0 ± 0.0 | +8.4d ± 3.1d | 0.0d ± 0.0d | 21.0d ± 0.0d | 18.4s ± 0.8s |
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

*Note on Timing Signs & Discretization:* `Lead Time to Alarm` $= t_{\text{degrade}} - t_{\text{first\_alarm}}$ (positive indicates early warning); `Adaptation Lag` $= t_{\text{first\_adapt}} - t_{\text{degrade}}$ (positive indicates elapsed delay before retraining); `Recovery Time` $= t_{\text{recovered}} - t_{\text{degrade}}$. In $S_2$ ($d=7\text{d}$), ground-truth degradation manifested uniformly at Day 119 across all 5 seeds in `grid_results_summary.csv`, while the early warning alarm fired at Day 105 in Seed 1 ($+14.0\text{d}$ lead time) and Day 112 in Seeds 2–5 ($+7.0\text{d}$ lead time), producing an empirical sample mean lead time of $+8.4\text{d} \pm 3.1\text{d}$ subject to 7-day evaluation window discretization.

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
| **$d = 14\text{d}$**| $S_1$ (Concept) | $B_0$ Static | 0.5996 | 0.4375 | 72.97% | 0.0 | N/A | Never (-1) |
| | | $B_1$ Periodic | 0.5996 | 0.6558 | 109.37% | 10.0 | -7.0d | 14.0d |
| | | **$P_1$ AGA** | 0.5996 | 0.4375 | **72.97%** | **0.0** | N/A | Never (-1) |
| | $S_2$ (Mixed) | $B_0$ Static | 0.5996 | 0.4907 | 81.84% | 0.0 | N/A | 14.0d |
| | | $B_1$ Periodic | 0.5996 | 0.7297 | 121.70% | 10.0 | -7.0d | 14.0d |
| | | **$P_1$ AGA** | 0.5996 | 0.6775 | **112.98%** | **4.0** | +7.0d | **14.0d** |
| **$d = 28\text{d}$**| $S_1$ (Concept) | $B_0$ Static | 0.3543 | 0.3177 | 89.67% | 0.0 | N/A | Never (-1) |
| | | $B_1$ Periodic | 0.3543 | 0.4540 | 128.14% | 10.0 | -14.0d | 14.0d |
| | | **$P_1$ AGA** | 0.3543 | 0.3177 | **89.67%** | **0.0** | N/A | Never (-1) |
| | $S_2$ (Mixed) | $B_0$ Static | 0.3543 | 0.5422 | 153.03% | 0.0 | N/A | N/A |
| | | $B_1$ Periodic | 0.3543 | 0.6729 | 189.92% | 10.0 | N/A | N/A |
| | | **$P_1$ AGA** | 0.3543 | 0.6243 | **176.21%** | **2.0** | N/A | N/A |

### Scientific Takeaways on Label Latency & Feature Dynamics:
1. **Lookback Truncation Confound at Extended Delays ($d=28$):**
   Baseline pre-drift recall drops systematically with label delay: $0.7484$ ($d=7$) $\to 0.5996$ ($d=14$) $\to 0.3543$ ($d=28$). Audit analysis reveals this is driven by the 28-day label maturity boundary: during training days 35..55, the feature lookback $[D - d - 28, \, D - d]$ for $d=28$ has zero matured labels prior to day 0. This truncates available terminal history, collapsing $t_{\text{risk}}$ feature importance by $97\%$ and substantially reducing baseline discriminatory power.
2. **Delayed EWS Trigger Dynamics:**
   In pure concept drift ($S_1$), drift detection is mediated entirely through the label-derived $t_{\text{risk}}$ feature. Under $d=14$ and $d=28$, label arrival latency postpones $t_{\text{risk}}$ divergence; consequently, EWS remains below the adaptation threshold ($\theta_2 = 0.35$) over the remaining test horizon (0 retrains). In mixed drift ($S_2$), where transaction covariates and score distributions shift simultaneously, EWS detects drift and triggers targeted adaptation even under delayed feedback ($4.0$ retrains at $d=14$, $2.0$ retrains at $d=28$).
3. **Degradation Sensitivity & Noise Suppression (1-Window vs. 2-Window Rule):**
   Under a strict 1-window degradation rule ($\text{Recall} < 0.80 \times \text{Base}$), single-window sample variance (e.g., 49 frauds in a 14-day window under $0.38\%$ prevalence) can trigger transient dips (as observed in $S_1$ $d=28$ at Day 70). Implementing a 2-consecutive-window persistence criterion successfully filters out these stochastic single-window transients, confirming that degradation protocol definitions must account for finite-sample binomial variance in low-prevalence fraud streams.

---

## 5. Diagnostic Figures & Visualizations

Diagnostic charts were generated by `fd_experiments.py` and are stored under [`outputs/fraud_detection/figures/`](file:///d:/project/KND_CDAP_DS_09/outputs/fraud_detection/figures/):

1. **`recall_trajectory_S1_d7.png` & `recall_trajectory_S2_d7.png`**:
   Plots Recall@1%FPR trajectories across all 16 test weeks for $B_0, B_1, B_2, B_4, P_1,$ and $P_1\text{-b}$, with vertical markers showing drift onset (Day 105) and subsequent model recovery.
2. **`retrain_counts_by_scenario.png`**:
   Categorical bar chart showing total retrain counts across methods by scenario, showing retrain suppression by $P_1$ on $S_3$ and Control.
3. **`delay_sensitivity_s1.png`**:
   Comparative line chart illustrating retention stability across Delays 7, 14, and 28.
4. **Per-Delay Recall Trajectories (`recall_trajectory_{scenario}_d{14,28}.png`)**:
   Documenting recovery trajectories under longer label blind periods.

---

## 6. Empirical Stream Evaluation (IEEE-CIS Status & Sparkov Fallback)

### 6.1 IEEE-CIS Protocol Check (Spec Section 14)
- **Phase 9 Status:** **IEEE-CIS check not performed (data unavailable); Sparkov fallback only**
- **Dataset File:** `data/fraud_detection/raw/ieee_cis/train_transaction.csv`
- **Execution Status:** `DATA_MISSING` / `DATA_NOT_FOUND`
- **Honesty Disclosure:** Real-data validation was **NOT performed** because `train_transaction.csv` was unavailable (requires manual user authentication and Kaggle competition rules acceptance). As specified in Section 3.2 and Section 14 of the specification, IEEE-CIS cannot serve as real-degradation validation in this project.

### 6.2 Sparkov Dataset Fallback Evaluation (Spec §14 & §17)
- **Dataset Classification:** **SYNTHETIC DATA** (NOT real data, and NOT part of the original specification).
- **Origin & Acquisition:** Acquired as a practical fallback from Kaggle ([`kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection)) as provided by the user. The file [`data/fraud_detection/raw/sparkov/fraudTest.csv`](file:///d:/project/KND_CDAP_DS_09/data/fraud_detection/raw/sparkov/fraudTest.csv) (150.3 MB, 555,719 rows, 2,145 frauds, $0.386\%$ fraud prevalence, spanning 194 days) was generated using the open-source Sparkov synthetic credit card transaction simulator.
- **Evaluation Protocol:** Base LightGBM trained on days $< 42$ (119,597 transactions, 466 frauds). Evaluated across 10 rolling 14-day evaluation windows spanning days 49..188.

#### Rolling 14-Day Performance Check (`sparkov_degradation_check.csv`):

| Window | Day Range | Total Tx | Frauds | Prevalence % | Recall@1% FPR | Rel. to Window 0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Days 49..62 | 39,359 | 141 | 0.358% | 0.7092 | 100.0% (Baseline) |
| 1 | Days 63..76 | 37,686 | 186 | 0.494% | 0.8495 | 119.8% |
| 2 | Days 77..90 | 32,267 | 130 | 0.403% | 0.9154 | 129.1% |
| 3 | Days 91..104 | 31,902 | 171 | 0.536% | 0.7427 | 104.7% |
| 4 | Days 105..118 | 31,875 | 232 | 0.728% | 0.7284 | 102.7% |
| 5 | Days 119..132 | 31,871 | 126 | 0.395% | 0.8810 | 124.2% |
| 6 | Days 133..146 | 32,019 | 129 | 0.403% | 0.8372 | 118.0% |
| 7 | Days 147..160 | 31,445 | 154 | 0.490% | 0.8896 | 125.4% |
| 8 | Days 161..174 | 62,192 | 171 | 0.275% | 0.7602 | 107.2% |
| 9 | Days 175..188 | 63,388 | 87 | 0.137% | 0.8851 | 124.8% |

- **Degradation Verdict:** **`False`** (Recall@1% FPR remained between $70.9\%$ and $91.5\%$, mean $81.98\%$; zero windows dropped below $80\%$ of baseline). The dataset contains **no real performance degradation**.

#### Label-Free Monitoring Signals on Sparkov Stream (`sparkov_signals_check.csv`):

| Window | Day Range | Mean PSI | Max PSI | Score JS | Relative Entropy | EWS Score | Detector Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Days 49..62 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1644 | **WAIT** |
| 1 | Days 63..76 | 0.0008 | 0.0026 | 0.0060 | 0.0099 | 0.1578 | **WAIT** |
| 2 | Days 77..90 | 0.0006 | 0.0011 | 0.0073 | 0.1336 | 0.1405 | **WAIT** |
| 3 | Days 91..104 | 0.0005 | 0.0010 | 0.0083 | 0.1395 | 0.1420 | **WAIT** |
| 4 | Days 105..118 | 0.0005 | 0.0012 | 0.0116 | 0.1277 | 0.1507 | **WAIT** |
| 5 | Days 119..132 | 0.0006 | 0.0012 | 0.0084 | 0.1419 | 0.1408 | **WAIT** |
| 6 | Days 133..146 | 0.0006 | 0.0014 | 0.0111 | 0.0596 | 0.1625 | **WAIT** |
| 7 | Days 147..160 | 0.0005 | 0.0010 | 0.0153 | 0.0636 | 0.1741 | **WAIT** |
| 8 | Days 161..174 | 0.0008 | 0.0015 | 0.0118 | 0.0301 | 0.1680 | **WAIT** |
| 9 | Days 175..188 | 0.0004 | 0.0008 | 0.0089 | 0.0618 | 0.1626 | **WAIT** |

- **Critical Limitation on False Alarms:** Across all 10 windows, EWS remained between $0.140$ and $0.174$ ($< \theta_1 = 0.22$), producing $100\%$ WAIT decisions (0 false alarms, 0 retrains). **However, "0 false alarms" is NOT evidence of robustness because no degradation exists in this synthetic stream.** On a stationary synthetic stream without drift, a detector staying quiet does not prove it can handle real-world distribution shifts or maintain adequate sensitivity under complex non-stationary environments.

---

## 7. Scientific Limitations & Ethical Reporting (Spec §17)

In compliance with the mandatory honesty rules in **Spec Section 17**:
1. **Synthetic Nature of Benchmarks:**
   All concept and virtual drift scenarios ($S_1, S_2, S_3$) in the primary benchmark were synthetically injected into the Fraud Detection Handbook dataset under controlled onset conditions. Similarly, the Sparkov fallback stream represents synthetic transaction simulation data rather than bank-internal telemetry. Real-world fraud dynamics frequently exhibit multimodal, non-stationary, and gradual shifts.
2. **Inability of Label-Free Signals to Directly Detect Pure Concept Drift ($S_1$):**
   Label-free signals (PSI, Score JS, Entropy) cannot observe shifts in $P(Y|X)$ when $P(X)$ is unchanged ($S_1$). In our framework, $S_1$ detection is mediated through the label-derived terminal-risk feature ($t_{\text{risk}}$), which requires delayed labels ($t - d$) to mature before EWS can trigger.
3. **SHAP Feature Dropping Ablation & Null Selection Result:**
   Dropping top-$k$ drifted features ($P_1\text{-b}$ `DROP_D`) severely harmed recall under concept drift ($75.3\%$ retention in $S_1$), because drifted features often contain the strongest fraud signals. Furthermore, in 58 of 60 evaluation streams, candidate model selection was identical between $P_1$ and $P_1\text{-noSHAP}$ (`DROP_D` was chosen 0 times by $P_1$). SHAP attributions act as an interpretability aid and diagnostic indicator rather than an active driver of candidate selection under delayed feedback.
4. **CBPE Calibration under Class Imbalance ($B_3$ Baseline):**
   Standard CBPE estimation on uncalibrated probabilities under heavy class weighting (`scale_pos_weight=20`) produces probability scale distortion (~$0.256$ estimate vs. ~$0.598$ empirical recall). Comparing uncalibrated estimates directly against empirical baseline recall causes continuous retraining ($11/11$ weeks). Calibrating CBPE against its own baseline estimate resolves this artifact.
5. **Computational Trade-Offs:**
   While AGA reduces model maintenance events (retrain counts) by $55\%$ to $67\%$, each adaptation event trains and evaluates 3 candidate models on holdout data. As a result, per-adaptation wall-clock runtime is comparable to periodic retraining.
6. **Dual Recovery Timing Metrics:**
   Recovery time is formally reported using two distinct, unambiguous metrics:
   - **Degradation MTTR:** $t_{\text{recovered}} - t_{\text{degrade}}$ (measuring total outage duration from degradation onset).
   - **Post-Adaptation Recovery:** $t_{\text{recovered}} - t_{\text{first\_adapt}}$ (measuring model turnaround time once adaptation is deployed).

---

## 8. What is NOT Validated

To maintain complete scientific honesty and prevent overinterpretation, the following aspects were **NOT validated** in this study:
1. **Real Data:** No evaluation on genuine, unanonymized real-world banking telemetry or unperturbed real-world concept drift was performed. Both the Fraud Detection Handbook and the Sparkov fallback are synthetic datasets.
2. **Human-Reviewed Alerts:** No operational simulation of alert triage capacity, investigator review fatigue, alert queue backlog, or financial costs of false alarms under production volumes was conducted.
3. **Production Deployment:** No live streaming infrastructure, distributed latency testing, online asynchronous inference, concurrent drift handling, or database lock performance was tested.
4. **Significance Tests:** Formal statistical significance testing (e.g., paired t-tests or Wilcoxon signed-rank tests across evaluation streams) was not conducted; all reported values represent empirical sample means and sample standard deviations across 5 random seeds.

---

## 9. Final Verification Checklist

- [x] All 101 Component 04 unit tests passing (`pytest tests/fraud_detection -q`).
- [x] Full factorial grid (660 runs) completed and checked in `grid_results_summary.csv`.
- [x] Both timing conventions (`Lead Time to First Alarm` and `Adaptation Lag`) separated and reported.
- [x] Fixed PSI ablation interface ensuring feature PSI is passed and logged.
- [x] IEEE-CIS data access restrictions handled per Spec §14 (`DATA_MISSING` reported plainly).
- [x] Sparkov fallback acquired, evaluated, and clearly labeled as synthetic data without degradation.
- [x] Diagnostic visualization plots saved under `outputs/fraud_detection/figures/`.
- [x] Honesty rules, null results, and "What is NOT Validated" limitations fully documented without overclaiming.
- [x] Repository scope boundaries and non-leakage temporal constraints tested by unit assertions.

