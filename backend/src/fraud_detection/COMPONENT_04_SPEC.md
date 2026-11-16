# COMPONENT 04 SPEC: Fraud Detection

## 1. Goal (what the code must do)

Fraud labels arrive 7 to 28 days after a transaction (label delay). During this "blind period" a
deployed fraud model can degrade without anyone knowing. This component:

1. Monitors the deployed model WITHOUT labels (PSI, score shift, entropy, SHAP profile shift).
2. Combines signals into an early-warning score and a decision: WAIT / MONITOR / ADAPT.
3. On ADAPT, runs AGA (Attribution-Guided Adaptation), using ONLY labels that have matured at that time.
4. Evaluates all of this on public data against baselines, by drift type and label delay.

### Claims policy (important for comments, docstrings, reports)
- Do NOT claim a new drift-detection algorithm, "world first", or that no prior work exists.
- Contribution = integration + controlled decision-level evaluation + drift-type detectability analysis.
- SHAP is a "look here" hint, not causal proof. SHAP benefit must be tested by ablation, and a null result is acceptable.
- Prior work to be aware of: SAMM (2019), PRODEM (2025), Explanation Shift (2023), Solozobov (2026),
  Dal Pozzolo et al. (IEEE 2015/2018), SHAP-guided adaptive ensemble (2026).

## 2. Repo scope
Write only under: backend/src/fraud_detection/, data/fraud_detection/, tests/fraud_detection/,
models/fraud_detection/, outputs/fraud_detection/. Everything else is read-only.
Extra dependencies go in backend/src/fraud_detection/requirements-fd.txt (never root requirements).

## 3. Data

### 3.1 Main: Fraud Detection Handbook (simulated)
- Location: data/fraud_detection/raw/handbook/*.pkl (183 daily files, 2018-04-01 onward)
- Columns (9): TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT,
  TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD, TX_FRAUD_SCENARIO
- Verified expectations after concatenating ALL files:
  rows = 1,754,155 · TX_TIME_DAYS = 0..182 · TX_FRAUD sum = 14,681 (0.84%)
  TX_FRAUD_SCENARIO counts: 0 -> 1,739,474 · 1 -> 973 · 2 -> 9,077 · 3 -> 4,631
- The loader MUST fail loudly (clear error) if these checks do not match, because a single daily file
  (about 9,500 rows) is not enough for this project.
- The default data has NO meaningful concept drift (static model stays stable for 16 weeks).
  Drift must be injected (Section 8).
- NEVER feed TX_FRAUD_SCENARIO, TX_TIME_DAYS, TX_TIME_SECONDS, TRANSACTION_ID, CUSTOMER_ID or TERMINAL_ID
  directly into the model as features. SCENARIO is analysis-only (leakage).

### 3.2 Real-data validation: IEEE-CIS
- User downloads manually from Kaggle (login required): put train_transaction.csv in
  data/fraud_detection/raw/ieee_cis/. Do NOT attempt to download it. Use train_transaction.csv only
  (test labels are not public). TransactionDT is a time delta in seconds, not a calendar date.
- See Section 14.

### 3.3 Optional backup: Bank Account Fraud (BAF)
- Account-opening fraud, a different domain. Only add a loader stub; implement only if asked.

## 4. Module map (backend/src/fraud_detection/)

Existing stub files (read first, extend, do not overwrite non-empty code):
- fd_data_loader.py        load + validate Handbook; IEEE-CIS loader
- fd_preprocessing.py      sorting, windows/splits, label-delay simulation, drift injection hooks
- fd_feature_engineering.py leakage-safe features
- fd_training.py           LightGBM training, retrain helpers
- fd_monitoring.py         label-free signals: PSI, score JS, entropy
- fd_explainability.py     SHAP profile + shift
- fd_detection.py          early-warning score + WAIT/MONITOR/ADAPT decision
- fd_evaluation.py         metrics, lead time, baselines, ablations

New files to add:
- fd_config.py             all constants and dataclasses (single source of truth)
- fd_drift_injection.py    S1, S2, S3 drift generators
- fd_adaptation.py         AGA candidates + selection
- fd_baselines.py          B0..B5
- fd_experiments.py        experiment grid runner + CLI (python -m ... )
- fd_ieee_check.py         IEEE-CIS degradation check
- COMPONENT_04_SPEC.md     this file

## 5. Time design and the leakage rule

Parameters (in fd_config.py): DELAY d in {7, 14, 28}; window = 7 days (also support 3 days);
TRAIN_START = 35, TRAIN_END = 62; WARMUP = days 0..34 (feature history only);
REFERENCE = 21 days right after the gap (trusted, pre-drift); DRIFT_ONSET configurable.

LEAKAGE RULE (non-negotiable): at decision time t, any label used (for training, calibration, terminal
risk features, adaptation selection) must satisfy label_day <= t - d. Implement it in ONE helper
(e.g. `matured_mask(df, t, d)`) and use it everywhere. Write tests that prove it.

Splits for d = 7 (verified setup): train days 35..62 -> gap 63..69 -> test weeks 10..25 (days 70..181);
reference weeks 10..12 (days 70..90); drift onset day 105 (start of week 15).
For other delays: test_start = ceil((63 + d) / 7) * 7, drift_onset = 112, reference = 21 days from test_start.

## 6. Feature engineering (leakage-safe)

Model features (exactly these to start):
TX_AMOUNT, hour, dow, amt_ratio, t_risk, t_vol, and for w in (1D, 7D, 30D): c_mean_w, c_cnt_w.

- hour, dow from TX_DATETIME.
- Customer rolling mean/count over the PREVIOUS w of the same customer (closed='left', never include
  the current transaction).
- amt_ratio = TX_AMOUNT / (c_mean_30D.fillna(TX_AMOUNT) + 1e-6)
- Terminal risk uses LABELS, so it must be delayed:
  for each day D, over days [D - d - 28, D - d]: t_vol = transactions, t_risk = frauds / transactions.

Reference implementation (verified; vectorize, do not loop over rows):

```python
tmp = df[["CUSTOMER_ID","TX_DATETIME","TX_AMOUNT"]].copy(); tmp["rid"] = df.index
tmp = tmp.sort_values(["CUSTOMER_ID","TX_DATETIME"])
for w in ["1D","7D","30D"]:
    r = tmp.set_index("TX_DATETIME").groupby("CUSTOMER_ID")["TX_AMOUNT"].rolling(w, closed="left")
    tmp[f"c_mean_{w}"] = r.mean().values; tmp[f"c_cnt_{w}"] = r.count().values
for c in [c for c in tmp.columns if c.startswith("c_")]:
    df.loc[tmp.rid.values, c] = tmp[c].values

nd, nt = int(df.TX_TIME_DAYS.max()) + 1, int(df.TERMINAL_ID.max()) + 1
tx = np.zeros((nt, nd)); fr = np.zeros((nt, nd))
np.add.at(tx, (df.TERMINAL_ID.values, df.TX_TIME_DAYS.values), 1)
np.add.at(fr, (df.TERMINAL_ID.values, df.TX_TIME_DAYS.values), df.TX_FRAUD.values)
ctx, cfr = tx.cumsum(1), fr.cumsum(1)
cum = lambda a, d: a[:, d] if d >= 0 else np.zeros(a.shape[0])
risk = np.zeros((nt, nd)); vol = np.zeros((nt, nd))
for D in range(nd):                      # DELAY, WIN = d, 28
    hi, lo = D - DELAY, D - DELAY - WIN
    f = cum(cfr, hi) - cum(cfr, lo); t = cum(ctx, hi) - cum(ctx, lo)
    vol[:, D] = t; risk[:, D] = np.where(t > 0, f / np.maximum(t, 1), 0)
df["t_risk"] = risk[df.TERMINAL_ID.values, df.TX_TIME_DAYS.values]
df["t_vol"]  = vol [df.TERMINAL_ID.values, df.TX_TIME_DAYS.values]
```

IMPORTANT: after ANY drift injection (amounts, times or labels change), ALL features must be rebuilt.
Recompute TX_TIME_DAYS from TX_DATETIME after any time shift.

## 7. Baseline model and metrics

- Model: LightGBM binary, learning_rate 0.05, num_leaves 31, n_estimators 200, subsample 0.8
  (subsample_freq 1), colsample_bytree 0.8, scale_pos_weight 20, random_state = seed, n_jobs configurable.
- B0 = train once on days 35..62, never retrain, evaluate weekly.
- Primary metric: recall at 1% FPR per window:
  thr = quantile(p[y==0], 0.99); recall = mean(p[y==1] > thr).
- Also report average precision (AP) AND AP / prevalence. NEVER judge degradation by AP alone
  (AP rises mechanically when the fraud rate rises).
- "True degradation" (post-hoc, evaluation only, never visible to the method): the first window where
  recall@1%FPR falls below 80% of its pre-drift mean (mean of weeks before onset after the gap).

## 8. Drift injection (fd_drift_injection.py)

Each generator takes (raw_df, onset_day, seed) and returns a NEW dataframe (never mutate input).
Rebuild features afterwards. Scenarios: control, S1, S2, S3.

- S1 CONCEPT (label rule change, inputs unchanged):
  rows with day >= onset, TX_AMOUNT > 180 and TX_FRAUD == 0  ->  TX_FRAUD = 1, TX_FRAUD_SCENARIO = 1.
- S2 MIXED (new fraud pattern, inputs change for fraud):
  each day D >= onset pick 12 customers (rng with seed) and mark them compromised for days [D, D+14);
  their transactions in that period get TX_AMOUNT *= 3, TX_FRAUD = 1, TX_FRAUD_SCENARIO = 4.
- S3 VIRTUAL COVARIATE (inputs shift, labels untouched):
  pick 50% of customers (seed); for rows with day >= onset shift TX_DATETIME by +6 hours.
  Recompute TX_TIME_DAYS; re-sort by TX_DATETIME.

Invariants to test: S1 changes no input columns; S3 changes no labels; S2 raises the fraud rate to about 4%
after onset; control equals the original data exactly.

## 9. Monitoring signals (label-free; fd_monitoring.py, fd_explainability.py)

Computed per window against a trusted REFERENCE window (labels never used):
- PSI per feature: 10 quantile bins from the reference, clip window values to reference range,
  fill NaN with -1, epsilon 1e-6 in bins. Report mean PSI and max PSI.
- Score JS: scipy.spatial.distance.jensenshannon between binned model-score histograms
  (bins = reference quantiles).
- Entropy: mean binary entropy of predicted probabilities (clip 1e-6); report relative change vs reference.
- SHAP profile: shap.TreeExplainer(model); sample up to 10,000 rows of the window (fixed seed) from ALL
  traffic (labels are unavailable); handle both list and ndarray outputs; profile = mean |SHAP| per feature,
  normalized to sum 1; shift = JS distance to the reference profile; per-feature delta = profile_w - profile_ref.

## 10. Early-warning score and decision (fd_detection.py)

- Standardize each signal (z-score vs reference/calibration statistics).
- EWS = logistic regression over signals trained on CALIBRATION streams (separate seeds and onsets from the
  evaluation streams) to predict "recall@1%FPR in the next window is >= 20% below baseline".
  Use only matured labels. Provide a simple weighted-sum fallback with equal weights.
- Thresholds theta1 < theta2 are chosen on calibration streams, then LOCKED (stored in a JSON file) before any
  evaluation stream is touched. Add a test that evaluation code never refits thresholds.
- Decision: EWS < theta1 -> WAIT; theta1 <= EWS < theta2 -> MONITOR (increase monitoring frequency,
  e.g. 3-day windows; no retrain); EWS >= theta2 -> ADAPT (run AGA). Cooldown: no second ADAPT within 14 days.

## 11. AGA: Attribution-Guided Adaptation (fd_adaptation.py)

Input: current model, window with SHAP delta, time t, delay d. Steps:
1. D = top-k features by |delta SHAP| (k configurable, default 3).
2. Build candidate models using ONLY matured labels (label_day <= t - d), last 28 matured days:
   (a) FULL: retrain on all features.
   (b) DROP_D: retrain without features in D.
   (c) RECENCY: retrain on all features with exponentially increasing sample weights for recent days.
3. Select the candidate with the best recall@1%FPR on the most recent matured labeled window
   (hold out the last 7 matured days from training for this selection).
4. Return chosen model + record (which candidate, D, training window, timestamp) for the report.
Also expose each candidate as a standalone policy (P1-a, P1-b, P1-c) for reporting.
SHAP-top-k must be comparable with RANDOM-k and PSI-top-k variants (ablation, Section 12).

## 12. Baselines and ablations (fd_baselines.py)

- B0 never retrain · B1 retrain every 7 days (matured labels) · B2 PSI/KS trigger (conventional drift trigger)
- B3 NannyML-CBPE-style confidence-based performance estimate (implement a simple calibrated version; note that
  CBPE assumes no concept drift, so failure under S1 is expected and must be reported fairly)
- B4 SHAP-shift-only trigger (Explanation-Shift style)
- B5 PRODEM-style: meta-model trained on matured labels to predict the base model's errors (call it
  "PRODEM-style"; the original code/data are not available)
- P1 full method · P1-noSHAP (signals without SHAP, feature selection by PSI) · SHAP-top-k vs random-k vs PSI-top-k.

## 13. Experiment grid and outputs (fd_experiments.py)

Grid: delay in {7, 14, 28} x scenario in {control, S1, S2, S3} x seeds in {1..5}; windows 7 days (sensitivity: 3 days).
Calibration seeds and evaluation seeds must be disjoint.
Report per run: recall@1%FPR, AP, AP/prevalence, true-degradation window, first alarm window per signal and per
method, lead time (in days; negative if late), false-alarm count in no-degradation windows, retrain count,
recovery time (days until recall returns to >= 90% of baseline after ADAPT).
Save under outputs/fraud_detection/: results CSV (long format), per-window signal CSVs, PNG figures
(recall over time with onset/alarm markers, signal trajectories, lead-time bars, retrain-count bars).
Everything must be reproducible from a CLI: `python -m backend.src.fraud_detection.fd_experiments --delay 7 --scenario S1 --seed 1`
(adapt the module path to the repo's convention after reading pyproject/shared).

## 14. IEEE-CIS check (fd_ieee_check.py)

Load train_transaction.csv; day = TransactionDT // 86400; numeric columns only (fillna not needed for LightGBM);
train on days < 42; evaluate 14-day windows starting day 49 with recall@1%FPR. Print the table and a verdict:
degradation present if recall falls >= 20% relative to the first window and stays lower. If not present, print that
IEEE-CIS cannot serve as real-degradation validation and the BAF backup should be considered. Never claim
degradation that the numbers do not show.

## 15. Tests (tests/fraud_detection/) with acceptance criteria

- loader: wrong row count raises a clear error; correct data passes the verified expectations.
- leakage: for random times t, no label with day > t - d influences any feature, training row or selection set.
- features: customer features exclude the current row; terminal features equal the manual computation on a toy case.
- drift invariants (Section 8).
- metrics: recall_at_fpr and PSI on toy data with known answers.
- thresholds locked: evaluation code cannot change them.
- regression on the verified experiment (d = 7, onset 105, seed 1; tolerance: +/- 0.06 absolute on recall,
  +/- 10 percentage points on the relative drop):
  control recall@1%FPR about 0.75 before and after (change within about 5%);
  S1 about 0.75 -> 0.53 (about -29%); S2 about 0.75 -> 0.46 (about -38%); S3 about 0.75 -> 0.735 (about -2%).
  Qualitative: PSI does NOT alarm on S1; PSI alarms strongly on S3 although performance does not drop;
  score/entropy/SHAP signals alarm on S1 and S2 and stay mostly quiet on S3.
  If results differ a lot, investigate leakage or window mistakes before proceeding.

## 16. Engineering requirements

- Python 3.10+; type hints; docstrings; `logging` (no print in library code); pathlib; no hard-coded absolute paths.
- All constants in fd_config.py; seeds always explicit; deterministic where possible.
- Memory/time: 1.75M rows. Use float32 where safe, vectorized pandas/numpy, no per-row Python loops,
  cache processed data to data/fraud_detection/processed/ (gitignored, parquet via pyarrow or pickle).
- Add data/fraud_detection/.gitignore ignoring raw/* and processed/* (keep README.md).
- Follow conventions you find in backend/src/shared (read-only). Do not import from other components.
- Provide functions callable by the backend later, e.g. `run_early_warning(window_df, model, reference) -> decision`
  but do NOT add API routes or touch the frontend.

## 17. Honesty rules for all text you write
State limitations: synthetic Handbook drift; simulated label delay; IEEE-CIS anonymized and possibly without
real degradation; label-free signals cannot see pure concept drift (Solozobov); S1 detection partly passes through
the label-derived terminal-risk feature. Report negative results. Do not overclaim.
