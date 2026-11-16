# Component 02 supervised classification experiment

## Finding

The seven demographic and socioeconomic predictors contain limited information about a label constructed from five financial inclusion scores. The models are predicting a downstream profile from broad demographics, so overlap among profiles is expected. The baseline confusion matrices show substantial cross-class confusion. XGBoost's higher accuracy comes with strong preference for class 1: held-out recall is 83% for class 1 but 14% for class 0 and 11% for class 4. The classes are imbalanced (class 1 is 41.7%; class 4 is 8.9%), but class imbalance is not the only limit: the baseline Logistic Regression already uses `class_weight="balanced"` and achieves more even recall with modest overall accuracy.

Preprocessing confirmed in source: categorical predictors (`female`, `educ`, `inc_q`, `emp_in`, `urbanicity`, `regionwb`) are most-frequent imputed and one-hot encoded; `age` is median imputed and standardized. The split is stratified 80:20 with seed 42. The derived target dataset has 56,503 rows and predictor missingness below 1.2%. The target comes from the saved five-dimensional K-Means labels. No dimension scores enter this experiment.

## Held-out comparison

All rows below refer to the same 11,301-row held-out test split. Baseline values are the preserved existing results; tuned values are from this experiment. Macro precision and weighted-F1 for the baseline are from the saved classification report.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression baseline | 0.4039 | 0.3783 | 0.36 | 0.3783 | 0.3624 | 0.4161 |
| Logistic Regression tuned | 0.4032 | 0.3767 | 0.3603 | 0.3767 | 0.3610 | 0.4151 |
| Random Forest baseline | 0.3544 | 0.3080 | 0.30 | 0.3080 | 0.3031 | 0.3640 |
| Random Forest tuned | 0.3918 | 0.3678 | 0.3532 | 0.3678 | 0.3525 | 0.4060 |
| XGBoost baseline | 0.4729 | 0.3336 | 0.40 | 0.3336 | 0.3354 | 0.4243 |
| XGBoost tuned | 0.4723 | 0.3318 | 0.3965 | 0.3318 | 0.3320 | 0.4225 |

The tuned Random Forest improved over its baseline but remained below Logistic Regression on balanced accuracy and macro-F1. XGBoost retained the highest accuracy, while its class balance remained poor. Tuned Logistic Regression is the most balanced interpretable candidate, but the original baseline Logistic Regression is marginally better on the principal metrics. Therefore the dashboard's saved baseline remains the production artifact; the separate tuned pipelines are experiment artifacts and do not alter dashboard inference.

## Tuning and feature analysis

Randomized search used stratified two-fold CV on the training split, optimizing balanced accuracy. Two folds and compact candidate counts (4 Logistic Regression, 4 Random Forest, 3 XGBoost candidates) were used to keep the experiment executable in the local environment. These are exploratory searches, not an exhaustive optimization. CatBoost was not available. Both Logistic Regression and Random Forest searches considered class weights; XGBoost used its native multiclass objective without class weighting. SMOTE was not used because synthetic interpolation across one-hot categorical features can create invalid records. Resampling must be separately evaluated with a categorical-aware method and strictly inside CV folds.

Best configurations:

- Logistic Regression: `C=0.01`, `class_weight="balanced"`.
- Random Forest: 500 trees, unlimited depth, `min_samples_leaf=8`, `max_features="sqrt"`, `class_weight="balanced_subsample"`.
- XGBoost: 250 trees, depth 5, learning rate 0.1, subsample 0.8, column subsample 1.0, min child weight 1, lambda 1.

One-at-a-time Logistic Regression ablation suggests `educ` and `regionwb` contribute most among these predictors. Removing `regionwb` lowers balanced accuracy from 0.3783 to 0.3351; removing `educ` lowers it to 0.3526. Removing `female` slightly increases the metrics, which suggests it adds little predictive signal in this model. These are predictive associations, not causal claims.

The source microdata includes `economy` and `economycode`, but this run did not add either. Country identifiers could let a random respondent split learn country-specific profile prevalence and overstate performance for unseen economies; they also are not part of the current dashboard form. A country-held-out evaluation and codebook-based predictor review should precede an expanded experiment. Survey weight was not used as an ordinary predictor. No financial-inclusion dimension indicators or scores were used.

## Class-wise tuned Logistic Regression

| Cluster | Precision | Recall | F1 | Test support |
|---:|---:|---:|---:|---:|
| 0 | 0.30 | 0.26 | 0.28 | 2,022 |
| 1 | 0.64 | 0.48 | 0.55 | 4,715 |
| 2 | 0.33 | 0.40 | 0.36 | 1,801 |
| 3 | 0.32 | 0.37 | 0.34 | 1,756 |
| 4 | 0.21 | 0.36 | 0.27 | 1,007 |

Rows in the saved confusion matrix are actual labels and columns are predictions, in order 0–4. Full class reports and matrices for all tuned models are in `classification_reports.txt` and `confusion_matrices.txt`.

## SHAP

SHAP was regenerated for the exact saved tuned Logistic Regression pipeline, with 100 held-out examples and 100 training background rows. The values have shape `(100, 22, 5)` for encoded features and `(100, 7, 5)` after grouping to the original predictors. All five profile classes are present. Maximum decision-score additivity error was `8.88e-16`; grouped contribution conservation error was `6.66e-16`. SHAP explains the model's multiclass decision scores, not causal drivers or probabilities. Outputs are in `shap/`.

## Reproduction

From the repository root, with project dependencies installed:

```powershell
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_tuning.py
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_feature_ablation.py
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_tuned_shap.py
```

The tuning script writes to `models/financial_inclusion/supervised_tuned/` and `outputs/financial_inclusion/experiments/supervised_tuning/`. It does not overwrite the existing baseline model files or metrics.
