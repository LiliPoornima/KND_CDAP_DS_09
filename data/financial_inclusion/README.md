# Financial Inclusion Data

## Clone setup for Component 02

The repository intentionally excludes the Global Findex respondent microdata and generated respondent-level files. They are not present in a fresh clone. In the current working copy, the following runtime artifacts exist locally but are ignored by Git:

- `data/financial_inclusion/raw/findex_microdata_2025_labelled_update112425.csv` (source used to reproduce preprocessing)
- `data/financial_inclusion/processed/fi_final_analytical_dataset.csv` (required to open the stakeholder and research dashboard)
- `data/financial_inclusion/processed/fi_clustering_input.csv` and `fi_dimension_dataset.csv` (needed to reproduce the final profile artifacts)
- `outputs/financial_inclusion/metrics/fi_clustering_labels_5D_k5.csv`
- `models/financial_inclusion/kmeans_5D_k5.joblib` (required for an individual assessment)
- `outputs/financial_inclusion/metrics/fi_supervised_baseline_metrics.csv`, the corresponding classification/confusion-matrix text reports, and `models/financial_inclusion/logistic_regression.joblib` (used by supervised research and SHAP generation)
- `outputs/financial_inclusion/reports/05_shap_explainability/fi_shap_global_importance.csv`, `fi_shap_local_explanations.csv`, `fi_shap_predictions.csv`, and `fi_shap_validation_summary.json`

Profile definitions, KMeans metadata, pipeline scripts, and several reviewed reports are tracked. The application and scripts resolve these locations relative to the repository; they do not use a machine-specific absolute path. The cross-machine failure is caused by ignored local data/model/result files, not by the working-directory calculation or missing Python packages.

### Run with the existing saved results

1. Ask the project data owner for the approved Component 02 runtime-artifact bundle and confirm that its redistribution to the team is permitted. Do not commit or upload Global Findex microdata, respondent-level derived files, or model artifacts until their licenses and project rules explicitly allow it.
2. Copy the bundle into this repository while retaining the paths listed above. Include the SHAP CSV/JSON files if the SHAP research panel is required. The dashboard reports the expected path when a required result is unavailable; it does not substitute example values.
3. From the repository root, create/activate a Python environment, install `requirements.txt`, and run `streamlit run frontend/app.py`.

### Rebuild artifacts from authorized source data

Place the authorized source CSV at `data/financial_inclusion/raw/findex_microdata_2025_labelled_update112425.csv`. From the repository root, install dependencies and run the existing scripts in their numbered stage order. The source, generated processed data, metrics, model files, and reports remain local because the repository `.gitignore` excludes them. The final analytical dataset stage consumes the saved dimension dataset, clustering input, and 5D k=5 labels; the KMeans model save stage consumes the clustering input, labels, and tracked final profile definitions. The supervised and SHAP panels additionally require their saved supervised model, dataset audit, and generated explanation reports.

```powershell
python -m pip install -r requirements.txt
python backend/src/financial_inclusion/01_data_preparation/fi_preprocessing.py
python backend/src/financial_inclusion/02_dimension_creation/fi_dimension_creation.py
python backend/src/financial_inclusion/03_clustering/fi_clustering.py
python backend/src/financial_inclusion/03_clustering/fi_clustering_save_candidate_labels.py
python backend/src/financial_inclusion/03_clustering/fi_save_final_kmeans.py
python backend/src/financial_inclusion/01_data_preparation/fi_final_dataset.py
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_baseline_models.py
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_dataset_audit.py
python backend/src/financial_inclusion/04_supervised_learning/fi_shap_explanations_run.py
streamlit run frontend/app.py
```

The preprocessing scripts may require additional intermediate audit/selection steps depending on which source artifact was supplied; follow any missing-input error paths from the stage scripts and do not bypass validation. The derived data is respondent-level and must be handled under the source dataset's license and the team's data policy. Git LFS is not configured; use it only after permission to redistribute each artifact and repository storage suitability have been confirmed.
