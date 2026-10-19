# Financial inclusion scripts

Run scripts from the repository root with their implementation-stage folder in the path:

```powershell
python backend/src/financial_inclusion/01_data_preparation/fi_preprocessing.py
python backend/src/financial_inclusion/02_dimension_creation/fi_dimension_creation.py
python backend/src/financial_inclusion/03_clustering/fi_clustering.py
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_baseline_models.py
```

The stage folders are `01_data_preparation`, `02_dimension_creation`, `03_clustering`, `04_supervised_learning`, and `05_analysis_visualisation`. Project-root paths are resolved from each script location; generated reports are written under the corresponding stage folder in `outputs/financial_inclusion/reports/`.
