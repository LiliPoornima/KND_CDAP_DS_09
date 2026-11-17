# KND_CDAP_DS_09

Final-year research project on **Explainable and Adaptive AI for Financial
Inclusion and Risk Management**. The project combines machine learning,
explainable AI (XAI), fairness analysis, and publicly available financial
datasets in an interactive research dashboard.

## About the project

The project investigates how transparent and data-driven methods can support
financial decision-making while making model behaviour, subgroup performance,
and analytical limitations visible to researchers and users.

The current README documents the following research components:

### Component 1: Credit-risk scoring

Component 1 evaluates credit-risk models using multiple public datasets:

- Give Me Some Credit
- Taiwan credit-card default
- Home Credit

The component includes dataset-specific preprocessing, feature engineering,
model comparison, held-out evaluation, calibration analysis, subgroup
fairness metrics, bootstrap confidence intervals, and SHAP-based explanations.
The dashboard also includes a research-only Home Credit application assessment
prototype. Its estimates are not lending decisions, credit approvals, or
causal conclusions.

Relevant resources:

- [`backend/src/credit_risk/`](backend/src/credit_risk/)
- [`outputs/credit_risk/`](outputs/credit_risk/)
- [`docs/research/credit_risk.md`](docs/research/credit_risk.md)
- [`docs/research/credit_risk_final_report.md`](docs/research/credit_risk_final_report.md)

### Component 2: Digital financial inclusion profiling

Component 2 uses Global Findex individual microdata to profile digital
financial inclusion. It creates interpretable inclusion dimensions and
profiles, examines population distributions and subgroup gaps, and provides
explainable decision-support views.

The dashboard includes:

- inclusion overview and profile exploration;
- dimension and population-distribution analysis;
- demographic, income, and regional subgroup comparisons;
- explainability views for model outputs;
- gap analysis and evidence-based recommendations.

Relevant resources:

- [`backend/src/financial_inclusion/`](backend/src/financial_inclusion/)
- [`frontend/financial_inclusion/`](frontend/financial_inclusion/)
- [`data/financial_inclusion/`](data/financial_inclusion/)
- [`outputs/financial_inclusion/`](outputs/financial_inclusion/)
- [`docs/research/financial_inclusion.md`](docs/research/financial_inclusion.md)

### Component 3: Multi-task financial market forecasting

Component 3 analyses financial-market behaviour using return, volatility, and
risk measures. Its backend modules support data loading and preprocessing,
feature engineering, forecasting, evaluation, backtesting, volatility
analysis, and value-at-risk (VaR) analysis.

Relevant resources:

- [`backend/src/market_risk/`](backend/src/market_risk/)
- [`data/market_risk/`](data/market_risk/)
- [`outputs/market_risk/`](outputs/market_risk/)
- [`docs/research/market_risk.md`](docs/research/market_risk.md)

## Dashboard

The Streamlit dashboard provides a single entry point for exploring the
research outputs. Use the sidebar to select the Home page or one of the
research components described above.

## Requirements

- Python 3.11 or newer
- `pip`
- A virtual environment is recommended
- The project dependencies listed in [`requirements.txt`](requirements.txt)

## Installation

From the repository root, create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the dashboard

Run this command from the repository root after activating the virtual
environment:

```powershell
python -m streamlit run frontend/app.py


Set-Location "C:\Users\Poornima Liyanage\Documents\GitHub\KND_CDAP_DS_09"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Streamlit will print a local URL, normally
`http://localhost:8501`. Open that URL in a browser. Keep the terminal
running while using the dashboard.

On Windows, if the environment is not activated, run the project interpreter
explicitly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

If the dashboard still shows `No module named 'plotly'`, stop any older
Streamlit process and restart it with the command above. The dashboard must
be launched by the project `.venv` interpreter, where Plotly is installed.

## Run the financial-inclusion processing stages

The financial-inclusion preparation and modelling scripts are run from the
repository root:

```bash
python backend/src/financial_inclusion/01_data_preparation/fi_preprocessing.py
python backend/src/financial_inclusion/02_dimension_creation/fi_dimension_creation.py
python backend/src/financial_inclusion/03_clustering/fi_clustering.py
python backend/src/financial_inclusion/04_supervised_learning/fi_supervised_baseline_models.py
```

Generated reports and processed outputs are written under
`outputs/financial_inclusion/`. The dashboard expects the validated
analytical data and saved research outputs to be available in the repository.

## Repository structure

```text
backend/src/       Research and modelling code
data/              Dataset documentation and local data folders
docs/              Architecture, setup, integration, and research notes
frontend/          Streamlit application and dashboard pages
models/            Saved model artefacts and model documentation
outputs/           Evaluation metrics, reports, figures, and audit results
tests/             Automated tests
```

## Data and generated artefacts

Datasets are organised by research component under [`data/`](data/).
Generated metrics, figures, reports, and model outputs are organised under
[`outputs/`](outputs/) and [`models/`](models/). Some source datasets may not
be included in the repository because of size or redistribution
restrictions. Follow the relevant component documentation for expected file
locations and preparation steps.

## Research-use notice

This is a research and demonstration project. Dashboard predictions,
rankings, explanations, fairness results, and recommendations should not be
treated as production financial advice, lending decisions, or guarantees of
future market performance. Results depend on the supplied datasets, their
definitions, preprocessing choices, and the evaluation design.

## Further documentation

- [`docs/development/setup_guide.md`](docs/development/setup_guide.md)
- [`docs/development/integration_guide.md`](docs/development/integration_guide.md)
- [`docs/architecture/system_architecture.md`](docs/architecture/system_architecture.md)
