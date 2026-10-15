# Component 02 — Digital Financial Inclusion Gap Profiling

## Research objective

This component investigates digital financial inclusion gaps using Global Findex 2025 individual microdata.

## Research workflow

1. Set up the Python environment.
2. Validate the source dataset structure.
3. Reproduce and validate the cohort selection.
4. Validate derived indicators against the codebook.
5. Select variables for the five proposed research dimensions.
6. Prepare and validate the final analysis dataset.
7. Construct dimensions and conduct subsequent analyses.

## Project structure

- `data/raw/`: Local source data; do not commit raw survey files.
- `data/processed/`: Local intermediate datasets.
- `src/`: Python implementation.
- `reports/generated/`: Locally generated audit outputs.
- `tests/`: Tests and validation documentation.

## Data access

Obtain the authorized source dataset separately and place it in `data/raw/`. Do not upload restricted or licensed data to GitHub.

## Current milestone

Initial dataset structure validation. No imputation, dimension scoring, or predictive modelling is performed at this stage.