"""Create provenance manifests for completed Component 1 dataset runs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_FILES = (
    "backend/src/credit_risk/common/cr_evaluation.py",
    "backend/src/credit_risk/common/cr_calibration.py",
    "backend/src/credit_risk/common/cr_data_loader.py",
    "backend/src/credit_risk/fairness/cr_fairness.py",
    "backend/src/credit_risk/fairness/cr_explanation_stability.py",
    "backend/src/credit_risk/explainability/cr_explainability.py",
    "backend/src/credit_risk/experiments/run_inspection.py",
    "backend/src/credit_risk/experiments/cr_ablation.py",
    "backend/src/credit_risk/experiments/write_manifests.py",
    "backend/src/credit_risk/give_me_some_credit/experiment.py",
    "backend/src/credit_risk/give_me_some_credit/feature_engineering.py",
    "backend/src/credit_risk/give_me_some_credit/preprocessing.py",
    "backend/src/credit_risk/give_me_some_credit/model.py",
    "backend/src/credit_risk/credit_card_default/experiment.py",
    "backend/src/credit_risk/credit_card_default/preprocessing.py",
    "backend/src/credit_risk/credit_card_default/training.py",
    "backend/src/credit_risk/credit_card_default/feature_engineering.py",
    "backend/src/credit_risk/home_credit/experiment.py",
    "backend/src/credit_risk/home_credit/data_loader.py",
    "backend/src/credit_risk/home_credit/feature_engineering.py",
    "backend/src/credit_risk/home_credit/preprocessing.py",
    "backend/src/credit_risk/home_credit/training.py",
    "requirements.txt",
)

DATASETS = {
    "give_me_some_credit": {
        "title": "Give Me Some Credit",
        "target": "SeriousDlqin2yrs",
        "target_definition": "Serious delinquency within two years",
        "geography": "Not identified in the supplied competition files",
        "observation_unit": "Borrower-level competition record",
        "source_members": ["GiveMeSomeCredit/cs-training.csv"],
        "audit_attributes": ["age", "NumberOfDependents"],
        "feature_count_expected": 14,
        "preprocessing": [
            "Drop unique exported row index Unnamed: 0",
            "Fit median imputation on training predictors only",
            "Add two delinquency summaries, one combined-account count, and a dependent indicator",
        ],
        "history_aggregation": None,
        "logistic_regression": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        "random_forest": {
            "pipeline": "RandomForestClassifier on train-imputed numeric predictors",
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "credit_card_default": {
        "title": "Default of Credit Card Clients (Taiwan)",
        "target": "default.payment.next.month",
        "target_definition": "Next-month credit-card payment default",
        "geography": "Taiwan, as identified by the dataset documentation",
        "observation_unit": "Taiwan credit-card client",
        "source_members": [
            "default+of+credit+card+clients/default of credit card clients.xls"
        ],
        "audit_attributes": ["SEX", "AGE", "MARRIAGE", "EDUCATION"],
        "feature_count_expected": 23,
        "preprocessing": [
            "Exclude ID and audit attributes from the predictor matrix",
            "Fit median imputation on training rows only",
            "Add repayment-behaviour summaries from six months of history",
        ],
        "history_aggregation": None,
        "logistic_regression": {
            "pipeline": "StandardScaler -> LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        "random_forest": {
            "pipeline": "RandomForestClassifier on train-imputed engineered predictors",
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "home_credit": {
        "title": "Home Credit Default Risk",
        "target": "TARGET",
        "target_definition": "Payment difficulty on early installments of a sampled loan",
        "geography": "Not established by the supplied competition files",
        "observation_unit": "Home Credit loan application",
        "source_members": [
            "home-credit-default-risk/application_train.csv",
            "home-credit-default-risk/bureau.csv",
            "home-credit-default-risk/bureau_balance.csv",
            "home-credit-default-risk/previous_application.csv",
            "home-credit-default-risk/installments_payments.csv",
            "home-credit-default-risk/POS_CASH_balance.csv",
            "home-credit-default-risk/credit_card_balance.csv",
        ],
        "audit_attributes": ["CODE_GENDER", "AGE_YEARS"],
        "feature_count_expected": 199,
        "preprocessing": [
            "Exclude application ID and direct audit fields from predictors",
            "Fit numeric median and categorical encoding in training-only pipelines",
            "Aggregate linked histories using relative dates no later than application time",
        ],
        "history_aggregation": {
            "included": True,
            "chunksize": 100000,
            "feature_count": 75,
            "tables": [
                "bureau.csv",
                "bureau_balance.csv",
                "previous_application.csv",
                "installments_payments.csv",
                "POS_CASH_balance.csv",
                "credit_card_balance.csv",
            ],
        },
        "logistic_regression": {
            "pipeline": "ColumnTransformer(numeric/categorical preprocessing) -> LogisticRegression",
            "class_weight": "balanced",
            "solver": "newton-cholesky",
            "max_iter": 1000,
            "tol": 0.0001,
            "random_state": 42,
            "numeric_imputer": "median, add_indicator=True, keep_empty_features=True",
            "numeric_scaler": "StandardScaler(with_mean=False)",
            "categorical_imputer": "constant __MISSING__",
            "one_hot_encoder": "handle_unknown=ignore, min_frequency=20",
        },
        "random_forest": {
            "pipeline": "ColumnTransformer(numeric/categorical preprocessing) -> RandomForestClassifier",
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "numeric_imputer": "median, add_indicator=True, keep_empty_features=True",
            "categorical_imputer": "constant __MISSING__",
            "one_hot_encoder": "handle_unknown=ignore, min_frequency=20",
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    package_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "shap": "shap",
        "xlrd": "xlrd",
        "streamlit": "streamlit",
        "plotly": "plotly",
    }
    versions: dict[str, str | None] = {}
    for label, distribution in package_names.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _installed_distributions() -> list[dict[str, str]]:
    return sorted(
        [
            {
                "name": distribution.metadata.get("Name", "unknown"),
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
        ],
        key=lambda package: package["name"].lower(),
    )


def _split_artifact(path: Path, target: str | None) -> dict:
    artifact = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if target is not None:
        target_counts: dict[str, int] = {}
        row_count = 0
        for chunk in pd.read_csv(path, usecols=[target], chunksize=100_000):
            row_count += len(chunk)
            for label, count in chunk[target].value_counts(dropna=False).items():
                key = "[missing]" if pd.isna(label) else str(int(label))
                target_counts[key] = target_counts.get(key, 0) + int(count)
        artifact["rows"] = row_count
        artifact["target_counts"] = dict(sorted(target_counts.items()))
    else:
        with path.open("rb") as stream:
            artifact["rows"] = max(0, sum(1 for _ in stream) - 1)
    return artifact


def _archive_description(source: Path, members_used: list[str]) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Source archive not found: {source}")
    with ZipFile(source) as archive:
        listing = {entry.filename: entry.file_size for entry in archive.infolist()}
        missing = [member for member in members_used if member not in listing]
        if missing:
            raise ValueError(f"Expected source member(s) are missing from {source.name}: {missing}")
        members = [
            {"path_in_archive": member, "uncompressed_size_bytes": listing[member]}
            for member in members_used
        ]
    return {
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
        "members_used": members,
    }


def _git_state() -> dict:
    import subprocess

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


def cr_write_component1_run_manifests(
    *,
    output_root: str | Path,
    source_archives: dict[str, str | Path],
) -> dict[str, Path]:
    """Write per-dataset manifests for saved splits and generated analyses."""
    root = Path(output_root).resolve()
    git_state = _git_state()
    written: dict[str, Path] = {}
    for dataset, config in DATASETS.items():
        if dataset not in source_archives:
            raise ValueError(f"A source archive path is required for {dataset}.")
        run_dir = root / dataset / "completion_run"
        prepared_dir = run_dir / "prepared"
        metrics_dir = run_dir / ("metrics" if dataset != "give_me_some_credit" else "")
        comparison_path = metrics_dir / "model_comparison.csv"
        if not comparison_path.is_file():
            raise FileNotFoundError(f"Model comparison output is missing: {comparison_path}")

        source = Path(source_archives[dataset]).expanduser().resolve()
        archive = _archive_description(source, config["source_members"])
        split_files = {}
        for split in ("train", "validation", "test"):
            processed = prepared_dir / f"{split}_processed.csv"
            if not processed.is_file():
                raise FileNotFoundError(f"Processed {split} split is missing: {processed}")
            split_files[f"{split}_processed"] = _split_artifact(processed, config["target"])
            group_file = prepared_dir / f"{split}_groups.csv"
            if group_file.is_file():
                split_files[f"{split}_audit_groups"] = _split_artifact(group_file, None)

        header = pd.read_csv(prepared_dir / "train_processed.csv", nrows=0).columns.tolist()
        if config["target"] not in header:
            raise ValueError(f"Target {config['target']} is missing from the saved training split.")
        feature_count = len(header) - 1
        if feature_count != config["feature_count_expected"]:
            raise ValueError(
                f"{dataset} split has {feature_count} predictors, while the audited "
                f"expectation is {config['feature_count_expected']}."
            )
        comparison = pd.read_csv(comparison_path)
        thresholds = {
            str(row["Model"]): float(row["Validation-selected threshold"])
            for _, row in comparison.iterrows()
        }

        code_hashes = {
            filename: _sha256(PROJECT_ROOT / filename)
            for filename in CODE_FILES
            if (PROJECT_ROOT / filename).is_file()
        }
        artifact_dirs = [metrics_dir, run_dir / "explainability"]
        generated = []
        for artifact_dir in artifact_dirs:
            if artifact_dir.is_dir():
                for path in sorted(artifact_dir.rglob("*.csv")):
                    if "prepared" in path.parts:
                        continue
                    generated.append(
                        {
                            "path": path.relative_to(PROJECT_ROOT).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                    )
        manifest = {
            "manifest_schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "key": dataset,
                "title": config["title"],
                "observation_unit": config["observation_unit"],
                "target_column": config["target"],
                "target_definition": config["target_definition"],
                "geographic_scope": config["geography"],
                "source_archive": archive,
            },
            "preparation": {
                "split_method": "two-stage stratified random split by target",
                "train_fraction": 0.70,
                "validation_fraction": 0.10,
                "test_fraction": 0.20,
                "split_random_state": 42,
                "target_stratified": True,
                "target_column": config["target"],
                "feature_count": feature_count,
                "feature_count_expected_from_audit": config["feature_count_expected"],
                "audit_attributes_kept_out_of_predictors": config["audit_attributes"],
                "preprocessing": config["preprocessing"],
                "history_aggregation": config["history_aggregation"],
                "saved_splits": split_files,
            },
            "experiments": {
                "models": {
                    "logistic_regression": {
                        "estimator": "sklearn.linear_model.LogisticRegression",
                        **config["logistic_regression"],
                    },
                    "random_forest": {
                        "estimator": "sklearn.ensemble.RandomForestClassifier",
                        **config["random_forest"],
                    },
                },
                "threshold_selection": {
                    "data_partition": "validation",
                    "candidate_thresholds": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
                    "objective": "maximize F1-score",
                    "tie_breakers": ["higher precision", "higher threshold"],
                    "selected_thresholds": thresholds,
                    "test_partition_used_for_threshold_selection": False,
                },
                "performance_uncertainty": {
                    "method": "percentile bootstrap stratified within observed test target class",
                    "replicates": 1000,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "conditioned_on": ["fitted model", "selected threshold", "observed test class counts"],
                },
                "paired_performance_uncertainty": {
                    "method": "paired percentile bootstrap stratified within observed test target class",
                    "models": ["Logistic Regression", "Random Forest"],
                    "difference": "Random Forest minus Logistic Regression",
                    "replicates": 1000,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "same_resampled_test_rows_for_both_models": True,
                    "thresholds_selected_on": "validation",
                    "conditioned_on": ["fitted models", "selected thresholds", "observed test class counts"],
                    "multiple_comparison_adjustment": None,
                },
                "feature_ablation_uncertainty": {
                    "method": "paired percentile bootstrap stratified within observed test target class",
                    "comparison": "each reduced-feature scenario minus its full-feature model",
                    "replicates": 1000,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "same_resampled_test_rows_for_both_scenarios": True,
                    "scenario_thresholds_selected_on": "validation",
                    "conditioned_on": ["fitted scenario models", "selected scenario thresholds", "observed test class counts"],
                    "multiple_comparison_adjustment": None,
                },
                "feature_ablation_fairness_uncertainty": {
                    "method": "paired percentile bootstrap stratified by observed audit group and target class",
                    "comparison": "each reduced-feature scenario disparity gap minus the full-feature model gap",
                    "metrics": [
                        "selection_rate_difference",
                        "true_positive_rate_difference",
                        "false_positive_rate_difference",
                        "equalized_odds_difference",
                    ],
                    "replicates": 1000,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "minimum_group_size": 30,
                    "minimum_class_support": 5,
                    "same_resampled_test_rows_for_both_scenarios": True,
                    "conditioned_on": [
                        "fitted scenario models",
                        "selected scenario thresholds",
                        "observed group-by-outcome counts",
                    ],
                    "multiple_comparison_adjustment": None,
                },
                "fairness_uncertainty": {
                    "method": "within-group percentile bootstrap",
                    "replicates": 1000,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "minimum_group_size": 30,
                    "minimum_class_support": 5,
                },
                "subgroup_calibration_uncertainty": {
                    "method": "within-group percentile bootstrap stratified by observed target class",
                    "metrics": [
                        "brier_score",
                        "log_loss",
                        "expected_calibration_error",
                        "calibration_in_the_large",
                    ],
                    "ece_bins": 10,
                    "replicates": 1000,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "random_state_schedule": "base + dataset_index*1000 + model_index*100 + attribute_index",
                    "minimum_group_size": 30,
                    "minimum_class_support": 5,
                    "conditioned_on": [
                        "fitted model",
                        "predicted probabilities",
                        "observed within-group outcome counts",
                    ],
                    "multiple_comparison_adjustment": None,
                },
                "explanation_stability_random_states": [42, 43, 44],
            },
            "software": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
                "all_installed_distributions": _installed_distributions(),
                "requirements_file": "requirements.txt",
            },
            "implementation": {
                **git_state,
                "source_file_sha256": code_hashes,
            },
            "generated_summary_artifacts": generated,
            "shared_coverage_artifact": {
                "path": (root / "subgroup_support_coverage.csv").relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(root / "subgroup_support_coverage.csv"),
            } if (root / "subgroup_support_coverage.csv").is_file() else None,
            "limitations": [
                "Dataset results are not pooled because populations and target definitions differ.",
                "Bootstrap intervals do not capture model-refit uncertainty, threshold-selection uncertainty, or population shift.",
                "Feature-ablation intervals are not adjusted for multiple comparisons across scenarios or metrics.",
                "Subgroup calibration intervals are not adjusted for multiple comparisons across groups or metrics.",
                "These historical public datasets do not establish global population representativeness.",
            ],
        }
        destination = run_dir / "run_manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written[dataset] = destination
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/credit_risk")
    parser.add_argument("--give-me-some-credit-source", required=True)
    parser.add_argument("--credit-card-source", required=True)
    parser.add_argument("--home-credit-source", required=True)
    args = parser.parse_args()
    paths = cr_write_component1_run_manifests(
        output_root=args.output_root,
        source_archives={
            "give_me_some_credit": args.give_me_some_credit_source,
            "credit_card_default": args.credit_card_source,
            "home_credit": args.home_credit_source,
        },
    )
    for dataset, path in paths.items():
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
