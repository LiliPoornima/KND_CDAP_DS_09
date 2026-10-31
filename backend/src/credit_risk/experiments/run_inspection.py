"""Run reproducible, dataset-specific Component 1 credit-risk baselines."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
    cr_bootstrap_binary_performance,
    cr_bootstrap_paired_binary_performance,
)
from backend.src.credit_risk.common.cr_calibration import (
    cr_bootstrap_group_calibration,
    cr_group_calibration_summary,
)
from backend.src.credit_risk.credit_card_default.experiment import (
    cr_prepare_credit_card_dataset_from_file,
    cr_run_credit_card_baselines,
)
from backend.src.credit_risk.credit_card_default.feature_engineering import (
    BILL_AMOUNT_COLUMNS,
    PAYMENT_AMOUNT_COLUMNS,
    PAYMENT_STATUS_COLUMNS,
)
from backend.src.credit_risk.credit_card_default.training import (
    cr_train_credit_card_logistic_regression,
    cr_train_credit_card_random_forest,
)
from backend.src.credit_risk.experiments.cr_ablation import (
    cr_run_validation_selected_feature_ablation,
)
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_analyze_group_fairness,
    cr_build_subgroup_support_coverage,
    cr_bootstrap_group_fairness,
    cr_bin_age_groups,
)
from backend.src.credit_risk.fairness.cr_explanation_stability import (
    cr_compare_explanation_stability,
)
from backend.src.credit_risk.explainability.cr_explainability import (
    cr_explain_model,
)
from backend.src.credit_risk.give_me_some_credit.experiment import (
    cr_prepare_dataset,
)
from backend.src.credit_risk.give_me_some_credit.model import (
    TARGET_COLUMN as GIVE_ME_SOME_CREDIT_TARGET,
    cr_evaluate_model,
    cr_train_logistic_regression,
    cr_train_random_forest,
)
from backend.src.credit_risk.home_credit.experiment import (
    cr_prepare_home_credit_dataset_from_archive,
    cr_run_home_credit_baselines,
)
from backend.src.credit_risk.home_credit.preprocessing import (
    TARGET_COLUMN as HOME_CREDIT_TARGET,
)
from backend.src.credit_risk.home_credit.training import (
    cr_create_home_credit_random_forest,
    cr_train_home_credit_logistic_regression,
    cr_train_home_credit_random_forest,
)
from backend.src.credit_risk.credit_card_default.preprocessing import (
    TARGET_COLUMN as CREDIT_CARD_TARGET,
)


def _select_threshold(validation: pd.DataFrame, target: str, model: Any) -> tuple[float, pd.DataFrame]:
    predictors = [column for column in validation.columns if column != target]
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(validation[predictors])[:, positive_index]
    threshold_table = cr_analyze_probability_thresholds(
        validation[target],
        probabilities,
    )
    best = threshold_table.sort_values(
        ["F1-score", "Precision", "Threshold"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    return float(best["Threshold"]), threshold_table


def cr_run_give_me_some_credit(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Reprepare and evaluate Give Me Some Credit without touching old outputs."""

    output = Path(output_dir)
    train, test, metadata = cr_prepare_dataset(
        pd.read_csv(source_csv),
        output_dir=output / "prepared",
        random_state=random_state,
    )
    validation = metadata["validation_processed"]
    model_metrics: dict[str, dict[str, Any]] = {}
    comparisons = []
    threshold_tables = []
    subgroup_calibration_metrics = []
    subgroup_calibration_bins = []
    models = {
        "Logistic Regression": cr_train_logistic_regression(train),
        "Random Forest": cr_train_random_forest(train),
    }

    for model_name, model in models.items():
        threshold, threshold_table = _select_threshold(
            validation,
            GIVE_ME_SOME_CREDIT_TARGET,
            model,
        )
        threshold_table.insert(0, "Model", model_name)
        threshold_tables.append(threshold_table)
        metrics = cr_evaluate_model(
            model,
            test,
            threshold=threshold,
        )
        positive_index = list(model.classes_).index(1)
        test_probabilities = model.predict_proba(
            test.drop(columns=[GIVE_ME_SOME_CREDIT_TARGET])
        )[:, positive_index]
        test_predictions = (test_probabilities >= threshold).astype(int)
        cr_bootstrap_binary_performance(
            test[GIVE_ME_SOME_CREDIT_TARGET],
            test_probabilities,
            threshold=threshold,
            random_state=random_state,
        ).to_csv(
            output
            / f"{model_name.lower().replace(' ', '_')}_performance_confidence_intervals.csv",
            index=False,
        )
        model_metrics[model_name] = metrics
        comparisons.append(
            {
                "Model": model_name,
                "Validation-selected threshold": threshold,
                "Test accuracy": metrics["accuracy"],
                "Test precision": metrics["precision"],
                "Test recall": metrics["recall"],
                "Test F1-score": metrics["f1_score"],
                "Test ROC-AUC": metrics["roc_auc"],
                "Test PR-AUC": metrics["pr_auc_average_precision"],
                "Test Brier score": metrics["brier_score"],
                "Test log loss": metrics["log_loss"],
                "Test expected calibration error": metrics[
                    "expected_calibration_error"
                ],
            }
        )

        for group_name, groups in (
            ("age", cr_bin_age_groups(test["age"])),
            (
                "dependents",
                test["NumberOfDependents"].map(
                    lambda value: "0" if value == 0 else ("1" if value == 1 else "2+")
                ),
            ),
        ):
            fairness = cr_analyze_group_fairness(
                test[GIVE_ME_SOME_CREDIT_TARGET],
                test_predictions,
                groups,
                attribute_name=group_name,
            )
            calibration = cr_group_calibration_summary(
                test[GIVE_ME_SOME_CREDIT_TARGET],
                test_probabilities,
                groups,
                attribute_name=group_name,
            )
            calibration_metrics = calibration["group_metrics"].copy()
            calibration_metrics.insert(0, "Model", model_name)
            subgroup_calibration_metrics.append(calibration_metrics)
            calibration_bins = calibration["calibration_bins"].copy()
            calibration_bins.insert(0, "Model", model_name)
            subgroup_calibration_bins.append(calibration_bins)
            uncertainty = cr_bootstrap_group_fairness(
                test[GIVE_ME_SOME_CREDIT_TARGET],
                test_predictions,
                groups,
                attribute_name=group_name,
                random_state=42,
            )
            fairness["group_metrics"].to_csv(
                output / f"{model_name.lower().replace(' ', '_')}_{group_name}_fairness.csv",
                index=False,
            )
            safe_name = model_name.lower().replace(" ", "_")
            uncertainty["group_confidence_intervals"].to_csv(
                output / f"{safe_name}_{group_name}_group_confidence_intervals.csv",
                index=False,
            )
            uncertainty["disparity_confidence_intervals"].to_csv(
                output / f"{safe_name}_{group_name}_gap_confidence_intervals.csv",
                index=False,
            )

        pd.DataFrame(metrics["calibration_bins"]).to_csv(
            output / f"{model_name.lower().replace(' ', '_')}_calibration_bins.csv",
            index=False,
        )

    pd.concat(threshold_tables, ignore_index=True).to_csv(
        output / "threshold_analysis.csv",
        index=False,
    )
    pd.concat(subgroup_calibration_metrics, ignore_index=True).to_csv(
        output / "subgroup_calibration_metrics.csv", index=False
    )
    pd.concat(subgroup_calibration_bins, ignore_index=True).to_csv(
        output / "subgroup_calibration_bins.csv", index=False
    )
    pd.DataFrame(comparisons).set_index("Model").to_csv(
        output / "model_comparison.csv"
    )
    return {
        "models": models,
        "test_metrics": model_metrics,
        "comparison": pd.DataFrame(comparisons).set_index("Model"),
    }


def cr_run_explanation_stability(
    model: Any,
    train_df: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_column: str,
    random_states: tuple[int, ...] = (42, 43, 44),
    sample_size: int = 100,
    background_size: int = 100,
) -> dict[str, Any]:
    """Refit a model under fixed seeds and save sampled SHAP stability results."""

    if len(random_states) < 2 or len(set(random_states)) != len(random_states):
        raise ValueError("At least two distinct random states are required.")
    if target_column not in train_df.columns:
        raise ValueError(f"Training data must include {target_column}.")
    predictors = train_df.drop(columns=[target_column])
    target = train_df[target_column]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    attribution_runs = []
    global_tables = []

    for run_number, random_state in enumerate(random_states, start=1):
        fitted = clone(model)
        seed_parameter = next(
            (
                name
                for name in ("classifier__random_state", "random_state")
                if name in fitted.get_params()
            ),
            None,
        )
        if seed_parameter is not None:
            fitted.set_params(**{seed_parameter: random_state})
        fitted.fit(predictors, target)
        explanation = cr_explain_model(
            fitted,
            evaluation_features,
            sample_size=sample_size,
            background_size=background_size,
            random_state=42,
        )
        attribution_runs.append(explanation["attributions"])
        global_table = explanation["global_importance"].copy()
        global_table.insert(0, "Run", run_number)
        global_table.insert(1, "Random state", random_state)
        global_tables.append(global_table)

    stability = cr_compare_explanation_stability(attribution_runs)
    pd.concat(global_tables, ignore_index=True).to_csv(
        output / "global_shap_importance_by_run.csv",
        index=False,
    )
    stability["pairwise_stability"].to_csv(
        output / "explanation_stability_pairwise.csv",
        index=False,
    )
    stability["feature_importance_by_run"].to_csv(
        output / "explanation_stability_feature_ranks.csv",
        index=False,
    )
    return stability


def cr_run_training_subsample_explanation_stability(
    model_template: Pipeline,
    train_df: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_column: str,
    training_fraction: float = 0.70,
    resample_states: tuple[int, ...] = (42, 43, 44, 45, 46),
    sample_size: int = 100,
    background_size: int = 100,
) -> dict[str, Any]:
    """Compare explanations after stratified training-row resampling.

    The preprocessor is fitted once on the complete training partition and
    reused across resamples, so the feature space remains identical. Each
    classifier is refitted on a stratified subset; the held-out evaluation
    rows and a training-only SHAP background are fixed across all fits.
    """

    if not isinstance(model_template, Pipeline):
        raise TypeError("model_template must be a scikit-learn Pipeline.")
    if not isinstance(train_df, pd.DataFrame) or train_df.empty:
        raise ValueError("train_df must be a non-empty DataFrame.")
    if target_column not in train_df.columns:
        raise ValueError(f"Training data must include {target_column}.")
    predictors = train_df.drop(columns=[target_column])
    target = train_df[target_column].astype(int)
    if not isinstance(evaluation_features, pd.DataFrame) or evaluation_features.empty:
        raise ValueError("evaluation_features must be a non-empty DataFrame.")
    if list(evaluation_features.columns) != list(predictors.columns):
        raise ValueError("Evaluation predictors must match training columns and order.")
    if (
        not isinstance(training_fraction, (int, float))
        or isinstance(training_fraction, bool)
        or not 0.1 <= training_fraction < 1
    ):
        raise ValueError("training_fraction must be between 0.1 and 1.0.")
    if len(resample_states) < 3 or len(set(resample_states)) != len(resample_states):
        raise ValueError("At least three distinct resample states are required.")

    evaluation = evaluation_features.sample(
        n=min(sample_size, len(evaluation_features)),
        random_state=101,
    )
    background_positions = np.random.default_rng(202).choice(
        len(predictors),
        size=min(background_size, len(predictors)),
        replace=False,
    )
    background_source = predictors.iloc[background_positions]

    if len(model_template.steps) > 1:
        preprocessor = clone(model_template[:-1])
        preprocessor.fit(predictors)
        transformed_train = preprocessor.transform(predictors)
        transformed_evaluation = preprocessor.transform(evaluation)
        transformed_background = preprocessor.transform(background_source)
        try:
            feature_names = list(
                preprocessor.get_feature_names_out(predictors.columns)
            )
        except (AttributeError, ValueError):
            feature_names = [f"feature_{index}" for index in range(transformed_train.shape[1])]
    else:
        transformed_train = predictors.to_numpy()
        transformed_evaluation = evaluation.to_numpy()
        transformed_background = background_source.to_numpy()
        feature_names = list(predictors.columns)

    def _to_frame(values: Any, index: pd.Index) -> pd.DataFrame:
        if hasattr(values, "toarray"):
            values = values.toarray()
        return pd.DataFrame(
            np.asarray(values),
            index=index,
            columns=feature_names,
        )

    evaluation_frame = _to_frame(transformed_evaluation, evaluation.index)
    background_frame = _to_frame(transformed_background, background_source.index)
    classifier_template = model_template.steps[-1][1]
    attribution_runs = []
    importance_tables = []

    for run_number, resample_state in enumerate(resample_states, start=1):
        selected_positions, _ = train_test_split(
            np.arange(len(target)),
            train_size=training_fraction,
            random_state=resample_state,
            stratify=target.to_numpy(),
        )
        fitted_classifier = clone(classifier_template)
        if "random_state" in fitted_classifier.get_params():
            # Keep estimator randomness fixed to focus this experiment on row resampling.
            fitted_classifier.set_params(random_state=42)
        fitted_classifier.fit(
            transformed_train[selected_positions],
            target.iloc[selected_positions],
        )
        explanation = cr_explain_model(
            fitted_classifier,
            evaluation_frame,
            background_data=background_frame,
            sample_size=len(evaluation_frame),
            background_size=len(background_frame),
            random_state=42,
        )
        attribution_runs.append(explanation["attributions"])
        importance = explanation["global_importance"].copy()
        importance.insert(0, "Run", run_number)
        importance.insert(1, "Resample state", resample_state)
        importance.insert(2, "Training rows", len(selected_positions))
        importance_tables.append(importance)

    stability = cr_compare_explanation_stability(attribution_runs)
    pairwise = stability["pairwise_stability"].copy()
    state_by_run = dict(enumerate(resample_states, start=1))
    pairwise.insert(1, "Resample state A", pairwise["run_a"].map(state_by_run))
    pairwise.insert(3, "Resample state B", pairwise["run_b"].map(state_by_run))
    summary = pd.DataFrame(
        [
            {
                **stability["summary"],
                "training_fraction": training_fraction,
                "resample_states": ";".join(map(str, resample_states)),
                "evaluation_rows": len(evaluation_frame),
                "background_rows": len(background_frame),
                "background_source": "training partition",
                "preprocessing_fit": "full training partition, fixed across resamples",
                "estimator_random_state": 42,
            }
        ]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "training_subsample_stability_summary.csv", index=False)
    pairwise.to_csv(
        output / "training_subsample_stability_pairwise.csv", index=False
    )
    stability["feature_importance_by_run"].to_csv(
        output / "training_subsample_feature_ranks.csv", index=False
    )
    pd.concat(importance_tables, ignore_index=True).to_csv(
        output / "training_subsample_global_importance_by_run.csv", index=False
    )
    return {"summary": summary, "pairwise_stability": pairwise}


def cr_run_saved_split_explanation_resampling(
    output_root: str | Path = "outputs/credit_risk",
    *,
    training_fraction: float = 0.70,
    resample_states: tuple[int, ...] = (42, 43, 44, 45, 46),
) -> dict[str, dict[str, Any]]:
    """Run the training-row resampling stability check on all saved datasets."""

    root = Path(output_root)
    results = {}
    for dataset, target in (
        ("give_me_some_credit", GIVE_ME_SOME_CREDIT_TARGET),
        ("credit_card_default", CREDIT_CARD_TARGET),
        ("home_credit", HOME_CREDIT_TARGET),
    ):
        run_dir = root / dataset / "completion_run"
        prepared_dir = run_dir / "prepared"
        train = pd.read_csv(prepared_dir / "train_processed.csv")
        test = pd.read_csv(prepared_dir / "test_processed.csv")
        if dataset == "home_credit":
            template = cr_create_home_credit_random_forest(
                train,
                target_column=target,
            )
        else:
            template = Pipeline(
                [
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    )
                ]
            )
        output_dir = (
            run_dir
            if dataset == "give_me_some_credit"
            else run_dir / "explainability"
        )
        results[dataset] = cr_run_training_subsample_explanation_stability(
            template,
            train,
            test.drop(columns=[target]),
            output_dir,
            target_column=target,
            training_fraction=training_fraction,
            resample_states=resample_states,
            sample_size=100,
            background_size=100,
        )
    return results


def cr_run_saved_split_group_calibration(
    output_root: str | Path = "outputs/credit_risk",
    *,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Audit held-out subgroup calibration and estimate within-group intervals."""

    root = Path(output_root)
    configurations = {
        "give_me_some_credit": {
            "target": GIVE_ME_SOME_CREDIT_TARGET,
            "trainers": {
                "Logistic Regression": cr_train_logistic_regression,
                "Random Forest": cr_train_random_forest,
            },
            "metrics_dir": Path("completion_run"),
        },
        "credit_card_default": {
            "target": CREDIT_CARD_TARGET,
            "trainers": {
                "Logistic Regression": cr_train_credit_card_logistic_regression,
                "Random Forest": cr_train_credit_card_random_forest,
            },
            "metrics_dir": Path("completion_run/metrics"),
        },
        "home_credit": {
            "target": HOME_CREDIT_TARGET,
            "trainers": {
                "Logistic Regression": cr_train_home_credit_logistic_regression,
                "Random Forest": cr_train_home_credit_random_forest,
            },
            "metrics_dir": Path("completion_run/metrics"),
        },
    }
    results = {}
    for dataset_index, (dataset, config) in enumerate(configurations.items()):
        run_dir = root / dataset / "completion_run"
        prepared_dir = run_dir / "prepared"
        train = pd.read_csv(prepared_dir / "train_processed.csv")
        test = pd.read_csv(prepared_dir / "test_processed.csv")
        target = config["target"]
        if dataset == "give_me_some_credit":
            groups = pd.DataFrame(
                {
                    "age": test["age"],
                    "dependents": test["NumberOfDependents"].map(
                        lambda value: "0"
                        if value == 0
                        else ("1" if value == 1 else "2+")
                    ),
                }
            )
        else:
            groups = pd.read_csv(prepared_dir / "test_groups.csv")

        metric_tables = []
        bin_tables = []
        interval_tables = []
        predictors = [column for column in train.columns if column != target]
        for model_index, (model_name, trainer) in enumerate(config["trainers"].items()):
            model = trainer(train, target_column=target)
            positive_index = list(model.classes_).index(1)
            probability = model.predict_proba(test[predictors])[:, positive_index]
            for attribute_index, attribute in enumerate(groups.columns):
                group_values = groups[attribute]
                if str(attribute).strip().lower() in {"age", "age_years"}:
                    group_values = cr_bin_age_groups(group_values)
                summary = cr_group_calibration_summary(
                    test[target],
                    probability,
                    group_values,
                    attribute_name=str(attribute),
                )
                metrics = summary["group_metrics"].copy()
                metrics.insert(0, "Model", model_name)
                metric_tables.append(metrics)
                intervals = cr_bootstrap_group_calibration(
                    test[target],
                    probability,
                    group_values,
                    attribute_name=str(attribute),
                    n_bootstrap=1000,
                    random_state=(
                        random_state
                        + dataset_index * 1000
                        + model_index * 100
                        + attribute_index
                    ),
                )
                intervals.insert(0, "Model", model_name)
                interval_tables.append(intervals)
                bins = summary["calibration_bins"].copy()
                bins.insert(0, "Model", model_name)
                bin_tables.append(bins)

        metrics = pd.concat(metric_tables, ignore_index=True)
        bins = pd.concat(bin_tables, ignore_index=True)
        intervals = pd.concat(interval_tables, ignore_index=True)
        output_dir = root / dataset / config["metrics_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output_dir / "subgroup_calibration_metrics.csv", index=False)
        bins.to_csv(output_dir / "subgroup_calibration_bins.csv", index=False)
        intervals.to_csv(
            output_dir / "subgroup_calibration_confidence_intervals.csv",
            index=False,
        )
        results[dataset] = metrics
    return results


def cr_run_extended_dataset_analyses(
    output_root: str | Path = "outputs/credit_risk",
    *,
    include_explanation_stability: bool = True,
) -> dict[str, Any]:
    """Run saved-split ablations and optionally Home Credit explanation stability.

    This uses the saved leakage-aware split files. Thresholds are selected on
    each scenario's validation predictions, and held-out test partitions are
    reserved for the reported comparison.
    """

    root = Path(output_root)
    results: dict[str, Any] = {}

    # Give Me Some Credit feature ablations are already part of the completed
    # run. Refit the same scenarios only to attach held-out subgroup results.
    gmsc_dir = root / "give_me_some_credit" / "completion_run"
    gmsc_prepared = gmsc_dir / "prepared"
    gmsc_train = pd.read_csv(gmsc_prepared / "train_processed.csv")
    gmsc_validation = pd.read_csv(gmsc_prepared / "validation_processed.csv")
    gmsc_test = pd.read_csv(gmsc_prepared / "test_processed.csv")
    gmsc_groups = pd.DataFrame(
        {
            "age": gmsc_test["age"],
            "dependents": gmsc_test["NumberOfDependents"].map(
                lambda value: "0" if value == 0 else ("1" if value == 1 else "2+")
            ),
        }
    )
    gmsc_feature_groups = {
        "delinquency_aggregate": [
            "TotalDelinquencyCount",
            "SevereDelinquencyCount",
        ],
        "account_aggregate": ["TotalOpenCreditAccounts"],
        "dependents_indicator": ["HasDependents"],
    }
    results["give_me_some_credit_ablation_subgroups"] = (
        cr_run_validation_selected_feature_ablation(
            gmsc_train,
            gmsc_validation,
            gmsc_test,
            gmsc_feature_groups,
            trainer=cr_train_random_forest,
            model_name="Random Forest",
            target_column=GIVE_ME_SOME_CREDIT_TARGET,
            test_groups=gmsc_groups,
            subgroup_output_path=gmsc_dir / "feature_ablation_subgroup_metrics.csv",
            subgroup_calibration_output_path=(
                gmsc_dir / "feature_ablation_subgroup_calibration.csv"
            ),
            subgroup_calibration_bins_output_path=(
                gmsc_dir / "feature_ablation_subgroup_calibration_bins.csv"
            ),
            paired_comparison_output_path=(
                gmsc_dir / "feature_ablation_paired_comparison_confidence_intervals.csv"
            ),
            paired_fairness_comparison_output_path=(
                gmsc_dir
                / "feature_ablation_paired_fairness_gap_confidence_intervals.csv"
            ),
        )
    )

    credit_card_dir = root / "credit_card_default" / "completion_run"
    credit_card_prepared = credit_card_dir / "prepared"
    credit_card_train = pd.read_csv(
        credit_card_prepared / "train_processed.csv"
    )
    credit_card_validation = pd.read_csv(
        credit_card_prepared / "validation_processed.csv"
    )
    credit_card_test = pd.read_csv(
        credit_card_prepared / "test_processed.csv"
    )
    credit_card_test_groups = pd.read_csv(
        credit_card_prepared / "test_groups.csv"
    )
    credit_card_target = CREDIT_CARD_TARGET
    credit_card_features = [
        column for column in credit_card_train.columns
        if column != credit_card_target
    ]
    credit_card_groups = {
        "repayment_status_history": [
            *[column for column in PAYMENT_STATUS_COLUMNS if column in credit_card_features],
            "BehavioralDelinquencyMonths",
            "MaximumReportedDelayLevel",
        ],
        "monthly_amount_history": [
            "LIMIT_BAL",
            *[column for column in BILL_AMOUNT_COLUMNS if column in credit_card_features],
            *[column for column in PAYMENT_AMOUNT_COLUMNS if column in credit_card_features],
        ],
        "derived_payment_summaries": [
            "MonthsWithPositivePayment",
            "MeanPositiveBillPaymentRatio",
        ],
    }
    results["credit_card_default_ablation"] = (
        cr_run_validation_selected_feature_ablation(
            credit_card_train,
            credit_card_validation,
            credit_card_test,
            credit_card_groups,
            trainer=cr_train_credit_card_random_forest,
            model_name="Random Forest",
            target_column=credit_card_target,
            output_path=credit_card_dir / "metrics" / "feature_ablation.csv",
            test_groups=credit_card_test_groups,
            subgroup_output_path=credit_card_dir / "metrics" / "feature_ablation_subgroup_metrics.csv",
            subgroup_calibration_output_path=(
                credit_card_dir / "metrics" / "feature_ablation_subgroup_calibration.csv"
            ),
            subgroup_calibration_bins_output_path=(
                credit_card_dir / "metrics" / "feature_ablation_subgroup_calibration_bins.csv"
            ),
            paired_comparison_output_path=(
                credit_card_dir
                / "metrics"
                / "feature_ablation_paired_comparison_confidence_intervals.csv"
            ),
            paired_fairness_comparison_output_path=(
                credit_card_dir
                / "metrics"
                / "feature_ablation_paired_fairness_gap_confidence_intervals.csv"
            ),
        )
    )

    home_credit_dir = root / "home_credit" / "completion_run"
    home_credit_prepared = home_credit_dir / "prepared"
    home_credit_train = pd.read_csv(home_credit_prepared / "train_processed.csv")
    home_credit_validation = pd.read_csv(
        home_credit_prepared / "validation_processed.csv"
    )
    home_credit_test = pd.read_csv(home_credit_prepared / "test_processed.csv")
    home_credit_test_groups = pd.read_csv(
        home_credit_prepared / "test_groups.csv"
    )
    home_credit_features = [
        column for column in home_credit_train.columns
        if column != HOME_CREDIT_TARGET
    ]
    history_prefixes = (
        "BUREAU_",
        "PREVIOUS_",
        "INSTALLMENT_",
        "POS_CASH_",
        "CREDIT_CARD_",
    )
    history_features = [
        column for column in home_credit_features
        if column.startswith(history_prefixes)
    ]
    if not history_features:
        raise ValueError("No aggregated Home Credit history features were found.")
    results["home_credit_ablation"] = (
        cr_run_validation_selected_feature_ablation(
            home_credit_train,
            home_credit_validation,
            home_credit_test,
            {"prior_repayment_history": history_features},
            trainer=cr_train_home_credit_random_forest,
            model_name="Random Forest",
            target_column=HOME_CREDIT_TARGET,
            output_path=home_credit_dir / "metrics" / "feature_ablation.csv",
            test_groups=home_credit_test_groups,
            subgroup_output_path=home_credit_dir / "metrics" / "feature_ablation_subgroup_metrics.csv",
            subgroup_calibration_output_path=(
                home_credit_dir / "metrics" / "feature_ablation_subgroup_calibration.csv"
            ),
            subgroup_calibration_bins_output_path=(
                home_credit_dir / "metrics" / "feature_ablation_subgroup_calibration_bins.csv"
            ),
            paired_comparison_output_path=(
                home_credit_dir
                / "metrics"
                / "feature_ablation_paired_comparison_confidence_intervals.csv"
            ),
            paired_fairness_comparison_output_path=(
                home_credit_dir
                / "metrics"
                / "feature_ablation_paired_fairness_gap_confidence_intervals.csv"
            ),
        )
    )

    if include_explanation_stability:
        model_template = cr_create_home_credit_random_forest(
            home_credit_train,
            target_column=HOME_CREDIT_TARGET,
        )
        results["home_credit_explanation_stability"] = cr_run_explanation_stability(
            model_template,
            home_credit_train,
            home_credit_test.drop(columns=[HOME_CREDIT_TARGET]),
            home_credit_dir / "explainability",
            target_column=HOME_CREDIT_TARGET,
            random_states=(42, 43, 44),
            sample_size=100,
            background_size=100,
        )
    return results


def cr_run_saved_performance_uncertainty(
    output_root: str | Path = "outputs/credit_risk",
    *,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Refit baselines on saved training splits and write test metric intervals.

    Validation splits select the threshold anew; test splits are used only for
    metric estimation and the stratified bootstrap intervals.
    """
    root = Path(output_root)
    specifications = {
        "give_me_some_credit": (
            GIVE_ME_SOME_CREDIT_TARGET,
            cr_train_logistic_regression,
            cr_train_random_forest,
            False,
        ),
        "credit_card_default": (
            CREDIT_CARD_TARGET,
            cr_train_credit_card_logistic_regression,
            cr_train_credit_card_random_forest,
            False,
        ),
        "home_credit": (
            HOME_CREDIT_TARGET,
            cr_train_home_credit_logistic_regression,
            cr_train_home_credit_random_forest,
            True,
        ),
    }
    results: dict[str, pd.DataFrame] = {}
    for dataset, (target, logistic_trainer, forest_trainer, home_credit) in specifications.items():
        base = root / dataset / "completion_run"
        prepared = base / "prepared"
        train = pd.read_csv(prepared / "train_processed.csv")
        validation = pd.read_csv(prepared / "validation_processed.csv")
        test = pd.read_csv(prepared / "test_processed.csv")
        destination = base / ("metrics" if dataset != "give_me_some_credit" else "")
        destination.mkdir(parents=True, exist_ok=True)
        trainers = {
            "logistic_regression": logistic_trainer,
            "random_forest": forest_trainer,
        }
        rows = []
        probabilities_by_model: dict[str, np.ndarray] = {}
        thresholds_by_model: dict[str, float] = {}
        for model_key, trainer in trainers.items():
            model = (
                trainer(train, target_column=target)
                if home_credit
                else trainer(train)
            )
            threshold, _ = _select_threshold(validation, target, model)
            positive_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(
                test.drop(columns=[target])
            )[:, positive_index]
            probabilities_by_model[model_key] = probabilities
            thresholds_by_model[model_key] = threshold
            intervals = cr_bootstrap_binary_performance(
                test[target],
                probabilities,
                threshold=threshold,
                random_state=random_state,
            )
            intervals.insert(0, "model", model_key)
            intervals.insert(0, "dataset", dataset)
            intervals.to_csv(
                destination / f"{model_key}_performance_confidence_intervals.csv",
                index=False,
            )
            rows.append(intervals)

        paired_intervals = cr_bootstrap_paired_binary_performance(
            test[target],
            probabilities_by_model["logistic_regression"],
            probabilities_by_model["random_forest"],
            threshold_a=thresholds_by_model["logistic_regression"],
            threshold_b=thresholds_by_model["random_forest"],
            model_a_name="Logistic Regression",
            model_b_name="Random Forest",
            random_state=random_state,
        )
        paired_intervals.insert(0, "dataset", dataset)
        paired_intervals.to_csv(
            destination / "paired_model_comparison_confidence_intervals.csv",
            index=False,
        )
        results[dataset] = pd.concat(rows, ignore_index=True)

    cr_build_subgroup_support_coverage(
        str(root),
        str(root / "subgroup_support_coverage.csv"),
    )
    return results


def cr_run_component1_inspection(
    *,
    output_root: str | Path,
    give_me_some_credit_csv: str | Path | None = None,
    credit_card_archive: str | Path | None = None,
    home_credit_archive: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run each supplied dataset independently and save isolated outputs."""

    root = Path(output_root)
    results: dict[str, Any] = {}
    if give_me_some_credit_csv is not None:
        results["give_me_some_credit"] = cr_run_give_me_some_credit(
            give_me_some_credit_csv,
            root / "give_me_some_credit" / "completion_run",
            random_state=random_state,
        )
    if credit_card_archive is not None:
        destination = root / "credit_card_default" / "completion_run"
        train, test, metadata = cr_prepare_credit_card_dataset_from_file(
            credit_card_archive,
            output_dir=destination / "prepared",
            random_state=random_state,
        )
        results["credit_card_default"] = cr_run_credit_card_baselines(
            train,
            metadata["validation_processed"],
            test,
            test_groups=metadata["audit_groups"]["test"],
            output_dir=destination / "metrics",
        )
    if home_credit_archive is not None:
        destination = root / "home_credit" / "completion_run"
        train, test, metadata = cr_prepare_home_credit_dataset_from_archive(
            home_credit_archive,
            output_dir=destination / "prepared",
            random_state=random_state,
        )
        results["home_credit"] = cr_run_home_credit_baselines(
            train,
            metadata["validation_processed"],
            test,
            test_groups=metadata["audit_groups"]["test"],
            output_dir=destination / "metrics",
        )
    if not results:
        raise ValueError("Provide at least one dataset path to run.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/credit_risk")
    parser.add_argument("--give-me-some-credit-csv")
    parser.add_argument("--credit-card-archive")
    parser.add_argument("--home-credit-archive")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--run-extended-analysis",
        action="store_true",
        help="Run saved-split feature ablations for all datasets and Home Credit explanation stability.",
    )
    parser.add_argument(
        "--complete-ablation-uncertainty-review",
        action="store_true",
        help="Write paired performance and subgroup-fairness intervals for each feature ablation versus its full-feature model.",
    )
    parser.add_argument(
        "--complete-uncertainty-review",
        action="store_true",
        help="Write held-out performance intervals and subgroup support coverage from saved splits.",
    )
    parser.add_argument(
        "--complete-group-calibration-review",
        action="store_true",
        help="Write held-out subgroup calibration metrics, reliability bins, and within-group bootstrap intervals.",
    )
    parser.add_argument(
        "--complete-explanation-resampling-review",
        action="store_true",
        help="Compare Random Forest explanations across stratified training-row resamples.",
    )
    args = parser.parse_args()
    has_baseline_input = any(
        (
            args.give_me_some_credit_csv,
            args.credit_card_archive,
            args.home_credit_archive,
        )
    )
    if (
        not has_baseline_input
        and not args.run_extended_analysis
        and not args.complete_ablation_uncertainty_review
        and not args.complete_uncertainty_review
        and not args.complete_group_calibration_review
        and not args.complete_explanation_resampling_review
    ):
        parser.error(
            "Provide a dataset input or use --run-extended-analysis with saved splits."
        )
    results = (
        cr_run_component1_inspection(
            output_root=args.output_root,
            give_me_some_credit_csv=args.give_me_some_credit_csv,
            credit_card_archive=args.credit_card_archive,
            home_credit_archive=args.home_credit_archive,
            random_state=args.random_state,
        )
        if has_baseline_input
        else {}
    )
    for dataset, result in results.items():
        print(f"\n{dataset}")
        print(result["comparison"].to_string())
    if args.run_extended_analysis:
        extended = cr_run_extended_dataset_analyses(args.output_root)
        for analysis, result in extended.items():
            print(f"\n{analysis}")
            if isinstance(result, pd.DataFrame):
                print(result.to_string(index=False))
            else:
                print(result["summary"])
    elif args.complete_ablation_uncertainty_review:
        extended = cr_run_extended_dataset_analyses(
            args.output_root,
            include_explanation_stability=False,
        )
        for analysis, result in extended.items():
            if analysis.endswith("ablation") or "ablation" in analysis:
                print(f"\n{analysis}")
                if isinstance(result, pd.DataFrame):
                    print(result.to_string(index=False))
    if args.complete_uncertainty_review:
        uncertainty = cr_run_saved_performance_uncertainty(args.output_root)
        for dataset, table in uncertainty.items():
            print(f"\n{dataset} performance intervals")
            print(table.to_string(index=False))
    if args.complete_group_calibration_review:
        calibration = cr_run_saved_split_group_calibration(
            args.output_root,
            random_state=args.random_state,
        )
        for dataset, table in calibration.items():
            print(f"\n{dataset} subgroup calibration")
            print(table.to_string(index=False))
    if args.complete_explanation_resampling_review:
        stability = cr_run_saved_split_explanation_resampling(args.output_root)
        for dataset, result in stability.items():
            print(f"\n{dataset} explanation stability across training resamples")
            print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
