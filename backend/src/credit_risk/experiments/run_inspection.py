"""Run reproducible, dataset-specific Component 1 credit-risk baselines."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone

from backend.src.credit_risk.common.cr_evaluation import (
    cr_analyze_probability_thresholds,
)
from backend.src.credit_risk.credit_card_default.experiment import (
    cr_prepare_credit_card_dataset_from_file,
    cr_run_credit_card_baselines,
)
from backend.src.credit_risk.fairness.cr_fairness import (
    cr_analyze_group_fairness,
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
    args = parser.parse_args()
    results = cr_run_component1_inspection(
        output_root=args.output_root,
        give_me_some_credit_csv=args.give_me_some_credit_csv,
        credit_card_archive=args.credit_card_archive,
        home_credit_archive=args.home_credit_archive,
        random_state=args.random_state,
    )
    for dataset, result in results.items():
        print(f"\n{dataset}")
        print(result["comparison"].to_string())


if __name__ == "__main__":
    main()
