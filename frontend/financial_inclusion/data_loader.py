
"""Load validated Component 02 datasets, reports, and analysis summaries."""

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "financial_inclusion"
    / "processed"
    / "fi_final_analytical_dataset.csv"
)

SUPERVISED_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "financial_inclusion"
    / "metrics"
    / "fi_supervised_baseline_metrics.csv"
)

SUPERVISED_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "financial_inclusion"
    / "metrics"
    / "fi_supervised_classification_reports.txt"
)

SUPERVISED_MATRIX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "financial_inclusion"
    / "metrics"
    / "fi_supervised_confusion_matrices.txt"
)

CLUSTERING_REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "financial_inclusion"
    / "reports"
    / "03_clustering"
)
CLUSTER_SELECTION_DECISION_PATH = (
    CLUSTERING_REPORT_DIR / "fi_clustering_candidate_selection_decision.md"
)
CLUSTER_SELECTION_MATRIX_PATH = (
    CLUSTERING_REPORT_DIR / "fi_clustering_candidate_selection_matrix.csv"
)
CLUSTER_STABILITY_PATH = (
    CLUSTERING_REPORT_DIR / "fi_clustering_stability_summary.csv"
)
CLUSTER_ALGORITHM_AGREEMENT_PATH = (
    CLUSTERING_REPORT_DIR / "fi_clustering_algorithm_agreement.json"
)
CLUSTER_EXPLORATORY_SUMMARY_PATH = (
    CLUSTERING_REPORT_DIR / "fi_clustering_summary.json"
)
CLUSTER_MODEL_METADATA_PATH = (
    PROJECT_ROOT / "models" / "financial_inclusion"
    / "kmeans_5D_k5_metadata.json"
)


SHAP_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "financial_inclusion"
    / "reports"
    / "05_shap_explainability"
)

SHAP_GLOBAL_IMPORTANCE_PATH = (
    SHAP_OUTPUT_DIR / "fi_shap_global_importance.csv"
)

SHAP_LOCAL_EXPLANATIONS_PATH = (
    SHAP_OUTPUT_DIR / "fi_shap_local_explanations.csv"
)

SHAP_PREDICTIONS_PATH = (
    SHAP_OUTPUT_DIR / "fi_shap_predictions.csv"
)

SHAP_VALIDATION_SUMMARY_PATH = (
    SHAP_OUTPUT_DIR / "fi_shap_validation_summary.json"
)


CONTEXT_COLUMNS = [
    "wpid_random",
    "economy",
    "economycode",
    "regionwb",
    "wgt",
    "female",
    "age",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
]

DIMENSION_COLUMNS = [
    "score_connectivity",
    "score_financial_access",
    "score_digital_financial_usage",
    "score_digital_safety",
    "score_financial_resilience",
]

DIMENSION_LABELS = {
    "score_connectivity": "Connectivity",
    "score_financial_access": "Financial access",
    "score_digital_financial_usage": "Digital financial usage",
    "score_digital_safety": "Digital safety",
    "score_financial_resilience": "Financial resilience",
}

REQUIRED_COLUMNS = (
    CONTEXT_COLUMNS
    + DIMENSION_COLUMNS
    + ["cluster_id"]
)

CLUSTER_IDS = (0, 1, 2, 3, 4)
SUPERVISED_MODEL_ORDER = (
    "Majority Baseline",
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
)
SUPERVISED_METRIC_COLUMNS = (
    "Accuracy",
    "Balanced Accuracy",
    "Macro-F1",
    "Weighted-F1",
)


def prepare_supervised_model_comparison(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and order the four saved model results for dashboard display."""
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame.")
    required_columns = {"Model", *SUPERVISED_METRIC_COLUMNS}
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"Supervised metrics are missing columns: {sorted(missing_columns)}"
        )

    available_models = set(metrics["Model"].dropna())
    missing_models = [
        model for model in SUPERVISED_MODEL_ORDER
        if model not in available_models
    ]
    if missing_models:
        raise ValueError(
            f"Supervised metrics are missing expected models: {missing_models}"
        )

    result = metrics.set_index("Model").loc[
        list(SUPERVISED_MODEL_ORDER),
        list(SUPERVISED_METRIC_COLUMNS),
    ].reset_index()
    if result[list(SUPERVISED_METRIC_COLUMNS)].isna().any().any():
        raise ValueError("Supervised metrics contain missing metric values.")
    return result


def load_clustering_evidence() -> dict[str, object]:
    """Load the final selection's saved validation and stability evidence.

    The exploratory summary is checked separately because it predates the
    final decision report and labels itself provisional.
    """
    required_paths = [
        CLUSTER_SELECTION_DECISION_PATH,
        CLUSTER_SELECTION_MATRIX_PATH,
        CLUSTER_STABILITY_PATH,
        CLUSTER_ALGORITHM_AGREEMENT_PATH,
        CLUSTER_EXPLORATORY_SUMMARY_PATH,
        CLUSTER_MODEL_METADATA_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Clustering evidence files are missing: {missing_paths}"
        )

    decision_text = CLUSTER_SELECTION_DECISION_PATH.read_text(encoding="utf-8")
    if "Selected candidate: 5D_k5" not in decision_text or (
        "Final model selected: Yes" not in decision_text
    ):
        raise ValueError("Final clustering decision report does not confirm 5D_k5.")

    candidates = pd.read_csv(CLUSTER_SELECTION_MATRIX_PATH)
    required_candidate_columns = {
        "candidate",
        "algorithm",
        "scaling",
        "k",
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "kmeans_gmm_ari",
    }
    missing = required_candidate_columns.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"Clustering candidate report is missing columns: {sorted(missing)}"
        )
    selected = candidates.loc[candidates["candidate"] == "5D_k5"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one 5D_k5 candidate validation row.")
    selected_row = selected.iloc[0]

    stability = pd.read_csv(CLUSTER_STABILITY_PATH)
    required_stability_columns = {
        "candidate",
        "seed_runs",
        "pairwise_comparisons",
        "mean_pairwise_ari",
        "min_pairwise_ari",
    }
    missing = required_stability_columns.difference(stability.columns)
    if missing:
        raise ValueError(
            f"Clustering stability report is missing columns: {sorted(missing)}"
        )
    stability_row = stability.loc[stability["candidate"] == "5D_k5"]
    if len(stability_row) != 1:
        raise ValueError("Expected exactly one 5D_k5 stability row.")

    with CLUSTER_ALGORITHM_AGREEMENT_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        agreement = json.load(file)
    algorithm_row = agreement.get("5D_k5")
    if not isinstance(algorithm_row, dict):
        raise ValueError("Algorithm-agreement report does not include 5D_k5.")

    with CLUSTER_MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if (
        metadata.get("candidate") != "5D_k5"
        or metadata.get("validated") is not True
        or metadata.get("n_clusters") != 5
    ):
        raise ValueError("Saved K-Means metadata does not validate 5D_k5.")

    with CLUSTER_EXPLORATORY_SUMMARY_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        exploratory_summary = json.load(file)
    has_legacy_conflict = exploratory_summary.get("final_model_selected") is False

    row = stability_row.iloc[0]
    return {
        "candidate": "5D_k5",
        "algorithm": str(selected_row["algorithm"]),
        "scaling": str(selected_row["scaling"]),
        "clusters": int(selected_row["k"]),
        "silhouette_score": float(selected_row["silhouette_score"]),
        "davies_bouldin_score": float(selected_row["davies_bouldin_score"]),
        "calinski_harabasz_score": float(
            selected_row["calinski_harabasz_score"]
        ),
        "seed_runs": int(row["seed_runs"]),
        "pairwise_comparisons": int(row["pairwise_comparisons"]),
        "mean_pairwise_ari": float(row["mean_pairwise_ari"]),
        "min_pairwise_ari": float(row["min_pairwise_ari"]),
        "kmeans_gmm_ari": float(algorithm_row["adjusted_rand_index_kmeans_vs_gmm"]),
        "metadata_validated": metadata["validated"],
        "legacy_summary_conflict": has_legacy_conflict,
    }


def calculate_community_analysis(
    df: pd.DataFrame,
    economy: str,
    *,
    filter_column: str | None = None,
    filter_value: object | None = None,
    cluster_id: int | None = None,
    exclude_selected_economy: bool = False,
) -> dict[str, object]:
    """Compare one economy with the same-sample reference population.

    Optional demographic filtering is applied before both the selected-economy
    and reference means are calculated. The reference is the analytical sample
    mean, not an official benchmark. Missing dimension values are excluded from
    each mean independently and their valid counts are returned.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The analytical dataset cannot be empty.")
    if not isinstance(economy, str) or not economy.strip():
        raise ValueError("economy must be a non-empty string.")

    required = ["economy", "cluster_id", *DIMENSION_COLUMNS]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Analytical dataset is missing required columns: {missing_columns}"
        )

    population = df
    if filter_column is not None:
        if filter_column not in df.columns:
            raise ValueError(f"Unknown filter column: {filter_column}")
        if filter_column in {"economy", "cluster_id", *DIMENSION_COLUMNS}:
            raise ValueError(f"Unsupported filter column: {filter_column}")
        if filter_value is not None:
            population = df.loc[df[filter_column] == filter_value]

    if cluster_id is not None:
        if cluster_id not in CLUSTER_IDS:
            raise ValueError(f"Unknown saved cluster ID: {cluster_id}")
        population = population.loc[population["cluster_id"] == cluster_id]

    if population.empty:
        raise ValueError("No respondents match the selected filter.")

    selected = population.loc[population["economy"] == economy]
    if selected.empty:
        raise ValueError(
            f"No respondents for economy {economy!r} match the selected filter."
        )

    reference_population = population
    if exclude_selected_economy:
        reference_population = population.loc[population["economy"] != economy]
        if reference_population.empty:
            raise ValueError(
                "No respondents remain in the reference after excluding the selected economy."
            )

    score_values = reference_population[DIMENSION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    selected_values = selected[DIMENSION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    comparison_rows = []
    for column in DIMENSION_COLUMNS:
        reference = score_values[column]
        local = selected_values[column]
        reference_mean = reference.mean()
        economy_mean = local.mean()
        comparison_rows.append(
            {
                "dimension": DIMENSION_LABELS[column],
                "column": column,
                "economy_mean": economy_mean,
                "reference_mean": reference_mean,
                "difference": economy_mean - reference_mean,
                "economy_valid_respondents": int(local.notna().sum()),
                "reference_valid_respondents": int(reference.notna().sum()),
                "economy_missing_respondents": int(local.isna().sum()),
                "reference_missing_respondents": int(reference.isna().sum()),
            }
        )

    clusters = pd.to_numeric(selected["cluster_id"], errors="coerce")
    if clusters.isna().any() or not clusters.isin(CLUSTER_IDS).all():
        raise ValueError("Selected economy has missing or invalid cluster IDs.")
    counts = clusters.astype(int).value_counts().reindex(CLUSTER_IDS, fill_value=0)
    cluster_distribution = pd.DataFrame(
        {
            "cluster_id": list(CLUSTER_IDS),
            "respondents": counts.to_numpy(dtype=int),
        }
    )
    cluster_distribution["percent_of_economy"] = (
        cluster_distribution["respondents"] / len(selected) * 100
    )

    return {
        "economy": economy,
        "respondents": len(selected),
        "reference_respondents": len(reference_population),
        "filter_column": filter_column,
        "filter_value": filter_value,
        "cluster_id": cluster_id,
        "exclude_selected_economy": exclude_selected_economy,
        "comparison": pd.DataFrame(comparison_rows),
        "cluster_distribution": cluster_distribution,
    }


def build_community_recommendations(
    comparison: pd.DataFrame,
    *,
    reference_label: str = "analytical-sample reference",
) -> list[dict[str, str | float]]:
    """Create stakeholder hypotheses only for observed negative score gaps."""
    required = {"dimension", "difference"}
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(
            f"Comparison data is missing required columns: {sorted(missing)}"
        )

    guidance = {
        "Connectivity": (
            "Could phone access, internet availability, or affordability be "
            "useful topics for local stakeholder review?",
            "Policymakers, community organizations, connectivity providers",
            "Phone and internet access indicators in a future comparable survey",
        ),
        "Financial access": (
            "Could account availability, affordability, documentation, or "
            "registration barriers be explored with residents?",
            "Financial service providers, policymakers, community organizations",
            "Account-access indicator and financial-access score in a future "
            "comparable survey",
        ),
        "Digital financial usage": (
            "Could awareness, usability, merchant acceptance, transaction "
            "costs, or service availability be investigated?",
            "Financial service providers, merchants, community organizations",
            "Digital-payment usage indicator and digital-usage score in a "
            "future comparable survey",
        ),
        "Digital safety": (
            "Could accessible phone-protection guidance or consumer-safety "
            "support be worth validating with residents?",
            "Consumer-protection bodies, financial service providers, "
            "community organizations",
            "Available phone-protection indicator and digital-safety score "
            "in a future comparable survey",
        ),
        "Financial resilience": (
            "Could opportunities to save or reported emergency financial "
            "capacity be explored further?",
            "Financial service providers, policymakers, community organizations",
            "Underlying resilience indicators and resilience score in a "
            "future comparable survey",
        ),
    }

    recommendations = []
    for row in comparison.loc[comparison["difference"] < 0].sort_values(
        "difference"
    ).itertuples(index=False):
        dimension = row.dimension
        if dimension not in guidance:
            raise ValueError(f"Unsupported dimension for recommendations: {dimension}")
        action, stakeholder, monitor = guidance[dimension]
        recommendations.append(
            {
                "finding_id": f"negative_mean_gap_{dimension.lower().replace(' ', '_')}",
                "dimension": dimension,
                "comparison": f"selected population versus {reference_label}",
                "difference": float(row.difference),
                "evidence": (
                    f"Selected population mean is {abs(float(row.difference)):.3f} "
                    "points below the reference mean."
                ),
                "interpretation": (
                    "This is a descriptive difference that may help prioritize "
                    "questions for local validation; it does not identify a cause."
                ),
                "observed_gap": (
                    f"{dimension} is {abs(float(row.difference)):.3f} points "
                    f"below the {reference_label} mean."
                ),
                "action": action,
                "stakeholder": stakeholder,
                "monitor": monitor,
                "limitation": (
                    "This is a hypothesis for stakeholder validation, not a "
                    "proven causal explanation or intervention."
                ),
            }
        )
    return recommendations


def format_suggested_actions(
    actions: list[dict[str, object]],
) -> pd.DataFrame:
    """Select readable action fields instead of stringifying dictionaries."""
    columns = ["dimension", "reason", "action", "stakeholder", "monitor"]
    if not isinstance(actions, list):
        raise TypeError("actions must be a list of action dictionaries.")

    rows = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise TypeError(f"Action {index} must be a dictionary.")
        missing = [column for column in columns if column not in action]
        if missing:
            raise ValueError(f"Action {index} is missing fields: {missing}")
        rows.append({column: action[column] for column in columns})
    return pd.DataFrame(rows, columns=columns)


def load_final_dataset() -> pd.DataFrame:
    """Load and validate the final respondent-level analytical dataset."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Final analytical dataset not found: {DATASET_PATH}. "
            "Run the Component 02 final dataset preparation script first."
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Final analytical dataset is missing columns: {missing_columns}"
        )

    if df["wpid_random"].isna().any():
        raise ValueError("The dataset contains missing respondent IDs.")

    if df["wpid_random"].duplicated().any():
        raise ValueError("The dataset contains duplicate respondent IDs.")

    if df["cluster_id"].isna().any():
        raise ValueError("The dataset contains missing cluster labels.")

    observed_clusters = set(df["cluster_id"].unique())
    expected_clusters = {0, 1, 2, 3, 4}

    if observed_clusters != expected_clusters:
        raise ValueError(
            "Expected the final 5D_k5 cluster IDs {0, 1, 2, 3, 4}; "
            f"found {sorted(observed_clusters)}."
        )

    return df


def calculate_dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dimension statistics directly from respondent data."""
    rows = []

    for column in DIMENSION_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")

        rows.append(
            {
                "variable": column,
                "mean": values.mean(),
                "median": values.median(),
                "distinct_nonmissing_values": values.nunique(),
                "missing_percent": values.isna().mean() * 100,
            }
        )

    return pd.DataFrame(rows)


def calculate_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final cluster sizes and dimension means from the dataset."""
    profile = (
        df.groupby("cluster_id")
        .agg(
            respondents=("wpid_random", "nunique"),
            economies_represented=("economy", "nunique"),
            **{
                f"{column}_mean": (column, "mean")
                for column in DIMENSION_COLUMNS
            },
        )
        .reset_index()
    )

    total = profile["respondents"].sum()
    profile["percent_of_candidate"] = (
        profile["respondents"] / total * 100
    )

    return profile


def calculate_cluster_demographics(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calculate demographic composition by cluster, excluding missing values."""
    if column not in df.columns:
        raise ValueError(f"Unknown demographic column: {column}")

    result = (
        df.dropna(subset=[column])
        .groupby(["cluster_id", column])
        .size()
        .rename("respondents")
        .reset_index()
    )

    denominators = result.groupby("cluster_id")["respondents"].transform("sum")
    result["percent_within_cluster"] = (
        result["respondents"] / denominators * 100
    )

    return result


def load_supervised_metrics() -> pd.DataFrame:
    """Load the saved supervised evaluation metrics from the reproducible pipeline."""
    if not SUPERVISED_METRICS_PATH.is_file():
        raise FileNotFoundError(
            "Supervised metrics file not found: "
            f"{SUPERVISED_METRICS_PATH}. Run the supervised evaluation pipeline first."
        )

    metrics = pd.read_csv(SUPERVISED_METRICS_PATH)

    required_columns = {
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    }
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise ValueError(
            "Supervised metrics file is missing columns: "
            f"{sorted(missing)}"
        )

    renamed = metrics.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted-F1",
        }
    )

    return renamed[
        ["Model", "Accuracy", "Balanced Accuracy", "Macro-F1", "Weighted-F1"]
    ]


def load_supervised_classification_reports() -> dict:
    """Load per-model classification reports from the saved text output."""
    if not SUPERVISED_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "Supervised classification report file not found: "
            f"{SUPERVISED_REPORT_PATH}"
        )

    content = SUPERVISED_REPORT_PATH.read_text(encoding="utf-8")
    reports = {
        match.group(1).strip(): match.group(2).strip()
        for match in re.finditer(
            r"(?ms)^===\s*(.*?)\s*===\s*$\s*(.*?)(?=^===|\Z)",
            content,
        )
        if "precision" in match.group(2).lower()
        or "accuracy" in match.group(2).lower()
    }
    return reports


def load_supervised_confusion_matrices() -> dict:
    """Parse confusion matrix blocks from the saved evaluation output."""
    if not SUPERVISED_MATRIX_PATH.is_file():
        raise FileNotFoundError(
            "Supervised confusion matrices file not found: "
            f"{SUPERVISED_MATRIX_PATH}"
        )

    text = SUPERVISED_MATRIX_PATH.read_text(encoding="utf-8")
    matrices = {}
    sections = re.finditer(
        r"(?ms)^===\s*(.*?)\s*===\s*$\s*(.*?)(?=^===|\Z)",
        text,
    )
    for match in sections:
        name, body = match.group(1).strip(), match.group(2)
        if "Rows = actual clusters" not in body:
            continue
        matrix_block = re.search(r"\[\[.*?\]\]", body, flags=re.DOTALL)
        if matrix_block is None:
            continue
        rows = [
            [int(value) for value in row.split()]
            for row in re.findall(
                r"\[\s*(-?\d+(?:\s+-?\d+){4})\s*\]",
                matrix_block.group(0),
            )
        ]
        if rows:
            if len(rows) != len(CLUSTER_IDS) or any(
                len(row) != len(CLUSTER_IDS) for row in rows
            ):
                raise ValueError(
                    f"Confusion matrix for {name} must be 5 by 5."
                )
            matrices[name] = rows
    return matrices


def load_shap_global_importance() -> pd.DataFrame:
    """Load saved global SHAP importance results."""
    required = {
        "cluster_id",
        "cluster_label",
        "feature",
        "mean_absolute_shap_decision_score",
        "rank_within_cluster",
    }

    if not SHAP_GLOBAL_IMPORTANCE_PATH.is_file():
        raise FileNotFoundError(
            f"SHAP global importance file not found: "
            f"{SHAP_GLOBAL_IMPORTANCE_PATH}"
        )

    df = pd.read_csv(SHAP_GLOBAL_IMPORTANCE_PATH)
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"SHAP global importance file is missing columns: "
            f"{sorted(missing)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError("SHAP global importance contains missing values.")

    if not df["cluster_id"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("SHAP global importance has unexpected cluster IDs.")

    return df


def load_shap_local_explanations() -> pd.DataFrame:
    """Load saved per-respondent SHAP contributions."""
    required = {
        "row_index",
        "predicted_cluster_id",
        "predicted_cluster_label",
        "explained_class_id",
        "feature",
        "shap_decision_score_contribution",
        "source_row_index",
    }

    if not SHAP_LOCAL_EXPLANATIONS_PATH.is_file():
        raise FileNotFoundError(
            f"SHAP local explanations file not found: "
            f"{SHAP_LOCAL_EXPLANATIONS_PATH}"
        )

    df = pd.read_csv(SHAP_LOCAL_EXPLANATIONS_PATH)
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"SHAP local explanations file is missing columns: "
            f"{sorted(missing)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError("SHAP local explanations contain missing values.")

    if not df["predicted_cluster_id"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("SHAP local explanations have unexpected cluster IDs.")

    return df


def load_shap_predictions() -> pd.DataFrame:
    """Load saved classifier predictions for the explained sample."""
    required = {
        "row_index",
        "predicted_cluster_id",
        "predicted_cluster_label",
        "predicted_cluster_probability",
        "source_row_index",
    }

    if not SHAP_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"SHAP predictions file not found: {SHAP_PREDICTIONS_PATH}"
        )

    df = pd.read_csv(SHAP_PREDICTIONS_PATH)
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"SHAP predictions file is missing columns: {sorted(missing)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError("SHAP predictions contain missing values.")

    if not df["predicted_cluster_id"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("SHAP predictions have unexpected cluster IDs.")

    probability = df["predicted_cluster_probability"]

    if not probability.between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    return df


def load_shap_validation_summary() -> dict:
    """Load metadata describing the saved SHAP validation run."""
    import json

    if not SHAP_VALIDATION_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"SHAP validation summary not found: "
            f"{SHAP_VALIDATION_SUMMARY_PATH}"
        )

    with SHAP_VALIDATION_SUMMARY_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        summary = json.load(file)

    if not isinstance(summary, dict):
        raise ValueError("SHAP validation summary must contain a JSON object.")

    # Keep failed or incomplete diagnostics available for the UI to report;
    # a failed audit is not the same as a missing artifact.

    return summary
