
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from .common import profile_frame
from ..constants import CLUSTER_IDS, CLUSTER_LABELS
from .. import data_loader as loader


def _render_clustering_evaluation() -> None:
    """Present saved 5D_k5 selection and quality evidence without reclustering."""
    st.subheader("Clustering evaluation")
    st.caption(
        "Unsupervised K-Means groups respondents by similarities across five financial "
        "inclusion dimensions. Classifier and SHAP results explain predictions of these "
        "saved profiles, not how K-Means formed them."
    )
    try:
        evidence = loader.load_clustering_evidence()
        candidates = pd.read_csv(loader.CLUSTER_SELECTION_MATRIX_PATH)
        decision = loader.CLUSTER_SELECTION_DECISION_PATH.read_text(encoding="utf-8")
        metadata = json.loads(loader.CLUSTER_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        st.warning(
            "Saved clustering evaluation evidence is missing or inconsistent. "
            "No clustering metrics are shown until the saved selection, stability, "
            "agreement, and model metadata can be validated."
        )
        return

    algorithm = str(metadata.get("algorithm", evidence["algorithm"]))
    if algorithm == "KMeans":
        algorithm = "K-Means"
    scaling = str(metadata.get("scaling", evidence["scaling"]))
    if scaling.lower().startswith("raw"):
        scaling = "Raw dimension scores"
    cluster_count = metadata.get("n_clusters", evidence["clusters"])
    st.markdown(
        f"**Selected solution: {algorithm} · {cluster_count} clusters · {scaling}**"
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Silhouette score", f"{evidence['silhouette_score']:.3f}")
    metric_columns[1].metric("Davies–Bouldin index", f"{evidence['davies_bouldin_score']:.3f}")
    metric_columns[2].metric("Calinski–Harabasz score", f"{evidence['calinski_harabasz_score']:,.1f}")
    for column, helper in zip(
        metric_columns,
        ("Higher is generally better.", "Lower is generally better.", "Higher is generally better."),
    ):
        column.caption(helper)

    consistency_columns = st.columns(2)
    if (
        evidence["seed_runs"] == 10
        and evidence["mean_pairwise_ari"] == 1.0
        and evidence["min_pairwise_ari"] == 1.0
    ):
        stability_value = "ARI 1.000"
    else:
        stability_value = (
            f"Mean {evidence['mean_pairwise_ari']:.3f} · "
            f"min {evidence['min_pairwise_ari']:.3f}"
        )
    consistency_columns[0].metric("Stability across 10 runs", stability_value)
    consistency_columns[0].caption("Mean and minimum agreement across saved seed runs.")
    consistency_columns[1].metric(
        "K-Means / GMM agreement", f"ARI {evidence['kmeans_gmm_ari']:.3f}"
    )
    consistency_columns[1].caption("Agreement between saved cluster assignments.")
    st.caption("ARI measures agreement between assignments; it does not establish substantive validity.")

    with st.expander("Why were five clusters selected?"):
        same_population = candidates.loc[
            candidates["candidate"].isin(["5D_k4", "5D_k5", "5D_k6"])
        ].copy()
        display_columns = [
            "candidate",
            "cluster_count",
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "smallest_cluster_percent",
            "kmeans_gmm_ari",
        ]
        if set(display_columns).issubset(same_population.columns):
            same_population = same_population[display_columns].rename(
                columns={
                    "candidate": "Saved candidate",
                    "cluster_count": "Number of clusters",
                    "silhouette_score": "Silhouette score",
                    "davies_bouldin_score": "Davies–Bouldin index",
                    "calinski_harabasz_score": "Calinski–Harabasz score",
                    "smallest_cluster_percent": "Smallest cluster share (%)",
                    "kmeans_gmm_ari": "K-Means / GMM ARI",
                }
            ).sort_values("Number of clusters")
            st.dataframe(
                same_population.style.format(
                    {
                        "Silhouette score": "{:.3f}",
                        "Davies–Bouldin index": "{:.3f}",
                        "Calinski–Harabasz score": "{:,.1f}",
                        "Smallest cluster share (%)": "{:.2f}",
                        "K-Means / GMM ARI": "{:.3f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning("Candidate comparison metrics are incomplete in the saved selection matrix.")
        rationale = decision.split("## Selection rationale", 1)
        if len(rationale) == 2:
            st.markdown(rationale[1].split("\n\n-", 1)[0].strip())
        else:
            st.warning("The saved selection rationale is unavailable.")
        alternatives = decision.split("## Comparison with alternatives", 1)
        if len(alternatives) == 2:
            st.markdown(alternatives[1].split("\n## ", 1)[0].strip())
        else:
            st.warning("The saved candidate-selection rationale is unavailable.")
        st.markdown(
            "The selected solution does not have the best value on every metric. "
            "The decision balances clustering quality, consistency, cluster sizes, "
            "interpretability, and research relevance."
        )

    with st.expander("Technical details"):
        st.caption(
            f"Initialization count: {metadata.get('n_init', 'Not recorded')} · "
            f"Random seed: {metadata.get('random_state', 'Not recorded')}."
        )
        selected_row = candidates.loc[candidates["candidate"] == evidence["candidate"]]
        if (
            len(selected_row) == 1
            and {"sample_size", "economies"}.issubset(selected_row.columns)
        ):
            sample_size = int(selected_row.iloc[0]["sample_size"])
            economy_count = int(selected_row.iloc[0]["economies"])
            st.caption(
                f"Analytical sample: {sample_size:,} respondents across {economy_count:,} "
                "economies, using five-dimension complete-case eligibility. This does not "
                "imply national representativeness."
            )
        if evidence.get("legacy_summary_conflict"):
            st.caption(
                "The earlier exploratory summary marked its k=6 silhouette leader as provisional "
                "and recorded no final model selection. The final report, candidate matrix, and "
                "validated model metadata agree on 5D_k5; displayed figures use that final evidence."
            )


def render(df: pd.DataFrame) -> None:
    

    # ---------------------------------------------------------
    # 1. Load saved profile summaries
    # ---------------------------------------------------------
    profiles = profile_frame(df)

    # Initialize the shared selection before rendering any
    # interactive controls that use this session-state key.
    if "profile_explorer_cluster" not in st.session_state:
        st.session_state["profile_explorer_cluster"] = CLUSTER_IDS[0]

    # Validate the saved selection against available cluster IDs.
    if st.session_state["profile_explorer_cluster"] not in CLUSTER_IDS:
        st.session_state["profile_explorer_cluster"] = CLUSTER_IDS[0]

    # ---------------------------------------------------------
    # 2. Interactive profile cards
    # ---------------------------------------------------------
    st.subheader("Explore the Five Inclusion Profiles")
    st.caption(
        "Select a profile card to explore its observed dimension "
        "patterns. Strengths and gaps are relative to the five "
        "dimension means, not policy thresholds."
    )

    card_columns = st.columns(5, gap="small")

    for card_col, cluster_id in zip(card_columns, CLUSTER_IDS):
        matched = profiles.loc[profiles["cluster_id"] == cluster_id]

        if matched.empty:
            with card_col:
                st.warning(f"Saved profile {cluster_id} is unavailable.")
            continue

        profile_row = matched.iloc[0]
        profile_name = CLUSTER_LABELS[cluster_id]
        is_selected = (
            st.session_state["profile_explorer_cluster"] == cluster_id
        )

        # Derive strengths and gaps from the validated profile means.
        dimension_values = {
            loader.DIMENSION_LABELS[column]: float(
                profile_row[f"{column}_mean"]
            )
            for column in loader.DIMENSION_COLUMNS
            if pd.notna(profile_row[f"{column}_mean"])
        }

        with card_col:
            with st.container(border=True):
                if is_selected:
                    st.markdown("**✓ SELECTED PROFILE**")
                else:
                    st.caption(f"Cluster {cluster_id}")

                st.markdown(f"### {profile_name}")

                st.metric(
                    label="Respondents",
                    value=f"{int(profile_row['respondents']):,}",
                    delta=(
                        f"{profile_row['percent_of_candidate']:.1f}% "
                        "of sample"
                    ),
                )

                if dimension_values:
                    highest_score = max(dimension_values.values())
                    lowest_score = min(dimension_values.values())
                    highest_dimensions = [
                        name
                        for name, score in dimension_values.items()
                        if abs(score - highest_score) < 1e-9
                    ]
                    lowest_dimensions = [
                        name
                        for name, score in dimension_values.items()
                        if abs(score - lowest_score) < 1e-9
                    ]

                    st.markdown("**Strongest observed dimension(s)**")
                    st.caption(
                        f"↑ {', '.join(highest_dimensions)}: {highest_score:.2f}"
                    )

                    st.markdown("**Weakest observed dimension(s)**")
                    st.caption(
                        f"↓ {', '.join(lowest_dimensions)}: {lowest_score:.2f}"
                    )
                else:
                    st.caption("Dimension means unavailable.")

                if st.button(
                    "Selected" if is_selected else "View profile",
                    key=f"select_profile_card_{cluster_id}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state["profile_explorer_cluster"] = (
                        cluster_id
                    )
                    st.rerun()

    st.divider()

    # ---------------------------------------------------------
    # 3. Saved-cluster selector
    # ---------------------------------------------------------
    selected = st.selectbox(
        "Saved cluster",
        CLUSTER_IDS,
        format_func=lambda cluster_id: (
            f"Cluster {cluster_id} — {CLUSTER_LABELS[cluster_id]}"
        ),
        key="profile_explorer_cluster",
    )

    selected_rows = profiles.loc[profiles["cluster_id"] == selected]

    if selected_rows.empty:
        st.warning("No saved summary is available for this profile.")
        return

    row = selected_rows.iloc[0]

    # ---------------------------------------------------------
    # 4. Selected profile dimension values
    # ---------------------------------------------------------
    values = pd.DataFrame(
        {
            "Dimension": [
                loader.DIMENSION_LABELS[column]
                for column in loader.DIMENSION_COLUMNS
            ],
            "Mean score": [
                row[f"{column}_mean"]
                for column in loader.DIMENSION_COLUMNS
            ],
        }
    )

    values["Mean score"] = pd.to_numeric(
        values["Mean score"],
        errors="coerce",
    )

    valid_values = values.dropna(subset=["Mean score"]).copy()

    

    

    # ---------------------------------------------------------
    # 6. Horizontal dimension-mean bar chart
    # ---------------------------------------------------------
    st.subheader("Dimension Means")

    if not valid_values.empty:
        bar_fig = px.bar(
            valid_values.sort_values("Mean score"),
            x="Mean score",
            y="Dimension",
            orientation="h",
            range_x=[0, 1],
            text=valid_values.sort_values("Mean score")["Mean score"].map(
                lambda score: f"{score:.2f}"
            ),
            title="Selected profile dimension means",
        )

        bar_fig.update_traces(textposition="outside", cliponaxis=False)
        bar_fig.update_layout(
            xaxis_title="Mean score (0 to 1)",
            yaxis_title="",
            margin=dict(l=10, r=30, t=55, b=35),
        )

        st.plotly_chart(
            bar_fig,
            config={"responsive": True},
        )

       
    else:
        st.warning("Dimension means are unavailable for this profile.")

    # ---------------------------------------------------------
    # 7. Saved profile description
    # ---------------------------------------------------------
    st.subheader("Profile interpretation")

    try:
        labels_path = (
            loader.CLUSTERING_REPORT_DIR
            / "fi_clustering_profile_labels_5D_k5.json"
        )

        labels_data = json.loads(
            labels_path.read_text(encoding="utf-8")
        )

        description = labels_data["cluster_profiles"][str(selected)][
            "description"
        ]

        st.info(f"Saved profile description: {description}")

    except (OSError, ValueError, KeyError, TypeError):
        st.caption(
            "Saved profile descriptions are unavailable. "
            "The dimension summaries above are calculated from "
            "the analytical dataset."
        )

    _render_clustering_evaluation()

    # ---------------------------------------------------------
