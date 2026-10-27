
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .common import profile_frame
from ..constants import CLUSTER_IDS, CLUSTER_LABELS
from .. import data_loader as loader


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
                    highest_dimension = max(
                        dimension_values,
                        key=dimension_values.get,
                    )
                    lowest_dimension = min(
                        dimension_values,
                        key=dimension_values.get,
                    )

                    st.markdown("**Relative strength**")
                    st.caption(
                        f"↑ {highest_dimension}: "
                        f"{dimension_values[highest_dimension]:.2f}"
                    )

                    st.markdown("**Relative gap**")
                    st.caption(
                        f"↓ {lowest_dimension}: "
                        f"{dimension_values[lowest_dimension]:.2f}"
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
            use_container_width=True,
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

    # ---------------------------------------------------------
    # 8. User-friendly individual scoring and profile inference
    # ---------------------------------------------------------
    with st.expander(
        "Try individual profile scoring",
        expanded=False,
        icon=":material/person_search:",
    ):
        st.markdown("#### Explore a respondent's inclusion profile")
        st.caption(
            "Answer the questions below using the respondent's known "
            "information. Choose 'Not available' when an answer is "
            "unknown or was not collected. The result is a descriptive "
            "inclusion profile, not a credit or loan decision."
        )

        root = Path(__file__).resolve().parents[3]

        for folder in ("02_dimension_creation", "03_clustering"):
            path = (
                root
                / "backend"
                / "src"
                / "financial_inclusion"
                / folder
            )
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from fi_individual_profiling import profile_individual

        # Human-readable options mapped to the backend's expected codes.
        yes_no_options = {
            "Not available": None,
            "Yes": 1,
            "No": 2,
        }

        binary_options = {
            "Not available": None,
            "Yes": 1,
            "No": 0,
        }

        emergency_options = {
            "Not available": None,
            "Very difficult": 1,
            "Somewhat difficult": 2,
            "Not difficult at all": 3,
            "Don't know": 4,
            "Prefer not to answer": 5,
        }

        income_loss_options = {
            "Not available": None,
            "Less than 2 weeks": 1,
            "About 1 month": 2,
            "About 2 months": 3,
            "More than 2 months": 4,
            "Don't know": 5,
            "Prefer not to answer": 6,
        }

        with st.form("fi_individual_profile_form", border=True):
            # -------------------------------------------------
            # A. Access and connectivity
            # -------------------------------------------------
            st.markdown("### 1. Access and connectivity")
            st.caption(
                "Tell us about the respondent's access to a phone "
                "and the internet."
            )

            col1, col2 = st.columns(2)

            with col1:
                mobile_answer = st.selectbox(
                    "Does the respondent personally own a mobile phone?",
                    options=list(yes_no_options.keys()),
                    index=0,
                    help=(
                        "Choose Yes if the respondent owns a mobile "
                        "phone, No if they do not, or Not available "
                        "if this information is unknown."
                    ),
                )

            with col2:
                internet_answer = st.selectbox(
                    "Has the respondent used the internet recently?",
                    options=list(binary_options.keys()),
                    index=0,
                    help=(
                        "Use the validated internet-use indicator "
                        "from your prepared dataset. Yes means the "
                        "indicator is 1; No means it is 0."
                    ),
                )

            st.divider()

            # -------------------------------------------------
            # B. Financial access and digital activity
            # -------------------------------------------------
            st.markdown("### 2. Financial access and digital activity")
            st.caption(
                "Describe account access, digital payments, and "
                "phone protection."
            )

            col3, col4 = st.columns(2)

            with col3:
                account_answer = st.selectbox(
                    "Does the respondent have a financial account?",
                    options=list(binary_options.keys()),
                    index=0,
                    help=(
                        "Use the validated account indicator. "
                        "Yes = 1; No = 0."
                    ),
                )

                payment_answer = st.selectbox(
                    "Has the respondent made or received digital payments?",
                    options=list(binary_options.keys()),
                    index=0,
                    help=(
                        "Use the validated digital-payment indicator. "
                        "Yes = 1; No = 0."
                    ),
                )

            with col4:
                protection_answer = st.selectbox(
                    "Does the respondent use phone security protection?",
                    options=list(yes_no_options.keys()),
                    index=0,
                    help=(
                        "Use the project's validated phone-protection "
                        "coding: Yes = 1 and No = 2."
                    ),
                )

                saving_answer = st.selectbox(
                    "Has the respondent saved money?",
                    options=list(binary_options.keys()),
                    index=0,
                    help=(
                        "Use the validated saving indicator. "
                        "Yes = 1; No = 0."
                    ),
                )

            st.divider()

            # -------------------------------------------------
            # C. Financial resilience
            # -------------------------------------------------
            st.markdown("### 3. Financial resilience")
            st.caption(
                "These questions help describe how the respondent "
                "could manage an unexpected expense or loss of income."
            )

            resilience_col1, resilience_col2 = st.columns(2)

            with resilience_col1:
                st.markdown("**Emergency funds within 30 days**")
                st.caption(
                    "How difficult would it be to find emergency "
                    "money within the next 30 days?"
                )

                emergency_answer = st.selectbox(
                    "Ability to find emergency funds",
                    options=list(emergency_options.keys()),
                    index=0,
                    help=(
                        "Very difficult, Somewhat difficult, or Not "
                        "difficult at all are the three substantive "
                        "responses. The question may only have been "
                        "asked of respondents who reported a possible "
                        "source of emergency funds."
                    ),
                )

            with resilience_col2:
                st.markdown("**Coping with loss of income**")
                st.caption(
                    "If the household lost its main income source, "
                    "how long could it cover expenses?"
                )

                income_loss_answer = st.selectbox(
                    "How long could expenses be covered?",
                    options=list(income_loss_options.keys()),
                    index=0,
                    help=(
                        "Choose the closest response: less than "
                        "2 weeks, about 1 month, about 2 months, "
                        "or more than 2 months."
                    ),
                )

            st.info(
                "Choose 'Not available' if the respondent was not "
                "asked a question or the answer is unknown. Do not "
                "select 'No' as a substitute for missing information. "
                "'Don't know' and 'Prefer not to answer' are separate "
                "responses where provided."
            )

            submitted = st.form_submit_button(
                "Calculate inclusion profile",
                type="primary",
                use_container_width=True,
            )

        # -----------------------------------------------------
        # Convert friendly labels to the expected backend codes
        # -----------------------------------------------------

        if submitted:
            entries = {
                "con1": yes_no_options[mobile_answer],
                "internet_use": binary_options[internet_answer],
                "account": binary_options[account_answer],
                "anydigpayment": binary_options[payment_answer],
                "con18": yes_no_options[protection_answer],
                "saved": binary_options[saving_answer],
                "fin24a": emergency_options[emergency_answer],
                "fin24b": income_loss_options[income_loss_answer],
            }

            st.divider()
            st.subheader("Digital Financial Inclusion Assessment")

            # -----------------------------------------------------
            # 1. Validate substantive resilience responses
            # -----------------------------------------------------
            invalid_resilience = []
            if entries["fin24a"] not in (1, 2, 3):
                invalid_resilience.append("Emergency-funds difficulty")
            if entries["fin24b"] not in (1, 2, 3, 4):
                invalid_resilience.append("Household income-loss buffer")

            if invalid_resilience:
                st.warning(
                    "The assessment is incomplete because the following "
                    "resilience answers are missing or non-substantive: "
                    + ", ".join(invalid_resilience)
                    + ". Select a substantive response only when it is known "
                    "and applicable. 'Don't know', refusal, and not asked "
                    "must not be converted into a substantive score."
                )
            else:
                result = profile_individual(entries)

                if result.get("status") != "success":
                    st.error(
                        "The saved profile model could not complete the "
                        "assessment: "
                        + str(result.get("message", "Unknown error"))
                    )
                    unavailable = result.get("unavailable_dimensions", [])
                    if unavailable:
                        st.caption(
                            "Dimensions requiring more information: "
                            + ", ".join(unavailable)
                        )
                else:
                    assigned_profile = result["profile"]
                    assigned_cluster = assigned_profile["cluster_id"]
                    dimension_scores = result.get("dimension_scores", {})

                    # -------------------------------------------------
                    # 2. Map backend scores to the project's dimensions
                    # -------------------------------------------------
                    score_lookup = {}
                    for key, value in dimension_scores.items():
                        try:
                            numeric_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        if pd.notna(numeric_value):
                            score_lookup[str(key).lower().strip()
                                         ] = numeric_value

                    ordered_scores = []
                    for column in loader.DIMENSION_COLUMNS:
                        label = loader.DIMENSION_LABELS[column]
                        candidates = {
                            str(column).lower().strip(),
                            str(label).lower().strip(),
                        }
                        score = None

                        for candidate in candidates:
                            if candidate in score_lookup:
                                score = score_lookup[candidate]
                                break

                        if score is None:
                            for key, value in score_lookup.items():
                                normalized_key = key.replace(
                                    "_score", "").strip()
                                if normalized_key in candidates:
                                    score = value
                                    break

                        if score is not None and (
                            pd.isna(score) or score < 0 or score > 1
                        ):
                            score = None

                        ordered_scores.append(
                            {
                                "key": str(column).lower().strip(),
                                "dimension": label,
                                "score": score,
                            }
                        )

                    missing_dimensions = [
                        item["dimension"]
                        for item in ordered_scores
                        if item["score"] is None
                    ]

                    # The executive summary is rendered with the validated scores below.
                   

                    if missing_dimensions:
                        with st.container(border=True):
                            profile_col, cluster_col = st.columns([3, 1])
                            with profile_col:
                                st.caption("SAVED K-MEANS PROFILE")
                                st.markdown(f"### {CLUSTER_LABELS[assigned_cluster]}")
                            with cluster_col:
                                st.metric("Cluster ID", str(assigned_cluster))

                        st.warning(
                            "The overall score and rule-based assessment are "
                            "withheld because these dimension scores could not "
                            "be validated: " + ", ".join(missing_dimensions)
                            + ". Verify the score keys returned by "
                            "profile_individual() and the dimension metadata."
                        )
                        available_scores = [
                            item for item in ordered_scores
                            if item["score"] is not None
                        ]
                        if available_scores:
                            st.markdown("#### Available dimension scores")
                            cols = st.columns(min(5, len(available_scores)))
                            for i, item in enumerate(available_scores):
                                with cols[i % len(cols)]:
                                    st.metric(item["dimension"],
                                              f"{item['score']:.2f}")
                    else:
                        # Provisional index: equal-weight mean of all five dimensions.
                        overall_score = sum(
                            item["score"] for item in ordered_scores
                        ) / len(ordered_scores)
                        score_out_of_100 = overall_score * 100
                        strongest = max(
                            ordered_scores, key=lambda x: x["score"])
                        weakest = min(ordered_scores, key=lambda x: x["score"])
                        strengths = [
                            x for x in ordered_scores if x["score"] >= 0.75]
                        gaps = sorted(
                            [x for x in ordered_scores if x["score"] < 0.50],
                            key=lambda x: x["score"],
                        )
                        critical = [
                            x for x in ordered_scores if x["score"] < 0.25]

                        if critical:
                            overall_status = "Priority inclusion gaps"
                            status_explanation = (
                                "At least one dimension is below 0.25. Review the "
                                "dimension-level results rather than relying on the average."
                            )
                        elif overall_score >= 0.75 and gaps:
                            overall_status = "Strong average, specific gaps remain"
                            status_explanation = (
                                "The overall mean is high, but one or more dimensions "
                                "remain below 0.50 and deserve review."
                            )
                        elif overall_score >= 0.75:
                            overall_status = "Strong overall inclusion pattern"
                            status_explanation = (
                                "The overall mean is high under the provisional rules. "
                                "This is not proof of complete inclusion."
                            )
                        elif overall_score >= 0.50:
                            overall_status = "Partial inclusion"
                            status_explanation = (
                                "The overall mean is intermediate. Focus on the "
                                "lowest-scoring dimensions and the respondent's needs."
                            )
                        else:
                            overall_status = "Inclusion gaps identified"
                            status_explanation = (
                                "The overall mean is low under the provisional rules. "
                                "Review the underlying indicators and possible barriers."
                            )

                        # Executive summary: keep all four items in one row.
                        # Markdown blocks are used for the longer dimension names
                        # so values wrap instead of being clipped by metric widgets.
                        summary_cols = st.columns([3.2, 1, 2, 2], gap="medium")

                        with summary_cols[0]:
                            with st.container(border=True):
                                st.caption("SAVED K-MEANS PROFILE")
                                st.markdown(f"### {CLUSTER_LABELS[assigned_cluster]}")
                                st.caption(
                                    "Descriptive assignment from the saved clustering model"
                                )

                        with summary_cols[1]:
                            st.metric("Cluster ID", str(assigned_cluster))

                        with summary_cols[2]:
                            st.markdown("**Strongest area**")
                            st.markdown(
                                f"<div style='font-size:1.35rem; line-height:1.3; "
                                f"overflow-wrap:anywhere; word-break:normal;'>"
                                f"{strongest['dimension']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Score: {strongest['score']:.2f}")

                        with summary_cols[3]:
                            st.markdown("**Priority gap**")
                            st.markdown(
                                f"<div style='font-size:1.35rem; line-height:1.3; "
                                f"overflow-wrap:anywhere; word-break:normal;'>"
                                f"{weakest['dimension']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Score: {weakest['score']:.2f}")

                        st.info(status_explanation)

                        # -------------------------------------------------
                        # 4. Readiness gauge — visible without an expander
                        # -------------------------------------------------
                        gauge_fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=score_out_of_100,
                                number={"suffix": "/100", "valueformat": ".1f"},
                                title={"text": "Overall dimension mean"},
                                gauge={
                                    "axis": {"range": [0, 100], "tickwidth": 1},
                                    "bar": {"color": "#0ea5e9"},
                                    "steps": [
                                        {"range": [0, 25], "color": "#fecaca"},
                                        {"range": [25, 50], "color": "#fed7aa"},
                                        {"range": [50, 75], "color": "#fef3c7"},
                                        {"range": [75, 100], "color": "#bbf7d0"},
                                    ],
                                },
                            )
                        )
                        gauge_fig.update_layout(
                            height=300,
                            margin=dict(t=35, b=20, l=25, r=25),
                        )
                        st.plotly_chart(
                            gauge_fig,
                            use_container_width=True,
                            config={"responsive": True},
                        )

                        

                        # -------------------------------------------------
                        # 5. Professional dimension cards
                        # -------------------------------------------------
                        st.markdown("### Five-dimension health check")

                        def score_level(score):
                            if score < 0.25:
                                return "Priority gap", "🔴"
                            if score < 0.50:
                                return "Needs attention", "🟠"
                            if score < 0.75:
                                return "Intermediate", "🟡"
                            return "Relatively high", "🟢"

                        dimension_cols = st.columns(5, gap="small")
                        for col, item in zip(dimension_cols, ordered_scores):
                            level, marker_emoji = score_level(item["score"])
                            with col:
                                with st.container(border=True):
                                    st.caption(item["dimension"])
                                    st.markdown(f"## {marker_emoji}")
                                    st.markdown(
                                        f"### {item['score'] * 100:.0f}%")
                                    st.caption(level)
                                    st.progress(float(item["score"]))

                        # -------------------------------------------------
                        # 6. Strengths and gaps in a two-column layout
                        # -------------------------------------------------
                        st.markdown("### Strengths and gaps")
                        left, right = st.columns(2, gap="medium")

                        with left:
                            with st.container(border=True):
                                st.markdown("#### Key strengths")
                                if strengths:
                                    for item in sorted(
                                        strengths,
                                        key=lambda x: x["score"],
                                        reverse=True,
                                    ):
                                        st.markdown(
                                            f"🟢 **{item['dimension']}**  "
                                            f"`{item['score']:.2f}`"
                                        )
                                else:
                                    st.caption(
                                        "No dimension reached the provisional 0.75 threshold."
                                    )

                        with right:
                            with st.container(border=True):
                                st.markdown("#### Gaps to review")
                                if gaps:
                                    for item in gaps:
                                        _, marker_emoji = score_level(
                                            item["score"])
                                        st.markdown(
                                            f"{marker_emoji} **{item['dimension']}**  "
                                            f"`{item['score']:.2f}`"
                                        )
                                else:
                                    st.caption(
                                        "No dimension is below the provisional 0.50 threshold."
                                    )

                        

                        # -------------------------------------------------
                        # 8. Results-based recommended actions
                        # -------------------------------------------------
                        st.markdown("### Recommended actions")

                        recommendation_catalog = {
                            "connectivity": (
                                "Check whether phone ownership, reliable internet, "
                                "data costs, or access to assisted services are barriers. "
                                "Offer practical connectivity or non-digital service options."
                            ),
                            "access": (
                                "Discuss barriers to opening or using a financial account. "
                                "Explain available account options, fees, documentation, "
                                "and access requirements in clear language."
                            ),
                            "usage": (
                                "Ask whether cost, trust, usability, access, or familiarity "
                                "limits digital payments. If wanted, offer a short demonstration "
                                "covering fees and safe use."
                            ),
                            "safety": (
                                "Offer practical guidance on device locks, secure authentication, "
                                "fraud recognition, and reporting suspicious transactions."
                            ),
                            "resilience": (
                                "Discuss the respondent's ability to handle an unexpected expense. "
                                "If support is wanted, explain feasible saving options and any "
                                "associated costs or access conditions; do not default to borrowing."
                            ),
                        }

                        def recommendation_category(item):
                            combined = f"{item['key']} {item['dimension']}".lower()
                            if "connectivity" in combined or "internet" in combined:
                                return "connectivity"
                            if "safety" in combined or "security" in combined:
                                return "safety"
                            if "resilience" in combined or "saving" in combined:
                                return "resilience"
                            if "usage" in combined or "payment" in combined or "digital" in combined:
                                return "usage"
                            if "access" in combined or "account" in combined:
                                return "access"
                            return "other"

                        # If a dimension is below the provisional 0.50 threshold,
                        # prioritize the lowest such dimension. Otherwise, use the
                        # lowest dimension as a discussion prompt, not as a flagged gap.
                        priority_item = gaps[0] if gaps else weakest
                        priority_category = recommendation_category(priority_item)
                        recommendation = recommendation_catalog.get(priority_category)

                        with st.container(border=True):
                            if gaps:
                                st.markdown(
                                    f"#### First priority: review {priority_item['dimension'].lower()}"
                                )
                                st.caption(
                                    f"Dimension score: {priority_item['score']:.2f} · "
                                    "Below the provisional 0.50 review threshold"
                                )
                                if recommendation:
                                    st.write(recommendation)
                                else:
                                    st.write(
                                        "Review the underlying indicators for this dimension "
                                        "and discuss the respondent's needs before deciding on support."
                                    )
                                st.markdown(
                                    "**Next step:** Confirm the specific barrier with the respondent, "
                                    "then offer relevant, voluntary support."
                                )
                            else:
                                st.markdown("#### No priority gap flagged")
                                st.write(
                                    "No dimension is below the provisional 0.50 threshold. "
                                    f"The lowest observed dimension is {weakest['dimension']} "
                                    f"({weakest['score']:.2f}); discuss it with the respondent "
                                    "and offer optional support only if it is needed."
                                )

                        

                        st.caption(result.get("interpretation_limit", ""))
