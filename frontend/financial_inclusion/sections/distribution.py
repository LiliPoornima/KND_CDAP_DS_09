
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from .common import show_caution
from ..constants import CLUSTER_LABELS


# Explicit aliases for economy names that may differ from ISO country names.
# Add aliases here only after checking the actual economy values in your dataset.
ECONOMY_ISO3_ALIASES = {
    "Bolivia": "BOL",
    "Bolivia, Plurinational State of": "BOL",
    "Brunei": "BRN",
    "Congo, Dem. Rep.": "COD",
    "Congo, Democratic Republic of the": "COD",
    "Congo, Rep.": "COG",
    "Congo, Republic of the": "COG",
    "Côte d'Ivoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "Czech Republic": "CZE",
    "Egypt, Arab Rep.": "EGY",
    "Hong Kong SAR, China": "HKG",
    "Iran, Islamic Rep.": "IRN",
    "Korea, Rep.": "KOR",
    "Korea, Democratic People's Republic of": "PRK",
    "Kyrgyz Republic": "KGZ",
    "Lao PDR": "LAO",
    "Micronesia, Fed. Sts.": "FSM",
    "Moldova": "MDA",
    "North Macedonia": "MKD",
    "Russian Federation": "RUS",
    "Slovak Republic": "SVK",
    "The Gambia": "GMB",
    "Türkiye": "TUR",
    "Turkey": "TUR",
    "Venezuela, RB": "VEN",
    "West Bank and Gaza": None,
    "Yemen, Rep.": "YEM",
}


# Categorical colors: cluster IDs are profile labels, not an ordinal scale.
CLUSTER_COLORS = {
    0: "#E76F51",
    1: "#2A9D8F",
    2: "#457B9D",
    3: "#8E6CBB",
    4: "#E9C46A",
}


def normalize_name(value: str) -> str:
    """Normalize economy names for exact matching."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def get_iso3(economy_name: str) -> str | None:
    """
    Resolve an economy to an ISO-3 code.

    Uses explicit aliases first, then an exact pycountry lookup.
    It does not use fuzzy matching to avoid silently assigning
    an economy to the wrong country.
    """
    name = str(economy_name).strip()

    # Explicit aliases, compared without case sensitivity.
    aliases = {
        normalize_name(key): value
        for key, value in ECONOMY_ISO3_ALIASES.items()
    }

    if normalize_name(name) in aliases:
        return aliases[normalize_name(name)]

    try:
        import pycountry

        # Exact name lookup.
        try:
            country = pycountry.countries.lookup(name)
            return country.alpha_3
        except LookupError:
            pass

        # Exact normalized matching against common country names.
        target = normalize_name(name)

        for country in pycountry.countries:
            possible_names = [
                getattr(country, "name", ""),
                getattr(country, "official_name", ""),
                getattr(country, "common_name", ""),
            ]

            if any(
                candidate and normalize_name(candidate) == target
                for candidate in possible_names
            ):
                return country.alpha_3

    except ImportError:
        pass

    # No validated mapping found.
    return None


def make_cluster_label(cluster_id) -> str:
    """Return a readable label for a saved cluster ID."""
    try:
        cluster_id = int(cluster_id)
    except (TypeError, ValueError):
        return "Unknown profile"

    return CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")


def render(df: pd.DataFrame) -> None:
    

    st.caption(
        "Explore how the five inclusion profiles are distributed across "
        "economies in the observed analytical sample. Results are unweighted "
        "sample summaries, not national prevalence estimates or country rankings."
    )

    # ---------------------------------------------------------
    # 1. Validate and prepare the data
    # ---------------------------------------------------------
    required_columns = {"economy", "cluster_id"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        st.error(
            "Cannot display the global distribution. Missing required "
            f"columns: {', '.join(sorted(missing_columns))}"
        )
        return

    data = df.loc[:, ["economy", "cluster_id"]].copy()
    data = data.dropna(subset=["economy", "cluster_id"])

    if data.empty:
        st.warning("No valid economy and cluster records are available.")
        return

    data["economy"] = data["economy"].astype(str).str.strip()
    data = data[data["economy"] != ""]

    if data.empty:
        st.warning("No valid economy names are available.")
        return

    # Ensure cluster IDs are numeric when possible.
    data["cluster_id"] = pd.to_numeric(
        data["cluster_id"], errors="coerce"
    )
    data = data.dropna(subset=["cluster_id"])
    data["cluster_id"] = data["cluster_id"].astype(int)

    if data.empty:
        st.warning("No valid cluster IDs are available.")
        return

    # ---------------------------------------------------------
    # 2. Calculate economy-level cluster distributions
    # ---------------------------------------------------------
    table = (
        data.groupby("economy")
        .agg(sample_size=("cluster_id", "size"))
        .reset_index()
    )

    parts = (
        data.groupby(["economy", "cluster_id"])
        .size()
        .rename("respondents")
        .reset_index()
    )

    parts["percent_within_economy"] = (
        parts["respondents"]
        / parts.groupby("economy")["respondents"].transform("sum")
        * 100
    )

    # Deterministic tie-break: the lower cluster ID wins a tie.
    dominant = (
        parts.sort_values(
            ["economy", "respondents", "cluster_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates("economy")
        [["economy", "cluster_id"]]
        .rename(columns={"cluster_id": "dominant_cluster_id"})
    )

    table = table.merge(dominant, on="economy", how="left")

    table["dominant_cluster"] = table["dominant_cluster_id"].map(
        make_cluster_label
    )

    table["dominant_share_pct"] = table["economy"].map(
        parts.merge(
            dominant,
            left_on=["economy", "cluster_id"],
            right_on=["economy", "dominant_cluster_id"],
            how="inner",
        )
        .drop_duplicates("economy")
        .set_index("economy")["percent_within_economy"]
    )

    # Build a complete cluster table, including zero counts.
    cluster_ids = sorted(CLUSTER_LABELS.keys())

    wide_counts = (
        parts.pivot(
            index="economy",
            columns="cluster_id",
            values="respondents",
        )
        .reindex(columns=cluster_ids, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    wide_counts.columns = [
        f"Cluster {int(cluster)} respondents"
        for cluster in wide_counts.columns
    ]

    wide_shares = (
        parts.pivot(
            index="economy",
            columns="cluster_id",
            values="percent_within_economy",
        )
        .reindex(columns=cluster_ids, fill_value=0)
        .fillna(0)
    )
    wide_shares.columns = [
        f"Cluster {int(cluster)} share (%)"
        for cluster in wide_shares.columns
    ]

    table = (
        table.merge(wide_counts.reset_index(), on="economy", how="left")
        .merge(wide_shares.reset_index(), on="economy", how="left")
    )

    # ---------------------------------------------------------
    # 3. Resolve economy names to ISO-3 map identifiers
    # ---------------------------------------------------------
    table["iso_alpha"] = table["economy"].map(get_iso3)

    mapped_table = table.dropna(subset=["iso_alpha"]).copy()
    unmapped_economies = sorted(
        table.loc[table["iso_alpha"].isna(), "economy"].unique()
    )

    table["dominant_cluster_id"] = table[
        "dominant_cluster_id"
    ].astype(int)

    # ---------------------------------------------------------
    # 4. World map at the top of the section
    # ---------------------------------------------------------
    st.subheader("World Map: Dominant Inclusion Profile")

    st.caption(
        "Each country is colored by the profile with the largest observed "
        "sample count in that economy. Select a country on the map to explore "
        "its sample composition below."
    )

    if mapped_table.empty:
        st.warning(
            "No economy names could be matched to validated ISO-3 codes. "
            "The map cannot be displayed until the economy-name mapping is reviewed."
        )
    else:
        mapped_table["map_profile"] = mapped_table[
            "dominant_cluster_id"
        ].map(make_cluster_label)

        profile_color_map = {
            make_cluster_label(cluster_id): color
            for cluster_id, color in CLUSTER_COLORS.items()
        }

        fig = px.choropleth(
            mapped_table,
            locations="iso_alpha",
            locationmode="ISO-3",
            color="map_profile",
            hover_name="economy",
            hover_data={
                "iso_alpha": False,
                "sample_size": True,
                "map_profile": True,
                "dominant_share_pct": ":.1f",
                "dominant_cluster_id": False,
            },
            custom_data=[
                "economy",
                "sample_size",
                "dominant_cluster",
                "dominant_cluster_id",
            ],
            color_discrete_map=profile_color_map,
            category_orders={
                "map_profile": [
                    make_cluster_label(cluster_id)
                    for cluster_id in cluster_ids
                ]
            },
            title="Dominant profile by economy",
        )

        fig.update_traces(
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Sample size: %{customdata[1]:,}<br>"
                "Dominant profile: %{customdata[2]}<br>"
                "Share in dominant profile: %{z:.1f}%"
                "<extra></extra>"
            ),
        )

        # Use custom hover data for a reliable country tooltip.
        # The country selection event is handled using ISO-3 locations.
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#AAAAAA",
            showcountries=True,
            countrycolor="#D0D0D0",
            showland=True,
            landcolor="#F0F2F5",
            showocean=True,
            oceancolor="#FFFFFF",
            projection_type="natural earth",
        )

        fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=55, b=0),
            legend_title_text="Dominant profile",
        )

        map_event = st.plotly_chart(
            fig,
            key="global_cluster_world_map",
            on_select="rerun",
            selection_mode="points",
            config={
                "responsive": True,
                "displayModeBar": True,
            },
        )

        # When a new map point is selected, update the economy selector.
        selected_points = (
            map_event.selection.points
            if map_event is not None
            else []
        )

        if selected_points:
            selected_iso = selected_points[0].get("location")
            iso_to_economy = dict(
                zip(mapped_table["iso_alpha"], mapped_table["economy"])
            )
            clicked_economy = iso_to_economy.get(selected_iso)

            # Avoid repeatedly overwriting a user's dropdown selection
            # with an old map selection on subsequent reruns.
            if (
                clicked_economy
                and st.session_state.get("_last_map_economy")
                != clicked_economy
            ):
                st.session_state["global_economy"] = clicked_economy
                st.session_state["_last_map_economy"] = clicked_economy

    # ---------------------------------------------------------
    # 5. Select an economy and show short insights
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Explore an economy")

    economies = sorted(table["economy"].unique(), key=str.casefold)

    # Initialize a safe default before creating the selectbox.
    if (
        "global_economy" not in st.session_state
        or st.session_state["global_economy"] not in economies
    ):
        st.session_state["global_economy"] = economies[0]

    economy = st.selectbox(
        "Select an economy",
        options=economies,
        key="global_economy",
    )

    selected_row = table.loc[table["economy"] == economy].iloc[0]
    sample = int(selected_row["sample_size"])
    show_caution(sample)

    selected = parts.loc[parts["economy"] == economy].copy()
    selected["Profile"] = selected["cluster_id"].map(make_cluster_label)

    # Include all known clusters so missing profiles appear as zero.
    selected = (
        selected.set_index("cluster_id")
        .reindex(cluster_ids, fill_value=0)
        .rename_axis("cluster_id")
        .reset_index()
    )
    selected["Profile"] = selected["cluster_id"].map(make_cluster_label)

    # Recalculate missing cluster shares as zero after reindexing.
    selected["respondents"] = selected["respondents"].fillna(0).astype(int)
    selected["percent_within_economy"] = (
        selected["percent_within_economy"].fillna(0)
    )

    dominant_id = int(selected_row["dominant_cluster_id"])
    dominant_label = make_cluster_label(dominant_id)
    dominant_share = float(
        selected.loc[
            selected["cluster_id"] == dominant_id,
            "percent_within_economy",
        ].iloc[0]
    )

    second_largest = (
        selected.sort_values(
            ["respondents", "cluster_id"],
            ascending=[False, True],
        )
        .iloc[1]
    )

    # Three compact headline metrics.
    
    metric1, metric2, metric3 = st.columns([1, 2.5, 1])

    with metric1:
        st.metric("Sample size", f"{sample:,}")

    with metric2:
        st.metric("Dominant profile", dominant_label)

    with metric3:
        st.metric("Dominant profile share", f"{dominant_share:.1f}%")

    # ---------------------------------------------------------
    # 6. Short, data-driven insights
    # ---------------------------------------------------------
    st.markdown("**Key insights**")

    insights = []

    insights.append(
        f"**Largest profile:** {dominant_label} represents "
        f"{dominant_share:.1f}% of this economy's observed sample."
    )

    if int(second_largest["cluster_id"]) != dominant_id:
        second_label = make_cluster_label(second_largest["cluster_id"])
        second_share = float(second_largest["percent_within_economy"])

        insights.append(
            f"**Next largest profile:** {second_label} accounts for "
            f"{second_share:.1f}% of the sample."
        )

    if dominant_share >= 60:
        insights.append(
            "The sample is strongly concentrated in one profile."
        )
    elif dominant_share >= 40:
        insights.append(
            "One profile is the largest group, but the sample also includes "
            "a substantial mix of other profiles."
        )
    else:
        insights.append(
            "No single profile exceeds 40% of the sample, indicating a "
            "more distributed profile composition."
        )

    non_dominant_share = 100 - dominant_share
    insights.append(
        f"The remaining profiles together represent approximately "
        f"{non_dominant_share:.1f}% of the observed sample."
    )

    for insight in insights:
        st.markdown(f"- {insight}")

    st.caption(
        "Insights describe the observed analytical sample only. They do not "
        "establish national prevalence, explain causes, or imply that one "
        "economy performs better than another."
    )

    # ---------------------------------------------------------
    # 7. Economy's cluster composition chart
    # ---------------------------------------------------------
    st.markdown("**Cluster composition**")

    selected_chart = selected.sort_values(
        "cluster_id"
    )

    fig_bar = px.bar(
        selected_chart,
        x="Profile",
        y="percent_within_economy",
        color="Profile",
        color_discrete_map=profile_color_map,
        custom_data=["respondents"],
        labels={
            "percent_within_economy": "Share of observed sample (%)",
            "Profile": "Inclusion profile",
        },
        title=f"Observed sample composition: {economy}",
    )

    fig_bar.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Sample share: %{y:.1f}%<br>"
            "Respondents: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    fig_bar.update_layout(
        showlegend=False,
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig_bar,
        key="global_cluster_composition_bar",
        config={"responsive": True},
        width="stretch",
    )

    st.caption(
        f"Selected economy: {economy}. Sample n={sample:,}. "
        "If multiple profiles tie for the largest count, the lowest saved "
        "cluster ID is selected as the dominant profile."
    )

    # ---------------------------------------------------------
    # 8. Detailed table, hidden by default
    # ---------------------------------------------------------
    with st.expander("View detailed economy-level data"):
        display_table = table.drop(columns=["iso_alpha"], errors="ignore")
        display_table = display_table.rename(
            columns={
                "economy": "Economy",
                "sample_size": "Sample size",
                "dominant_cluster_id": "Dominant cluster ID",
                "dominant_cluster": "Dominant sample profile",
                "dominant_share_pct": "Dominant profile share (%)",
            }
        )

        st.dataframe(
            display_table,
            hide_index=True,
            width="stretch",
        )

   
