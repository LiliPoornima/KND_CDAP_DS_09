"""Dataset & Data Quality section (all numbers come from the training run's metadata)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.src.financial_capability.fc_config import (
    ADMIN_FIELDS,
    FWB_SCALE_ITEMS,
    OUTCOME_ADJACENT_FIELDS,
    SENSITIVE_FIELDS,
)
from backend.src.financial_capability.fc_feature_selection import data_dictionary
from frontend.financial_capability.data_loader import get_metadata, show_not_trained


def render() -> None:
    st.subheader("Dataset & Data Quality")
    meta = get_metadata()
    if meta is None:
        show_not_trained()
        st.markdown("The approved feature list below does not need the trained model.")
        _feature_tables()
        return

    ds, v, t = meta["dataset"], meta["dataset"]["validation"], meta["target"]
    st.markdown(f"**Source:** {ds['name']}  \n**File:** `{ds['file_name']}` "
                f"(SHA-256 `{ds['file_sha256'][:16]}...`)  \n**Page:** {ds['source_page']}")
    a, b, c, d = st.columns(4)
    a.metric("Rows in file", f"{v['n_rows']:,}")
    b.metric("Columns in file", f"{v['n_columns']:,}")
    c.metric("Duplicate rows", v["n_duplicate_rows"])
    d.metric("Missing cells (NaN)", v["n_missing_cells"])
    st.caption(f"Rows dropped because FWBscore is a special code (-4 not written / -1 refused): "
               f"{t['rows_dropped_invalid_target']} {v['target_special_code_counts']}. "
               f"Rows used: {meta['split']['n_train'] + meta['split']['n_test']:,}. "
               f"Selected features: {meta['features']['n_features']}.")

    st.markdown("### Target class distribution")
    rows = []
    for split_name, key in (("All", "class_distribution_all"), ("Train", "class_distribution_train"),
                            ("Test", "class_distribution_test")):
        for cls, vals in t[key].items():
            rows.append({"split": split_name, "class": cls, "count": vals["count"],
                         "percent": vals["percent"]})
    dist = pd.DataFrame(rows)
    fig = px.bar(dist, x="class", y="count", color="split", barmode="group", text="percent",
                 category_orders={"class": ["Low", "Medium", "High"]})
    fig.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Grouping rule: Low {t['band_rule']['Low']}, Medium {t['band_rule']['Medium']}, "
               f"High {t['band_rule']['High']}. {t['band_rule_status']} "
               "Class balance is moderately uneven (Medium is the smallest class), so macro metrics "
               "are reported next to accuracy.")

    st.markdown("### Refused / special codes treated as missing (selected features)")
    special = v["refused_or_missing_code_counts"]
    if special:
        sp = (pd.DataFrame({"variable": list(special), "rows": list(special.values())})
              .sort_values("rows", ascending=False))
        st.dataframe(sp, hide_index=True, use_container_width=True)
    if v["out_of_codebook_values"]:
        st.error(f"Values outside the codebook: {v['out_of_codebook_values']}")
    else:
        st.success("No values outside the codebook were found in the selected features.")
    st.markdown("### Feature groups, data dictionary and exclusions")
    _feature_tables()


def _feature_tables() -> None:
    dd = data_dictionary()
    grp = dd.groupby("group").size().reset_index(name="features")
    st.dataframe(grp, hide_index=True, use_container_width=True)
    with st.expander("Data dictionary of the approved features (from the official codebook)"):
        st.dataframe(dd, hide_index=True, use_container_width=True)
    with st.expander("Columns that are NOT model inputs, and why"):
        st.markdown(
            f"- **Leakage - the 10 score items:** {', '.join(FWB_SCALE_ITEMS)}\n"
            "- **Leakage - target / target copies:** FWBscore and any target-derived field\n"
            f"- **Outcome-like hardship / stress measures:** {', '.join(OUTCOME_ADJACENT_FIELDS)}\n"
            f"- **Identifiers / weights / frame data:** {', '.join(ADMIN_FIELDS)}\n"
            f"- **Sensitive demographics (kept out of the model):** {', '.join(SENSITIVE_FIELDS)}\n"
            "- All other survey columns were not selected for Phase 1."
        )

