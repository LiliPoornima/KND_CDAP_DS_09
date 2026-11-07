"""What-if Analysis section: interactive simulation comparing original and simulated capability behaviors."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.src.financial_capability.fc_config import CLASS_NAMES, FEATURE_SPEC_BY_NAME
from backend.src.financial_capability.fc_predictor import InvalidResponseError
from backend.src.financial_capability.fc_what_if import (
    CHANGEABLE_FEATURES,
    WHAT_IF_DISCLAIMER,
    run_what_if,
)
from frontend.financial_capability.data_loader import require_bundle, require_response


def _comparison_chart(original_probs: dict, simulated_probs: dict) -> go.Figure:
    """Create a side-by-side grouped bar chart comparing Original vs Simulated probabilities."""
    categories = list(CLASS_NAMES)
    orig_vals = [original_probs.get(c, 0.0) for c in categories]
    sim_vals = [simulated_probs.get(c, 0.0) for c in categories]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Original Assessment",
                x=categories,
                y=orig_vals,
                marker_color="#94A3B8",
                text=[f"{v:.1%}" for v in orig_vals],
                textposition="auto",
            ),
            go.Bar(
                name="Simulated Assessment",
                x=categories,
                y=sim_vals,
                marker_color="#1E3A8A",
                text=[f"{v:.1%}" for v in sim_vals],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=dict(text="Probability Comparison: Original vs Simulated", font=dict(size=14, color="#0F172A")),
        yaxis=dict(range=[0, 1], title="Probability", gridcolor="#F1F5F9"),
        xaxis=dict(title=""),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig


def render() -> None:
    """Render the redesigned What-if Simulation Lab."""
    st.subheader("What-if Simulation Lab")
    st.info(f"ℹ️ {WHAT_IF_DISCLAIMER}")

    bundle = require_bundle()
    if bundle is None:
        return
    response = require_response()
    if response is None:
        st.info("Please complete the **Individual Assessment** first to establish your baseline.")
        return

    st.markdown("### 1. Select Capability Behaviors to Modify")
    st.write(
        "Explore how changes in your saving, budgeting, knowledge, or goal planning habits "
        "affect the machine learning model's prediction. Background demographics remain fixed."
    )

    fields = st.multiselect(
        "Choose up to 4 questions to modify in this scenario:",
        options=list(CHANGEABLE_FEATURES),
        format_func=lambda n: f"{FEATURE_SPEC_BY_NAME[n].group} → {FEATURE_SPEC_BY_NAME[n].label}",
        max_selections=4,
    )

    changes = {}
    if fields:
        cols = st.columns(min(len(fields), 2))
        for idx, name in enumerate(fields):
            spec = FEATURE_SPEC_BY_NAME[name]
            current = response.get(name)
            with cols[idx % 2]:
                changes[name] = st.selectbox(
                    f"New Answer: {spec.label}",
                    options=list(spec.options),
                    index=list(spec.options).index(current) if current in spec.options else 0,
                    format_func=lambda v, s=spec: s.options[v],
                    key=f"fc_whatif_field_{name}",
                    help=f"Current baseline: {spec.options.get(current, 'Not answered')}",
                )

    run_sim = st.button("Run What-if Simulation", type="primary", disabled=not fields)
    if not run_sim:
        return

    try:
        out = run_what_if(bundle, response, changes)
    except InvalidResponseError as exc:
        st.error(f"Simulation error: {exc}")
        return

    st.divider()
    st.markdown("### 2. Simulation Results: Side-by-Side Comparison")

    # Changed variables summary
    st.markdown("#### Modified Behaviors in this Scenario")
    change_rows = [
        {
            "Area": FEATURE_SPEC_BY_NAME[n].group,
            "Behavior": FEATURE_SPEC_BY_NAME[n].label,
            "Original Answer": c["from"],
            "Simulated Answer": c["to"],
        }
        for n, c in out["changes"].items()
    ]
    st.dataframe(pd.DataFrame(change_rows), hide_index=True, use_container_width=True)

    # Side-by-Side Comparison Columns
    col_orig, col_sim = st.columns(2)
    with col_orig:
        with st.container(border=True):
            st.markdown("#### 👤 Original Assessment")
            st.metric(
                "Predicted Well-being",
                out["original"]["predicted_class"],
                f"Confidence: {out['original']['confidence']:.1%}",
            )
            for c in CLASS_NAMES:
                st.caption(f"{c}: {out['original']['probabilities'][c]:.1%}")

    with col_sim:
        with st.container(border=True):
            st.markdown("#### 🧪 Simulated Assessment")
            diff_label = (
                f"{out['what_if']['confidence'] - out['original']['confidence']:+.1%}"
                if out["what_if"]["predicted_class"] == out["original"]["predicted_class"]
                else f"Shifted to {out['what_if']['predicted_class']}"
            )
            st.metric(
                "Simulated Well-being",
                out["what_if"]["predicted_class"],
                diff_label,
            )
            for c in CLASS_NAMES:
                delta = out["probability_change"][c]
                sign = "+" if delta > 0 else ""
                st.caption(f"{c}: {out['what_if']['probabilities'][c]:.1%} ({sign}{delta:.1%})")

    # Comparison grouped chart
    st.plotly_chart(_comparison_chart(out["original"]["probabilities"], out["what_if"]["probabilities"]), use_container_width=True)

    # Probability shifts table
    st.markdown("#### Detailed Class Probability Movements")
    delta_df = pd.DataFrame(
        {
            "Category": list(CLASS_NAMES),
            "Original Probability": [out["original"]["probabilities"][c] for c in CLASS_NAMES],
            "Simulated Probability": [out["what_if"]["probabilities"][c] for c in CLASS_NAMES],
            "Absolute Shift": [out["probability_change"][c] for c in CLASS_NAMES],
        }
    )
    st.dataframe(
        delta_df.style.format(
            {
                "Original Probability": "{:.1%}",
                "Simulated Probability": "{:.1%}",
                "Absolute Shift": "{:+.1%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    if out["class_changed"]:
        st.success(
            f"🎯 **Category Transition**: The model prediction moved from **{out['original']['predicted_class']}** "
            f"to **{out['what_if']['predicted_class']}**."
        )
    else:
        st.info("ℹ️ The predicted category remained the same, but internal class probabilities moved as shown above.")

    st.caption("🔒 *Note: Original assessment answers were not modified in your session.*")
