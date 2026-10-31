"""Component 04: Explainable Early Warning and Attribution-Guided Adaptation for Delayed-Label Fraud Detection."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.fraud_detection.data_loader import (
    load_decisions,
    load_grid_long,
    load_grid_summary,
    load_locked_thresholds,
    load_sparkov_data,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection — Delayed-Label Adaptation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Component 04: Delayed-Label Fraud Detection")
st.caption(
    "Explainable Early Warning and Attribution-Guided Adaptation for Non-Stationary Fraud Streams Under Delayed Labels"
)

st.markdown(
    """
    In production payment fraud detection, labels arrive with significant latency (**7 to 28 days**).
    This research dashboard evaluates label-free drift monitoring (PSI, Score JS divergence, Prediction Entropy,
    and TreeSHAP profile shift), a calibrated **Early-Warning Score (EWS)** decision policy, and
    **Attribution-Guided Adaptation (AGA)** under strict non-leakage temporal constraints.
    """
)

# ---------------------------------------------------------------------------
# Load Cached Data
# ---------------------------------------------------------------------------
try:
    summary_df = load_grid_summary()
    long_df = load_grid_long()
    thresholds = load_locked_thresholds()
    sparkov_deg, sparkov_sig = load_sparkov_data()
except Exception as exc:
    st.error(f"Failed to load research data: {exc}")
    st.info("Ensure benchmark CSV files exist under outputs/fraud_detection/experiments/.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Global Controls
# ---------------------------------------------------------------------------
st.sidebar.header("🕹️ Benchmark Parameters")

scenario_labels = {
    "S2": "S2: Mixed Drift (Compromised Terminals)",
    "S1": "S1: Pure Concept Drift (Seasonal Fraud P(Y|X))",
    "S3": "S3: Virtual Covariate Shift (Time Shift P(X))",
    "control": "Control: Stationary Baseline Stream",
}

selected_scenario = st.sidebar.selectbox(
    "Drift Scenario",
    options=["S2", "S1", "S3", "control"],
    format_func=lambda x: scenario_labels.get(x, x),
    index=0,
)

selected_delay = st.sidebar.selectbox(
    "Label Delay (d)",
    options=[7, 14, 28],
    index=0,
    help="Days before transaction labels become matured and available for retraining.",
)

selected_seed = st.sidebar.selectbox(
    "Random Evaluation Seed",
    options=[1, 2, 3, 4, 5],
    index=0,
    help="Evaluation seed disjoint from threshold calibration streams.",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Specification Policy:**
    - Locked EWS Alert: $\\theta_1 = 0.22$
    - Locked EWS Adapt: $\\theta_2 = 0.35$
    - Cooldown Period: 14 Days
    - Test Horizon: Weeks 10..25 (Days 70..181)
    """
)

# ---------------------------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🚨 Live Drift Monitor & EWS",
    "🔄 Adaptation Benchmark ($P_1$ vs Baselines)",
    "⏱️ Delay Scaling & Feature Dynamics",
    "📊 Fallback Stream (Sparkov Check)",
    "📑 Methodology & Scope Limits",
])

# ===========================================================================
# TAB 1: Live Drift Monitor & Early-Warning Simulation
# ===========================================================================
with tabs[0]:
    st.subheader("Early-Warning Drift Monitoring & Decision Simulation")
    st.markdown(
        f"Simulating label-free drift detection on **{scenario_labels[selected_scenario]}** with **{selected_delay}-day label delay** (Seed {selected_seed})."
    )

    decisions_df = load_decisions(selected_scenario, selected_delay, selected_seed)
    
    if decisions_df is not None and not decisions_df.empty:
        # Week slider for step-by-step playback
        min_week = int(decisions_df["week"].min())
        max_week = int(decisions_df["week"].max())
        
        sim_week = st.slider(
            "Evaluation Week (Drift Onset occurs at Week 15 / Day 105)",
            min_value=min_week,
            max_value=max_week,
            value=min(16, max_week),
            step=1,
            help="Select week to inspect real-time signals, EWS score, and detector decision.",
        )
        
        matching_rows = decisions_df[decisions_df["week"] == sim_week]
        if not matching_rows.empty:
            week_row = matching_rows.iloc[0]
        else:
            week_row = decisions_df.iloc[0]
        
        ews_val = float(week_row.get("ews", 0.0))
        decision_val = str(week_row.get("decision", "WAIT"))
        recall_val = float(week_row.get("recall_at_1fpr", 0.0))
        cooldown_val = bool(week_row.get("cooldown_active", False))
        has_sig = bool(week_row.get("has_signals", True))
        candidate_val = str(week_row.get("candidate_chosen", "NONE"))
        action_val = str(week_row.get("action", "NONE"))
        
        # Color badge styling
        if decision_val == "ADAPT":
            badge_color = "#ff4b4b"
            badge_text = "🔴 ADAPT (Retraining Triggered)"
        elif decision_val == "MONITOR":
            badge_color = "#ffa500"
            badge_text = "🟡 MONITOR (Early Warning Alert)"
        else:
            badge_color = "#00cc66"
            badge_text = "🟢 WAIT (System Normal)"
            
        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Current Window", f"Week {sim_week} (Day {int(week_row.get('day_start', 0))})")
        if has_sig:
            kpi2.metric("Early-Warning Score (EWS)", f"{ews_val:.4f}", delta=f"{ews_val - 0.22:.4f} vs Alert Thresh")
        else:
            kpi2.metric("Retrain Triggered", str(week_row.get("retrain_triggered", False)))
        kpi3.metric("Model Recall@1% FPR", f"{recall_val:.4f}")
        kpi4.metric("Candidate Model Chosen", candidate_val if candidate_val != "NONE" else "None (Static)")

        st.markdown(
            f"""
            <div style="background-color: {badge_color}22; border-left: 5px solid {badge_color}; padding: 10px; border-radius: 4px; margin-bottom: 20px;">
                <h4 style="margin: 0; color: {badge_color};">{badge_text}</h4>
                <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #555;">
                    Current Day: {int(week_row.get('day_start', 0))} | Action: {action_val} | Candidate Model Chosen: {candidate_val}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if has_sig:
            col_gauge, col_signals = st.columns([1, 2])

            with col_gauge:
                # Plotly Gauge Chart for EWS
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=ews_val,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "EWS Risk Gauge", "font": {"size": 18}},
                        gauge={
                            "axis": {"range": [0, 1.0], "tickwidth": 1, "tickcolor": "darkblue"},
                            "bar": {"color": badge_color},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "gray",
                            "steps": [
                                {"range": [0, 0.22], "color": "#e8f8f0"},
                                {"range": [0.22, 0.35], "color": "#fef6e7"},
                                {"range": [0.35, 1.0], "color": "#fde8e8"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 0.35,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_signals:
                # Timeline of the 4 monitoring signals
                timeline_cols = ["week", "mean_psi", "score_js", "entropy_rel_change", "shap_shift", "ews"]
                avail_cols = [c for c in timeline_cols if c in decisions_df.columns]
                tdf = decisions_df[avail_cols].copy()
                
                fig_timeline = go.Figure()
                if "mean_psi" in tdf:
                    fig_timeline.add_trace(go.Scatter(x=tdf["week"], y=tdf["mean_psi"], mode="lines+markers", name="Feature PSI (x100)", yaxis="y1"))
                if "score_js" in tdf:
                    fig_timeline.add_trace(go.Scatter(x=tdf["week"], y=tdf["score_js"], mode="lines+markers", name="Score JS Div", yaxis="y1"))
                if "entropy_rel_change" in tdf:
                    fig_timeline.add_trace(go.Scatter(x=tdf["week"], y=tdf["entropy_rel_change"], mode="lines+markers", name="Rel. Entropy", yaxis="y1"))
                if "shap_shift" in tdf:
                    fig_timeline.add_trace(go.Scatter(x=tdf["week"], y=tdf["shap_shift"], mode="lines+markers", name="TreeSHAP Shift", yaxis="y1"))
                if "ews" in tdf:
                    fig_timeline.add_trace(go.Scatter(x=tdf["week"], y=tdf["ews"], mode="lines+markers", name="Combined EWS", line=dict(color="red", width=3)))

                # Add vertical lines for drift onset (Week 15) and current selected week
                fig_timeline.add_vline(x=15, line_width=2, line_dash="dash", line_color="black", annotation_text="Drift Onset (W15)")
                fig_timeline.add_vline(x=sim_week, line_width=2, line_dash="dot", line_color=badge_color, annotation_text=f"Selected (W{sim_week})")

                fig_timeline.update_layout(
                    title="Weekly Monitoring Signals Trajectory",
                    xaxis_title="Evaluation Week",
                    yaxis_title="Signal Value",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            # Feature Attribution Delta Breakdown
            st.markdown("#### Feature-Level Attribution & PSI Breakdown (Selected Week)")
            psi_feature_cols = [c for c in week_row.index if c.startswith("psi_")]
            delta_feature_cols = [c for c in week_row.index if c.startswith("delta_")]
            
            if psi_feature_cols or delta_feature_cols:
                feat_names = [c.replace("psi_", "") for c in psi_feature_cols]
                psi_vals = [float(week_row.get(f"psi_{f}", 0.0)) for f in feat_names]
                delta_vals = [float(week_row.get(f"delta_{f}", 0.0)) for f in feat_names]
                
                feat_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Feature PSI": psi_vals,
                    "SHAP Profile Delta": delta_vals,
                }).sort_values(by="Feature PSI", ascending=False)
                
                fig_bar = px.bar(
                    feat_df,
                    x="Feature",
                    y=["Feature PSI", "SHAP Profile Delta"],
                    barmode="group",
                    title=f"Feature Drift Attribution in Week {sim_week}",
                    labels={"value": "Drift Magnitude", "variable": "Signal Type"},
                    height=320,
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Longitudinal recall trajectory when detailed SHAP signals were not generated
            fig_grid_traj = go.Figure()
            fig_grid_traj.add_trace(go.Scatter(
                x=decisions_df["week"],
                y=decisions_df["recall_at_1fpr"],
                mode="lines+markers",
                name="P1 AGA Recall@1%",
                line=dict(color="#0066cc", width=3),
            ))
            fig_grid_traj.add_vline(x=15, line_width=2, line_dash="dash", line_color="black", annotation_text="Drift Onset (W15)")
            fig_grid_traj.add_vline(x=sim_week, line_width=2, line_dash="dot", line_color=badge_color, annotation_text=f"Selected (W{sim_week})")
            fig_grid_traj.update_layout(
                title=f"Longitudinal Recall Trajectory ({selected_scenario}, Delay {selected_delay}d, Seed {selected_seed})",
                xaxis_title="Evaluation Week",
                yaxis_title="Recall@1% FPR",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_grid_traj, use_container_width=True)

            st.info(
                f"""
                💡 **Signal Telemetry Note:** High-resolution TreeSHAP Profile Shift and per-feature PSI breakdowns were logged for the reference benchmark combination (**Delay 7d, Seed 1**).  
                For Delay **{selected_delay}d**, the dashboard displays the exact evaluated performance trajectory and adaptation decisions from the 660-run factorial grid.  
                *(To inspect the live TreeSHAP/PSI risk gauge and per-feature breakdown, switch **Label Delay** to **7** in the sidebar).*
                """
            )
            
    else:
        st.warning("Per-window signal playback CSV not found for this combination. Showing grid summary data.")

# ===========================================================================
# TAB 2: Adaptation Benchmark (P1 vs Baselines)
# ===========================================================================
with tabs[1]:
    st.subheader("Adaptation Performance Benchmark ($P_1$ AGA vs. Baselines)")
    st.markdown(
        f"Evaluating **{scenario_labels[selected_scenario]}** under **Delay {selected_delay}d** across evaluated baselines and adaptation policies."
    )

    # Filter longitudinal trajectory
    sub_long = long_df[
        (long_df["scenario"] == selected_scenario)
        & (long_df["delay"] == selected_delay)
        & (long_df["seed"] == selected_seed)
    ].copy()

    if not sub_long.empty:
        # Plotly Recall Trajectory over time
        methods_to_plot = ["B0_static", "B1_periodic", "B2_psi_trigger", "B4_shap_shift", "P1_aga_full", "P1_b_drop_only"]
        plot_df = sub_long[sub_long["method"].isin(methods_to_plot)].copy()
        
        method_names_map = {
            "B0_static": "B0: Static (No Retraining)",
            "B1_periodic": "B1: Periodic Weekly Retraining",
            "B2_psi_trigger": "B2: Conventional PSI Trigger",
            "B4_shap_shift": "B4: SHAP-Shift Trigger",
            "P1_aga_full": "P1: Attribution-Guided Adaptation (AGA)",
            "P1_b_drop_only": "P1-b: Drop Drifted Features (Ablation)",
        }
        plot_df["Method Label"] = plot_df["method"].map(lambda m: method_names_map.get(m, m))

        fig_rec = px.line(
            plot_df,
            x="week",
            y="recall_at_1fpr",
            color="Method Label",
            markers=True,
            title=f"Recall@1% FPR Trajectory Across 16 Evaluation Weeks ({selected_scenario}, Delay {selected_delay}d, Seed {selected_seed})",
            labels={"week": "Evaluation Week", "recall_at_1fpr": "Recall@1% FPR"},
            height=420,
        )
        fig_rec.add_vline(x=15, line_width=2, line_dash="dash", line_color="black", annotation_text="Drift Onset (Week 15)")
        fig_rec.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=80),
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    # Retrain Count Bar Chart & Summary Table from summary_df
    sub_summary = summary_df[
        (summary_df["scenario"] == selected_scenario)
        & (summary_df["delay"] == selected_delay)
    ].copy()

    col_tbl, col_retrains = st.columns([3, 2])

    with col_tbl:
        st.markdown("#### Performance Metrics Summary (Aggregated over Seeds 1..5)")
        agg_cols = ["method", "base_recall", "post_onset_mean_recall", "relative_retention_pct", "retrain_count", "lead_time_to_first_alarm_days", "recovery_time_days"]
        avail_agg = [c for c in agg_cols if c in sub_summary.columns]
        
        display_summary = (
            sub_summary.groupby("method")[avail_agg[1:]]
            .mean()
            .reset_index()
            .sort_values(by="post_onset_mean_recall", ascending=False)
        )
        display_summary.columns = [
            c.replace("_", " ").title() for c in display_summary.columns
        ]
        st.dataframe(display_summary.style.format(precision=4), use_container_width=True)

    with col_retrains:
        st.markdown("#### Maintenance Retrain Count")
        retrain_bar = px.bar(
            display_summary,
            x="Method",
            y="Retrain Count",
            color="Retrain Count",
            title="Total Retrain Events by Method",
            height=340,
        )
        retrain_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(retrain_bar, use_container_width=True)

    # Honest Takeaway Alert Box
    st.info(
        """
        **Scientific Takeaway & Trade-Off:**
        - $P_1$ substantially cuts retrain events ($3.6$ in $S_1$, $5.0$ in $S_2$ vs. 11.0 for weekly periodic $B_1$), and completely suppresses retrains on benign shift ($S_3$) and Control.
        - **However, $P_1$ does NOT beat periodic retraining ($B_1$) on raw recall under active drift.** When matured labels are available every week, continuous periodic retraining achieves higher recall ($0.785$ vs $0.694$ in $S_1$).
        - $P_1$'s value is operational maintenance reduction on benign shifts, not superior recall under persistent drift.
        """
    )

# ===========================================================================
# TAB 3: Delay Scaling & Feature Dynamics
# ===========================================================================
with tabs[2]:
    st.subheader("Impact of Label Latency ($d \in \{7, 14, 28\}$ Days)")
    st.markdown(
        "Evaluating how delayed feedback influences model retention, detection lead times, and feature availability."
    )

    # Delay comparison across delays for the chosen scenario
    delay_compare = summary_df[
        (summary_df["scenario"] == selected_scenario)
        & (summary_df["method"].isin(["B0_static", "B1_periodic", "P1_aga_full"]))
    ].copy()

    delay_agg = (
        delay_compare.groupby(["delay", "method"])[["post_onset_mean_recall", "retrain_count", "relative_retention_pct"]]
        .mean()
        .reset_index()
    )

    fig_delay = px.bar(
        delay_agg,
        x="delay",
        y="post_onset_mean_recall",
        color="method",
        barmode="group",
        title=f"Post-Onset Recall Across Delays ({selected_scenario})",
        labels={"delay": "Label Delay (Days)", "post_onset_mean_recall": "Recall@1% FPR", "method": "Policy"},
        height=360,
    )
    fig_delay.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_delay, use_container_width=True)

    # Terminal risk feature confound explanation card
    st.markdown("#### The Terminal-Risk Lookback Confound at $d=28$")
    st.markdown(
        """
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
            <h5 style="margin-top: 0;">Lookback Truncation Phenomenon</h5>
            <p style="font-size: 0.95em; color: #444;">
                Baseline pre-drift recall drops systematically with label delay:
                <b>0.7484</b> (d=7) &rarr; <b>0.5996</b> (d=14) &rarr; <b>0.3543</b> (d=28).
            </p>
            <p style="font-size: 0.95em; color: #444;">
                <b>Causal Mechanism:</b> In accordance with the non-leakage rule, the terminal risk lookback window
                is strictly <code>[D - d - 28, D - d]</code>. At delay <code>d=28</code>, during the base training period (days 35..55),
                there are zero matured transactions prior to Day 0. Consequently, historical lookback is truncated,
                collapsing <code>t_risk</code> feature importance by <b>97%</b> and drastically reducing initial discriminatory power.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 4: Fallback Stream (Sparkov Check)
# ===========================================================================
with tabs[3]:
    st.subheader("Empirical Stream Fallback (Sparkov Credit Card Dataset)")
    
    # IEEE-CIS Status Box
    st.warning(
        """
        **IEEE-CIS Check Status: Not Performed (Data Unavailable)**  
        Real-data validation on IEEE-CIS was **NOT performed** because `train_transaction.csv` requires manual Kaggle login and interactive rules acceptance. Per Spec Section 14, IEEE-CIS cannot serve as real-degradation validation in this project.
        """
    )

    st.markdown(
        """
        **Sparkov Dataset Classification:** **SYNTHETIC DATA** (NOT real data, and NOT part of original specification).  
        Acquired from Kaggle (`kartik2112/fraud-detection`, file `fraudTest.csv`, 150.3 MB, 555,719 rows, 2,145 frauds, spanning 194 days) as provided by the user, generated using the open-source Sparkov synthetic credit card transaction simulator.
        """
    )

    if not sparkov_deg.empty:
        col_s1, col_s2 = st.columns([3, 2])
        
        with col_s1:
            st.markdown("#### Rolling 14-Day Performance Check (`sparkov_degradation_check.csv`)")
            st.dataframe(sparkov_deg.style.format(precision=4), use_container_width=True)
            
        with col_s2:
            st.markdown("#### Degradation Verdict")
            st.success("**Degradation Verdict: FALSE** (Mean Recall: 81.98%, 0 windows below 80% baseline).")
            st.error(
                """
                **Critical Scientific Caveat on False Alarms:**  
                Across all 10 rolling windows, EWS remained between 0.140 and 0.174, issuing **WAIT in 100% of windows (0 false alarms)**.  
                **However, "0 false alarms" is NOT evidence of robustness because no degradation exists in this synthetic stream.**
                """
            )

        if not sparkov_sig.empty:
            st.markdown("#### Label-Free Monitoring Signals on Sparkov Stream (`sparkov_signals_check.csv`)")
            st.dataframe(sparkov_sig.style.format(precision=4), use_container_width=True)

# ===========================================================================
# TAB 5: Methodology & Scope Limits
# ===========================================================================
with tabs[4]:
    st.subheader("Research Methodology & Scientific Boundaries")

    col_meth, col_not_val = st.columns(2)

    with col_meth:
        st.markdown("#### Methodological Principles (Spec §1–13)")
        st.markdown(
            """
            - **Strict Non-Leakage Rule:** At decision day $t$ with delay $d$, available labels are strictly $\\text{label\\_day} \\le t - d$.
            - **Temporal Partitioning:** Candidate retraining splits the last 28 matured days into a 21-day candidate training set and a 7-day holdout validation set.
            - **Calibrated Multi-Signal EWS:** Combines Feature PSI, Score JS Divergence, Binary Entropy, and TreeSHAP Profile Shift.
            - **Locked Control Thresholds:** Thresholds ($\theta_1=0.22, \theta_2=0.35$) were calibrated on control streams and locked in `locked_thresholds.json`.
            - **Unit Test Coverage:** All 101 unit tests passing in `tests/fraud_detection/`.
            """
        )

    with col_not_val:
        st.markdown("#### ⚠️ What is NOT Validated")
        st.markdown(
            """
            To maintain scientific integrity and prevent overclaiming, the following aspects were **NOT validated**:
            1. **Real Data:** No evaluation on genuine, unanonymized real-world banking telemetry or unperturbed real-world concept drift was performed (Handbook and Sparkov are synthetic).
            2. **Human-Reviewed Alerts:** No operational simulation of alert triage queues, analyst review fatigue, or operational costs of false alarms under production volumes was conducted.
            3. **Production Deployment:** No live streaming infrastructure, distributed latency testing, online asynchronous inference, or concurrent drift handling was tested.
            4. **Significance Tests:** Formal statistical significance testing (e.g. paired t-tests across seeds) was not conducted; metrics report empirical sample means and sample standard deviations across 5 seeds.
            """

