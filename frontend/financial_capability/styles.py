"""Modern Fintech Design System and CSS styles for Component 05.

Provides a clean Navy Blue & White palette, subtle card shadows,
rounded corners, badge pills, and visual hierarchy inspired by
modern financial analytics applications.
"""

from __future__ import annotations

import streamlit as st


FINTECH_CSS = """
<style>
/* ------------------------------------------------ Main Typography & Backgrounds */
:root {
    --navy-primary: #0F172A;
    --navy-brand: #1E3A8A;
    --navy-light: #F8FAFC;
    --border-subtle: #E2E8F0;
    --text-primary: #0F172A;
    --text-muted: #64748B;
    --amber-warning: #D97706;
    --amber-light: #FEF3C7;
    --blue-accent: #2563EB;
    --blue-light: #EFF6FF;
    --green-success: #059669;
    --green-light: #ECFDF5;
    --coral-negative: #EF4444;
}

/* Card Container Styles */
.fc-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
}
.fc-card:hover {
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}

/* 2x2 Grid Insight Card */
.fc-insight-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 140px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.fc-insight-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.fc-insight-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 8px;
}
.fc-insight-count {
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 4px;
}
.fc-insight-label {
    font-size: 0.85rem;
    color: #64748B;
}

/* Hero Top Priority Card */
.fc-hero-card {
    background: #FFFFFF;
    border: 1.5px solid #F59E0B;
    border-radius: 12px;
    padding: 22px 24px;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
    margin-bottom: 20px;
}
.fc-hero-badge {
    background: #FEF3C7;
    color: #92400E;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
}
.fc-hero-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 14px;
}

/* Secondary & Healthy Cards */
.fc-secondary-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.fc-healthy-card {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Badge Pills */
.fc-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-high-priority {
    background-color: #FEF3C7;
    color: #B45309;
}
.badge-lower-priority {
    background-color: #F1F5F9;
    color: #475569;
}
.badge-good {
    background-color: #DCFCE7;
    color: #15803D;
}
.badge-insufficient {
    background-color: #F1F5F9;
    color: #64748B;
}
</style>
"""


def inject_fintech_styles() -> None:
    """Inject the global navy-and-white fintech CSS styling into the Streamlit session."""
    st.markdown(FINTECH_CSS, unsafe_allow_html=True)

