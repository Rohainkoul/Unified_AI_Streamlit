from __future__ import annotations

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Unified AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
<style>

/* Main background */

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 15% 8%,
            rgba(37, 99, 235, 0.10),
            transparent 27%
        ),
        radial-gradient(
            circle at 87% 10%,
            rgba(13, 148, 136, 0.10),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #f8fafc 0%,
            #ffffff 50%,
            #f8fafc 100%
        );
}


/* Page width */

.block-container {
    max-width: 1220px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}


/* Titles */

h1 {
    color: #0f172a;
    font-weight: 800 !important;
    letter-spacing: -0.035em;
}

h2, h3 {
    color: #1e293b;
}


/* Native cards */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid #e2e8f0 !important;
    background: rgba(255,255,255,0.94);
    box-shadow:
        0 12px 35px
        rgba(15, 23, 42, 0.07);
}


/* Buttons */

div.stButton > button {
    border-radius: 11px;
    min-height: 3rem;
    font-weight: 700;
    border: none;
    transition: all 0.2s ease;
}

div.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            90deg,
            #2563eb,
            #0f766e
        );
    color: white;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow:
        0 8px 24px
        rgba(15, 23, 42, 0.15);
}


/* Metrics */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem;
    box-shadow:
        0 6px 18px
        rgba(15, 23, 42, 0.05);
}


/* Hide default footer */

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
#
# Use native Streamlit text so HTML can never display as code.
# ============================================================

st.caption(
    "⚡ AI AUTOMATION SUITE"
)

st.title(
    "Unified AI Platform"
)

st.markdown(
    "### Document Intelligence + Computer Vision"
)

st.write(
    "One intelligent workspace for invoice automation and "
    "AI-assisted vehicle inspection. Select a workflow below "
    "to get started."
)


# ============================================================
# FEATURE STRIP
# ============================================================

f1, f2, f3, f4 = st.columns(4)

f1.info(
    "⚡ Production AI"
)

f2.info(
    "🧠 Intelligent Automation"
)

f3.info(
    "📊 Structured Results"
)

f4.info(
    "🛡️ Human-Friendly Reports"
)


st.markdown("")


# ============================================================
# WORKFLOW CARDS
# ============================================================

invoice_col, vehicle_col = st.columns(
    2,
    gap="large",
)


# ------------------------------------------------------------
# INVOICE
# ------------------------------------------------------------

with invoice_col:

    with st.container(
        border=True
    ):

        st.subheader(
            "🧾 Invoice Intelligence AI"
        )

        st.caption(
            "DOCUMENT INTELLIGENCE"
        )

        st.write(
            "Automatically extract invoice information, "
            "discover additional document fields, reconcile "
            "financial values and generate structured output."
        )

        st.markdown(
            """
**Key capabilities**

- 16 trained invoice fields
- Automatic dynamic-field discovery
- Financial and GST reconciliation
- Line-item extraction
- Structured JSON output
"""
        )

        st.info(
            "Built for invoice and ERP document workflows."
        )

        if st.button(
            "Open Invoice Intelligence AI →",
            type="primary",
            use_container_width=True,
            key="open_invoice_ai",
        ):

            st.switch_page(
                "pages/1_Invoice_Intelligence_AI.py"
            )


# ------------------------------------------------------------
# VEHICLE
# ------------------------------------------------------------

with vehicle_col:

    with st.container(
        border=True
    ):

        st.subheader(
            "🚗 Vehicle Inspection AI"
        )

        st.caption(
            "COMPUTER VISION"
        )

        st.write(
            "Analyze vehicle photographs, identify visible "
            "exterior damage, associate damage with affected "
            "parts and produce a clear repair assessment."
        )

        st.markdown(
            """
**Key capabilities**

- Vehicle-part segmentation
- Visible damage detection
- Damage-to-part association
- Severity assessment
- Human-readable inspection report
"""
        )

        st.success(
            "Parts AI + Damage AI + inspection decision engine."
        )

        if st.button(
            "Open Vehicle Inspection AI →",
            type="primary",
            use_container_width=True,
            key="open_vehicle_ai",
        ):

            st.switch_page(
                "pages/2_Vehicle_Inspection_AI.py"
            )


# ============================================================
# PLATFORM SUMMARY
# ============================================================

st.markdown("")

st.divider()

s1, s2, s3 = st.columns(3)


s1.metric(
    "AI Workflows",
    "2",
)


s2.metric(
    "Platform",
    "Unified",
)


s3.metric(
    "Output",
    "Actionable",
)


st.caption(
    "Unified AI Platform • Invoice Intelligence • Vehicle Inspection"
)
