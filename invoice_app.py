from __future__ import annotations

import html
import json
import math
import os
import re
import tempfile

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


# ============================================================
# PADDLE / PADDLEX CLOUD STARTUP
# MUST BE SET BEFORE invoice_engine IMPORT
# ============================================================

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"

# ============================================================
# STREAMLIT CLOUD OPENCV HEADLESS BOOTSTRAP
# ============================================================
#
# PaddleX installs a GUI OpenCV build transitively.
# Streamlit Cloud cannot currently rely on apt/libGL, so we
# install a server-safe OpenCV build into writable /tmp.
#
# This executes BEFORE invoice_engine is imported.
# It does NOT modify model/inference logic.
# ============================================================

def _prepare_streamlit_headless_opencv():
    import importlib
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    if os.name == "nt":
        return

    # Streamlit reruns this file in the same Python process.
    # If our working cv2 is already loaded, leave it untouched.
    existing_cv2 = sys.modules.get("cv2")

    if existing_cv2 is not None:
        return

    target = (
        Path(tempfile.gettempdir())
        / "invoice_ai_opencv_standard_headless"
    )

    marker_path = (
        target
        / ".ready"
    )

    if not marker_path.exists():

        print(
            "Preparing isolated headless OpenCV runtime..."
        )

        target.mkdir(
            parents=True,
            exist_ok=True,
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(target),
                "--upgrade",
                "opencv-python-headless==4.10.0.84",
            ],
            check=True,
        )

        marker_path.write_text(
            "ready",
            encoding="utf-8",
        )

    target_string = str(
        target
    )

    if target_string in sys.path:
        sys.path.remove(
            target_string
        )

    sys.path.insert(
        0,
        target_string,
    )

    importlib.invalidate_caches()

    import cv2

    print(
        "? Isolated OpenCV ready:",
        cv2.__version__,
    )

    print(
        "? OpenCV loaded from:",
        cv2.__file__,
    )


_prepare_streamlit_headless_opencv()




# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import streamlit as st

from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from invoice_engine import (
    EXPECTED_FIELDS,
    load_v3_production_engine,
    process_invoice_dynamic_final,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Invoice Intelligence AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(markup: str) -> None:
    st.html(markup.strip())


# ============================================================
# CSS
# ============================================================

st.html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(220, 38, 38, 0.06),
                transparent 34rem
            ),
            #f7f8fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .hero {
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1f2937 64%,
                #6b1f2a 130%
            );

        color: white;
        border-radius: 22px;
        padding: 1.65rem 1.8rem;
        margin-bottom: 1.35rem;

        box-shadow:
            0 12px 35px
            rgba(17, 24, 39, 0.12);
    }

    .hero-kicker {
        color: #fca5a5;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        color: white;
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
    }

    .hero-sub {
        color: #d1d5db;
        font-size: 0.98rem;
        max-width: 950px;
        margin-top: 0.7rem;
        line-height: 1.58;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 0.95rem;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;

        border:
            1px solid
            rgba(255, 255, 255, 0.18);

        background:
            rgba(255, 255, 255, 0.08);

        font-size: 0.82rem;
        color: #f3f4f6;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;

        box-shadow:
            0 0 0 3px
            rgba(52, 211, 153, 0.15);

        display: inline-block;
    }

    .section-card {
        background: rgba(255,255,255,0.97);
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 1.05rem 1.1rem;

        box-shadow:
            0 5px 18px
            rgba(17, 24, 39, 0.035);

        margin-bottom: 0.9rem;
    }

    .auto-card {
        background:
            linear-gradient(
                135deg,
                #ecfdf5 0%,
                #ffffff 110%
            );

        border: 1px solid #a7f3d0;
        border-radius: 15px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.9rem;
    }

    .auto-title {
        font-size: 0.76rem;
        font-weight: 800;
        color: #047857;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .auto-text {
        color: #374151;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 0.28rem;
    }

    .mini-label {
        color: #6b7280;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .mini-value {
        color: #111827;
        font-size: 1.06rem;
        font-weight: 700;
        margin-top: 0.28rem;
        word-break: break-word;
    }

    .not-detected {
        color: #9ca3af;
        font-style: italic;
        font-weight: 500;
    }

    .ready-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;

        box-shadow:
            0 5px 18px
            rgba(17,24,39,0.035);
    }

    .ready-title {
        color: #047857;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .ready-text {
        color: #374151;
        font-size: 0.98rem;
        margin-top: 0.3rem;
    }

    .preview-page {
        color: #6b7280;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.2rem 0 0.5rem 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.97);
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 0.85rem 0.95rem;

        box-shadow:
            0 5px 18px
            rgba(17,24,39,0.035);
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 0.35rem 0.8rem 0.8rem 0.8rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 44px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    footer {
        visibility: hidden;
    }

    

    /* INVOICE AI FINAL UI POLISH */

    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(255,255,255,0.72);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.32rem;
        margin-top: 0.35rem;
        margin-bottom: 0.85rem;
    }

    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        font-weight: 700 !important;
        min-height: 42px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        box-shadow:
            0 3px 10px
            rgba(17, 24, 39, 0.08);
    }

    .result-status-pass {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background:
            linear-gradient(
                135deg,
                #ecfdf5 0%,
                #ffffff 100%
            );
        border: 1px solid #a7f3d0;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin: 0.3rem 0 1rem 0;
    }

    .result-status-review {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background:
            linear-gradient(
                135deg,
                #fff7ed 0%,
                #ffffff 100%
            );
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin: 0.3rem 0 1rem 0;
    }

    .status-icon-large {
        font-size: 1.55rem;
        line-height: 1;
    }

    .status-heading-pass {
        font-size: 0.92rem;
        font-weight: 800;
        color: #047857;
    }

    .status-heading-review {
        font-size: 0.92rem;
        font-weight: 800;
        color: #c2410c;
    }

    .status-subtext {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 0.12rem;
    }

    .quick-header {
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1f2937 100%
            );
        border-radius: 17px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        color: white;
    }

    .quick-header-title {
        color: white;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .quick-header-sub {
        color: #d1d5db;
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    .field-count-pill {
        display: inline-block;
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.28rem 0.62rem;
        margin-bottom: 0.7rem;
    }

    .trained-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .dynamic-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: #f5f3ff;
        color: #7c3aed;
        font-size: 0.72rem;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        box-shadow:
            0 6px 22px
            rgba(17, 24, 39, 0.045);
    }

    div[data-testid="stMetric"] label {
        font-weight: 700 !important;
    }


    

    /* INVOICE AI COMPLETE UI V2 */

    :root {
        --card-bg: rgba(255, 255, 255, 0.96);
        --card-border: #e5e7eb;
        --text-main: #111827;
        --text-muted: #6b7280;
        --accent: #ef4444;
        --accent-soft: #fff1f2;
        --success: #059669;
        --success-soft: #ecfdf5;
        --warning: #d97706;
        --warning-soft: #fffbeb;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 92% 4%,
                rgba(239,68,68,0.08),
                transparent 31rem
            ),
            radial-gradient(
                circle at 4% 90%,
                rgba(59,130,246,0.045),
                transparent 34rem
            ),
            #f8fafc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                125deg,
                #0f172a 0%,
                #111827 53%,
                #3f1d28 100%
            );

        color: white;

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius: 24px;

        padding:
            2rem
            2.15rem;

        margin-bottom: 1.45rem;

        box-shadow:
            0 18px 45px
            rgba(15,23,42,0.15);
    }

    .hero::after {
        content: "";
        position: absolute;

        width: 330px;
        height: 330px;

        border-radius: 50%;

        right: -120px;
        top: -180px;

        background:
            radial-gradient(
                circle,
                rgba(248,113,113,0.24),
                rgba(248,113,113,0)
            );
    }

    .hero-kicker {
        color: #fca5a5;
        font-size: 0.73rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero-title {
        color: white;
        font-size: 2.35rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .hero-sub {
        color: #d1d5db;
        font-size: 0.98rem;
        line-height: 1.65;
        max-width: 900px;
        margin-top: 0.72rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;

        margin-top: 1rem;

        background:
            rgba(255,255,255,0.08);

        border:
            1px solid
            rgba(255,255,255,0.14);

        border-radius: 999px;

        padding:
            0.42rem
            0.8rem;

        color: #f3f4f6;

        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;

        box-shadow:
            0 0 0 4px
            rgba(52,211,153,0.12);
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #fafafa 100%
            );

        border-right:
            1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] hr {
        border-color: #e5e7eb;
    }


    /* ========================================================
       CARDS / METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: var(--card-bg);

        border:
            1px solid
            var(--card-border);

        border-radius: 17px;

        padding:
            0.9rem
            1rem;

        min-height: 100px;

        box-shadow:
            0 5px 18px
            rgba(15,23,42,0.045);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-1px);

        box-shadow:
            0 8px 24px
            rgba(15,23,42,0.07);
    }

    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 760 !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    div[data-testid="stFileUploader"] {
        background:
            linear-gradient(
                135deg,
                #ffffff 0%,
                #fafafa 100%
            );

        border:
            1px solid #e2e8f0;

        border-radius: 20px;

        padding:
            0.65rem
            1rem
            1rem
            1rem;

        box-shadow:
            0 6px 22px
            rgba(15,23,42,0.04);
    }

    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 15px !important;
        border: 1.5px dashed #cbd5e1 !important;
        background: #f8fafc !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    div.stButton > button {
        border-radius: 12px;
        min-height: 46px;
        font-weight: 750;
        transition: all 0.15s ease;
    }

    div.stButton > button[kind="primary"] {
        box-shadow:
            0 5px 15px
            rgba(239,68,68,0.18);
    }

    div.stDownloadButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 700;
    }


    /* ========================================================
       TABS
       ======================================================== */

    div[data-baseweb="tab-list"] {
        gap: 0.22rem;

        background:
            rgba(255,255,255,0.82);

        padding:
            0.34rem;

        border:
            1px solid #e5e7eb;

        border-radius: 14px;

        margin-top: 0.8rem;
        margin-bottom: 1rem;

        box-shadow:
            0 3px 12px
            rgba(15,23,42,0.035);
    }

    button[data-baseweb="tab"] {
        border-radius: 10px !important;

        padding:
            0.45rem
            0.72rem !important;

        font-weight: 650 !important;

        color: #475569 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #dc2626 !important;
        background: #fff1f2 !important;

        box-shadow:
            inset 0 0 0 1px
            rgba(239,68,68,0.18);
    }


    /* ========================================================
       TABLES
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border:
            1px solid #e5e7eb;

        border-radius: 14px;

        overflow: hidden;

        box-shadow:
            0 5px 18px
            rgba(15,23,42,0.035);
    }


    /* ========================================================
       STATUS BANNERS
       ======================================================== */

    .result-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;

        gap: 1rem;

        border-radius: 18px;

        padding:
            1rem
            1.15rem;

        margin:
            0.25rem
            0
            1rem
            0;
    }

    .result-banner-pass {
        background:
            linear-gradient(
                135deg,
                #ecfdf5,
                #ffffff
            );

        border:
            1px solid #a7f3d0;
    }

    .result-banner-review {
        background:
            linear-gradient(
                135deg,
                #fff7ed,
                #ffffff
            );

        border:
            1px solid #fed7aa;
    }

    .result-banner-title {
        font-weight: 800;
        font-size: 0.96rem;
        color: #111827;
    }

    .result-banner-sub {
        color: #64748b;
        font-size: 0.81rem;
        margin-top: 0.2rem;
    }

    .result-chip {
        border-radius: 999px;

        padding:
            0.3rem
            0.7rem;

        font-weight: 800;
        font-size: 0.72rem;
    }

    .result-chip-pass {
        background: #d1fae5;
        color: #047857;
    }

    .result-chip-review {
        background: #ffedd5;
        color: #c2410c;
    }


    /* ========================================================
       QUICK VIEW
       ======================================================== */

    .quick-panel {
        background:
            linear-gradient(
                135deg,
                #111827,
                #1f2937
            );

        color: white;

        border-radius: 18px;

        padding:
            1.15rem
            1.25rem;

        margin-bottom: 1rem;

        box-shadow:
            0 8px 25px
            rgba(15,23,42,0.12);
    }

    .quick-panel-title {
        color: white;
        font-size: 1.08rem;
        font-weight: 800;
    }

    .quick-panel-sub {
        color: #cbd5e1;
        font-size: 0.83rem;
        margin-top: 0.25rem;
    }


    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;

        font-size: 1.18rem;
        font-weight: 800;

        color: #1e293b;

        margin-top: 0.3rem;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.84rem;
        margin-bottom: 0.85rem;
    }


    /* ========================================================
       STREAMLIT CLEANUP
       ======================================================== */

    footer {
        visibility: hidden;
    }


</style>
    """
)


# ============================================================
# CONSTANTS
# ============================================================

FIELD_LABELS = {
    "VENDOR_NAME": "Vendor Name",
    "INVOICE_NUMBER": "Invoice Number",
    "INVOICE_DATE": "Invoice Date",
    "DUE_DATE": "Due Date",
    "CUSTOMER_NAME": "Customer Name",
    "ADDRESS": "Address",
    "CURRENCY": "Currency",
    "LINE_ITEM_DESC": "Line Item Description",
    "LINE_ITEM_QTY": "Line Item Quantity",
    "LINE_ITEM_UNIT_PRICE": "Line Item Unit Price",
    "LINE_ITEM_AMOUNT": "Line Item Amount",
    "TAX": "Tax",
    "DISCOUNT": "Discount",
    "SUBTOTAL": "Subtotal",
    "TOTAL_AMOUNT": "Total Amount",
    "PAYMENT_TERMS": "Payment Terms",
}


TABLE_KEYS = (
    "line_items",
    "line_item_table",
    "line_items_table",
    "material_table",
    "materials",
    "items",
)


VALUE_KEYS = (
    "value",
    "text",
    "prediction",
    "normalized_value",
    "final_value",
)


# ============================================================
# TRAINED-SCHEMA ALIASES
# DO NOT DUPLICATE THESE IN AUTO DYNAMIC OUTPUT
# ============================================================

TRAINED_SCHEMA_ALIASES = {
    "vendor",
    "vendor name",
    "seller",
    "seller name",
    "supplier",
    "supplier name",

    "invoice no",
    "invoice number",
    "invoice id",
    "invoice",
    "inv no",
    "inv number",

    "document no",
    "document number",
    "doc no",
    "doc number",

    "invoice date",
    "due date",

    "customer",
    "customer name",
    "buyer",
    "buyer name",

    "address",
    "billing address",
    "shipping address",

    "currency",

    "subtotal",
    "sub total",

    "total",
    "total amount",
    "grand total",
    "invoice total",

    "tax",
    "tax amount",

    "discount",
    "payment terms",
}


# ============================================================
# AUTO-DISCOVERY BLACKLIST
# ============================================================

AUTO_LABEL_BLACKLIST = {
    "",
    "no",
    "number",
    "code",
    "value",

    "to",
    "from",

    "description",
    "material",
    "material code",

    "quantity",
    "qty",
    "rate",
    "amount",

    "uom",
    "unit",
    "unit price",

    "sl",
    "sl no",
    "serial",
    "serial no",
    "serial number",

    "dear sir",
    "dear madam",

    "prepared by",
    "checked by",
}


# ============================================================
# ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine():

    return (
        load_v3_production_engine()
    )


def initialize_engine():

    try:

        with st.spinner(
            "Initializing Invoice AI production engine..."
        ):

            return get_engine()

    except Exception as error:

        st.error(
            "❌ Production engine initialization failed"
        )

        st.exception(
            error
        )

        st.stop()


# ============================================================
# JSON SAFE
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        Decimal,
    ):
        return float(value)

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                make_json_safe(item)

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            make_json_safe(item)

            for item
            in value
        ]

    if hasattr(
        value,
        "item",
    ):

        try:
            return make_json_safe(
                value.item()
            )

        except Exception:
            pass

    if hasattr(
        value,
        "tolist",
    ):

        try:
            return make_json_safe(
                value.tolist()
            )

        except Exception:
            pass

    if (
        isinstance(
            value,
            float,
        )
        and
        not math.isfinite(value)
    ):
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def clean_spaces(
    value: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalize_label(
    value: str,
) -> str:

    text = clean_spaces(
        value
    )

    if not text:
        return ""

    # --------------------------------------------------------
    # Common dotted acronyms
    # --------------------------------------------------------

    text = re.sub(
        r"(?i)\bP\s*\.\s*O\s*\.?",
        "PO",
        text,
    )

    text = re.sub(
        r"(?i)\bG\s*\.\s*R\s*\.?",
        "GR",
        text,
    )

    text = re.sub(
        r"(?i)\bB\s*\.\s*Value\b",
        "B Value",
        text,
    )

    text = re.sub(
        r"(?i)\bPh\s*\.?\s*No\.?",
        "Phone Number",
        text,
    )

    # --------------------------------------------------------
    # PDF occasionally joins PO + NO -> PONO
    # --------------------------------------------------------

    text = re.sub(
        r"(?i)\bPONO\.?\b",
        "PO Number",
        text,
    )

    text = re.sub(
        r"(?i)\bGRNO\.?\b",
        "GR Number",
        text,
    )

    text = re.sub(
        r"[:=]+$",
        "",
        text,
    ).strip()

    simple = (
        text.casefold()
    )

    simple = re.sub(
        r"[^a-z0-9]+",
        " ",
        simple,
    )

    simple = clean_spaces(
        simple
    )

    canonical = {

        # Purchase order
        "po":
            "PO Number",

        "po no":
            "PO Number",

        "po number":
            "PO Number",

        "pono":
            "PO Number",

        "purchase order":
            "PO Number",

        "purchase order no":
            "PO Number",

        "purchase order number":
            "PO Number",

        # GR
        "gr":
            "GR Number",

        "gr no":
            "GR Number",

        "gr number":
            "GR Number",

        "grno":
            "GR Number",

        "goods receipt no":
            "GR Number",

        "goods receipt number":
            "GR Number",

        # ODN
        "odn no":
            "ODN Number",

        "odn number":
            "ODN Number",

        # HSN
        "hsn":
            "HSN Code",

        "hsn code":
            "HSN Code",

        "hsn sac":
            "HSN Code",

        "sac code":
            "HSN Code",

        # Tax IDs
        "gst no":
            "GSTIN",

        "gst number":
            "GSTIN",

        "gstin":
            "GSTIN",

        "pan no":
            "PAN",

        "pan number":
            "PAN",

        "cin no":
            "CIN",

        "cin number":
            "CIN",

        # Bank
        "ifsc":
            "IFSC Code",

        # Contact
        "ph no":
            "Phone Number",

        "phone no":
            "Phone Number",

        "phone number":
            "Phone Number",

        "mobile no":
            "Phone Number",

        "mobile number":
            "Phone Number",

        "email id":
            "Email",

        "email address":
            "Email",

        # Values
        "b value":
            "B Value",

        "bvalue":
            "B Value",

        # Location
        "state code":
            "State Code",

        "state name":
            "State Name",

        "place of supply":
            "Place of Supply",

        # Tax components
        "cgst":
            "CGST",

        "sgst":
            "SGST",

        "igst":
            "IGST",

        # References
        "reference no":
            "Reference Number",

        "reference number":
            "Reference Number",

        "ref no":
            "Reference Number",
    }

    return canonical.get(
        simple,
        text,
    )


def normalized_label_key(
    value: str,
) -> str:

    text = normalize_label(
        value
    )

    text = text.casefold()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return clean_spaces(
        text
    )


# ============================================================
# VALID AUTO LABEL
# ============================================================

def valid_auto_label(
    value: str,
) -> bool:

    label = normalize_label(
        value
    )

    key = normalized_label_key(
        label
    )

    if not key:
        return False

    if key in TRAINED_SCHEMA_ALIASES:
        return False

    if key in AUTO_LABEL_BLACKLIST:
        return False

    if len(label) < 2:
        return False

    if len(label) > 45:
        return False

    if not re.search(
        r"[A-Za-z]",
        label,
    ):
        return False

    words = key.split()

    if len(words) > 6:
        return False

    if re.search(
        r"\b("
        r"please|thank|thanks|rupees|"
        r"faithfully|only|subject|description"
        r")\b",
        key,
    ):
        return False

    return True


# ============================================================
# NATIVE PDF TEXT / GEOMETRY
# ============================================================

def native_pdf_lines(
    uploaded_file,
) -> list[dict]:

    suffix = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
    )

    if suffix != ".pdf":
        return []

    document = fitz.open(
        stream=
            uploaded_file.getvalue(),
        filetype="pdf",
    )

    output = []

    try:

        for page_index in range(
            document.page_count
        ):

            page = (
                document.load_page(
                    page_index
                )
            )

            words = (
                page.get_text(
                    "words"
                )
            )

            groups = {}

            for word in words:

                if len(word) < 8:
                    continue

                (
                    x0,
                    y0,
                    x1,
                    y1,
                    text,
                    block_no,
                    line_no,
                    word_no,
                ) = word[:8]

                text = str(
                    text
                ).strip()

                if not text:
                    continue

                key = (
                    int(block_no),
                    int(line_no),
                )

                groups.setdefault(
                    key,
                    [],
                ).append(
                    {
                        "text":
                            text,

                        "word_no":
                            int(word_no),

                        "x0":
                            float(x0),

                        "y0":
                            float(y0),

                        "x1":
                            float(x1),

                        "y1":
                            float(y1),
                    }
                )

            page_lines = []

            for group in groups.values():

                group.sort(
                    key=lambda item: (
                        item["word_no"],
                        item["x0"],
                    )
                )

                text = " ".join(
                    item["text"]
                    for item
                    in group
                ).strip()

                if not text:
                    continue

                page_lines.append(
                    {
                        "page":
                            page_index + 1,

                        "text":
                            text,

                        "x0":
                            min(
                                item["x0"]
                                for item
                                in group
                            ),

                        "y0":
                            min(
                                item["y0"]
                                for item
                                in group
                            ),

                        "x1":
                            max(
                                item["x1"]
                                for item
                                in group
                            ),

                        "y1":
                            max(
                                item["y1"]
                                for item
                                in group
                            ),
                    }
                )

            page_lines.sort(
                key=lambda item: (
                    round(
                        item["y0"],
                        1,
                    ),
                    item["x0"],
                )
            )

            output.extend(
                page_lines
            )

    finally:
        document.close()

    return output


# ============================================================
# LABEL SEGMENT CLEANUP
# ============================================================

def clean_label_segment(
    segment: str,
) -> str:

    value = str(
        segment
    ).strip()

    value = value.strip(
        " ,;|"
    )

    # --------------------------------------------------------
    # If previous value and next label share a text line:
    #
    # email@x.com,Ph No
    #        -> Ph No
    # --------------------------------------------------------

    for delimiter in (
        ",",
        ";",
        "|",
    ):

        if delimiter in value:

            value = (
                value
                .split(delimiter)[-1]
                .strip()
            )

    # --------------------------------------------------------
    # Strip previous field value before a following label:
    #
    # 7000006657 GR No.
    # -> GR No.
    #
    # 0.00 CGST
    # -> CGST
    #
    # abc@email.com Ph No
    # -> Ph No
    # --------------------------------------------------------

    previous_value_pattern = re.compile(
        r"(?i)^("
        r"\S+@\S+"
        r"|"
        r"[-+]?"
        r"(?:₹|Rs\.?|INR|\$)?"
        r"\d[\d,./%\-]*"
        r"|"
        r"[A-Z0-9/_\-]*\d[A-Z0-9/_\-]*"
        r")\s+"
    )

    for _ in range(3):

        changed = (
            previous_value_pattern.sub(
                "",
                value,
                count=1,
            )
        )

        if changed == value:
            break

        value = changed.strip()

    return value


# ============================================================
# EXTRACT LABELS FROM ONE TEXT LINE
# ============================================================

def labels_from_line(
    text: str,
) -> list[str]:

    text = str(
        text
    )

    separator_matches = list(
        re.finditer(
            r"[:=]",
            text,
        )
    )

    if not separator_matches:
        return []

    output = []

    previous_separator_end = 0

    for match in separator_matches:

        segment = text[
            previous_separator_end:
            match.start()
        ]

        previous_separator_end = (
            match.end()
        )

        segment = (
            clean_label_segment(
                segment
            )
        )

        if not segment:
            continue

        label = normalize_label(
            segment
        )

        if not valid_auto_label(
            label
        ):
            continue

        output.append(
            label
        )

    return output


# ============================================================
# STRONG STANDALONE LABEL DISCOVERY
# ============================================================

def standalone_discovered_fields(
    lines: list[dict],
) -> list[str]:

    text = "\n".join(
        str(
            line.get(
                "text",
                "",
            )
        )
        for line
        in lines
    )

    normalized = (
        text.casefold()
    )

    normalized = re.sub(
        r"[._]+",
        " ",
        normalized,
    )

    normalized = clean_spaces(
        normalized
    )

    strong_patterns = {

        "GSTIN": [
            r"\bgstin\b",
        ],

        "CIN": [
            r"\bcin\b",
        ],

        "PAN": [
            r"\bpan\b",
        ],

        "HSN Code": [
            r"\bhsn\s+code\b",
            r"\bhsn\s*/\s*sac\b",
        ],

        "ODN Number": [
            r"\bodn\s+(?:no|number)\b",
        ],

        "PO Number": [
            r"\bpo\s+(?:no|number)\b",
            r"\bpono\b",
            r"\bpurchase\s+order\b",
        ],

        "GR Number": [
            r"\bgr\s+(?:no|number)\b",
            r"\bgrno\b",
        ],

        "B Value": [
            r"\bb\s+value\b",
            r"\bbvalue\b",
        ],

        "Email": [
            r"\bemail\b",
        ],

        "Phone Number": [
            r"\bph\s+(?:no|number)\b",
            r"\bphone\s+(?:no|number)\b",
            r"\bmobile\s+(?:no|number)\b",
        ],

        "CGST": [
            r"\bcgst\b",
        ],

        "SGST": [
            r"\bsgst\b",
        ],

        "IGST": [
            r"\bigst\b",
        ],

        "State Code": [
            r"\bstate\s+code\b",
        ],

        "State Name": [
            r"\bstate\s+name\b",
        ],

        "Place of Supply": [
            r"\bplace\s+of\s+supply\b",
        ],
    }

    # Reference Number is intentionally NOT added
    # unless an explicit label was found by labels_from_line().

    output = []

    for canonical, patterns in (
        strong_patterns.items()
    ):

        if any(
            re.search(
                pattern,
                normalized,
            )
            for pattern
            in patterns
        ):

            if valid_auto_label(
                canonical
            ):

                output.append(
                    canonical
                )

    return output


# ============================================================
# AUTOMATIC DYNAMIC FIELD DISCOVERY
# ============================================================

def discover_dynamic_fields(
    uploaded_file,
) -> list[str]:

    lines = (
        native_pdf_lines(
            uploaded_file
        )
    )

    discovered = []

    seen = set()

    def add_field(
        raw_field: str,
    ):

        field = normalize_label(
            raw_field
        )

        if not valid_auto_label(
            field
        ):
            return

        key = normalized_label_key(
            field
        )

        if not key:
            return

        if key in seen:
            return

        seen.add(
            key
        )

        discovered.append(
            field
        )

    # --------------------------------------------------------
    # First: actual colon / equals labels
    # --------------------------------------------------------

    for line in lines:

        for field in labels_from_line(
            line.get(
                "text",
                "",
            )
        ):

            add_field(
                field
            )

    # --------------------------------------------------------
    # Second: strong invoice-specific standalone labels
    # --------------------------------------------------------

    for field in standalone_discovered_fields(
        lines
    ):

        add_field(
            field
        )

    return discovered[:40]


# ============================================================
# MANUAL OVERRIDE
# ============================================================

def parse_manual_parameters(
    raw_text: str,
) -> list[str]:

    if not raw_text:
        return []

    pieces = re.split(
        r"[\n,;]+",
        raw_text,
    )

    output = []

    seen = set()

    for piece in pieces:

        field = normalize_label(
            piece
        )

        if not field:
            continue

        key = normalized_label_key(
            field
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            field
        )

    return output[:20]


# ============================================================
# FIELD NORMALIZATION
# ============================================================

def unwrap_field_value(
    value: Any,
) -> Any:

    if isinstance(
        value,
        dict,
    ):

        for key in VALUE_KEYS:

            if key in value:

                return unwrap_field_value(
                    value[key]
                )

        if len(value) == 1:

            return unwrap_field_value(
                next(
                    iter(
                        value.values()
                    )
                )
            )

        return value

    if isinstance(
        value,
        list,
    ):

        cleaned = [
            unwrap_field_value(item)

            for item
            in value

            if item is not None
        ]

        cleaned = [
            item
            for item
            in cleaned

            if str(item).strip()
        ]

        if not cleaned:
            return "NOT_DETECTED"

        if len(cleaned) == 1:
            return cleaned[0]

        return " | ".join(
            str(item)
            for item
            in cleaned
        )

    if value is None:
        return "NOT_DETECTED"

    if (
        isinstance(
            value,
            str,
        )
        and
        not value.strip()
    ):
        return "NOT_DETECTED"

    return value


# ============================================================
# FIND TRAINED FIELD DICTIONARY
# ============================================================

def field_dict_score(
    candidate: dict,
) -> int:

    return sum(
        1
        for field
        in EXPECTED_FIELDS
        if field
        in candidate
    )


def find_best_field_dict(
    payload: Any,
) -> dict:

    best = {}
    best_score = 0

    def walk(
        node: Any,
    ):

        nonlocal best, best_score

        if isinstance(
            node,
            dict,
        ):

            score = field_dict_score(
                node
            )

            if score > best_score:

                best = node
                best_score = score

            for value in node.values():
                walk(value)

        elif isinstance(
            node,
            list,
        ):

            for value in node:
                walk(value)

    walk(payload)

    return best


def normalized_fields(
    payload: Any,
) -> dict:

    source = (
        find_best_field_dict(
            payload
        )
    )

    return {
        field:
            unwrap_field_value(
                source.get(
                    field,
                    "NOT_DETECTED",
                )
            )

        for field
        in EXPECTED_FIELDS
    }


# ============================================================
# FIND LINE ITEMS
# ============================================================

def find_table(
    payload: Any,
):

    found = None

    def walk(
        node: Any,
    ):

        nonlocal found

        if found is not None:
            return

        if isinstance(
            node,
            dict,
        ):

            for key in TABLE_KEYS:

                if key not in node:
                    continue

                candidate = node[key]

                if (
                    isinstance(
                        candidate,
                        list,
                    )
                    and
                    candidate
                    and
                    all(
                        isinstance(
                            row,
                            dict,
                        )
                        for row
                        in candidate
                    )
                ):

                    found = candidate
                    return

                if (
                    isinstance(
                        candidate,
                        dict,
                    )
                    and
                    candidate
                ):

                    found = candidate
                    return

            for value in node.values():
                walk(value)

        elif isinstance(
            node,
            list,
        ):

            if (
                node
                and
                all(
                    isinstance(
                        row,
                        dict,
                    )
                    for row
                    in node
                )
            ):

                keys = {
                    str(key).lower()

                    for row
                    in node

                    for key
                    in row.keys()
                }

                signals = (
                    "description",
                    "desc",
                    "qty",
                    "quantity",
                    "rate",
                    "price",
                    "amount",
                    "material",
                    "item",
                )

                if any(
                    token in key

                    for key
                    in keys

                    for token
                    in signals
                ):

                    found = node
                    return

            for value in node:
                walk(value)

    walk(payload)

    return found


# ============================================================
# DISPLAY VALUE
# ============================================================

def display_value(
    value: Any,
) -> str:

    if value is None:
        return "NOT_DETECTED"

    if isinstance(
        value,
        (
            dict,
            list,
        ),
    ):

        return json.dumps(
            make_json_safe(
                value
            ),
            ensure_ascii=False,
        )

    text = str(
        value
    ).strip()

    if (
        not text
        or
        text.lower()
        in {
            "none",
            "null",
            "nan",
        }
    ):
        return "NOT_DETECTED"

    return text


# ============================================================
# PREVIEW
# ============================================================

def preview_pdf(
    uploaded_file,
):

    document = fitz.open(
        stream=
            uploaded_file.getvalue(),
        filetype="pdf",
    )

    try:

        if document.page_count == 0:

            st.warning(
                "PDF contains no pages."
            )

            return

        st.caption(
            f"{document.page_count} page"
            f"{'s' if document.page_count != 1 else ''}"
            " • rendered directly in the app"
        )

        for page_number in range(
            document.page_count
        ):

            page = document.load_page(
                page_number
            )

            pixmap = page.get_pixmap(
                matrix=
                    fitz.Matrix(
                        1.35,
                        1.35,
                    ),
                alpha=False,
            )

            render_html(
                f"""
                <div class="preview-page">
                    Page {page_number + 1}
                </div>
                """
            )

            st.image(
                pixmap.tobytes(
                    "png"
                ),
                use_container_width=True,
            )

            if (
                page_number
                <
                document.page_count - 1
            ):
                st.divider()

    finally:
        document.close()


def preview_image(
    uploaded_file,
):

    uploaded_file.seek(
        0
    )

    image = Image.open(
        uploaded_file
    )

    try:

        st.image(
            image,
            caption=
                uploaded_file.name,
            use_container_width=True,
        )

    finally:

        try:
            image.close()

        except Exception:
            pass


def preview_uploaded_file(
    uploaded_file,
):

    suffix = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
    )

    try:

        if suffix == ".pdf":

            preview_pdf(
                uploaded_file
            )

        else:

            preview_image(
                uploaded_file
            )

    except Exception as error:

        st.warning(
            "Preview could not be displayed."
        )

        st.caption(
            f"Preview error: {error}"
        )


# ============================================================
# EXPLICIT HSN REPAIR
# ============================================================

def find_explicit_hsn_value(
    uploaded_file,
) -> dict | None:

    lines = native_pdf_lines(
        uploaded_file
    )

    for index, line in enumerate(
        lines
    ):

        text = str(
            line.get(
                "text",
                "",
            )
        )

        normalized = (
            text.casefold()
        )

        normalized = re.sub(
            r"[._\-]+",
            " ",
            normalized,
        )

        normalized = clean_spaces(
            normalized
        )

        if (
            "hsn code"
            not in normalized
            and
            "hsn sac"
            not in normalized
        ):
            continue

        same_line = re.search(
            r"(?i)"
            r"HSN"
            r"(?:\s*/\s*SAC)?"
            r"(?:\s+Code)?"
            r"\s*[:=\-]?\s*"
            r"(\d{4,8})",
            text,
        )

        if same_line:

            value = same_line.group(1)

            if len(value) in {
                4,
                6,
                8,
            }:

                return {
                    "value": value,
                    "page":
                        line.get("page"),
                    "evidence":
                        text,
                }

        # ----------------------------------------------------
        # For known PDF text ordering, inspect next lines.
        # ----------------------------------------------------

        for offset in (
            1,
            2,
            3,
        ):

            candidate_index = (
                index + offset
            )

            if candidate_index >= len(
                lines
            ):
                break

            candidate = lines[
                candidate_index
            ]

            if (
                candidate.get("page")
                !=
                line.get("page")
            ):
                break

            candidate_text = str(
                candidate.get(
                    "text",
                    "",
                )
            ).strip()

            match = re.fullmatch(
                r"\d{4,8}",
                candidate_text,
            )

            if not match:
                continue

            value = match.group(0)

            if len(value) not in {
                4,
                6,
                8,
            }:
                continue

            return {
                "value":
                    value,

                "page":
                    candidate.get(
                        "page"
                    ),

                "evidence":
                    candidate_text,
            }

    return None


# ============================================================
# PRODUCTION GST COMPONENT DISCOVERY
# ============================================================

def find_verified_gst_components(
    payload: Any,
) -> dict:

    found = {}

    def consume_component(
        item: Any,
    ):

        if not isinstance(
            item,
            dict,
        ):
            return

        component_type = str(
            item.get(
                "type",
                "",
            )
        ).upper().strip()

        if component_type not in {
            "CGST",
            "SGST",
            "IGST",
        }:
            return

        rate = item.get(
            "rate_percent"
        )

        amount = item.get(
            "amount"
        )

        if amount is None:
            return

        # Keep first verified component for each tax type.
        if component_type not in found:

            found[
                component_type
            ] = {
                "rate":
                    rate,

                "amount":
                    amount,

                "page":
                    item.get(
                        "page"
                    ),

                "source":
                    item.get(
                        "source",
                        "PRODUCTION_GST_RECONCILIATION",
                    ),

                "evidence":
                    item.get(
                        "row_text"
                    ),
            }

    def walk(
        node: Any,
    ):

        if isinstance(
            node,
            dict,
        ):

            for key, value in (
                node.items()
            ):

                if (
                    key
                    in {
                        "gst_components",
                        "gst_components_selected",
                    }
                    and
                    isinstance(
                        value,
                        list,
                    )
                ):

                    for item in value:
                        consume_component(item)

                walk(value)

        elif isinstance(
            node,
            list,
        ):

            for item in node:
                walk(item)

    walk(payload)

    return found


def format_number(
    value: Any,
) -> str:

    try:

        number = float(value)

        if number.is_integer():

            return (
                f"{number:,.2f}"
            )

        return (
            f"{number:,.2f}"
        )

    except Exception:

        return str(value)


# ============================================================
# POST-PROCESS DYNAMIC RESULT
# ============================================================

def clean_dynamic_result(
    result: dict,
    uploaded_file,
):

    dynamic_fields = result.setdefault(
        "dynamic_fields",
        {},
    )

    discovered = result.get(
        "auto_discovered_parameters",
        [],
    )

    # --------------------------------------------------------
    # Canonicalize + deduplicate dynamic result keys
    # --------------------------------------------------------

    cleaned_dynamic = {}

    for (
        raw_name,
        information,
    ) in dynamic_fields.items():

        name = normalize_label(
            raw_name
        )

        if not valid_auto_label(
            name
        ):
            continue

        key = normalized_label_key(
            name
        )

        existing = cleaned_dynamic.get(
            key
        )

        candidate = {
            "name":
                name,

            "information":
                information,
        }

        if existing is None:

            cleaned_dynamic[
                key
            ] = candidate

            continue

        # Prefer DETECTED over NOT_DETECTED / AMBIGUOUS.
        existing_info = (
            existing[
                "information"
            ]
            if isinstance(
                existing[
                    "information"
                ],
                dict,
            )
            else
            {}
        )

        candidate_info = (
            information
            if isinstance(
                information,
                dict,
            )
            else
            {}
        )

        existing_detected = (
            existing_info.get(
                "status"
            )
            ==
            "DETECTED"
        )

        candidate_detected = (
            candidate_info.get(
                "status"
            )
            ==
            "DETECTED"
        )

        if (
            candidate_detected
            and
            not existing_detected
        ):

            cleaned_dynamic[
                key
            ] = candidate

            continue

        if (
            candidate_detected
            ==
            existing_detected
        ):

            existing_confidence = float(
                existing_info.get(
                    "confidence",
                    0.0,
                )
                or
                0.0
            )

            candidate_confidence = float(
                candidate_info.get(
                    "confidence",
                    0.0,
                )
                or
                0.0
            )

            if (
                candidate_confidence
                >
                existing_confidence
            ):

                cleaned_dynamic[
                    key
                ] = candidate

    dynamic_fields = {
        value[
            "name"
        ]:
            value[
                "information"
            ]

        for value
        in cleaned_dynamic.values()
    }

    # --------------------------------------------------------
    # HSN explicit-label repair
    # --------------------------------------------------------

    if "HSN Code" in discovered:

        repaired_hsn = (
            find_explicit_hsn_value(
                uploaded_file
            )
        )

        if repaired_hsn:

            dynamic_fields[
                "HSN Code"
            ] = {
                "value":
                    repaired_hsn[
                        "value"
                    ],

                "status":
                    "DETECTED",

                "confidence":
                    1.0,

                "page":
                    repaired_hsn[
                        "page"
                    ],

                "source":
                    "AUTO_EXPLICIT_LABEL",

                "evidence":
                    repaired_hsn[
                        "evidence"
                    ],
            }

    # --------------------------------------------------------
    # Replace CGST / SGST / IGST with reconciled production
    # components whenever available.
    # --------------------------------------------------------

    core_result = result.get(
        "production_result",
        result,
    )

    verified_gst = (
        find_verified_gst_components(
            core_result
        )
    )

    for tax_type in (
        "CGST",
        "SGST",
        "IGST",
    ):

        if tax_type not in discovered:
            continue

        component = verified_gst.get(
            tax_type
        )

        if not component:
            continue

        rate = component.get(
            "rate"
        )

        amount = component.get(
            "amount"
        )

        if rate is not None:

            try:

                rate_text = (
                    f"{float(rate):g}%"
                )

            except Exception:

                rate_text = (
                    f"{rate}%"
                )

            value = (
                f"{rate_text} / "
                f"{format_number(amount)}"
            )

        else:

            value = (
                format_number(
                    amount
                )
            )

        dynamic_fields[
            tax_type
        ] = {
            "value":
                value,

            "status":
                "DETECTED",

            "confidence":
                1.0,

            "page":
                component.get(
                    "page"
                ),

            "source":
                "PRODUCTION_GST_RECONCILIATION",

            "evidence":
                (
                    component.get(
                        "evidence"
                    )
                    or
                    component.get(
                        "source"
                    )
                ),
        }

    # --------------------------------------------------------
    # Clean discovered schema too
    # --------------------------------------------------------

    clean_discovered = []

    seen = set()

    for field in discovered:

        canonical = normalize_label(
            field
        )

        if not valid_auto_label(
            canonical
        ):
            continue

        key = normalized_label_key(
            canonical
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        clean_discovered.append(
            canonical
        )

    result[
        "dynamic_fields"
    ] = dynamic_fields

    result[
        "auto_discovered_parameters"
    ] = clean_discovered

    result[
        "dynamic_schema_mode"
    ] = (
        "AUTOMATIC_SCHEMA_DISCOVERY"
    )

    result[
        "dynamic_discovery_mode"
    ] = (
        "AUTOMATIC_SCHEMA_DISCOVERY"
    )

    return result


# ============================================================
# PROCESS UPLOAD
# ============================================================

def process_upload(
    uploaded_file,
):
    """
    FINAL FRONTEND -> UNIFIED ENGINE BRIDGE.

    The frontend supplies only the uploaded document.

    invoice_engine.py performs:
      - LayoutLMv3 trained extraction
      - automatic dynamic schema discovery
      - dynamic value extraction
      - V7 reconciliation
      - trained + dynamic field merger
      - unified JSON generation

    No frontend field schema is supplied.
    """

    suffix = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
        or
        ".bin"
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = Path(
                temp_file.name
            )

        result = (
            process_invoice_dynamic_final(
                str(
                    temp_path
                )
            )
        )

        return make_json_safe(
            result
        )

    finally:

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass


# ============================================================
# RESULT HELPERS
# ============================================================

def production_payload(
    result: Any,
) -> Any:
    """
    Unified V3 Dynamic result is already the production payload.

    Backward-compatible fallback remains for older cached results.
    """

    if not isinstance(
        result,
        dict,
    ):
        return result

    if (
        result.get(
            "schema_version"
        )
        ==
        "invoice_ai_v3_dynamic_unified_v1"
    ):
        return result

    if (
        "production_result"
        in result
    ):
        return result[
            "production_result"
        ]

    return result


def unified_fields_payload(
    result: Any,
) -> dict:
    """
    Return the one combined trained + dynamic field dictionary.
    """

    if not isinstance(
        result,
        dict,
    ):
        return {}

    fields = result.get(
        "fields",
        {},
    )

    if isinstance(
        fields,
        dict,
    ):
        return fields

    return {}


def dynamic_payload(
    result: Any,
) -> dict:
    """
    Return only fields whose unified origin is AUTO_DYNAMIC.

    Older wrapper results are still supported.
    """

    if not isinstance(
        result,
        dict,
    ):
        return {}

    fields = result.get(
        "fields"
    )

    if isinstance(
        fields,
        dict,
    ):

        return {
            field_name:
                information

            for (
                field_name,
                information,
            ) in fields.items()

            if (
                isinstance(
                    information,
                    dict,
                )
                and
                information.get(
                    "origin"
                )
                ==
                "AUTO_DYNAMIC"
            )
        }

    dynamic = result.get(
        "dynamic_fields",
        {},
    )

    if isinstance(
        dynamic,
        dict,
    ):
        return dynamic

    return {}


def trained_payload(
    result: Any,
) -> dict:
    """
    Return only the fixed trained neural-schema fields.
    """

    if not isinstance(
        result,
        dict,
    ):
        return {}

    fields = result.get(
        "fields",
        {},
    )

    if not isinstance(
        fields,
        dict,
    ):
        return {}

    output = {}

    for field_name in EXPECTED_FIELDS:

        if field_name in fields:

            output[
                field_name
            ] = fields[
                field_name
            ]

    return output


# ============================================================
# SUMMARY
# ============================================================

def render_summary(
    fields: dict,
):

    st.subheader(
        "Invoice Summary"
    )

    cards = [
        (
            "Vendor",
            display_value(
                fields.get(
                    "VENDOR_NAME"
                )
            ),
        ),

        (
            "Invoice #",
            display_value(
                fields.get(
                    "INVOICE_NUMBER"
                )
            ),
        ),

        (
            "Invoice Date",
            display_value(
                fields.get(
                    "INVOICE_DATE"
                )
            ),
        ),

        (
            "Total Amount",
            display_value(
                fields.get(
                    "TOTAL_AMOUNT"
                )
            ),
        ),
    ]

    columns = st.columns(
        4
    )

    for (
        column,
        (
            label,
            value,
        ),
    ) in zip(
        columns,
        cards,
    ):

        with column:

            safe_label = html.escape(
                str(label)
            )

            if (
                value
                ==
                "NOT_DETECTED"
            ):

                safe_value = (
                    '<span class="not-detected">'
                    'NOT_DETECTED'
                    '</span>'
                )

            else:

                safe_value = html.escape(
                    str(value)
                )

            render_html(
                f"""
                <div class="section-card">

                    <div class="mini-label">
                        {safe_label}
                    </div>

                    <div class="mini-value">
                        {safe_value}
                    </div>

                </div>
                """
            )


# ============================================================
# TRAINED FIELD TABLE
# ============================================================

def render_fields(
    fields: dict,
):

    rows = [
        {
            "Field":
                FIELD_LABELS.get(
                    field,
                    field,
                ),

            "Value":
                display_value(
                    fields.get(
                        field
                    )
                ),
        }

        for field
        in EXPECTED_FIELDS
    ]

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DYNAMIC FIELD TABLE
# ============================================================

def render_dynamic_fields(
    dynamic_fields: dict,
):

    if not dynamic_fields:

        st.info(
            "No additional fields were automatically discovered."
        )

        return

    rows = []

    for (
        field_name,
        information,
    ) in dynamic_fields.items():

        if not isinstance(
            information,
            dict,
        ):

            continue

        confidence = information.get(
            "confidence",
            0.0,
        )

        try:

            confidence_text = (
                f"{float(confidence) * 100:.1f}%"
            )

        except Exception:

            confidence_text = str(
                confidence
            )

        rows.append(
            {
                "Field":
                    field_name,

                "Value":
                    display_value(
                        information.get(
                            "value"
                        )
                    ),

                "Status":
                    display_value(
                        information.get(
                            "status"
                        )
                    ),

                "Confidence":
                    confidence_text,

                "Page":
                    (
                        information.get(
                            "page"
                        )
                        or
                        "—"
                    ),

                "Source":
                    display_value(
                        information.get(
                            "source"
                        )
                    ),

                "Evidence":
                    display_value(
                        information.get(
                            "evidence"
                        )
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            rows
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# LINE ITEMS
# ============================================================

def render_line_items(
    result,
):

    table = find_table(
        result
    )

    if table is None:

        st.info(
            "No structured line-item table "
            "was returned for this invoice."
        )

        return

    if isinstance(
        table,
        dict,
    ):

        try:

            dataframe = pd.DataFrame(
                table
            )

        except Exception:

            dataframe = pd.DataFrame(
                [
                    {
                        "Key":
                            key,

                        "Value":
                            display_value(
                                value
                            ),
                    }

                    for key, value
                    in table.items()
                ]
            )

    else:

        dataframe = pd.DataFrame(
            table
        )

    if dataframe.empty:

        st.info(
            "No structured line-item rows were returned."
        )

        return

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RESULTS
# ============================================================


def render_results(
    result,
):
    """
    Final production result interface.

    FRONTEND ONLY.
    No extraction or inference changes.
    """

    core_result = production_payload(
        result
    )

    all_fields = unified_fields_payload(
        result
    )

    trained_fields = trained_payload(
        result
    )

    dynamic_fields = dynamic_payload(
        result
    )


    # ========================================================
    # SAFE FIELD VALUE EXTRACTOR
    # ========================================================

    def field_value(
        information,
    ):

        if information is None:
            return "Not present"

        status = ""

        if isinstance(
            information,
            dict,
        ):

            status = str(
                information.get(
                    "status",
                    "",
                )
                or
                ""
            ).upper()

            value = information.get(
                "value"
            )

            if value is None:

                for key in VALUE_KEYS:

                    candidate = information.get(
                        key
                    )

                    if candidate not in (
                        None,
                        "",
                    ):
                        value = candidate
                        break

        else:

            value = information


        # ----------------------------------------------------
        # CLEAN BUSINESS-FACING MISSING VALUES
        # ----------------------------------------------------

        missing_values = {
            None,
            "",
            "?",
            "??",
            "???",
            "NOT_DETECTED",
            "NOT_PRESENT",
            "NONE",
            "NULL",
            "N/A",
            "NA",
        }

        value_text = (
            str(value).strip()
            if value is not None
            else
            ""
        )

        is_simple_missing = (
            value is None
            or
            (
                isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                )
                and
                value_text.upper()
                in {
                    "",
                    "?",
                    "??",
                    "???",
                    "NOT_DETECTED",
                    "NOT_PRESENT",
                    "NONE",
                    "NULL",
                    "N/A",
                    "NA",
                }
            )
        )

        if (
            is_simple_missing
            or
            status
            in {
                "NOT_PRESENT",
                "NOT_DETECTED",
                "NO_EVIDENCE",
            }
        ):
            return "Not present"


        if isinstance(
            value,
            list,
        ):

            cleaned = []

            for item in value:

                item_text = str(
                    item
                ).strip()

                if (
                    not item_text
                    or
                    item_text
                    in {
                        "?",
                        "??",
                        "???",
                    }
                    or
                    item_text.upper()
                    in {
                        "NOT_DETECTED",
                        "NOT_PRESENT",
                        "NONE",
                        "NULL",
                        "N/A",
                        "NA",
                    }
                ):
                    cleaned.append(
                        "Not present"
                    )

                else:
                    cleaned.append(
                        item_text
                    )

            return " | ".join(
                cleaned
            )


        if isinstance(
            value,
            dict,
        ):

            return json.dumps(
                value,
                ensure_ascii=False,
            )


        return value_text


    def named_value(
        key,
    ):

        return field_value(
            all_fields.get(
                key
            )
        )


    # ========================================================
    # STATUS
    # ========================================================

    result_status = str(
        result.get(
            "status",
            "",
        )
        if isinstance(
            result,
            dict,
        )
        else
        ""
    ).upper()


    if not result_status:

        validation = (
            result.get(
                "validation",
                {},
            )
            if isinstance(
                result,
                dict,
            )
            else
            {}
        )

        if isinstance(
            validation,
            dict,
        ):

            result_status = str(
                validation.get(
                    "overall_status",
                    "",
                )
            ).upper()


    if result_status == "PASS":

        render_html(
            """
            <div class="result-banner result-banner-pass">

                <div>
                    <div class="result-banner-title">
                        Invoice extraction completed successfully
                    </div>

                    <div class="result-banner-sub">
                        Core extraction and production validation
                        completed without blocking issues.
                    </div>
                </div>

                <div class="result-chip result-chip-pass">
                    PASS
                </div>

            </div>
            """
        )

    else:

        safe_status = html.escape(
            result_status
            or
            "REVIEW_REQUIRED"
        )

        render_html(
            f"""
            <div class="result-banner result-banner-review">

                <div>
                    <div class="result-banner-title">
                        Invoice processed successfully
                    </div>

                    <div class="result-banner-sub">
                        Some extracted information should be reviewed.
                    </div>
                </div>

                <div class="result-chip result-chip-review">
                    {safe_status}
                </div>

            </div>
            """
        )


    # ========================================================
    # CORE SUMMARY
    # ========================================================

    render_summary(
        trained_fields
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    field_summary = (
        result.get(
            "field_summary",
            {},
        )
        if isinstance(
            result,
            dict,
        )
        else
        {}
    )

    total_fields = field_summary.get(
        "total_fields",
        len(all_fields),
    )

    trained_count = field_summary.get(
        "trained_schema_fields",
        len(trained_fields),
    )

    dynamic_count = field_summary.get(
        "auto_dynamic_fields",
        len(dynamic_fields),
    )

    resolved_count = field_summary.get(
        "resolved_fields",
        0,
    )


    m1, m2, m3, m4 = st.columns(
        4
    )

    m1.metric(
        "\U0001F4CB  Unified Fields",
        total_fields,
    )

    m2.metric(
        "\U0001F9E0  Trained Fields",
        trained_count,
    )

    m3.metric(
        "\u2728  Dynamic Fields",
        dynamic_count,
    )

    m4.metric(
        "\u2705  Resolved Fields",
        resolved_count,
    )


    # ========================================================
    # TABS
    # ========================================================

    (
        tab_all,
        tab_trained,
        tab_dynamic,
        tab_items,
        tab_quick,
        tab_json,
    ) = st.tabs(
        [
            "\U0001F4CB Unified Fields",
            "\U0001F9E0 Trained Schema",
            "\u2728 Dynamic Fields",
            "\U0001F9FE Line Items",
            "\U0001F441 Quick View",
            "\U0001F9E9 Final JSON",
        ]
    )


    # ========================================================
    # UNIFIED FIELDS
    # ========================================================

    with tab_all:

        st.markdown(
            "### \U0001F4CB Unified Invoice Fields"
        )

        st.caption(
            "One combined view containing trained fields "
            "and automatically discovered dynamic fields."
        )

        rows = []

        for (
            field_name,
            information,
        ) in all_fields.items():

            if isinstance(
                information,
                dict,
            ):

                origin = str(
                    information.get(
                        "origin",
                        "",
                    )
                    or
                    ""
                )

                status = str(
                    information.get(
                        "status",
                        "",
                    )
                    or
                    ""
                )

                source_name = str(
                    information.get(
                        "source",
                        "",
                    )
                    or
                    ""
                )

            else:

                origin = ""
                status = ""
                source_name = ""


            if origin == "TRAINED_SCHEMA":
                origin_text = "Trained"

            elif origin == "AUTO_DYNAMIC":
                origin_text = "Dynamic"

            else:
                origin_text = origin or "?"


            rows.append(
                {
                    "Field":
                        FIELD_LABELS.get(
                            field_name,
                            field_name,
                        ),

                    "Value":
                        field_value(
                            information
                        ),

                    "Type":
                        origin_text,

                    "Status":
                        status or "?",

                    "Source":
                        source_name or "?",
                }
            )


        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # TRAINED FIELDS
    # ========================================================

    with tab_trained:

        st.markdown(
            "### \U0001F9E0 LayoutLMv3 Trained Schema"
        )

        st.caption(
            "The fixed neural extraction schema containing "
            "the 16 trained invoice fields."
        )

        render_fields(
            trained_fields
        )


    # ========================================================
    # DYNAMIC FIELDS
    # ========================================================

    with tab_dynamic:

        st.markdown(
            "### \u2728 Automatically Discovered Fields"
        )

        st.caption(
            "Additional parameters discovered automatically "
            "by the inference engine at runtime."
        )

        discovered = (
            result.get(
                "auto_discovered_parameters",
                [],
            )
            if isinstance(
                result,
                dict,
            )
            else
            []
        )


        if discovered:

            with st.expander(
                f"Detected dynamic schema ({len(discovered)} fields)",
                expanded=False,
            ):

                for field in discovered:

                    st.write(
                        f"- {field}"
                    )


        render_dynamic_fields(
            dynamic_fields
        )


    # ========================================================
    # LINE ITEMS
    # ========================================================

    with tab_items:

        st.markdown(
            "### \U0001F9FE Structured Line Items"
        )

        st.caption(
            "Material, quantity, unit price and line-value "
            "information recovered from the invoice."
        )

        render_line_items(
            core_result
        )


    # ========================================================
    # QUICK VIEW
    # ========================================================

    with tab_quick:

        render_html(
            """
            <div class="quick-panel">

                <div class="quick-panel-title">
                    &#128065; Invoice Quick View
                </div>

                <div class="quick-panel-sub">
                    A simplified business-facing view of the
                    extracted invoice without opening the JSON.
                </div>

            </div>
            """
        )


        q1, q2, q3 = st.columns(
            3
        )

        q1.metric(
            "\U0001F3E2  Vendor",
            named_value(
                "VENDOR_NAME"
            ),
        )

        q2.metric(
            "\U0001F9FE  Invoice Number",
            named_value(
                "INVOICE_NUMBER"
            ),
        )

        q3.metric(
            "\U0001F4C5  Invoice Date",
            named_value(
                "INVOICE_DATE"
            ),
        )


        q4, q5, q6 = st.columns(
            3
        )

        q4.metric(
            "\U0001F464  Customer",
            named_value(
                "CUSTOMER_NAME"
            ),
        )

        q5.metric(
            "\U0001F4B1  Currency",
            named_value(
                "CURRENCY"
            ),
        )

        q6.metric(
            "\U0001F4B0  Total Amount",
            named_value(
                "TOTAL_AMOUNT"
            ),
        )


        st.markdown(
            "### \U0001F5C2 All Extracted Fields"
        )

        st.caption(
            "Simple Field / Value view for business users."
        )


        quick_rows = []

        for (
            field_name,
            information,
        ) in all_fields.items():

            if isinstance(
                information,
                dict,
            ):

                origin = str(
                    information.get(
                        "origin",
                        "",
                    )
                    or
                    ""
                )

            else:

                origin = ""


            field_type = (
                "Trained"
                if origin == "TRAINED_SCHEMA"
                else
                "Dynamic"
                if origin == "AUTO_DYNAMIC"
                else
                origin
                or
                "?"
            )


            quick_rows.append(
                {
                    "Field":
                        FIELD_LABELS.get(
                            field_name,
                            field_name,
                        ),

                    "Value":
                        field_value(
                            information
                        ),

                    "Type":
                        field_type,
                }
            )


        quick_dataframe = (
            pd.DataFrame(
                quick_rows
            )
        )


        st.dataframe(
            quick_dataframe,
            use_container_width=True,
            hide_index=True,
            height=min(
                850,
                80
                +
                len(
                    quick_dataframe
                )
                *
                35,
            ),
        )


        line_items = (
            core_result.get(
                "line_items",
                [],
            )
            if isinstance(
                core_result,
                dict,
            )
            else
            []
        )


        if (
            isinstance(
                line_items,
                list,
            )
            and
            line_items
        ):

            st.markdown(
                "### \U0001F6D2 Line Items"
            )

            st.caption(
                f"{len(line_items)} structured line item(s)"
            )

            st.dataframe(
                pd.DataFrame(
                    line_items
                ),
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # FINAL JSON
    # ========================================================

    with tab_json:

        st.markdown(
            "### \U0001F9E9 Final Unified JSON"
        )

        st.caption(
            "Complete production JSON returned by "
            "Invoice_AI_V3_Dynamic_Production."
        )

        st.json(
            result
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    json_bytes = (
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        "\u2B07 Download Unified JSON",
        data=json_bytes,
        file_name=
            "invoice_ai_v3_dynamic_result.json",
        mime=
            "application/json",
    )


# ============================================================
# START ENGINE
# ============================================================

engine = initialize_engine()


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-kicker">
            AI-Powered Document Intelligence
        </div>

        <div class="hero-title">
            🧾 Invoice Intelligence AI
        </div>

        <div class="hero-sub">
            Upload an invoice and automatically extract the
            verified Invoice AI V3 schema together with
            additional fields discovered dynamically from
            the document itself.
        </div>

        <div class="status-pill">

            <span class="status-dot"></span>

            Production engine ready ? Dynamic discovery enabled

        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧾 Invoice AI V3"
    )

    st.caption(
        "Production Engine"
    )

    st.success(
        "● Engine Ready"
    )

    left, right = st.columns(
        2
    )

    with left:

        st.metric(
            "Trained Fields",
            len(
                engine.get(
                    "fields",
                    EXPECTED_FIELDS,
                )
            ),
        )

    with right:

        st.metric(
            "BIO Labels",
            len(
                engine.get(
                    "label_list",
                    [],
                )
            ),
        )

    st.metric(
        "Parameters",
        f"{engine.get('parameter_count', 0):,}",
    )

    st.metric(
        "Device",
        str(
            engine.get(
                "device",
                "cpu",
            )
        ).upper(),
    )

    render_html(
        """
        <div class="auto-card">

            <div class="auto-title">
                Automatic Dynamic Schema
            </div>

            <div class="auto-text">
                Extra invoice fields are discovered
                automatically from every uploaded document.
                No manual field list is required.
            </div>

        </div>
        """
    )

    with st.expander(
        "Engine Details"
    ):

        st.write(
            {
                "Architecture":
                    engine[
                        "model"
                    ].__class__.__name__,

                "Runtime":
                    "V3 + V6.1",

                "Trained Neural Schema":
                    "16 fields",

                "Dynamic Schema":
                    "Automatic discovery",

                "Dynamic Discovery Mode":
                    "AUTOMATIC_SCHEMA_DISCOVERY",
            }
        )

    st.divider()

    if st.button(
        "Clear Current Result",
        use_container_width=True,
    ):

        st.session_state.pop(
            "invoice_result",
            None,
        )

        st.session_state.pop(
            "invoice_result_name",
            None,
        )

        st.rerun()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "### 1. Upload Invoice"
)

uploaded_file = st.file_uploader(
    "Choose a PDF or invoice image",

    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tiff",
        "tif",
    ],
)


if uploaded_file is None:

    render_html(
        """
        <div class="ready-card">

            <div class="ready-title">
                Ready
            </div>

            <div class="ready-text">
                Upload an invoice to begin automatic extraction.
            </div>

        </div>
        """
    )

    st.stop()


# ============================================================
# AUTOMATIC FIELD DISCOVERY
# ============================================================

# Automatic discovery now occurs inside invoice_engine.py.
auto_fields = []


# ============================================================
# FILE INFO
# ============================================================

file_size_mb = (
    len(
        uploaded_file.getvalue()
    )
    /
    (
        1024
        *
        1024
    )
)

file_col, size_col = st.columns(
    [
        3,
        1,
    ]
)

with file_col:

    st.caption(
        "FILE"
    )

    st.markdown(
        f"**{uploaded_file.name}**"
    )


with size_col:

    st.caption(
        "SIZE"
    )

    st.markdown(
        f"**{file_size_mb:.2f} MB**"
    )


# ============================================================
# PREVIEW + ACTION
# ============================================================

preview_col, action_col = st.columns(
    [
        2.15,
        1,
    ],
    gap="large",
)


with preview_col:

    st.markdown(
        "### 2. Invoice Preview"
    )

    with st.container(
        border=True
    ):

        preview_uploaded_file(
            uploaded_file
        )


with action_col:

    st.markdown(
        "### 3. Extract"
    )

    render_html(
        """
        <div class="auto-card">

            <div class="auto-title">
                Auto Discovery Active
            </div>

            <div class="auto-text">
                The system scans the invoice structure,
                identifies additional labels automatically
                and sends those fields to the dynamic
                extraction layer.
            </div>

        </div>
        """
    )

    st.info(
        "Additional invoice fields are discovered automatically "
        "inside the Invoice AI inference engine when processing runs."
    )

    st.caption(
        "No manual parameter list is required. "
        "Additional fields are discovered automatically "
        "inside the inference engine."
    )

    st.markdown(
        "**Production Pipeline**"
    )

    st.markdown(
        """
        - Native PDF text / geometry
        - OCR fallback
        - LayoutLMv3 16-field inference
        - GST / financial validation
        - Automatic schema discovery
        - Dynamic additional-field extraction
        - Duplicate / noise cleanup
        - Structured JSON output
        """
    )

    process_clicked = st.button(
        "🚀 Process Invoice",
        type="primary",
        use_container_width=True,
    )

    if process_clicked:

        try:

            with st.spinner(
                "Analyzing invoice and discovering fields..."
            ):

                result = process_upload(
                    uploaded_file,
                )

            st.session_state[
                "invoice_result"
            ] = result

            st.session_state[
                "invoice_result_name"
            ] = uploaded_file.name

        except Exception as error:

            st.error(
                "❌ Invoice processing failed"
            )

            st.exception(
                error
            )


# ============================================================
# STORED RESULT
# ============================================================

current_result = st.session_state.get(
    "invoice_result"
)

current_name = st.session_state.get(
    "invoice_result_name"
)


if (
    current_result is not None
    and
    current_name
    ==
    uploaded_file.name
):

    st.divider()

    st.markdown(
        "## 4. Extraction Results"
    )

    st.caption(
        "The verified 16-field production model and "
        "automatic dynamic-field discovery were both run."
    )

    render_results(
        current_result
    )