from __future__ import annotations

import io
import json
import sys

from collections import defaultdict
from pathlib import Path

import streamlit as st
from PIL import Image


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Vehicle Inspection AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL VISUAL THEME
# ============================================================

st.markdown(
    """
<style>

/* ---------------------------------------------------------
   MAIN BACKGROUND
   --------------------------------------------------------- */

[data-testid="stAppViewContainer"] {

    background:
        radial-gradient(
            circle at 90% 6%,
            rgba(13, 148, 136, 0.07),
            transparent 24%
        ),
        radial-gradient(
            circle at 10% 8%,
            rgba(37, 99, 235, 0.06),
            transparent 25%
        ),
        #f8fafc;
}


/* ---------------------------------------------------------
   MAIN WIDTH
   --------------------------------------------------------- */

.block-container {

    max-width: 1450px;

    padding-top: 2rem;

    padding-bottom: 4rem;
}


/* ---------------------------------------------------------
   TYPOGRAPHY
   --------------------------------------------------------- */

h1 {

    color: #0f172a;

    font-weight: 800 !important;

    letter-spacing: -0.025em;
}


h2, h3 {

    color: #1e293b;

    letter-spacing: -0.015em;
}


/* ---------------------------------------------------------
   SIDEBAR
   --------------------------------------------------------- */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 55%,
            #134e4a 100%
        );
}


[data-testid="stSidebar"] * {

    color: #f8fafc !important;
}


[data-testid="stSidebar"] hr {

    border-color:
        rgba(
            255,
            255,
            255,
            0.15
        );
}


/* ---------------------------------------------------------
   BUTTONS
   --------------------------------------------------------- */

div.stButton > button {

    border-radius: 10px;

    height: 2.9rem;

    font-weight: 700;

    border: none;

    transition:
        all
        0.2s
        ease;
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

    transform:
        translateY(
            -1px
        );

    box-shadow:
        0
        8px
        24px
        rgba(
            15,
            23,
            42,
            0.16
        );
}


/* ---------------------------------------------------------
   BORDER CONTAINERS
   --------------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {

    border-radius:
        16px !important;

    border:
        1px
        solid
        #e2e8f0 !important;

    background:
        rgba(
            255,
            255,
            255,
            0.96
        );

    box-shadow:
        0
        8px
        28px
        rgba(
            15,
            23,
            42,
            0.05
        );
}


/* ---------------------------------------------------------
   METRICS
   --------------------------------------------------------- */

div[data-testid="stMetric"] {

    background:
        white;

    border:
        1px
        solid
        #e2e8f0;

    border-radius:
        14px;

    padding:
        1rem;

    box-shadow:
        0
        6px
        18px
        rgba(
            15,
            23,
            42,
            0.05
        );
}


div[data-testid="stMetricLabel"] {

    color:
        #64748b;
}


div[data-testid="stMetricValue"] {

    color:
        #0f172a;

    font-weight:
        750;
}


/* ---------------------------------------------------------
   UPLOADER
   --------------------------------------------------------- */

[data-testid="stFileUploader"] {

    background:
        white;

    border-radius:
        14px;

    padding:
        0.6rem;
}


/* ---------------------------------------------------------
   IMAGES
   --------------------------------------------------------- */

[data-testid="stImage"] img {

    border-radius:
        14px;

    box-shadow:
        0
        8px
        25px
        rgba(
            15,
            23,
            42,
            0.10
        );
}


/* ---------------------------------------------------------
   DATAFRAME
   --------------------------------------------------------- */

[data-testid="stDataFrame"] {

    border-radius:
        14px;

    overflow:
        hidden;

    border:
        1px
        solid
        #e2e8f0;
}


/* ---------------------------------------------------------
   EXPANDER
   --------------------------------------------------------- */

[data-testid="stExpander"] {

    background:
        white;

    border-radius:
        12px;
}


/* ---------------------------------------------------------
   DOWNLOAD BUTTONS
   --------------------------------------------------------- */

div.stDownloadButton > button {

    border-radius:
        10px;

    font-weight:
        650;

    border:
        1px
        solid
        #cbd5e1;

    background:
        white;
}


/* ---------------------------------------------------------
   HIDE STREAMLIT BRANDING EXCESS
   --------------------------------------------------------- */

footer {

    visibility:
        hidden;
}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# ENGINE
# ============================================================

try:

    from vehicle.vehicle_engine import (
        create_vehicle_engine,
    )

except Exception as error:

    st.error(
        "Vehicle Inspection AI could not start."
    )

    st.exception(
        error
    )

    st.stop()


@st.cache_resource(show_spinner=False)
def get_vehicle_engine():

    return create_vehicle_engine(
        verify_hashes=True,
        optimize_fp16=True,
    )


# ============================================================
# HUMAN NAMES
# ============================================================

PART_NAMES = {

    "back_bumper": "Rear Bumper",
    "back_door": "Rear Door",
    "back_wheel": "Rear Wheel",
    "back_window": "Rear Window",
    "back_windshield": "Rear Windshield",

    "fender": "Fender",

    "front_bumper": "Front Bumper",
    "front_door": "Front Door",
    "front_wheel": "Front Wheel",
    "front_window": "Front Window",

    "grille": "Front Grille",
    "headlight": "Headlight",
    "hood": "Hood / Bonnet",
    "license_plate": "License Plate",
    "mirror": "Side Mirror",

    "quarter_panel": "Quarter Panel",
    "rocker_panel": "Lower Side Panel",

    "roof": "Roof",
    "tail_light": "Tail Light",
    "trunk": "Trunk / Boot",
    "windshield": "Front Windshield",

    "unassigned": "Unidentified Body Area",
}


DAMAGE_NAMES = {

    "STRUCTURAL_DAMAGE":
        "Structural Damage",

    "DEFORMATION":
        "Dent / Deformation",

    "SURFACE_DAMAGE":
        "Surface Damage",

    "CORROSION":
        "Corrosion",
}


DAMAGE_EXPLANATIONS = {

    "STRUCTURAL_DAMAGE":
        (
            "This component appears physically broken, "
            "cracked, displaced or heavily damaged."
        ),

    "DEFORMATION":
        (
            "This body panel appears bent, dented, "
            "warped or pushed out of its normal shape."
        ),

    "SURFACE_DAMAGE":
        (
            "Visible cosmetic damage such as scratches, "
            "paint chips or surface deterioration was detected."
        ),

    "CORROSION":
        (
            "The area shows visible signs consistent "
            "with rust or corrosion."
        ),
}


ACTION_TEXT = {

    "minor":
        (
            "Cosmetic repair is recommended."
        ),

    "moderate":
        (
            "Professional inspection and repair are recommended."
        ),

    "severe":
        (
            "Professional repair is strongly recommended. "
            "The affected component may need replacement."
        ),
}


SEVERITY_ORDER = {

    "minor": 1,
    "moderate": 2,
    "severe": 3,
}


def human_part(value):

    return PART_NAMES.get(
        value,
        value
        .replace("_", " ")
        .title(),
    )


def human_damage(value):

    return DAMAGE_NAMES.get(
        value,
        value
        .replace("_", " ")
        .title(),
    )


# ============================================================
# GROUP DAMAGE BY PART
# ============================================================

def group_issues(issues):

    grouped = defaultdict(list)

    for issue in issues:

        grouped[
            issue.get(
                "part",
                "unassigned",
            )
        ].append(
            issue
        )

    output = []

    for part, entries in grouped.items():

        entries = sorted(
            entries,
            key=lambda item: (
                SEVERITY_ORDER.get(
                    item.get(
                        "severity",
                        "minor",
                    ),
                    0,
                ),
                item.get(
                    "confidence",
                    0.0,
                ),
            ),
            reverse=True,
        )

        strongest = entries[0]

        damage_types = []

        for entry in entries:

            damage_type = entry.get(
                "damage_class",
                "UNKNOWN",
            )

            if damage_type not in damage_types:

                damage_types.append(
                    damage_type
                )

        output.append(
            {
                "part":
                    part,

                "severity":
                    strongest.get(
                        "severity",
                        "minor",
                    ),

                "confidence":
                    max(
                        entry.get(
                            "confidence",
                            0.0,
                        )
                        for entry
                        in entries
                    ),

                "damage_types":
                    damage_types,

                "detections":
                    len(entries),
            }
        )

    output.sort(
        key=lambda item: (
            SEVERITY_ORDER.get(
                item[
                    "severity"
                ],
                0,
            ),
            item[
                "confidence"
            ],
        ),
        reverse=True,
    )

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🚗 Vehicle Inspection"
    )

    st.caption(
        "AI-assisted visible exterior damage assessment"
    )

    st.divider()

    st.markdown(
        """
**For best results**

- Use a clear exterior vehicle photo
- Keep the damaged area visible
- Avoid motion blur
- Use good lighting
- Avoid extreme viewing angles
"""
    )

    st.divider()

    if st.button(
        "← Back to Home",
        use_container_width=True,
    ):

        st.switch_page(
            "app.py"
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚗 Vehicle Inspection AI"
)

st.write(
    "Upload a vehicle photo and receive a simple inspection "
    "report showing visible damage, affected parts, severity "
    "and recommended action."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.header(
    "1. Upload Vehicle Photo"
)

uploaded_file = st.file_uploader(
    "Choose a JPG, PNG or WEBP image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a vehicle photo to begin the inspection."
    )

    st.stop()


image_bytes = (
    uploaded_file.getvalue()
)


try:

    preview_image = Image.open(
        io.BytesIO(
            image_bytes
        )
    ).convert(
        "RGB"
    )

except Exception as error:

    st.error(
        "The uploaded image could not be opened."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# PREVIEW + ACTION
# ============================================================

preview_col, action_col = st.columns(
    [
        2,
        1,
    ],
    gap="large",
)


with preview_col:

    st.image(
        preview_image,
        caption="Uploaded Vehicle",
        use_container_width=True,
    )


with action_col:

    with st.container(
        border=True
    ):

        st.subheader(
            "Ready to Inspect"
        )

        st.write(
            "The AI will analyze visible body parts "
            "and identify areas showing exterior damage."
        )

        st.markdown(
            "**The report will include:**"
        )

        st.markdown(
            """
- Affected vehicle parts
- Damage type
- Severity
- Repair recommendation
- Annotated damage image
"""
        )

        process_clicked = st.button(
            "🔍 Inspect Vehicle",
            type="primary",
            use_container_width=True,
        )


# ============================================================
# RUN
# ============================================================

if process_clicked:

    try:

        with st.spinner(
            "Inspecting vehicle..."
        ):

            engine = (
                get_vehicle_engine()
            )

            output = engine.inspect(
                image_bytes,
                image_name=uploaded_file.name,
            )

        st.session_state[
            "vehicle_result"
        ] = output

        st.session_state[
            "vehicle_result_name"
        ] = uploaded_file.name

    except Exception as error:

        st.error(
            "Vehicle inspection failed."
        )

        st.exception(
            error
        )


# ============================================================
# RESULT STATE
# ============================================================

current_output = (
    st.session_state.get(
        "vehicle_result"
    )
)

current_name = (
    st.session_state.get(
        "vehicle_result_name"
    )
)


if (
    current_output is None
    or
    current_name
    !=
    uploaded_file.name
):

    st.stop()


result = (
    current_output[
        "result"
    ]
)

overlay = (
    current_output[
        "overlay"
    ]
)

issues = result.get(
    "issues",
    [],
)

grouped_issues = (
    group_issues(
        issues
    )
)

status = result.get(
    "inspection_status",
    "UNKNOWN",
)


st.divider()


# ============================================================
# OVERALL RESULT
# ============================================================

st.header(
    "2. Inspection Result"
)


if status == "PASS":

    st.success(
        "✅ No significant visible exterior damage detected."
    )

    st.write(
        "The AI did not identify any visible damage "
        "that passed the inspection acceptance criteria."
    )


elif status == "FAIL":

    st.error(
        "❌ Significant visible damage detected."
    )

    st.write(
        "One or more vehicle components show severe visible "
        "damage. Professional inspection and repair are recommended."
    )


else:

    st.warning(
        "⚠️ Visible damage detected."
    )

    st.write(
        "The vehicle has visible exterior damage that "
        "should be professionally inspected."
    )


# ============================================================
# SUMMARY METRICS
# ============================================================

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


metric1.metric(
    "Overall Result",
    (
        "Passed"
        if status == "PASS"
        else
        (
            "Repair Needed"
            if status == "FAIL"
            else
            "Attention Needed"
        )
    ),
)


metric2.metric(
    "Affected Parts",
    len(
        grouped_issues
    ),
)


metric3.metric(
    "Damage Areas",
    result.get(
        "accepted_damage_predictions",
        0,
    ),
)


metric4.metric(
    "Vehicle Parts Recognized",
    result.get(
        "parts_detected",
        0,
    ),
)


# ============================================================
# VISUAL INSPECTION
# ============================================================

st.header(
    "3. Visual Inspection"
)

original_col, ai_col = (
    st.columns(
        2,
        gap="large",
    )
)


with original_col:

    st.subheader(
        "Original Photo"
    )

    st.image(
        preview_image,
        use_container_width=True,
    )


with ai_col:

    st.subheader(
        "AI Damage View"
    )

    st.image(
        overlay,
        use_container_width=True,
    )


# ============================================================
# HUMAN DAMAGE REPORT
# ============================================================

st.header(
    "4. What Needs Attention"
)


if not grouped_issues:

    st.success(
        "No significant visible damage was detected."
    )


else:

    for index, item in enumerate(
        grouped_issues,
        start=1,
    ):

        part_name = human_part(
            item[
                "part"
            ]
        )

        severity = (
            item[
                "severity"
            ]
        )

        severity_name = (
            severity.title()
        )

        damage_types = (
            item[
                "damage_types"
            ]
        )

        damage_names = [
            human_damage(
                damage
            )
            for damage
            in damage_types
        ]

        primary_damage = (
            damage_types[0]
        )

        explanation = (
            DAMAGE_EXPLANATIONS.get(
                primary_damage,
                "Visible damage was detected.",
            )
        )

        recommendation = (
            ACTION_TEXT.get(
                severity,
                "Professional inspection is recommended.",
            )
        )

        with st.container(
            border=True
        ):

            top_left, top_right = (
                st.columns(
                    [
                        3,
                        1,
                    ]
                )
            )


            with top_left:

                st.subheader(
                    f"{index}. {part_name}"
                )


            with top_right:

                if severity == "severe":

                    st.error(
                        "Severe"
                    )

                elif severity == "moderate":

                    st.warning(
                        "Moderate"
                    )

                else:

                    st.success(
                        "Minor"
                    )


            st.markdown(
                "**Damage detected**"
            )

            st.write(
                ", ".join(
                    damage_names
                )
            )


            st.markdown(
                "**What this means**"
            )

            st.write(
                explanation
            )


            st.markdown(
                "**Recommended action**"
            )

            st.write(
                recommendation
            )


            st.progress(
                min(
                    max(
                        float(
                            item[
                                "confidence"
                            ]
                        ),
                        0.0,
                    ),
                    1.0,
                ),
                text=(
                    "AI confidence: "
                    f"{item['confidence']:.0%}"
                ),
            )


# ============================================================
# SIMPLE SUMMARY TABLE
# ============================================================

if grouped_issues:

    st.header(
        "5. Damage Summary"
    )

    rows = []

    for item in grouped_issues:

        rows.append(
            {
                "Affected Part":
                    human_part(
                        item[
                            "part"
                        ]
                    ),

                "Damage":
                    ", ".join(
                        human_damage(
                            damage
                        )
                        for damage
                        in item[
                            "damage_types"
                        ]
                    ),

                "Severity":
                    item[
                        "severity"
                    ].title(),

                "AI Confidence":
                    f"{item['confidence']:.0%}",

                "Recommended Action":
                    ACTION_TEXT.get(
                        item[
                            "severity"
                        ],
                        "Professional inspection recommended.",
                    ),
            }
        )


    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    "ℹ️ This system evaluates only damage visible in the "
    "uploaded photograph. Hidden mechanical, electrical or "
    "internal structural problems cannot be diagnosed from "
    "an exterior image alone."
)


# ============================================================
# DOWNLOAD
# ============================================================

st.header(
    "6. Download Report"
)


download_json, download_image = (
    st.columns(2)
)


json_bytes = json.dumps(
    result,
    indent=2,
).encode(
    "utf-8"
)


with download_json:

    st.download_button(
        "⬇ Download Inspection Data",
        data=json_bytes,
        file_name=(
            Path(
                uploaded_file.name
            ).stem
            +
            "_vehicle_inspection.json"
        ),
        mime="application/json",
        use_container_width=True,
    )


overlay_buffer = (
    io.BytesIO()
)

overlay.save(
    overlay_buffer,
    format="PNG",
)

overlay_buffer.seek(
    0
)


with download_image:

    st.download_button(
        "⬇ Download Annotated Image",
        data=overlay_buffer.getvalue(),
        file_name=(
            Path(
                uploaded_file.name
            ).stem
            +
            "_damage_overlay.png"
        ),
        mime="image/png",
        use_container_width=True,
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "🛠 Technical Details / JSON",
    expanded=False,
):

    st.caption(
        "For developers, integration testing and auditing."
    )

    st.json(
        result
    )
