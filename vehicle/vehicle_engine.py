from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRSegMedium


# ============================================================
# VEHICLE INSPECTION AI
# FINAL PRODUCTION INFERENCE ENGINE
#
# NO TRAINING IS PERFORMED HERE.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

PARTS_MODEL_PATH = (
    MODEL_DIR
    / "parts_rfdetr_seg_medium_v1_best_ema.pth"
)

DAMAGE_MODEL_PATH = (
    MODEL_DIR
    / "damage_rfdetr_seg_medium_v3_grouped_best_ema.pth"
)


EXPECTED_PARTS_SHA256 = (
    "a3a94e0552e43badb8452dad7bd3f75a"
    "fe442ddf494c6a7739fa9daa82939e51"
)

EXPECTED_DAMAGE_SHA256 = (
    "2c18e66377efd2d19e45583551e7f1fa"
    "fdef05af83652dcecd6d203053bb5fd5"
)


PART_CLASSES = [
    "Back-bumper",
    "Back-door",
    "Back-wheel",
    "Back-window",
    "Back-windshield",
    "Fender",
    "Front-bumper",
    "Front-door",
    "Front-wheel",
    "Front-window",
    "Grille",
    "Headlight",
    "Hood",
    "License-plate",
    "Mirror",
    "Quarter-panel",
    "Rocker-panel",
    "Roof",
    "Tail-light",
    "Trunk",
    "Windshield",
]


DAMAGE_CLASSES = [
    "STRUCTURAL_DAMAGE",
    "DEFORMATION",
    "SURFACE_DAMAGE",
    "CORROSION",
]


# ------------------------------------------------------------
# FINAL PRODUCTION SETTINGS
# ------------------------------------------------------------

PART_THRESHOLD = 0.40

RAW_DAMAGE_THRESHOLD = 0.25

DAMAGE_ACCEPTANCE_THRESHOLDS = {
    "STRUCTURAL_DAMAGE": 0.50,
    "DEFORMATION": 0.40,
    "SURFACE_DAMAGE": 0.45,
    "CORROSION": 0.40,
}

MIN_DAMAGE_AREA_RATIO = 0.0005
MIN_ASSOCIATION_COVERAGE = 0.05
DUPLICATE_MASK_IOU = 0.50


CONDITION_MAP = {
    "STRUCTURAL_DAMAGE": "structural_damage",
    "DEFORMATION": "dented_or_deformed",
    "SURFACE_DAMAGE": "surface_damage",
    "CORROSION": "corrosion",
}


SEVERITY_MAP = {
    "STRUCTURAL_DAMAGE": "severe",
    "DEFORMATION": "moderate",
    "SURFACE_DAMAGE": "minor",
    "CORROSION": "moderate",
}


DAMAGE_COLORS = {
    "STRUCTURAL_DAMAGE": (255, 40, 40),
    "DEFORMATION": (255, 140, 0),
    "SURFACE_DAMAGE": (255, 220, 0),
    "CORROSION": (150, 80, 30),
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_file(path: Union[str, Path]) -> str:
    path = Path(path)

    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def normalize_part_name(name: str) -> str:
    return (
        name
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_input_image(
    image_input: Union[
        str,
        Path,
        Image.Image,
        bytes,
        bytearray,
        io.BytesIO,
        Any,
    ]
) -> Image.Image:

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")

    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(
            io.BytesIO(image_input)
        ).convert("RGB")

    if hasattr(image_input, "read"):
        data = image_input.read()

        try:
            image_input.seek(0)
        except Exception:
            pass

        return Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    raise TypeError(
        "Unsupported image input type."
    )


def mask_iou(
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
) -> float:

    if mask_a is None or mask_b is None:
        return 0.0

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def association_metrics(
    damage_mask: Optional[np.ndarray],
    part_mask: Optional[np.ndarray],
) -> Tuple[float, float]:

    if damage_mask is None or part_mask is None:
        return 0.0, 0.0

    intersection = np.logical_and(
        damage_mask,
        part_mask,
    ).sum()

    damage_area = damage_mask.sum()

    union = np.logical_or(
        damage_mask,
        part_mask,
    ).sum()

    coverage = (
        intersection / damage_area
        if damage_area > 0
        else 0.0
    )

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    return (
        float(coverage),
        float(iou),
    )


def apply_mask(
    base_image: Image.Image,
    mask: Optional[np.ndarray],
    rgb: Tuple[int, int, int],
    alpha: float,
) -> Image.Image:

    if mask is None:
        return base_image

    array = np.array(
        base_image
    ).astype(np.float32)

    color = np.array(
        rgb,
        dtype=np.float32,
    )

    array[mask] = (
        array[mask] * (1.0 - alpha)
        +
        color * alpha
    )

    return Image.fromarray(
        np.clip(
            array,
            0,
            255,
        ).astype(np.uint8)
    )


# ============================================================
# ENGINE
# ============================================================

class VehicleInspectionEngine:

    def __init__(
        self,
        verify_hashes: bool = True,
        optimize_fp16: bool = True,
    ):

        self.parts_model_path = PARTS_MODEL_PATH
        self.damage_model_path = DAMAGE_MODEL_PATH

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.parts_model = None
        self.damage_model = None

        if not self.parts_model_path.exists():
            raise FileNotFoundError(
                f"Parts model missing: "
                f"{self.parts_model_path}"
            )

        if not self.damage_model_path.exists():
            raise FileNotFoundError(
                f"Damage model missing: "
                f"{self.damage_model_path}"
            )

        if verify_hashes:
            self._verify_models()

        self._load_models(
            optimize_fp16=optimize_fp16
        )


    # --------------------------------------------------------
    # MODEL VERIFICATION
    # --------------------------------------------------------

    def _verify_models(self) -> None:

        parts_hash = sha256_file(
            self.parts_model_path
        )

        damage_hash = sha256_file(
            self.damage_model_path
        )

        if parts_hash != EXPECTED_PARTS_SHA256:
            raise RuntimeError(
                "Parts model SHA256 mismatch."
            )

        if damage_hash != EXPECTED_DAMAGE_SHA256:
            raise RuntimeError(
                "Damage model SHA256 mismatch."
            )


    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------

    def _load_models(
        self,
        optimize_fp16: bool,
    ) -> None:

        self.parts_model = RFDETRSegMedium(
            num_classes=21,
            pretrain_weights=str(
                self.parts_model_path
            ),
        )

        self.damage_model = RFDETRSegMedium(
            num_classes=4,
            pretrain_weights=str(
                self.damage_model_path
            ),
        )

        if (
            self.device == "cuda"
            and optimize_fp16
        ):

            try:
                self.parts_model.inference(
                    dtype=torch.float16
                )
            except Exception:
                pass

            try:
                self.damage_model.inference(
                    dtype=torch.float16
                )
            except Exception:
                pass


    # --------------------------------------------------------
    # MASK PREPARATION
    # --------------------------------------------------------

    @staticmethod
    def _prepare_mask(
        mask: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:

        if mask is None:
            return None

        mask = np.asarray(
            mask
        ).astype(bool)

        mask = np.squeeze(
            mask
        )

        if mask.shape != (
            height,
            width,
        ):

            mask_image = Image.fromarray(
                mask.astype(np.uint8) * 255
            )

            mask_image = mask_image.resize(
                (
                    width,
                    height,
                ),
                Image.Resampling.NEAREST,
            )

            mask = (
                np.array(mask_image)
                > 0
            )

        return mask


    # --------------------------------------------------------
    # DETECTION EXTRACTION
    # --------------------------------------------------------

    def _extract_detections(
        self,
        detections: Any,
        class_names: List[str],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:

        output = []

        if detections is None:
            return output

        masks = getattr(
            detections,
            "mask",
            None,
        )

        class_ids = getattr(
            detections,
            "class_id",
            None,
        )

        confidences = getattr(
            detections,
            "confidence",
            None,
        )

        boxes = getattr(
            detections,
            "xyxy",
            None,
        )

        if class_ids is None:
            return output

        image_area = (
            width * height
        )

        for i in range(
            len(detections)
        ):

            class_id = int(
                class_ids[i]
            )

            if not (
                0 <= class_id < len(class_names)
            ):
                continue

            confidence = (
                float(confidences[i])
                if confidences is not None
                else 0.0
            )

            mask = None

            if (
                masks is not None
                and i < len(masks)
            ):
                mask = self._prepare_mask(
                    masks[i],
                    width,
                    height,
                )

            bbox = None

            if (
                boxes is not None
                and i < len(boxes)
            ):
                bbox = [
                    float(v)
                    for v in boxes[i]
                ]

            mask_pixels = (
                int(mask.sum())
                if mask is not None
                else 0
            )

            output.append(
                {
                    "class_id":
                        class_id,

                    "class_name":
                        class_names[class_id],

                    "confidence":
                        confidence,

                    "mask":
                        mask,

                    "bbox":
                        bbox,

                    "mask_pixels":
                        mask_pixels,

                    "mask_area_ratio":
                        (
                            mask_pixels
                            / image_area
                            if image_area > 0
                            else 0.0
                        ),
                }
            )

        return output


    # --------------------------------------------------------
    # DAMAGE FILTER
    # --------------------------------------------------------

    def _filter_damage(
        self,
        raw_damages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        """
        Adaptive production damage filter.

        Strategy:
        1. High-confidence detections always survive.
        2. Lower-confidence detections may survive when several
           predictions of the SAME class provide supporting evidence.
        3. Tiny masks are rejected.
        4. Highly overlapping duplicate predictions are suppressed.

        This keeps the clean-car false-positive guard while allowing
        heavily damaged vehicles to expose more than one real issue.
        """

        # ----------------------------------------------------
        # STRICT thresholds used for isolated predictions
        # ----------------------------------------------------

        strict_thresholds = {
            "STRUCTURAL_DAMAGE": 0.50,
            "DEFORMATION": 0.40,
            "SURFACE_DAMAGE": 0.45,
            "CORROSION": 0.40,
        }

        # ----------------------------------------------------
        # RELAXED thresholds used only when repeated evidence
        # exists for the same damage class
        # ----------------------------------------------------

        relaxed_thresholds = {
            "STRUCTURAL_DAMAGE": 0.30,
            "DEFORMATION": 0.25,
            "SURFACE_DAMAGE": 0.30,
            "CORROSION": 0.28,
        }

        # Number of supporting predictions required before
        # relaxed threshold becomes active.
        evidence_required = {
            "STRUCTURAL_DAMAGE": 3,
            "DEFORMATION": 3,
            "SURFACE_DAMAGE": 3,
            "CORROSION": 2,
        }

        # ----------------------------------------------------
        # Count supporting predictions for every class
        # ----------------------------------------------------

        support_counts = {
            class_name: 0
            for class_name in DAMAGE_CLASSES
        }

        for damage in raw_damages:

            class_name = damage["class_name"]

            if (
                damage["mask_area_ratio"]
                < MIN_DAMAGE_AREA_RATIO
            ):
                continue

            if (
                damage["confidence"]
                >= relaxed_thresholds[
                    class_name
                ]
            ):
                support_counts[
                    class_name
                ] += 1

        # ----------------------------------------------------
        # Select candidates
        # ----------------------------------------------------

        candidates = []

        for damage in raw_damages:

            class_name = damage["class_name"]

            confidence = float(
                damage["confidence"]
            )

            area_ratio = float(
                damage["mask_area_ratio"]
            )

            if (
                area_ratio
                < MIN_DAMAGE_AREA_RATIO
            ):
                continue

            strict_pass = (
                confidence
                >= strict_thresholds[
                    class_name
                ]
            )

            repeated_evidence_pass = (
                support_counts[
                    class_name
                ]
                >= evidence_required[
                    class_name
                ]
                and
                confidence
                >= relaxed_thresholds[
                    class_name
                ]
            )

            if (
                strict_pass
                or
                repeated_evidence_pass
            ):

                accepted_copy = dict(
                    damage
                )

                accepted_copy[
                    "acceptance_mode"
                ] = (
                    "strict"
                    if strict_pass
                    else "multi_evidence"
                )

                accepted_copy[
                    "class_support_count"
                ] = support_counts[
                    class_name
                ]

                candidates.append(
                    accepted_copy
                )

        # ----------------------------------------------------
        # Highest confidence first
        # ----------------------------------------------------

        candidates = sorted(
            candidates,
            key=lambda item:
                item["confidence"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Duplicate suppression
        #
        # Only suppress detections when the masks overlap
        # strongly. Different damaged areas remain separate.
        # ----------------------------------------------------

        accepted = []

        for candidate in candidates:

            duplicate = False

            for existing in accepted:

                overlap = mask_iou(
                    candidate["mask"],
                    existing["mask"],
                )

                # Same physical region.
                if overlap >= 0.60:

                    duplicate = True
                    break

            if not duplicate:

                accepted.append(
                    candidate
                )

        # ----------------------------------------------------
        # Avoid pathological prediction floods
        # ----------------------------------------------------

        MAX_DAMAGE_REGIONS = 8

        accepted = accepted[
            :MAX_DAMAGE_REGIONS
        ]

        return accepted


    # --------------------------------------------------------
    # DAMAGE -> PART ASSOCIATION
    # --------------------------------------------------------

    def _associate_part(
        self,
        damage: Dict[str, Any],
        parts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        best_part = None
        best_coverage = 0.0
        best_iou = 0.0

        for part in parts:

            coverage, iou = (
                association_metrics(
                    damage["mask"],
                    part["mask"],
                )
            )

            if coverage > best_coverage:

                best_part = part
                best_coverage = coverage
                best_iou = iou

        if (
            best_part is not None
            and
            best_coverage
            >= MIN_ASSOCIATION_COVERAGE
        ):

            return {
                "part":
                    best_part,

                "method":
                    "mask_overlap",

                "coverage":
                    best_coverage,

                "iou":
                    best_iou,
            }

        return {
            "part":
                None,

            "method":
                "unassigned",

            "coverage":
                0.0,

            "iou":
                0.0,
        }


    # --------------------------------------------------------
    # ISSUE CREATION
    # --------------------------------------------------------

    def _build_issues(
        self,
        accepted_damages: List[Dict[str, Any]],
        parts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        issues = []

        for damage in accepted_damages:

            association = (
                self._associate_part(
                    damage,
                    parts,
                )
            )

            part = association["part"]

            if part is None:

                part_name = "unassigned"
                part_confidence = 0.0

                combined_confidence = (
                    damage["confidence"]
                )

            else:

                part_name = normalize_part_name(
                    part["class_name"]
                )

                part_confidence = float(
                    part["confidence"]
                )

                combined_confidence = math.sqrt(
                    damage["confidence"]
                    *
                    part_confidence
                )

            issues.append(
                {
                    "part":
                        part_name,

                    "condition":
                        CONDITION_MAP[
                            damage["class_name"]
                        ],

                    "severity":
                        SEVERITY_MAP[
                            damage["class_name"]
                        ],

                    "confidence":
                        round(
                            float(
                                combined_confidence
                            ),
                            4,
                        ),

                    "damage_confidence":
                        round(
                            float(
                                damage["confidence"]
                            ),
                            4,
                        ),

                    "part_confidence":
                        round(
                            part_confidence,
                            4,
                        ),

                    "damage_class":
                        damage["class_name"],

                    "damage_mask_area_ratio":
                        round(
                            float(
                                damage[
                                    "mask_area_ratio"
                                ]
                            ),
                            6,
                        ),

                    "association_method":
                        association["method"],

                    "damage_mask_overlap_with_part":
                        round(
                            float(
                                association[
                                    "coverage"
                                ]
                            ),
                            4,
                        ),

                    "mask_iou":
                        round(
                            float(
                                association["iou"]
                            ),
                            4,
                        ),
                }
            )

        issues.sort(
            key=lambda item:
                item["confidence"],
            reverse=True,
        )

        return issues


    # --------------------------------------------------------
    # OVERALL CONDITION
    # --------------------------------------------------------

    @staticmethod
    def _overall_condition(
        issues: List[Dict[str, Any]],
    ) -> Tuple[str, str]:

        if not issues:
            return (
                "PASS",
                "no_visible_damage_detected",
            )

        severity_rank = {
            "minor": 1,
            "moderate": 2,
            "severe": 3,
        }

        highest = max(
            severity_rank[
                issue["severity"]
            ]
            for issue in issues
        )

        if highest >= 3:
            return (
                "FAIL",
                "requires_repair",
            )

        return (
            "ATTENTION",
            "repair_recommended",
        )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    def _build_overlay(
        self,
        image: Image.Image,
        parts: List[Dict[str, Any]],
        accepted_damages: List[Dict[str, Any]],
    ) -> Image.Image:

        """
        Clean human-readable inspection overlay.

        - Shows accepted damage only.
        - Does not paint all vehicle-part masks.
        - Uses lighter transparent damage highlighting.
        - Draws one label per affected vehicle part.
        """

        width, height = image.size

        overlay = image.copy()

        # ====================================================
        # DAMAGE MASKS ONLY
        # ====================================================

        for damage in accepted_damages:

            overlay = apply_mask(
                overlay,
                damage.get(
                    "mask"
                ),
                DAMAGE_COLORS[
                    damage[
                        "class_name"
                    ]
                ],
                0.25,
            )

        # ====================================================
        # DAMAGE BOUNDARIES
        # ====================================================

        overlay_array = np.array(
            overlay
        )

        for damage in accepted_damages:

            mask = damage.get(
                "mask"
            )

            if mask is None:
                continue

            mask = mask.astype(
                bool
            )

            up = np.roll(
                mask,
                1,
                axis=0,
            )

            down = np.roll(
                mask,
                -1,
                axis=0,
            )

            left = np.roll(
                mask,
                1,
                axis=1,
            )

            right = np.roll(
                mask,
                -1,
                axis=1,
            )

            interior = (
                mask
                &
                up
                &
                down
                &
                left
                &
                right
            )

            boundary = (
                mask
                &
                ~interior
            )

            color = DAMAGE_COLORS[
                damage[
                    "class_name"
                ]
            ]

            overlay_array[
                boundary
            ] = np.array(
                color,
                dtype=np.uint8,
            )

        overlay = Image.fromarray(
            overlay_array
        )

        draw = ImageDraw.Draw(
            overlay
        )

        # ====================================================
        # FONT
        # ====================================================

        try:

            font = ImageFont.truetype(
                "arialbd.ttf",
                max(
                    15,
                    int(
                        width * 0.013
                    ),
                ),
            )

        except Exception:

            try:

                font = ImageFont.truetype(
                    "arial.ttf",
                    max(
                        15,
                        int(
                            width * 0.013
                        ),
                    ),
                )

            except Exception:

                font = ImageFont.load_default()

        # ====================================================
        # HUMAN DAMAGE NAMES
        # ====================================================

        damage_names = {

            "STRUCTURAL_DAMAGE":
                "Structural damage",

            "DEFORMATION":
                "Dent / deformation",

            "SURFACE_DAMAGE":
                "Surface damage",

            "CORROSION":
                "Corrosion",
        }

        # ====================================================
        # ONE LABEL PER PART
        # ====================================================

        best_for_part = {}

        for damage in accepted_damages:

            association = (
                self._associate_part(
                    damage,
                    parts,
                )
            )

            part = association.get(
                "part"
            )

            if part is None:

                part_key = (
                    "visible_damage"
                )

                part_label = (
                    "Visible damage"
                )

            else:

                part_key = normalize_part_name(
                    part[
                        "class_name"
                    ]
                )

                part_label = (
                    part[
                        "class_name"
                    ]
                    .replace(
                        "-",
                        " "
                    )
                    .title()
                )

            existing = best_for_part.get(
                part_key
            )

            if (
                existing is None
                or
                damage[
                    "confidence"
                ]
                >
                existing[
                    "damage"
                ][
                    "confidence"
                ]
            ):

                best_for_part[
                    part_key
                ] = {

                    "part_label":
                        part_label,

                    "damage":
                        damage,
                }

        # ====================================================
        # LABEL DRAWING
        # ====================================================

        occupied_y = []

        for item in best_for_part.values():

            damage = item[
                "damage"
            ]

            bbox = damage.get(
                "bbox"
            )

            if bbox is None:
                continue

            x1, y1, _, _ = bbox

            label = (
                item[
                    "part_label"
                ]
                +
                " • "
                +
                damage_names.get(
                    damage[
                        "class_name"
                    ],
                    damage[
                        "class_name"
                    ],
                )
            )

            x = max(
                10,
                int(
                    x1
                ),
            )

            y = max(
                10,
                int(
                    y1
                )
                - 30,
            )

            # Prevent label pile-up.
            while any(
                abs(
                    y
                    -
                    previous_y
                )
                <
                30

                for previous_y
                in occupied_y
            ):

                y += 32

            y = min(
                y,
                max(
                    10,
                    height - 35,
                ),
            )

            occupied_y.append(
                y
            )

            text_box = draw.textbbox(
                (
                    x,
                    y,
                ),
                label,
                font=font,
            )

            draw.rounded_rectangle(
                (
                    text_box[0] - 7,
                    text_box[1] - 5,
                    text_box[2] + 7,
                    text_box[3] + 5,
                ),
                radius=6,
                fill=(
                    15,
                    23,
                    42,
                ),
            )

            draw.text(
                (
                    x,
                    y,
                ),
                label,
                font=font,
                fill=(
                    255,
                    255,
                    255,
                ),
            )

        # ====================================================
        # CLEAN CAR MESSAGE
        # ====================================================

        if not accepted_damages:

            message = (
                "No visible damage detected"
            )

            box = draw.textbbox(
                (
                    20,
                    20,
                ),
                message,
                font=font,
            )

            draw.rounded_rectangle(
                (
                    box[0] - 8,
                    box[1] - 6,
                    box[2] + 8,
                    box[3] + 6,
                ),
                radius=7,
                fill=(
                    22,
                    101,
                    52,
                ),
            )

            draw.text(
                (
                    20,
                    20,
                ),
                message,
                font=font,
                fill=(
                    255,
                    255,
                    255,
                ),
            )

        return overlay


    # --------------------------------------------------------
    # PUBLIC INFERENCE METHOD
    # --------------------------------------------------------

    def inspect(
        self,
        image_input: Any,
        image_name: str = "vehicle_image",
    ) -> Dict[str, Any]:

        image = load_input_image(
            image_input
        )

        width, height = image.size

        # ----------------------------------------
        # Parts inference
        # ----------------------------------------

        with torch.no_grad():

            part_predictions = (
                self.parts_model.predict(
                    image,
                    threshold=PART_THRESHOLD,
                )
            )

        # ----------------------------------------
        # Damage inference
        # ----------------------------------------

        with torch.no_grad():

            damage_predictions = (
                self.damage_model.predict(
                    image,
                    threshold=RAW_DAMAGE_THRESHOLD,
                )
            )

        # ----------------------------------------
        # Extract predictions
        # ----------------------------------------

        parts = self._extract_detections(
            part_predictions,
            PART_CLASSES,
            width,
            height,
        )

        raw_damages = (
            self._extract_detections(
                damage_predictions,
                DAMAGE_CLASSES,
                width,
                height,
            )
        )

        accepted_damages = (
            self._filter_damage(
                raw_damages
            )
        )

        issues = self._build_issues(
            accepted_damages,
            parts,
        )

        (
            inspection_status,
            overall_condition,
        ) = self._overall_condition(
            issues
        )

        overlay = self._build_overlay(
            image,
            parts,
            accepted_damages,
        )

        result = {
            "vehicle_type":
                "car",

            "image":
                image_name,

            "inspection_status":
                inspection_status,

            "overall_condition":
                overall_condition,

            "parts_detected":
                len(parts),

            "raw_damage_predictions":
                len(raw_damages),

            "accepted_damage_predictions":
                len(
                    accepted_damages
                ),

            "issues":
                issues,

            "models": {
                "parts": {
                    "architecture":
                        "RFDETRSegMedium",

                    "version":
                        "parts_rfdetr_seg_medium_v1",

                    "threshold":
                        PART_THRESHOLD,
                },

                "damage": {
                    "architecture":
                        "RFDETRSegMedium",

                    "version":
                        "damage_rfdetr_seg_medium_v3_grouped",

                    "raw_threshold":
                        RAW_DAMAGE_THRESHOLD,

                    "class_acceptance_thresholds":
                        DAMAGE_ACCEPTANCE_THRESHOLDS,
                },
            },

            "visible_damage_only":
                True,

            "decision_policy":
                (
                    "conservative_"
                    "false_positive_reduction"
                ),

            "note":
                (
                    "Assessment is based on visible "
                    "exterior image evidence only. "
                    "It does not diagnose hidden "
                    "mechanical or internal damage."
                ),
        }

        return {
            "result":
                result,

            "overlay":
                overlay,

            "original_image":
                image,
        }


# ============================================================
# SIMPLE FACTORY
# ============================================================

def create_vehicle_engine(
    verify_hashes: bool = True,
    optimize_fp16: bool = True,
) -> VehicleInspectionEngine:

    return VehicleInspectionEngine(
        verify_hashes=verify_hashes,
        optimize_fp16=optimize_fp16,
    )

