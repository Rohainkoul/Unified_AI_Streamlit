from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import zipfile

from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import torch

from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

try:
    import pymupdf
except ImportError:
    pymupdf = None


# ============================================================
# INVOICE AI V3 — FINAL PRODUCTION ENGINE
# ============================================================

EXPECTED_PARAMETER_COUNT = 125_942_945

EXPECTED_FIELDS = [
    "VENDOR_NAME",
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "DUE_DATE",
    "CUSTOMER_NAME",
    "ADDRESS",
    "CURRENCY",
    "LINE_ITEM_DESC",
    "LINE_ITEM_QTY",
    "LINE_ITEM_UNIT_PRICE",
    "LINE_ITEM_AMOUNT",
    "TAX",
    "DISCOUNT",
    "SUBTOTAL",
    "TOTAL_AMOUNT",
    "PAYMENT_TERMS",
]

EXPECTED_LABELS = ["O"]

for field in EXPECTED_FIELDS:
    EXPECTED_LABELS.extend(
        [
            f"B-{field}",
            f"I-{field}",
        ]
    )

EXPECTED_NUM_LABELS = len(EXPECTED_LABELS)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CACHE_ROOT = PROJECT_ROOT / ".invoice_ai_cache"

MODEL_CACHE_ROOT = CACHE_ROOT / "model"

RUNTIME_CACHE_ROOT = CACHE_ROOT / "production_runtime"

COLAB_COMPAT_ROOT = CACHE_ROOT / "colab_content"

RUNTIME_SIGNATURE_FILE = (
    RUNTIME_CACHE_ROOT
    / ".runtime_signature.json"
)

MODEL_ZIP_NAME = (
    "Invoice_AI_V3_AllRounder_FINAL.zip"
)

RUNTIME_ZIP_NAME = (
    "Invoice_AI_V3_FINAL_Production_Runtime.zip"
)

CANONICAL_RUNTIME_LOADER = (
    "load_v3_final_runtime.py"
)


# ============================================================
# ARTIFACT DISCOVERY
# ============================================================

def _find_artifact(
    exact_name: str,
    pattern: str,
) -> Path:

    exact_path = (
        ARTIFACTS_DIR
        / exact_name
    )

    if exact_path.is_file():
        return exact_path

    candidates = sorted(
        ARTIFACTS_DIR.glob(
            pattern
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Required artifact was not found.\n\n"
            f"Expected:\n{exact_path}"
        )

    raise RuntimeError(
        "Multiple matching artifact ZIPs were found.\n\n"
        +
        "\n".join(
            str(path)
            for path
            in candidates
        )
    )


def get_model_zip() -> Path:

    return _find_artifact(
        MODEL_ZIP_NAME,
        "Invoice_AI_V3_AllRounder_FINAL*.zip",
    )


def get_runtime_zip() -> Path:

    return _find_artifact(
        RUNTIME_ZIP_NAME,
        "Invoice_AI_V3_FINAL_Production_Runtime*.zip",
    )


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def _validate_zip_members(
    archive: zipfile.ZipFile,
):

    for member in archive.infolist():

        normalized = (
            member.filename
            .replace(
                "\\",
                "/",
            )
        )

        path = PurePosixPath(
            normalized
        )

        if (
            path.is_absolute()
            or
            ".." in path.parts
        ):

            raise RuntimeError(
                "Unsafe path inside ZIP:\n"
                f"{member.filename}"
            )


def _extract_fresh(
    zip_path: Path,
    destination: Path,
):

    temporary = (
        destination.parent
        /
        (
            destination.name
            +
            ".__extracting__"
        )
    )

    if temporary.exists():

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

    temporary.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:

            _validate_zip_members(
                archive
            )

            archive.extractall(
                temporary
            )

        if destination.exists():

            shutil.rmtree(
                destination,
                ignore_errors=True,
            )

        temporary.replace(
            destination
        )

    except Exception:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

        raise


# ============================================================
# MODEL DISCOVERY
# ============================================================

def _candidate_is_model(
    candidate: Path,
) -> bool:

    if not (
        candidate
        / "config.json"
    ).is_file():

        return False

    return (
        (
            candidate
            / "model.safetensors"
        ).is_file()
        or
        (
            candidate
            / "pytorch_model.bin"
        ).is_file()
    )


def find_model_directory() -> Path:

    if not MODEL_CACHE_ROOT.exists():

        raise FileNotFoundError(
            "Model cache does not exist."
        )

    matches = []

    for config_file in (
        MODEL_CACHE_ROOT.rglob(
            "config.json"
        )
    ):

        candidate = (
            config_file.parent
        )

        if _candidate_is_model(
            candidate
        ):

            matches.append(
                candidate
            )

    if not matches:

        raise RuntimeError(
            "No model directory containing "
            "config.json + model weights "
            "was found."
        )

    matches.sort(
        key=lambda path: (
            len(
                path.relative_to(
                    MODEL_CACHE_ROOT
                ).parts
            ),
            str(path).lower(),
        )
    )

    return matches[0]


def prepare_model() -> Path:

    try:
        return find_model_directory()

    except (
        FileNotFoundError,
        RuntimeError,
    ):
        pass

    print(
        "\nExtracting Final V3 model..."
    )

    _extract_fresh(
        get_model_zip(),
        MODEL_CACHE_ROOT,
    )

    model_dir = (
        find_model_directory()
    )

    print(
        "✅ Model extraction complete"
    )

    return model_dir


# ============================================================
# FINAL V3 MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_v3_model():

    model_dir = (
        prepare_model()
    )

    print(
        "\nLoading Final V3 model..."
    )

    processor = (
        LayoutLMv3Processor
        .from_pretrained(
            str(
                model_dir
            ),
            apply_ocr=False,
            local_files_only=True,
        )
    )

    model = (
        LayoutLMv3ForTokenClassification
        .from_pretrained(
            str(
                model_dir
            ),
            local_files_only=True,
        )
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(
        device
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    if (
        model.__class__.__name__
        !=
        "LayoutLMv3ForTokenClassification"
    ):

        raise RuntimeError(
            "Wrong model architecture.\n"
            f"Found: "
            f"{model.__class__.__name__}"
        )

    if (
        parameter_count
        !=
        EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(
            "Wrong model loaded.\n\n"
            f"Expected parameters: "
            f"{EXPECTED_PARAMETER_COUNT:,}\n"
            f"Found parameters: "
            f"{parameter_count:,}"
        )

    if (
        model.config.num_labels
        !=
        EXPECTED_NUM_LABELS
    ):

        raise RuntimeError(
            "Wrong BIO schema.\n\n"
            f"Expected labels: "
            f"{EXPECTED_NUM_LABELS}\n"
            f"Found labels: "
            f"{model.config.num_labels}"
        )

    id2label = {
        int(key): value
        for key, value
        in model.config.id2label.items()
    }

    label_list = [
        id2label[index]
        for index
        in range(
            model.config.num_labels
        )
    ]

    if (
        label_list
        !=
        EXPECTED_LABELS
    ):

        raise RuntimeError(
            "Final V3 label order does not "
            "match expected schema."
        )

    label2id = {
        label: index
        for index, label
        in enumerate(
            label_list
        )
    }

    print(
        "✅ Final V3 model loaded"
    )

    return {
        "model":
            model,

        "processor":
            processor,

        "device":
            device,

        "model_dir":
            model_dir,

        "parameter_count":
            parameter_count,

        "fields":
            list(
                EXPECTED_FIELDS
            ),

        "label_list":
            label_list,

        "id2label":
            id2label,

        "label2id":
            label2id,
    }


# ============================================================
# RUNTIME SIGNATURE
# ============================================================

def _sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024
                *
                1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _runtime_signature():

    runtime_zip = (
        get_runtime_zip()
    )

    stat = runtime_zip.stat()

    return {
        "filename":
            runtime_zip.name,

        "size":
            stat.st_size,

        "sha256":
            _sha256(
                runtime_zip
            ),
    }


def _read_runtime_signature():

    if not (
        RUNTIME_SIGNATURE_FILE.exists()
    ):

        return None

    try:

        return json.loads(
            RUNTIME_SIGNATURE_FILE
            .read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return None


def _write_runtime_signature(
    signature,
):

    RUNTIME_SIGNATURE_FILE.write_text(
        json.dumps(
            signature,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# RUNTIME EXTRACTION
# ============================================================

def find_runtime_loader() -> Path:

    if not (
        RUNTIME_CACHE_ROOT.exists()
    ):

        raise FileNotFoundError(
            "Runtime cache does not exist."
        )

    loaders = list(
        RUNTIME_CACHE_ROOT.rglob(
            CANONICAL_RUNTIME_LOADER
        )
    )

    if len(loaders) != 1:

        raise RuntimeError(
            "Expected exactly one "
            f"{CANONICAL_RUNTIME_LOADER}, "
            f"found {len(loaders)}."
        )

    return loaders[0]


def prepare_runtime(
    force_refresh=False,
) -> Path:

    signature = (
        _runtime_signature()
    )

    cache_valid = False

    if (
        RUNTIME_CACHE_ROOT.exists()
        and
        not force_refresh
    ):

        try:

            loader = (
                find_runtime_loader()
            )

            cache_valid = (
                loader.is_file()
                and
                _read_runtime_signature()
                ==
                signature
            )

        except Exception:

            cache_valid = False

    if not cache_valid:

        print(
            "\nExtracting clean "
            "production runtime..."
        )

        _extract_fresh(
            get_runtime_zip(),
            RUNTIME_CACHE_ROOT,
        )

        _write_runtime_signature(
            signature
        )

        print(
            "✅ Runtime extraction complete"
        )

    return find_runtime_loader()


# ============================================================
# COLAB PATH COMPATIBILITY
# ============================================================

def _patch_colab_runtime_paths():

    COLAB_COMPAT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    replacement = (
        COLAB_COMPAT_ROOT
        .resolve()
        .as_posix()
    )

    patchable_extensions = {
        ".py",
        ".ipynb",
        ".json",
    }

    patched_files = []

    for runtime_file in (
        RUNTIME_CACHE_ROOT.rglob(
            "*"
        )
    ):

        if not runtime_file.is_file():
            continue

        if (
            runtime_file.suffix.lower()
            not in patchable_extensions
        ):

            continue

        try:

            source = (
                runtime_file.read_text(
                    encoding="utf-8"
                )
            )

        except (
            UnicodeDecodeError,
            OSError,
        ):

            continue

        patched = source

        patched = patched.replace(
            '"/content',
            f'"{replacement}',
        )

        patched = patched.replace(
            "'/content",
            f"'{replacement}",
        )

        patched = patched.replace(
            '\\"/content',
            f'\\"{replacement}',
        )

        patched = patched.replace(
            "\\'/content",
            f"\\'{replacement}",
        )

        if patched != source:

            runtime_file.write_text(
                patched,
                encoding="utf-8",
            )

            patched_files.append(
                runtime_file.name
            )

    print(
        "✅ Colab filesystem compatibility ready"
    )

    print(
        "Patched runtime files:",
        len(
            patched_files
        ),
    )

    for filename in patched_files:

        print(
            " •",
            filename,
        )


def _redirect_colab_path(
    value,
) -> Path:

    normalized = (
        str(
            value
        )
        .replace(
            "\\",
            "/",
        )
    )

    if (
        normalized
        ==
        "/content"
        or
        normalized.startswith(
            "/content/"
        )
    ):

        relative = (
            normalized[
                len(
                    "/content"
                ):
            ]
            .lstrip(
                "/"
            )
        )

        return (
            COLAB_COMPAT_ROOT
            / relative
        )

    return Path(
        value
    )


# ============================================================
# RUNTIME SOURCE RESOLUTION
# ============================================================

def _suffix_match_score(
    requested: Path,
    candidate: Path,
):

    requested_parts = [
        part.lower()
        for part
        in requested.parts
    ]

    candidate_parts = [
        part.lower()
        for part
        in candidate.parts
    ]

    score = 0

    for left, right in zip(
        reversed(
            requested_parts
        ),
        reversed(
            candidate_parts
        ),
    ):

        if left != right:
            break

        score += 1

    return score


def _resolve_runtime_source(
    source,
    runtime_loader: Path,
) -> Path:

    redirected = (
        _redirect_colab_path(
            source
        )
    )

    if redirected.exists():
        return redirected

    original = Path(
        source
    )

    if not original.is_absolute():

        possibilities = [
            runtime_loader.parent
            /
            original,

            RUNTIME_CACHE_ROOT
            /
            original,

            PROJECT_ROOT
            /
            original,

            COLAB_COMPAT_ROOT
            /
            original,
        ]

        for candidate in possibilities:

            if candidate.exists():
                return candidate

    matches = [
        candidate
        for candidate
        in RUNTIME_CACHE_ROOT.rglob(
            original.name
        )
        if candidate.is_file()
    ]

    if not matches:

        raise FileNotFoundError(
            "Production runtime requested "
            "a source file that cannot be found.\n\n"
            f"Requested:\n{source}\n\n"
            f"Runtime root:\n"
            f"{RUNTIME_CACHE_ROOT}"
        )

    matches.sort(
        key=lambda candidate: (
            -_suffix_match_score(
                original,
                candidate,
            ),
            len(
                candidate.parts
            ),
            str(
                candidate
            ).lower(),
        )
    )

    if len(matches) > 1:

        first_score = (
            _suffix_match_score(
                original,
                matches[0],
            )
        )

        second_score = (
            _suffix_match_score(
                original,
                matches[1],
            )
        )

        if first_score == second_score:

            raise RuntimeError(
                "Ambiguous runtime source path.\n\n"
                f"Requested:\n{source}\n\n"
                "Matches:\n"
                +
                "\n".join(
                    str(path)
                    for path
                    in matches
                )
            )

    return matches[0]


# ============================================================
# SAFE SHUTIL PATCHES
# ============================================================

def _build_safe_copy2(
    runtime_loader,
    original_copy2,
):

    def safe_copy2(
        source,
        destination,
        *,
        follow_symlinks=True,
    ):

        source_path = (
            _resolve_runtime_source(
                source,
                runtime_loader,
            )
        )

        destination_path = (
            _redirect_colab_path(
                destination
            )
        )

        if (
            destination_path.exists()
            and
            destination_path.is_dir()
        ):

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        return original_copy2(
            str(
                source_path
            ),
            str(
                destination_path
            ),
            follow_symlinks=
                follow_symlinks,
        )

    return safe_copy2


def _build_safe_copy(
    runtime_loader,
    original_copy,
):

    def safe_copy(
        source,
        destination,
        *,
        follow_symlinks=True,
    ):

        source_path = (
            _resolve_runtime_source(
                source,
                runtime_loader,
            )
        )

        destination_path = (
            _redirect_colab_path(
                destination
            )
        )

        if (
            destination_path.exists()
            and
            destination_path.is_dir()
        ):

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        return original_copy(
            str(
                source_path
            ),
            str(
                destination_path
            ),
            follow_symlinks=
                follow_symlinks,
        )

    return safe_copy


def _build_safe_copyfile(
    runtime_loader,
    original_copyfile,
):

    def safe_copyfile(
        source,
        destination,
        *,
        follow_symlinks=True,
    ):

        source_path = (
            _resolve_runtime_source(
                source,
                runtime_loader,
            )
        )

        destination_path = (
            _redirect_colab_path(
                destination
            )
        )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return original_copyfile(
            str(
                source_path
            ),
            str(
                destination_path
            ),
            follow_symlinks=
                follow_symlinks,
        )

    return safe_copyfile


# ============================================================
# COMPLETE V3/V6.1 PRODUCTION ENGINE
# ============================================================

@lru_cache(maxsize=2)
def load_v3_production_engine(
    force_runtime_refresh=False,
):

    engine = (
        load_v3_model()
    )

    runtime_loader = (
        prepare_runtime(
            force_refresh=
                force_runtime_refresh
        )
    )

    _patch_colab_runtime_paths()

    print(
        "\nRuntime loader:"
    )

    print(
        runtime_loader
    )

    runtime_namespace = {
        "__name__":
            "invoice_ai_v3_production_runtime",

        "__file__":
            str(
                runtime_loader
            ),

        "__builtins__":
            __builtins__,

        "model":
            engine[
                "model"
            ],

        "processor":
            engine[
                "processor"
            ],

        "device":
            engine[
                "device"
            ],

        "id2label":
            engine[
                "id2label"
            ],

        "label2id":
            engine[
                "label2id"
            ],

        "MODEL":
            engine[
                "model"
            ],

        "PROCESSOR":
            engine[
                "processor"
            ],

        "DEVICE":
            engine[
                "device"
            ],

        "TARGET_FIELDS":
            list(
                EXPECTED_FIELDS
            ),

        "EXPECTED_FIELDS":
            list(
                EXPECTED_FIELDS
            ),

        "MODEL_ID2LABEL":
            engine[
                "id2label"
            ],

        "LABEL_LIST":
            engine[
                "label_list"
            ],

        "PARAMETERS":
            engine[
                "parameter_count"
            ],
    }

    runtime_source = (
        runtime_loader.read_text(
            encoding="utf-8"
        )
    )

    runtime_directory = str(
        runtime_loader.parent
    )

    inserted_runtime_path = False

    if runtime_directory not in sys.path:

        sys.path.insert(
            0,
            runtime_directory,
        )

        inserted_runtime_path = True

    original_copy2 = shutil.copy2
    original_copy = shutil.copy
    original_copyfile = shutil.copyfile

    shutil.copy2 = (
        _build_safe_copy2(
            runtime_loader,
            original_copy2,
        )
    )

    shutil.copy = (
        _build_safe_copy(
            runtime_loader,
            original_copy,
        )
    )

    shutil.copyfile = (
        _build_safe_copyfile(
            runtime_loader,
            original_copyfile,
        )
    )

    print(
        "\nLoading production runtime..."
    )

    try:

        exec(
            compile(
                runtime_source,
                str(
                    runtime_loader
                ),
                "exec",
            ),
            runtime_namespace,
            runtime_namespace,
        )

    finally:

        shutil.copy2 = original_copy2
        shutil.copy = original_copy
        shutil.copyfile = original_copyfile

        if inserted_runtime_path:

            try:

                sys.path.remove(
                    runtime_directory
                )

            except ValueError:

                pass

    process_invoice_final = (
        runtime_namespace.get(
            "process_invoice_final"
        )
    )

    if not callable(
        process_invoice_final
    ):

        raise RuntimeError(
            "Production runtime executed, "
            "but process_invoice_final() "
            "was not created."
        )

    print(
        "✅ Production runtime loaded"
    )

    return {
        **engine,

        "runtime_loader":
            runtime_loader,

        "runtime_namespace":
            runtime_namespace,

        "process_invoice_final":
            process_invoice_final,

        "runtime_ready":
            True,

        "colab_compat_root":
            COLAB_COMPAT_ROOT,
    }


# ============================================================
# DYNAMIC PARAMETER EXTRACTION
# ============================================================

DYNAMIC_NOT_DETECTED = (
    "NOT_DETECTED"
)

DYNAMIC_MIN_CONFIDENCE = (
    0.60
)

DYNAMIC_MAX_FIELDS = (
    50
)

DYNAMIC_MAX_VALUE_LENGTH = (
    220
)


_DYNAMIC_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "from",
    "field",
    "value",
    "details",
    "detail",
}


_DYNAMIC_GENERIC_TOKENS = {
    "number",
    "no",
    "num",
    "nr",
    "id",
    "identifier",
    "code",
    "reference",
    "ref",
    "value",
    "detail",
    "details",
}


_DYNAMIC_EQUIVALENTS = {

    "number": {
        "number",
        "no",
        "num",
        "nr",
    },

    "no": {
        "no",
        "number",
        "num",
        "nr",
    },

    "reference": {
        "reference",
        "ref",
    },

    "ref": {
        "ref",
        "reference",
    },

    "identifier": {
        "identifier",
        "id",
    },

    "id": {
        "id",
        "identifier",
    },

    "purchase": {
        "purchase",
        "po",
    },

    "order": {
        "order",
        "po",
    },

    "account": {
        "account",
        "acct",
        "ac",
    },

    "quantity": {
        "quantity",
        "qty",
    },

    "amount": {
        "amount",
        "amt",
        "value",
    },

    "phone": {
        "phone",
        "mobile",
        "telephone",
        "tel",
    },

    "date": {
        "date",
        "dated",
    },

    "hsn": {
        "hsn",
        "sac",
    },
}


_DYNAMIC_SPECIAL_VARIANTS = {

    "gstin": {
        "gstin",
        "gst no",
        "gst number",
        "gst registration number",
        "gst registration no",
    },

    "ifsc": {
        "ifsc",
        "ifsc code",
    },

    "ifsc code": {
        "ifsc",
        "ifsc code",
    },

    "pan": {
        "pan",
        "pan no",
        "pan number",
    },

    "hsn code": {
        "hsn code",
        "hsn",
        "hsn sac",
        "hsn / sac",
        "sac code",
    },

    "purchase order": {
        "purchase order",
        "po",
        "p o",
    },

    "purchase order number": {
        "purchase order number",
        "purchase order no",
        "po number",
        "po no",
        "p o number",
        "p o no",
    },

    "po number": {
        "po number",
        "po no",
        "p o number",
        "p o no",
        "purchase order number",
        "purchase order no",
    },

    "gr number": {
        "gr number",
        "gr no",
        "goods receipt number",
        "goods receipt no",
    },

    "bank account number": {
        "bank account number",
        "bank account no",
        "account number",
        "account no",
        "acct no",
        "ac no",
    },

    "vehicle number": {
        "vehicle number",
        "vehicle no",
        "vehicle registration",
        "vehicle registration no",
        "vehicle reg no",
    },

    "delivery challan number": {
        "delivery challan number",
        "delivery challan no",
        "challan number",
        "challan no",
        "dc no",
    },

    "b value": {
        "b value",
        "bvalue",
    },
}


_DYNAMIC_REQUIRED_SEMANTIC_GROUPS = {

    "gstin": (
        {
            "gstin",
            "gst",
        },
    ),

    "ifsc": (
        {
            "ifsc",
        },
    ),

    "ifsc code": (
        {
            "ifsc",
        },
    ),

    "pan": (
        {
            "pan",
        },
    ),

    "hsn code": (
        {
            "hsn",
            "sac",
        },
    ),

    "purchase order": (
        {
            "purchase",
            "po",
        },
        {
            "order",
            "po",
        },
    ),

    "purchase order number": (
        {
            "purchase",
            "po",
        },
        {
            "order",
            "po",
        },
    ),

    "po number": (
        {
            "po",
            "purchase",
        },
    ),

    "gr number": (
        {
            "gr",
            "goods",
        },
    ),

    "bank account number": (
        {
            "bank",
            "account",
            "acct",
            "ac",
        },
    ),

    "vehicle number": (
        {
            "vehicle",
            "registration",
            "reg",
        },
    ),

    "delivery challan number": (
        {
            "challan",
            "dc",
        },
    ),

    "b value": (
        {
            "b",
            "bvalue",
        },
    ),
}


# ============================================================
# PATTERNS
# ============================================================

_DYNAMIC_GSTIN_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"[0-9]{2}"
    r"[A-Z]{5}"
    r"[0-9]{4}"
    r"[A-Z]"
    r"[1-9A-Z]"
    r"Z"
    r"[0-9A-Z]"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)


_DYNAMIC_IFSC_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"[A-Z]{4}"
    r"0"
    r"[A-Z0-9]{6}"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)


_DYNAMIC_PAN_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"[A-Z]{5}"
    r"[0-9]{4}"
    r"[A-Z]"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)


_DYNAMIC_HSN_PATTERN = re.compile(
    r"(?<!\d)"
    r"\d{4,8}"
    r"(?!\d)"
)


_DYNAMIC_EMAIL_PATTERN = re.compile(
    r"\b"
    r"[A-Z0-9._%+\-]+"
    r"@"
    r"[A-Z0-9.\-]+"
    r"\."
    r"[A-Z]{2,}"
    r"\b",
    re.IGNORECASE,
)


_DYNAMIC_DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}"
    r"|"
    r"\d{4}[./\-]\d{1,2}[./\-]\d{1,2}"
    r")\b"
)


_DYNAMIC_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d{1,3}[\s\-]?)?"
    r"(?:\d[\s\-]?){8,12}"
    r"(?!\d)"
)


_DYNAMIC_AMOUNT_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)?"
    r"\s*"
    r"[-+]?"
    r"\d[\d,]*"
    r"(?:\.\d{1,4})?"
    r"(?!\w)",
    re.IGNORECASE,
)


_DYNAMIC_PERCENT_PATTERN = re.compile(
    r"(?<!\w)"
    r"[-+]?"
    r"\d+(?:\.\d+)?"
    r"\s*%"
    r"(?!\w)"
)


_DYNAMIC_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z0-9]"
    r"[A-Za-z0-9./_\-]{1,39}"
    r"(?![A-Za-z0-9])"
)


# ============================================================
# ACRONYM NORMALIZATION
# ============================================================

def _collapse_dotted_acronyms(
    text: str,
) -> str:

    pattern = re.compile(
        r"(?i)"
        r"\b"
        r"(?:"
        r"[a-z]"
        r"\s*\.\s*"
        r"){1,5}"
        r"[a-z]"
        r"(?:\s*\.)?"
        r"(?![a-z])"
    )

    def replace(
        match,
    ):

        return re.sub(
            r"[^A-Za-z]",
            "",
            match.group(
                0
            ),
        )

    return pattern.sub(
        replace,
        text,
    )


def _collapse_common_spaced_acronyms(
    text: str,
) -> str:

    replacements = [

        (
            r"(?i)\bP\s+O\b",
            "PO",
        ),

        (
            r"(?i)\bG\s+R\b",
            "GR",
        ),

        (
            r"(?i)\bI\s+F\s+S\s+C\b",
            "IFSC",
        ),

        (
            r"(?i)\bG\s+S\s+T\s+I\s+N\b",
            "GSTIN",
        ),

        (
            r"(?i)\bG\s+S\s+T\b",
            "GST",
        ),

        (
            r"(?i)\bD\s+C\b",
            "DC",
        ),

        (
            r"(?i)\bA\s+C\b",
            "AC",
        ),
    ]

    for (
        pattern,
        replacement,
    ) in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
        )

    return text


def _dynamic_pre_normalize(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(
            value
        ),
    )

    text = (
        _collapse_dotted_acronyms(
            text
        )
    )

    text = (
        _collapse_common_spaced_acronyms(
            text
        )
    )

    return text


# ============================================================
# NORMALIZATION
# ============================================================

def _dynamic_normalize_text(
    value: Any,
) -> str:

    text = (
        _dynamic_pre_normalize(
            value
        )
    )

    if not text:
        return ""

    text = text.lower()

    text = text.replace(
        "&",
        " and ",
    )

    text = text.replace(
        "_",
        " ",
    )

    text = text.replace(
        "-",
        " ",
    )

    text = re.sub(
        r"[^a-z0-9+#/ ]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _dynamic_clean_value(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(
            value
        ),
    )

    text = text.strip()

    text = re.sub(
        r"^[\s:;=|\-–—#]+",
        "",
        text,
    )

    text = re.sub(
        r"[\s:;=|\-–—]+$",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _dynamic_query_tokens(
    field_name: str,
):

    normalized = (
        _dynamic_normalize_text(
            field_name
        )
    )

    return [
        token
        for token
        in normalized.split()
        if token
        not in
        _DYNAMIC_STOPWORDS
    ]


# ============================================================
# FIELD VARIANTS
# ============================================================

def _dynamic_field_variants(
    field_name: str,
):

    normalized = (
        _dynamic_normalize_text(
            field_name
        )
    )

    if not normalized:
        return []

    variants = {
        normalized
    }

    for (
        special_name,
        special_variants,
    ) in (
        _DYNAMIC_SPECIAL_VARIANTS.items()
    ):

        normalized_special = (
            _dynamic_normalize_text(
                special_name
            )
        )

        normalized_aliases = {
            _dynamic_normalize_text(
                item
            )
            for item
            in special_variants
        }

        if (
            normalized
            ==
            normalized_special
            or
            normalized
            in
            normalized_aliases
            or
            normalized_special
            in
            normalized
        ):

            variants.update(
                normalized_aliases
            )

    tokens = (
        normalized.split()
    )

    for index, token in enumerate(
        tokens
    ):

        alternatives = (
            _DYNAMIC_EQUIVALENTS.get(
                token,
                set(),
            )
        )

        for alternative in alternatives:

            changed = list(
                tokens
            )

            changed[
                index
            ] = alternative

            variants.add(
                " ".join(
                    changed
                )
            )

    return sorted(
        {
            _dynamic_normalize_text(
                item
            )
            for item
            in variants
            if item
        },
        key=len,
        reverse=True,
    )


# ============================================================
# FIELD TYPE
# ============================================================

def _dynamic_parameter_type(
    field_name: str,
):

    name = (
        _dynamic_normalize_text(
            field_name
        )
    )

    tokens = set(
        name.split()
    )

    if (
        "gstin"
        in tokens
        or
        name
        in {
            "gst no",
            "gst number",
        }
    ):

        return "gstin"

    if "ifsc" in tokens:
        return "ifsc"

    if (
        "pan"
        in tokens
        and
        len(tokens)
        <=
        4
    ):

        return "pan"

    if (
        "hsn"
        in tokens
        or
        (
            "sac"
            in tokens
            and
            "code"
            in tokens
        )
    ):

        return "hsn"

    if "email" in tokens:
        return "email"

    if (
        "date"
        in tokens
        or
        "dated"
        in tokens
    ):

        return "date"

    if tokens.intersection(
        {
            "phone",
            "mobile",
            "telephone",
            "tel",
        }
    ):

        return "phone"

    if tokens.intersection(
        {
            "amount",
            "total",
            "value",
            "balance",
            "subtotal",
        }
    ):

        return "amount"

    if tokens.intersection(
        {
            "percentage",
            "percent",
        }
    ):

        return "percent"

    if tokens.intersection(
        {
            "quantity",
            "qty",
        }
    ):

        return "quantity"

    if tokens.intersection(
        {
            "number",
            "no",
            "num",
            "id",
            "identifier",
            "reference",
            "ref",
            "code",
        }
    ):

        return "identifier"

    return "generic"


def _dynamic_pattern_for_type(
    field_type: str,
):

    return {
        "gstin":
            _DYNAMIC_GSTIN_PATTERN,

        "ifsc":
            _DYNAMIC_IFSC_PATTERN,

        "pan":
            _DYNAMIC_PAN_PATTERN,

        "hsn":
            _DYNAMIC_HSN_PATTERN,

        "email":
            _DYNAMIC_EMAIL_PATTERN,

        "date":
            _DYNAMIC_DATE_PATTERN,

        "phone":
            _DYNAMIC_PHONE_PATTERN,

        "amount":
            _DYNAMIC_AMOUNT_PATTERN,

        "percent":
            _DYNAMIC_PERCENT_PATTERN,

    }.get(
        field_type
    )


# ============================================================
# SEMANTIC ANCHOR PROTECTION
# ============================================================

def _dynamic_required_semantic_groups(
    field_name: str,
):

    normalized = (
        _dynamic_normalize_text(
            field_name
        )
    )

    for (
        key,
        groups,
    ) in (
        _DYNAMIC_REQUIRED_SEMANTIC_GROUPS.items()
    ):

        normalized_key = (
            _dynamic_normalize_text(
                key
            )
        )

        if (
            normalized
            ==
            normalized_key
            or
            normalized_key
            in
            normalized
        ):

            return groups

    semantic_tokens = [
        token
        for token
        in
        _dynamic_query_tokens(
            field_name
        )
        if token
        not in
        _DYNAMIC_GENERIC_TOKENS
    ]

    if not semantic_tokens:
        return tuple()

    groups = []

    for token in semantic_tokens:

        equivalents = set(
            _DYNAMIC_EQUIVALENTS.get(
                token,
                set(),
            )
        )

        equivalents.add(
            token
        )

        groups.append(
            equivalents
        )

    return tuple(
        groups
    )


def _dynamic_semantic_anchor_ok(
    field_name: str,
    line_text: str,
) -> bool:

    groups = (
        _dynamic_required_semantic_groups(
            field_name
        )
    )

    if not groups:
        return True

    line_tokens = set(
        _dynamic_normalize_text(
            line_text
        ).split()
    )

    if not line_tokens:
        return False

    for group in groups:

        normalized_group = {
            _dynamic_normalize_text(
                token
            )
            for token
            in group
        }

        if not (
            line_tokens
            .intersection(
                normalized_group
            )
        ):

            return False

    return True


# ============================================================
# VALUE BOUNDARY DETECTION
# ============================================================

def _dynamic_trim_at_next_label(
    candidate: str,
):

    text = (
        _dynamic_clean_value(
            candidate
        )
    )

    if not text:
        return ""

    patterns = [

        r"\s+(?=[A-Za-z]{1,15}\s+No\.?\s*[:=])",

        r"\s+(?=[A-Za-z]{1,15}\s+Number\s*[:=])",

        r"\s+(?=[A-Za-z]{1,15}\.?\s*Value\s*[:=])",

        r"\s+(?=[A-Za-z]{1,20}\s+Code\s*[:=])",

        r"\s+(?=[A-Za-z]{1,20}\s+Date\s*[:=])",

        r"\s+(?=[A-Za-z]{1,20}\s+Amount\s*[:=])",

        r"\s+(?=[A-Za-z]{1,20}\s+Rate\s*[:=])",

        r"\s+(?=[A-Za-z]{1,20}\s*[:=])",
    ]

    earliest = None

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            if (
                earliest is None
                or
                match.start()
                <
                earliest
            ):

                earliest = (
                    match.start()
                )

    if earliest is not None:

        text = text[
            :
            earliest
        ]

    return (
        _dynamic_clean_value(
            text
        )
    )


def _dynamic_extract_identifier(
    candidate: str,
):

    candidate = (
        _dynamic_trim_at_next_label(
            candidate
        )
    )

    if not candidate:
        return ""

    matches = list(
        _DYNAMIC_IDENTIFIER_PATTERN
        .finditer(
            candidate
        )
    )

    for match in matches:

        token = (
            match.group(
                0
            )
            .strip(
                ".,;:"
            )
        )

        if not token:
            continue

        if not any(
            character.isdigit()
            for character
            in token
        ):

            continue

        if len(token) > 40:
            continue

        return token

    pieces = (
        candidate.split()
    )

    if not pieces:
        return ""

    return (
        _dynamic_clean_value(
            pieces[0]
        )
    )


def _dynamic_extract_typed_value(
    field_type: str,
    candidate: str,
):

    cleaned = (
        _dynamic_trim_at_next_label(
            candidate
        )
    )

    if not cleaned:
        return ""

    pattern = (
        _dynamic_pattern_for_type(
            field_type
        )
    )

    if pattern is not None:

        match = (
            pattern.search(
                cleaned
            )
        )

        if match:

            return (
                _dynamic_clean_value(
                    match.group(
                        0
                    )
                )
            )

    if field_type == "identifier":

        return (
            _dynamic_extract_identifier(
                cleaned
            )
        )

    return (
        cleaned[
            :
            DYNAMIC_MAX_VALUE_LENGTH
        ]
    )


def _dynamic_pattern_score(
    field_type: str,
    candidate: str,
):

    if not candidate:
        return -0.25

    pattern = (
        _dynamic_pattern_for_type(
            field_type
        )
    )

    if pattern is None:
        return 0.0

    if pattern.search(
        candidate
    ):

        if (
            field_type
            in {
                "gstin",
                "ifsc",
                "pan",
                "email",
            }
        ):

            return 0.30

        if field_type == "hsn":
            return 0.26

        return 0.18

    if (
        field_type
        in {
            "gstin",
            "ifsc",
            "pan",
            "email",
            "hsn",
        }
    ):

        return -0.30

    return -0.08


# ============================================================
# NATIVE PDF TEXT + GEOMETRY
# ============================================================

# ============================================================
# V8 STEP 3
# NATIVE PDF TEXT / GEOMETRY REQUEST CACHE
#
# Avoid reopening and reparsing the same PDF several times
# during one inference flow.
#
# This caches document geometry only.
# It does NOT cache neural predictions or final results.
# ============================================================

_DYNAMIC_PDF_LINES_CACHE = {}


def _dynamic_pdf_cache_key(
    input_path,
):

    path = Path(
        input_path
    )

    try:

        stat = path.stat()

        return (
            str(
                path.resolve()
            ),
            int(
                stat.st_size
            ),
            int(
                stat.st_mtime_ns
            ),
        )

    except Exception:

        return (
            str(path),
            None,
            None,
        )


def _dynamic_pdf_lines(
    input_path,
):

    if pymupdf is None:
        return []

    path = Path(
        input_path
    )

    if (
        not path.is_file()
        or
        path.suffix.lower()
        !=
        ".pdf"
    ):

        return []

    cache_key = (
        _dynamic_pdf_cache_key(
            path
        )
    )

    cached = (
        _DYNAMIC_PDF_LINES_CACHE.get(
            cache_key
        )
    )

    if cached is not None:

        # Return fresh dictionaries so callers cannot mutate
        # the cached source data accidentally.
        return [
            dict(item)
            for item
            in cached
        ]

    lines = []

    document = pymupdf.open(
        str(
            path
        )
    )

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

            grouped = {}

            for word in words:

                if len(word) < 8:
                    continue

                (
                    x0,
                    y0,
                    x1,
                    y1,
                    text,
                    block_number,
                    line_number,
                    word_number,
                ) = word[:8]

                text = (
                    str(
                        text
                    )
                    .strip()
                )

                if not text:
                    continue

                key = (
                    int(
                        block_number
                    ),
                    int(
                        line_number
                    ),
                )

                grouped.setdefault(
                    key,
                    [],
                ).append(
                    {
                        "x0":
                            float(
                                x0
                            ),

                        "y0":
                            float(
                                y0
                            ),

                        "x1":
                            float(
                                x1
                            ),

                        "y1":
                            float(
                                y1
                            ),

                        "text":
                            text,

                        "word_number":
                            int(
                                word_number
                            ),
                    }
                )

            page_lines = []

            for values in (
                grouped.values()
            ):

                values.sort(
                    key=lambda item: (
                        item[
                            "word_number"
                        ],
                        item[
                            "x0"
                        ],
                    )
                )

                text = " ".join(
                    item[
                        "text"
                    ]
                    for item
                    in values
                ).strip()

                if not text:
                    continue

                page_lines.append(
                    {
                        "page":
                            page_index
                            +
                            1,

                        "text":
                            text,

                        "x0":
                            min(
                                item[
                                    "x0"
                                ]
                                for item
                                in values
                            ),

                        "y0":
                            min(
                                item[
                                    "y0"
                                ]
                                for item
                                in values
                            ),

                        "x1":
                            max(
                                item[
                                    "x1"
                                ]
                                for item
                                in values
                            ),

                        "y1":
                            max(
                                item[
                                    "y1"
                                ]
                                for item
                                in values
                            ),

                        "source":
                            "NATIVE_PDF",
                    }
                )

            page_lines.sort(
                key=lambda item: (
                    round(
                        item[
                            "y0"
                        ],
                        1,
                    ),
                    item[
                        "x0"
                    ],
                )
            )

            lines.extend(
                page_lines
            )

    finally:

        document.close()

    # Keep the cache intentionally tiny. Streamlit uploads use
    # temporary paths, so old entries provide no long-term
    # benefit.
    _DYNAMIC_PDF_LINES_CACHE.clear()

    _DYNAMIC_PDF_LINES_CACHE[
        cache_key
    ] = tuple(
        dict(item)
        for item
        in lines
    )

    return [
        dict(item)
        for item
        in lines
    ]


# ============================================================
# PRODUCTION RESULT EVIDENCE
# ============================================================

def _dynamic_collect_result_text(
    payload,
):

    collected = []

    ignored_keys = {
        "confidence",
        "score",
        "source",
        "start_word_index",
        "end_word_index",
        "bbox",
        "boxes",
    }

    def walk(
        node,
    ):

        if isinstance(
            node,
            dict,
        ):

            for (
                key,
                value,
            ) in node.items():

                key_string = str(
                    key
                )

                if (
                    key_string.lower()
                    in
                    ignored_keys
                ):

                    continue

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                    ),
                ):

                    text = (
                        str(
                            value
                        )
                        .strip()
                    )

                    if (
                        text
                        and
                        text.upper()
                        !=
                        DYNAMIC_NOT_DETECTED
                    ):

                        collected.append(
                            {
                                "page":
                                    None,

                                "text":
                                    (
                                        f"{key_string}: "
                                        f"{text}"
                                    ),

                                "x0":
                                    0.0,

                                "y0":
                                    0.0,

                                "x1":
                                    0.0,

                                "y1":
                                    0.0,

                                "source":
                                    "PRODUCTION_RESULT",
                            }
                        )

                else:

                    walk(
                        value
                    )

        elif isinstance(
            node,
            list,
        ):

            for item in node:

                walk(
                    item
                )

    walk(
        payload
    )

    return collected


def _dynamic_document_lines(
    input_path,
    production_result=None,
):

    lines = (
        _dynamic_pdf_lines(
            input_path
        )
    )

    if production_result is not None:

        lines.extend(
            _dynamic_collect_result_text(
                production_result
            )
        )

    return lines


# ============================================================
# TOKEN SIMILARITY
# ============================================================

def _dynamic_token_similarity(
    query_tokens: Iterable[str],
    line_tokens: Iterable[str],
):

    query_tokens = list(
        query_tokens
    )

    line_tokens = list(
        line_tokens
    )

    if (
        not query_tokens
        or
        not line_tokens
    ):

        return 0.0

    used = set()

    scores = []

    for query_token in query_tokens:

        best = 0.0

        best_index = None

        equivalents = (
            _DYNAMIC_EQUIVALENTS.get(
                query_token,
                {
                    query_token
                },
            )
        )

        for (
            index,
            line_token,
        ) in enumerate(
            line_tokens
        ):

            if index in used:
                continue

            if (
                line_token
                in
                equivalents
                or
                query_token
                ==
                line_token
            ):

                score = 1.0

            else:

                score = (
                    difflib
                    .SequenceMatcher(
                        None,
                        query_token,
                        line_token,
                    )
                    .ratio()
                )

            if score > best:

                best = score
                best_index = index

        if (
            best_index
            is not None
            and
            best
            >=
            0.68
        ):

            used.add(
                best_index
            )

        scores.append(
            best
        )

    return (
        sum(
            scores
        )
        /
        len(
            scores
        )
    )


# ============================================================
# ANCHOR SIMILARITY
# ============================================================

def _dynamic_anchor_similarity(
    field_name,
    line_text,
):

    line_normalized = (
        _dynamic_normalize_text(
            line_text
        )
    )

    if not line_normalized:

        return (
            0.0,
            None,
        )

    if not (
        _dynamic_semantic_anchor_ok(
            field_name,
            line_text,
        )
    ):

        return (
            0.0,
            None,
        )

    line_tokens = (
        line_normalized.split()
    )

    variants = (
        _dynamic_field_variants(
            field_name
        )
    )

    best_score = 0.0
    best_variant = None

    for variant in variants:

        variant_tokens = [
            token
            for token
            in variant.split()
            if token
            not in
            _DYNAMIC_STOPWORDS
        ]

        if not variant_tokens:
            continue

        if variant in line_normalized:

            score = 1.0

        else:

            token_score = (
                _dynamic_token_similarity(
                    variant_tokens,
                    line_tokens,
                )
            )

            sequence_score = (
                difflib
                .SequenceMatcher(
                    None,
                    variant,
                    line_normalized,
                )
                .ratio()
            )

            score = max(
                token_score
                *
                0.96,
                sequence_score
                *
                0.78,
            )

        if score > best_score:

            best_score = score
            best_variant = variant

    return (
        best_score,
        best_variant,
    )


# ============================================================
# SAME-LINE VALUE
# ============================================================

def _dynamic_value_after_anchor(
    line_text,
    field_name,
):

    original = (
        _dynamic_pre_normalize(
            line_text
        )
    )

    separators = (
        ":",
        "=",
        " - ",
        " – ",
        " — ",
    )

    for separator in separators:

        if separator not in original:
            continue

        pieces = (
            original.split(
                separator
            )
        )

        if len(pieces) < 2:
            continue

        for piece_index in range(
            len(pieces)
            -
            1
        ):

            left = (
                separator.join(
                    pieces[
                        :
                        piece_index
                        +
                        1
                    ]
                )
            )

            similarity, _ = (
                _dynamic_anchor_similarity(
                    field_name,
                    left,
                )
            )

            if similarity < 0.66:
                continue

            right = (
                separator.join(
                    pieces[
                        piece_index
                        +
                        1
                        :
                    ]
                )
            )

            right = (
                _dynamic_trim_at_next_label(
                    right
                )
            )

            if right:
                return right

    raw_tokens = (
        original.split()
    )

    if not raw_tokens:
        return ""

    for start in range(
        len(
            raw_tokens
        )
    ):

        maximum_end = min(
            len(
                raw_tokens
            ),
            start
            +
            8,
        )

        for end in range(
            start
            +
            1,
            maximum_end
            +
            1,
        ):

            piece = (
                " ".join(
                    raw_tokens[
                        start:end
                    ]
                )
            )

            similarity, _ = (
                _dynamic_anchor_similarity(
                    field_name,
                    piece,
                )
            )

            if similarity >= 0.94:

                remainder = (
                    " ".join(
                        raw_tokens[
                            end:
                        ]
                    )
                )

                remainder = (
                    _dynamic_trim_at_next_label(
                        remainder
                    )
                )

                if remainder:
                    return remainder

    return ""


# ============================================================
# CANDIDATE QUALITY
# ============================================================

def _dynamic_candidate_quality(
    candidate,
    field_name,
):

    candidate = (
        _dynamic_clean_value(
            candidate
        )
    )

    if not candidate:
        return -1.0

    if (
        len(
            candidate
        )
        >
        DYNAMIC_MAX_VALUE_LENGTH
    ):

        return -0.25

    candidate_normalized = (
        _dynamic_normalize_text(
            candidate
        )
    )

    field_normalized = (
        _dynamic_normalize_text(
            field_name
        )
    )

    if (
        candidate_normalized
        ==
        field_normalized
    ):

        return -1.0

    score = 0.0

    length = len(
        candidate
    )

    if (
        2
        <=
        length
        <=
        80
    ):

        score += 0.12

    elif length <= 160:

        score += 0.05

    else:

        score -= 0.08

    candidate_tokens = set(
        candidate_normalized.split()
    )

    field_tokens = set(
        _dynamic_query_tokens(
            field_name
        )
    )

    if field_tokens:

        overlap = (
            len(
                candidate_tokens
                &
                field_tokens
            )
            /
            len(
                field_tokens
            )
        )

        if overlap >= 0.80:

            score -= 0.20

    return score


# ============================================================
# GEOMETRY BONUS
# ============================================================

def _dynamic_geometry_bonus(
    anchor,
    candidate,
):

    if (
        anchor.get(
            "page"
        )
        is None
        or
        candidate.get(
            "page"
        )
        is None
        or
        anchor.get(
            "page"
        )
        !=
        candidate.get(
            "page"
        )
    ):

        return 0.0

    ay0 = float(
        anchor.get(
            "y0",
            0.0,
        )
    )

    ay1 = float(
        anchor.get(
            "y1",
            0.0,
        )
    )

    ax0 = float(
        anchor.get(
            "x0",
            0.0,
        )
    )

    bx0 = float(
        candidate.get(
            "x0",
            0.0,
        )
    )

    by0 = float(
        candidate.get(
            "y0",
            0.0,
        )
    )

    if (
        abs(
            by0
            -
            ay0
        )
        <=
        10.0
        and
        bx0
        >=
        ax0
    ):

        return 0.10

    below_distance = (
        by0
        -
        ay1
    )

    if (
        0
        <=
        below_distance
        <=
        32.0
    ):

        return 0.07

    return 0.0


# ============================================================
# FINAL HSN RECOVERY
#
# Geometry-first.
#
# Critical rule:
# A value ABOVE "HSN Code:" is never selected.
# ============================================================

def _dynamic_hsn_candidates(
    lines,
):

    candidates = []

    for anchor in lines:

        anchor_text = str(
            anchor.get(
                "text",
                "",
            )
        )

        anchor_normalized = (
            _dynamic_normalize_text(
                anchor_text
            )
        )

        explicit_hsn = (
            "hsn code"
            in
            anchor_normalized
        )

        table_hsn = (
            "hsn"
            in
            anchor_normalized
            and
            "sac"
            in
            anchor_normalized
        )

        if not (
            explicit_hsn
            or
            table_hsn
        ):

            continue

        anchor_page = (
            anchor.get(
                "page"
            )
        )

        anchor_x0 = float(
            anchor.get(
                "x0",
                0.0,
            )
        )

        anchor_x1 = float(
            anchor.get(
                "x1",
                anchor_x0,
            )
        )

        anchor_y0 = float(
            anchor.get(
                "y0",
                0.0,
            )
        )

        anchor_y1 = float(
            anchor.get(
                "y1",
                anchor_y0,
            )
        )

        # ----------------------------------------------------
        # SAME-LINE HSN
        # ----------------------------------------------------

        same_line_match = re.search(
            r"(?i)"
            r"(?:"
            r"HSN"
            r"(?:\s*/\s*SAC)?"
            r"(?:\s+CODE)?"
            r")"
            r"\s*[:=\-]?\s*"
            r"(\d{4,8})",
            anchor_text,
        )

        if same_line_match:

            value = (
                same_line_match.group(
                    1
                )
            )

            if len(value) in {
                4,
                6,
                8,
            }:

                candidates.append(
                    {
                        "value":
                            value,

                        "score":
                            1.45,

                        "page":
                            anchor_page,

                        "source":
                            "HSN_SAME_LINE",

                        "anchor":
                            anchor_text,

                        "evidence":
                            anchor_text,

                        "matched_variant":
                            "hsn code",
                    }
                )

        # ----------------------------------------------------
        # GEOMETRY SEARCH BELOW THE ANCHOR
        # ----------------------------------------------------

        for candidate_line in lines:

            candidate_page = (
                candidate_line.get(
                    "page"
                )
            )

            if (
                anchor_page
                is not None
                and
                candidate_page
                is not None
                and
                anchor_page
                !=
                candidate_page
            ):

                continue

            if candidate_line is anchor:
                continue

            candidate_text = str(
                candidate_line.get(
                    "text",
                    "",
                )
            )

            candidate_normalized = (
                _dynamic_normalize_text(
                    candidate_text
                )
            )

            # Do not use another label as the value.
            if (
                "hsn code"
                in
                candidate_normalized
            ):

                continue

            candidate_y0 = float(
                candidate_line.get(
                    "y0",
                    0.0,
                )
            )

            candidate_y1 = float(
                candidate_line.get(
                    "y1",
                    candidate_y0,
                )
            )

            candidate_x0 = float(
                candidate_line.get(
                    "x0",
                    0.0,
                )
            )

            candidate_x1 = float(
                candidate_line.get(
                    "x1",
                    candidate_x0,
                )
            )

            # =================================================
            # CRITICAL FIX
            #
            # Candidate must be BELOW HSN Code.
            #
            # Material code above the anchor can never win.
            # =================================================

            vertical_distance = (
                candidate_y0
                -
                anchor_y1
            )

            if vertical_distance < -1.5:
                continue

            # Don't travel too far down the document.
            if vertical_distance > 70.0:
                continue

            matches = list(
                _DYNAMIC_HSN_PATTERN
                .finditer(
                    candidate_text
                )
            )

            if not matches:
                continue

            anchor_center_x = (
                anchor_x0
                +
                anchor_x1
            ) / 2.0

            candidate_center_x = (
                candidate_x0
                +
                candidate_x1
            ) / 2.0

            x_distance = abs(
                candidate_center_x
                -
                anchor_center_x
            )

            for match in matches:

                value = (
                    match.group(
                        0
                    )
                )

                # Standard HSN/SAC lengths.
                if len(value) not in {
                    4,
                    6,
                    8,
                }:

                    continue

                score = (
                    1.00
                    if explicit_hsn
                    else
                    0.78
                )

                # Strongest preference:
                # directly below.
                if (
                    0
                    <=
                    vertical_distance
                    <=
                    18
                ):

                    score += 0.38

                elif (
                    vertical_distance
                    <=
                    35
                ):

                    score += 0.22

                elif (
                    vertical_distance
                    <=
                    55
                ):

                    score += 0.10

                # Horizontal alignment.
                if x_distance <= 25:

                    score += 0.25

                elif x_distance <= 60:

                    score += 0.15

                elif x_distance <= 120:

                    score += 0.05

                # Penalize lines containing multiple unrelated
                # values / table content.
                numeric_tokens = re.findall(
                    r"\d{4,}",
                    candidate_text,
                )

                if len(numeric_tokens) > 1:

                    score -= 0.10

                candidates.append(
                    {
                        "value":
                            value,

                        "score":
                            score,

                        "page":
                            candidate_page,

                        "source":
                            "HSN_BELOW_ANCHOR",

                        "anchor":
                            anchor_text,

                        "evidence":
                            candidate_text,

                        "matched_variant":
                            (
                                "hsn code"
                                if explicit_hsn
                                else
                                "hsn / sac"
                            ),
                    }
                )

    return candidates


# ============================================================
# GLOBAL FORMAT RECOVERY
# ============================================================

def _dynamic_global_pattern_candidates(
    field_name,
    lines,
):

    field_type = (
        _dynamic_parameter_type(
            field_name
        )
    )

    if field_type == "hsn":

        return (
            _dynamic_hsn_candidates(
                lines
            )
        )

    pattern = (
        _dynamic_pattern_for_type(
            field_type
        )
    )

    if (
        pattern is None
        or
        field_type
        in {
            "amount",
            "date",
            "phone",
            "percent",
        }
    ):

        return []

    candidates = []

    for line in lines:

        text = str(
            line.get(
                "text",
                "",
            )
        )

        for match in (
            pattern.finditer(
                text
            )
        ):

            value = (
                _dynamic_clean_value(
                    match.group(
                        0
                    )
                )
            )

            if not value:
                continue

            candidates.append(
                {
                    "value":
                        value,

                    "score":
                        0.72,

                    "page":
                        line.get(
                            "page"
                        ),

                    "source":
                        "GLOBAL_FORMAT_MATCH",

                    "anchor":
                        None,

                    "evidence":
                        text,

                    "matched_variant":
                        None,
                }
            )

    return candidates


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def _dynamic_candidates_for_field(
    field_name,
    lines,
):

    field_type = (
        _dynamic_parameter_type(
            field_name
        )
    )

    candidates = []

    # HSN uses only geometry-aware HSN logic.
    #
    # We deliberately do NOT let generic nearby-number
    # matching compete with it anymore.
    if field_type == "hsn":

        return (
            _dynamic_hsn_candidates(
                lines
            )
        )

    for (
        index,
        line,
    ) in enumerate(
        lines
    ):

        line_text = str(
            line.get(
                "text",
                "",
            )
        )

        anchor_score, variant = (
            _dynamic_anchor_similarity(
                field_name,
                line_text,
            )
        )

        if anchor_score < 0.62:
            continue

        same_line_value = (
            _dynamic_value_after_anchor(
                line_text,
                field_name,
            )
        )

        if same_line_value:

            value = (
                _dynamic_extract_typed_value(
                    field_type,
                    same_line_value,
                )
            )

            score = (
                anchor_score
                *
                0.58
                +
                0.24
                +
                _dynamic_candidate_quality(
                    value,
                    field_name,
                )
                +
                _dynamic_pattern_score(
                    field_type,
                    same_line_value,
                )
            )

            candidates.append(
                {
                    "value":
                        value,

                    "score":
                        score,

                    "page":
                        line.get(
                            "page"
                        ),

                    "source":
                        "SAME_LINE_ANCHOR",

                    "anchor":
                        line_text,

                    "evidence":
                        line_text,

                    "matched_variant":
                        variant,
                }
            )

        line_normalized = (
            _dynamic_normalize_text(
                line_text
            )
        )

        field_variants = (
            _dynamic_field_variants(
                field_name
            )
        )

        clean_label_like = any(
            line_normalized
            ==
            field_variant
            for field_variant
            in field_variants
        )

        if (
            not clean_label_like
            and
            anchor_score
            <
            0.90
        ):

            continue

        for offset in (
            1,
            2,
        ):

            next_index = (
                index
                +
                offset
            )

            if (
                next_index
                >=
                len(
                    lines
                )
            ):

                continue

            next_line = (
                lines[
                    next_index
                ]
            )

            current_page = (
                line.get(
                    "page"
                )
            )

            next_page = (
                next_line.get(
                    "page"
                )
            )

            if (
                current_page
                is not None
                and
                next_page
                is not None
                and
                current_page
                !=
                next_page
            ):

                continue

            next_text = (
                _dynamic_clean_value(
                    next_line.get(
                        "text",
                        "",
                    )
                )
            )

            if not next_text:
                continue

            value = (
                _dynamic_extract_typed_value(
                    field_type,
                    next_text,
                )
            )

            distance_bonus = (
                0.11
                if offset
                ==
                1
                else
                0.05
            )

            score = (
                anchor_score
                *
                0.52
                +
                distance_bonus
                +
                _dynamic_candidate_quality(
                    value,
                    field_name,
                )
                +
                _dynamic_pattern_score(
                    field_type,
                    next_text,
                )
                +
                _dynamic_geometry_bonus(
                    line,
                    next_line,
                )
            )

            candidates.append(
                {
                    "value":
                        value,

                    "score":
                        score,

                    "page":
                        next_page,

                    "source":
                        (
                            "NEXT_LINE_ANCHOR"
                            if offset == 1
                            else
                            "SECOND_LINE_ANCHOR"
                        ),

                    "anchor":
                        line_text,

                    "evidence":
                        next_text,

                    "matched_variant":
                        variant,
                }
            )

    candidates.extend(
        _dynamic_global_pattern_candidates(
            field_name,
            lines,
        )
    )

    cleaned = []

    for candidate in candidates:

        value = (
            _dynamic_clean_value(
                candidate.get(
                    "value",
                    "",
                )
            )
        )

        if not value:
            continue

        if (
            len(
                value
            )
            >
            DYNAMIC_MAX_VALUE_LENGTH
        ):

            continue

        candidate[
            "value"
        ] = value

        cleaned.append(
            candidate
        )

    return cleaned


# ============================================================
# DEDUPLICATE CANDIDATES
# ============================================================

def _dynamic_dedupe_candidates(
    candidates,
):

    best_by_value = {}

    for candidate in candidates:

        key = (
            _dynamic_normalize_text(
                candidate.get(
                    "value",
                    "",
                )
            )
        )

        if not key:
            continue

        existing = (
            best_by_value.get(
                key
            )
        )

        if (
            existing is None
            or
            float(
                candidate.get(
                    "score",
                    0.0,
                )
            )
            >
            float(
                existing.get(
                    "score",
                    0.0,
                )
            )
        ):

            best_by_value[
                key
            ] = candidate

    return sorted(
        best_by_value.values(),
        key=lambda item: float(
            item.get(
                "score",
                0.0,
            )
        ),
        reverse=True,
    )


# ============================================================
# SCORE -> CONFIDENCE
# ============================================================

def _dynamic_score_to_confidence(
    score,
):

    confidence = (
        (
            float(
                score
            )
            -
            0.35
        )
        /
        0.75
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    return round(
        confidence,
        4,
    )


# ============================================================
# PUBLIC DYNAMIC API
# ============================================================

def extract_dynamic_parameters(
    input_path,
    requested_fields,
    *,
    production_result=None,
    min_confidence=
        DYNAMIC_MIN_CONFIDENCE,
):

    cleaned_requests = []

    seen = set()

    for raw_field in (
        requested_fields
        or
        []
    ):

        field = (
            str(
                raw_field
            )
            .strip()
        )

        if not field:
            continue

        key = (
            _dynamic_normalize_text(
                field
            )
        )

        if (
            not key
            or
            key
            in
            seen
        ):

            continue

        seen.add(
            key
        )

        cleaned_requests.append(
            field
        )

        if (
            len(
                cleaned_requests
            )
            >=
            DYNAMIC_MAX_FIELDS
        ):

            break

    if not cleaned_requests:
        return {}

    lines = (
        _dynamic_document_lines(
            input_path,
            production_result=
                production_result,
        )
    )

    output = {}

    if not lines:

        for field in cleaned_requests:

            output[
                field
            ] = {
                "value":
                    DYNAMIC_NOT_DETECTED,

                "status":
                    DYNAMIC_NOT_DETECTED,

                "confidence":
                    0.0,

                "page":
                    None,

                "source":
                    "NO_DOCUMENT_TEXT",

                "evidence":
                    None,
            }

        return output

    for field in cleaned_requests:

        candidates = (
            _dynamic_dedupe_candidates(
                _dynamic_candidates_for_field(
                    field,
                    lines,
                )
            )
        )

        if not candidates:

            output[
                field
            ] = {
                "value":
                    DYNAMIC_NOT_DETECTED,

                "status":
                    DYNAMIC_NOT_DETECTED,

                "confidence":
                    0.0,

                "page":
                    None,

                "source":
                    "NO_EVIDENCE",

                "evidence":
                    None,
            }

            continue

        best = (
            candidates[
                0
            ]
        )

        confidence = (
            _dynamic_score_to_confidence(
                best.get(
                    "score",
                    0.0,
                )
            )
        )

        ambiguous = False

        if len(candidates) >= 2:

            first_value = (
                _dynamic_normalize_text(
                    candidates[
                        0
                    ].get(
                        "value",
                        "",
                    )
                )
            )

            second_value = (
                _dynamic_normalize_text(
                    candidates[
                        1
                    ].get(
                        "value",
                        "",
                    )
                )
            )

            score_gap = (
                float(
                    candidates[
                        0
                    ].get(
                        "score",
                        0.0,
                    )
                )
                -
                float(
                    candidates[
                        1
                    ].get(
                        "score",
                        0.0,
                    )
                )
            )

            if (
                first_value
                !=
                second_value
                and
                score_gap
                <
                0.035
                and
                confidence
                <
                0.84
            ):

                ambiguous = True

        if (
            confidence
            <
            float(
                min_confidence
            )
            or
            ambiguous
        ):

            output[
                field
            ] = {
                "value":
                    DYNAMIC_NOT_DETECTED,

                "status":
                    (
                        "AMBIGUOUS"
                        if ambiguous
                        else
                        DYNAMIC_NOT_DETECTED
                    ),

                "confidence":
                    confidence,

                "page":
                    best.get(
                        "page"
                    ),

                "source":
                    best.get(
                        "source"
                    ),

                "evidence":
                    best.get(
                        "evidence"
                    ),
            }

            continue

        output[
            field
        ] = {
            "value":
                best.get(
                    "value",
                    DYNAMIC_NOT_DETECTED,
                ),

            "status":
                "DETECTED",

            "confidence":
                confidence,

            "page":
                best.get(
                    "page"
                ),

            "source":
                best.get(
                    "source"
                ),

            "evidence":
                best.get(
                    "evidence"
                ),
        }

    return output


# ============================================================
# COMBINED PRODUCTION + DYNAMIC API
# ============================================================

def process_invoice_with_dynamic(
    input_path,
    requested_fields=None,
    *,
    min_dynamic_confidence=
        DYNAMIC_MIN_CONFIDENCE,
):

    engine = (
        load_v3_production_engine()
    )

    process_invoice_final = (
        engine[
            "process_invoice_final"
        ]
    )

    production_result = (
        process_invoice_final(
            str(
                input_path
            )
        )
    )

    dynamic_fields = (
        extract_dynamic_parameters(
            input_path,
            requested_fields
            or
            [],
            production_result=
                production_result,
            min_confidence=
                min_dynamic_confidence,
        )
    )

    return {
        "production_result":
            production_result,

        "dynamic_fields":
            dynamic_fields,

        "dynamic_requested":
            list(
                requested_fields
                or
                []
            ),

        "dynamic_schema_mode":
            "RUNTIME_USER_DEFINED",

        "trained_schema_fields":
            list(
                EXPECTED_FIELDS
            ),
    }


# ============================================================
# AUTOMATIC DYNAMIC SCHEMA DISCOVERY
# BACKEND / INFERENCE-ENGINE LAYER
# ============================================================

AUTO_DYNAMIC_SCHEMA_MODE = "AUTOMATIC_SCHEMA_DISCOVERY"


_AUTO_TRAINED_SCHEMA_ALIASES = {
    "vendor",
    "vendor name",
    "seller",
    "seller name",
    "supplier",
    "supplier name",

    "invoice",
    "invoice no",
    "invoice number",
    "invoice id",
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


_AUTO_LABEL_BLACKLIST = {
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

    "narration",
}


def _auto_clean_spaces(value):

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def _auto_normalize_label(value):

    text = _auto_clean_spaces(
        value
    )

    if not text:
        return ""

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

    simple = text.casefold()

    simple = re.sub(
        r"[^a-z0-9]+",
        " ",
        simple,
    )

    simple = _auto_clean_spaces(
        simple
    )

    canonical = {

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

        "odn no":
            "ODN Number",

        "odn number":
            "ODN Number",

        "hsn":
            "HSN Code",

        "hsn code":
            "HSN Code",

        "hsn sac":
            "HSN Code",

        "sac code":
            "HSN Code",

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

        "ifsc":
            "IFSC Code",

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

        "b value":
            "B Value",

        "bvalue":
            "B Value",

        "state code":
            "State Code",

        "state name":
            "State Name",

        "place of supply":
            "Place of Supply",

        "cgst":
            "CGST",

        "sgst":
            "SGST",

        "igst":
            "IGST",

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


def _auto_label_key(value):

    text = _auto_normalize_label(
        value
    ).casefold()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return _auto_clean_spaces(
        text
    )


# ============================================================
# V8 DYNAMIC LABEL QUALITY FILTER
#
# Purpose:
#   Reject low-information / structural fragments discovered
#   as dynamic labels while preserving meaningful invoice
#   parameters.
#
# This layer:
#   - does NOT retrain LayoutLMv3
#   - does NOT modify model weights
#   - does NOT alter trained 16-field extraction
#   - does NOT alter GST / line-item / financial logic
# ============================================================

_AUTO_V8_SINGLE_TOKEN_REJECT = {
    "add",
    "in",
    "out",
    "at",
    "by",
    "on",
    "as",
    "for",
    "with",
    "without",
    "against",
    "through",
    "via",
    "per",
    "page",
    "date",
    "name",
    "type",
    "item",
    "items",
    "particular",
    "particulars",
    "details",
    "detail",
    "remarks",
    "remark",
    "note",
    "notes",
    "total",
    "basic",
    "gross",
    "net",
    "invoice",
    "bill",
    "party",
    "company",
    "customer",
    "vendor",
    "supplier",
    "buyer",
    "seller",
    "state",
    "place",
}

_AUTO_V8_SAFE_SINGLE_TOKEN_LABELS = {
    "gstin",
    "pan",
    "cin",
    "cgst",
    "sgst",
    "igst",
    "email",
    "irn",
    "hsn",
    "sac",
}

_AUTO_V8_BAD_EXACT_LABELS = {
    "add",
    "in",
    "from",
    "to",
    "date",
    "remarks",
    "remark",
    "note",
    "notes",
    "basic total",
    "total value",
    "bill amount",
    "invoice",
    "bill",
    "gst",
}

_AUTO_V8_TABLE_HEADER_LABELS = {
    "description",
    "item description",
    "material description",
    "material",
    "material code",
    "qty",
    "quantity",
    "rate",
    "unit price",
    "amount",
    "line amount",
    "uom",
    "unit",
    "hsn",
    "sac",
    "sl",
    "sl no",
    "serial no",
}

_AUTO_V8_MEANINGFUL_TOKENS = {
    "number",
    "no",
    "code",
    "date",
    "name",
    "mode",
    "terms",
    "address",
    "email",
    "phone",
    "mobile",
    "bank",
    "account",
    "ifsc",
    "gstin",
    "gst",
    "pan",
    "cin",
    "irn",
    "hsn",
    "sac",
    "eway",
    "way",
    "vehicle",
    "ack",
    "acknowledgement",
    "po",
    "purchase",
    "order",
    "gr",
    "goods",
    "receipt",
    "delivery",
    "state",
    "supply",
    "place",
    "posting",
    "freight",
    "round",
    "discount",
    "taxable",
    "tax",
    "cgst",
    "sgst",
    "igst",
    "amount",
    "words",
    "reference",
    "ref",
    "odn",
    "payment",
    "challan",
    "lr",
    "ewaybill",
}


def _auto_v8_label_has_semantic_value(
    label,
):
    key = _auto_label_key(
        label
    )

    if not key:
        return False

    if key in _AUTO_V8_BAD_EXACT_LABELS:
        return False

    if key in _AUTO_V8_TABLE_HEADER_LABELS:
        return False

    words = key.split()

    if (
        len(words) == 1
        and
        key in _AUTO_V8_SINGLE_TOKEN_REJECT
        and
        key not in _AUTO_V8_SAFE_SINGLE_TOKEN_LABELS
    ):
        return False

    # Acronyms / identifiers that are valid by themselves.
    if key in _AUTO_V8_SAFE_SINGLE_TOKEN_LABELS:
        return True

    # Two or more words are allowed when the phrase has
    # invoice-specific semantic content.
    if len(words) >= 2:

        if any(
            token
            in
            _AUTO_V8_MEANINGFUL_TOKENS
            for token
            in words
        ):
            return True

        # Keep explicit percentage-bearing financial labels,
        # e.g. "Discount @ 2.00%".
        if re.search(
            r"\d+(?:\.\d+)?\s*%",
            str(label),
        ):
            return True

        return False

    return True


def _auto_valid_label(value):

    label = _auto_normalize_label(
        value
    )

    key = _auto_label_key(
        label
    )

    if not key:
        return False

    if key in _AUTO_TRAINED_SCHEMA_ALIASES:
        return False

    if key in _AUTO_LABEL_BLACKLIST:
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

    if not _auto_v8_label_has_semantic_value(
        label
    ):
        return False

    return True


def _auto_clean_label_segment(segment):

    value = str(
        segment
    ).strip()

    value = value.strip(
        " ,;|"
    )

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


def _auto_labels_from_line(text):

    text = str(
        text
    )

    separators = list(
        re.finditer(
            r"[:=]",
            text,
        )
    )

    if not separators:
        return []

    output = []

    previous_end = 0

    for match in separators:

        segment = text[
            previous_end:
            match.start()
        ]

        previous_end = (
            match.end()
        )

        segment = (
            _auto_clean_label_segment(
                segment
            )
        )

        if not segment:
            continue

        label = (
            _auto_normalize_label(
                segment
            )
        )

        if not _auto_valid_label(
            label
        ):
            continue

        output.append(
            label
        )

    return output


def _auto_standalone_fields(lines):

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

    normalized = text.casefold()

    normalized = re.sub(
        r"[._]+",
        " ",
        normalized,
    )

    normalized = (
        _auto_clean_spaces(
            normalized
        )
    )

    patterns = {

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

    output = []

    for canonical, candidates in (
        patterns.items()
    ):

        if any(
            re.search(
                pattern,
                normalized,
            )
            for pattern
            in candidates
        ):

            if _auto_valid_label(
                canonical
            ):

                output.append(
                    canonical
                )

    return output


def discover_dynamic_fields(
    input_path,
    production_result=None,
):

    """
    Automatically discover additional invoice fields from
    the uploaded document itself.

    IMPORTANT:
    production_result is NOT used for schema discovery.

    This prevents internal engine metadata such as:
      schema_version
      document_id
      page_count
      line_number
      validation fields
      overall_status
      calculated_total
      etc.

    from being incorrectly treated as invoice fields.

    production_result remains in the signature for backward
    compatibility with the existing inference pipeline.
    """

    # ========================================================
    # DOCUMENT-ONLY DYNAMIC SCHEMA DISCOVERY
    # ========================================================

    lines = (
        _dynamic_pdf_lines(
            input_path
        )
    )

    discovered = []
    seen = set()

    def add_field(
        raw_field,
    ):

        field = (
            _auto_normalize_label(
                raw_field
            )
        )

        if not _auto_valid_label(
            field
        ):
            return

        key = (
            _auto_label_key(
                field
            )
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

    # ========================================================
    # EXPLICIT LABEL DISCOVERY
    # ========================================================

    for line in lines:

        text = str(
            line.get(
                "text",
                "",
            )
        )

        for field in (
            _auto_labels_from_line(
                text
            )
        ):

            add_field(
                field
            )

    # ========================================================
    # STRONG STANDALONE INVOICE LABELS
    # ========================================================

    for field in (
        _auto_standalone_fields(
            lines
        )
    ):

        add_field(
            field
        )

    return discovered[:40]


def _auto_find_explicit_hsn(
    input_path,
):

    lines = (
        _dynamic_pdf_lines(
            input_path
        )
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

        normalized = text.casefold()

        normalized = re.sub(
            r"[._\-]+",
            " ",
            normalized,
        )

        normalized = (
            _auto_clean_spaces(
                normalized
            )
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

            value = (
                same_line.group(
                    1
                )
            )

            if len(value) in {
                4,
                6,
                8,
            }:

                return {
                    "value":
                        value,

                    "page":
                        line.get(
                            "page"
                        ),

                    "evidence":
                        text,
                }

        for offset in (
            1,
            2,
            3,
        ):

            candidate_index = (
                index
                +
                offset
            )

            if (
                candidate_index
                >=
                len(
                    lines
                )
            ):
                break

            candidate = (
                lines[
                    candidate_index
                ]
            )

            if (
                candidate.get(
                    "page"
                )
                !=
                line.get(
                    "page"
                )
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

            value = (
                match.group(
                    0
                )
            )

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


def _auto_find_gst_components(
    payload,
):

    found = {}

    def consume(item):

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

        amount = (
            item.get(
                "amount"
            )
        )

        if amount is None:
            return

        if component_type in found:
            return

        found[
            component_type
        ] = {
            "rate":
                item.get(
                    "rate_percent"
                ),

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

    def walk(node):

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
                        consume(item)

                walk(value)

        elif isinstance(
            node,
            list,
        ):

            for item in node:
                walk(item)

    walk(
        payload
    )

    return found


def _auto_format_number(value):

    try:

        return (
            f"{float(value):,.2f}"
        )

    except Exception:

        return str(
            value
        )


def _auto_cleanup_dynamic_fields(
    input_path,
    discovered_fields,
    dynamic_fields,
    production_result,
):

    cleaned = {}
    seen = {}

    for raw_name, information in (
        dynamic_fields.items()
    ):

        name = (
            _auto_normalize_label(
                raw_name
            )
        )

        if not _auto_valid_label(
            name
        ):
            continue

        key = (
            _auto_label_key(
                name
            )
        )

        if not key:
            continue

        candidate_info = (
            information
            if isinstance(
                information,
                dict,
            )
            else
            {}
        )

        existing_name = (
            seen.get(
                key
            )
        )

        if existing_name is None:

            cleaned[
                name
            ] = information

            seen[
                key
            ] = name

            continue

        existing_info = (
            cleaned.get(
                existing_name,
                {}
            )
        )

        existing_detected = (
            isinstance(
                existing_info,
                dict,
            )
            and
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

        replace = False

        if (
            candidate_detected
            and
            not existing_detected
        ):

            replace = True

        elif (
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

                replace = True

        if replace:

            cleaned.pop(
                existing_name,
                None,
            )

            cleaned[
                name
            ] = information

            seen[
                key
            ] = name

    if (
        "HSN Code"
        in
        discovered_fields
    ):

        repaired = (
            _auto_find_explicit_hsn(
                input_path
            )
        )

        if repaired:

            cleaned[
                "HSN Code"
            ] = {
                "value":
                    repaired[
                        "value"
                    ],

                "status":
                    "DETECTED",

                "confidence":
                    1.0,

                "page":
                    repaired[
                        "page"
                    ],

                "source":
                    "AUTO_EXPLICIT_LABEL",

                "evidence":
                    repaired[
                        "evidence"
                    ],
            }

    verified_gst = (
        _auto_find_gst_components(
            production_result
        )
    )

    for tax_type in (
        "CGST",
        "SGST",
        "IGST",
    ):

        if (
            tax_type
            not in
            discovered_fields
        ):
            continue

        component = (
            verified_gst.get(
                tax_type
            )
        )

        if not component:
            continue

        rate = (
            component.get(
                "rate"
            )
        )

        amount = (
            component.get(
                "amount"
            )
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
                f"{_auto_format_number(amount)}"
            )

        else:

            value = (
                _auto_format_number(
                    amount
                )
            )

        cleaned[
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

    return cleaned


# Add useful aliases to the existing generic dynamic extractor.
_DYNAMIC_SPECIAL_VARIANTS.setdefault(
    "phone number",
    set(),
).update(
    {
        "phone number",
        "phone no",
        "ph no",
        "ph number",
        "mobile no",
        "mobile number",
        "telephone",
        "tel",
    }
)



# ============================================================
# V7 BLIND-TEST PRODUCTION HARDENING
# ============================================================

_V7_NUMBER_PATTERN = re.compile(
    r"-?\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)


def _v7_clean_text(value):

    return re.sub(
        r"\s+",
        " ",
        str(
            value
            if value is not None
            else
            ""
        ),
    ).strip()


def _v7_number(value):

    if value is None:
        return None

    text = str(
        value
    ).strip()

    text = (
        text
        .replace(",", "")
        .replace("?", "")
        .replace("INR", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .strip()
    )

    text = re.sub(
        r"^[^\d+\-.]+",
        "",
        text,
    )

    text = re.sub(
        r"[^\d+\-.]+$",
        "",
        text,
    )

    if not text:
        return None

    try:
        return float(text)

    except Exception:
        return None


def _v7_money(value):

    number = _v7_number(
        value
    )

    if number is None:
        return str(
            value
        )

    return (
        f"{number:,.2f}"
    )


def _v7_document_lines(
    input_path,
):

    """
    Document-only evidence.

    Native PDF lines are preferred.

    The production result JSON is NEVER injected into this
    evidence source.
    """

    return (
        _dynamic_pdf_lines(
            input_path
        )
    )


def _v7_line_texts(
    input_path,
):

    return [
        {
            "page":
                line.get(
                    "page"
                ),

            "text":
                _v7_clean_text(
                    line.get(
                        "text",
                        "",
                    )
                ),

            "x0":
                line.get(
                    "x0"
                ),

            "y0":
                line.get(
                    "y0"
                ),

            "x1":
                line.get(
                    "x1"
                ),

            "y1":
                line.get(
                    "y1"
                ),
        }

        for line
        in _v7_document_lines(
            input_path
        )

        if _v7_clean_text(
            line.get(
                "text",
                "",
            )
        )
    ]


def _v7_label_regex(
    label,
):

    words = re.findall(
        r"[A-Za-z0-9]+",
        str(
            label
        ),
    )

    if not words:
        return None

    body = (
        r"\s*[\s._/\-]*\s*"
        .join(
            re.escape(
                word
            )
            for word
            in words
        )
    )

    return re.compile(
        rf"(?i)"
        rf"(?<![A-Za-z0-9])"
        rf"{body}"
        rf"\s*[:=]\s*"
    )


def _v7_trim_value_at_next_label(
    value,
):

    text = _v7_clean_text(
        value
    )

    if not text:
        return ""

    # --------------------------------------------------------
    # Stop at another obvious label on the same physical line.
    #
    # Example:
    # GSTIN : xxx PAN : yyy CIN : zzz
    # --------------------------------------------------------

    next_label = re.search(
        r"\s+"
        r"(?="
        r"[A-Za-z][A-Za-z0-9 /&().@\-]{1,35}"
        r"\s*[:=]"
        r")",
        text,
    )

    if next_label:

        text = text[
            :
            next_label.start()
        ]

    return (
        text
        .strip(
            " \t\r\n,;|"
        )
    )


def _v7_exact_labeled_value(
    input_path,
    label,
):

    """
    Prefer an explicit exact label over fuzzy semantic matching.

    This is intentionally conservative and is used only as a
    post-extraction repair layer.
    """

    pattern = (
        _v7_label_regex(
            label
        )
    )

    if pattern is None:
        return None

    for line in (
        _v7_line_texts(
            input_path
        )
    ):

        text = (
            line[
                "text"
            ]
        )

        match = (
            pattern.search(
                text
            )
        )

        if not match:
            continue

        raw_value = (
            text[
                match.end()
                :
            ]
        )

        value = (
            _v7_trim_value_at_next_label(
                raw_value
            )
        )

        if not value:
            continue

        return {
            "value":
                value,

            "page":
                line.get(
                    "page"
                ),

            "evidence":
                text,
        }

    return None


def _v7_find_amount_line(
    input_path,
    aliases,
):

    """
    Find a financial amount from an explicit document label.

    Only the number after the label is considered.
    """

    lines = (
        _v7_line_texts(
            input_path
        )
    )

    for alias in aliases:

        pattern = (
            _v7_label_regex(
                alias
            )
        )

        if pattern is None:
            continue

        for line in lines:

            text = (
                line[
                    "text"
                ]
            )

            match = (
                pattern.search(
                    text
                )
            )

            if not match:
                continue

            remainder = (
                text[
                    match.end()
                    :
                ]
            )

            number_match = (
                _V7_NUMBER_PATTERN
                .search(
                    remainder
                )
            )

            if not number_match:
                continue

            number = (
                _v7_number(
                    number_match.group(
                        0
                    )
                )
            )

            if number is None:
                continue

            return {
                "value":
                    number,

                "page":
                    line.get(
                        "page"
                    ),

                "evidence":
                    text,

                "label":
                    alias,
            }

    return None


def _v7_find_gst_components(
    production_result,
):

    """
    Recover GST components from every verified runtime source.

    IMPORTANT blind-test fix:

    The runtime may detect valid components under
    v6_audit.gst_candidates_seen even when
    gst_components_selected is empty.

    The previous automatic cleanup ignored that source.
    """

    candidates = []

    def consume(
        item,
        source_bucket=None,
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

        amount = (
            _v7_number(
                item.get(
                    "amount"
                )
            )
        )

        if (
            amount is None
            or
            amount < 0
        ):
            return

        rate = (
            _v7_number(
                item.get(
                    "rate_percent"
                )
            )
        )

        candidates.append(
            {
                "type":
                    component_type,

                "rate":
                    rate,

                "amount":
                    amount,

                "page":
                    item.get(
                        "page"
                    ),

                "source":
                    (
                        item.get(
                            "source"
                        )
                        or
                        source_bucket
                        or
                        "V7_GST_COMPONENT"
                    ),

                "evidence":
                    (
                        item.get(
                            "row_text"
                        )
                        or
                        item.get(
                            "evidence"
                        )
                    ),

                "bucket":
                    source_bucket,
            }
        )

    def walk(
        node,
        parent_key=None,
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
                        "gst_candidates_seen",
                    }
                    and
                    isinstance(
                        value,
                        list,
                    )
                ):

                    for item in value:

                        consume(
                            item,
                            source_bucket=
                                key,
                        )

                walk(
                    value,
                    parent_key=
                        key,
                )

        elif isinstance(
            node,
            list,
        ):

            for item in node:

                walk(
                    item,
                    parent_key=
                        parent_key,
                )

    walk(
        production_result
    )

    best = {}

    # --------------------------------------------------------
    # Prefer selected/reconciled components when available.
    # Otherwise accept native GST candidate rows.
    # --------------------------------------------------------

    bucket_priority = {
        "gst_components_selected":
            3,

        "gst_components":
            2,

        "gst_candidates_seen":
            1,

        None:
            0,
    }

    for item in candidates:

        component_type = (
            item[
                "type"
            ]
        )

        existing = (
            best.get(
                component_type
            )
        )

        if existing is None:

            best[
                component_type
            ] = item

            continue

        new_priority = (
            bucket_priority.get(
                item.get(
                    "bucket"
                ),
                0,
            )
        )

        old_priority = (
            bucket_priority.get(
                existing.get(
                    "bucket"
                ),
                0,
            )
        )

        if new_priority > old_priority:

            best[
                component_type
            ] = item

            continue

        if (
            new_priority
            ==
            old_priority
            and
            item[
                "amount"
            ]
            <
            existing[
                "amount"
            ]
            *
            10
        ):

            # Conservative preference for sane invoice-tax
            # component magnitude when duplicates exist.
            best[
                component_type
            ] = item

    return best


def _v7_set_field(
    production_result,
    field_name,
    value,
    *,
    source,
    status="RULE_RECOVERED",
):

    if not isinstance(
        production_result,
        dict,
    ):
        return

    fields = (
        production_result.setdefault(
            "fields",
            {},
        )
    )

    if not isinstance(
        fields,
        dict,
    ):
        return

    fields[
        field_name
    ] = {
        "value":
            value,

        "status":
            status,

        "source":
            source,
    }


def _v7_normalize_total_amount(
    production_result,
):

    try:

        field = (
            production_result[
                "fields"
            ][
                "TOTAL_AMOUNT"
            ]
        )

    except Exception:
        return

    if not isinstance(
        field,
        dict,
    ):
        return

    raw_value = (
        field.get(
            "value"
        )
    )

    number = (
        _v7_number(
            raw_value
        )
    )

    if number is None:
        return

    field[
        "value"
    ] = (
        _v7_money(
            number
        )
    )


def _v7_line_item_sum(
    production_result,
):

    if not isinstance(
        production_result,
        dict,
    ):
        return None

    line_items = (
        production_result.get(
            "line_items"
        )
    )

    if not isinstance(
        line_items,
        list,
    ):
        return None

    amounts = []

    for item in line_items:

        if not isinstance(
            item,
            dict,
        ):
            continue

        amount = (
            _v7_number(
                item.get(
                    "line_amount"
                )
            )
        )

        if amount is None:
            continue

        amounts.append(
            amount
        )

    if not amounts:
        return None

    return sum(
        amounts
    )


def _v7_repair_subtotal(
    input_path,
    production_result,
):

    """
    SUBTOTAL means the pre-discount line-item subtotal.

    Priority:
      1. explicit 'Subtotal' / 'Basic Total'
      2. validated line-item sum
      3. leave existing production value untouched

    'Taxable Value After Discount' must never silently replace
    the trained SUBTOTAL field.
    """

    explicit = (
        _v7_find_amount_line(
            input_path,
            [
                "Subtotal",
                "Sub Total",
                "Basic Total",
            ],
        )
    )

    line_sum = (
        _v7_line_item_sum(
            production_result
        )
    )

    selected = None
    source = None

    if explicit is not None:

        selected = (
            explicit[
                "value"
            ]
        )

        source = (
            "V7_EXPLICIT_SUBTOTAL"
        )

        if line_sum is not None:

            tolerance = max(
                1.0,
                abs(
                    line_sum
                )
                *
                0.001,
            )

            if (
                abs(
                    selected
                    -
                    line_sum
                )
                <=
                tolerance
            ):

                selected = (
                    line_sum
                )

                source = (
                    "V7_EXPLICIT_SUBTOTAL_LINE_SUM_VALIDATED"
                )

    elif line_sum is not None:

        selected = (
            line_sum
        )

        source = (
            "V7_LINE_ITEM_SUBTOTAL"
        )

    if selected is None:
        return

    _v7_set_field(
        production_result,
        "SUBTOTAL",
        _v7_money(
            selected
        ),
        source=
            source,
        status=
            "RECONCILED",
    )

    normalized = (
        production_result.get(
            "normalized"
        )
    )

    if isinstance(
        normalized,
        dict,
    ):

        normalized[
            "subtotal"
        ] = float(
            selected
        )

    validation = (
        production_result.get(
            "validation"
        )
    )

    if isinstance(
        validation,
        dict,
    ):

        line_validation = (
            validation.get(
                "line_item_reconciliation"
            )
        )

        if isinstance(
            line_validation,
            dict,
        ):

            if line_sum is not None:

                difference = (
                    line_sum
                    -
                    selected
                )

                line_validation[
                    "line_amount_sum"
                ] = line_sum

                line_validation[
                    "difference"
                ] = difference

                line_validation[
                    "matches_subtotal"
                ] = (
                    abs(
                        difference
                    )
                    <=
                    max(
                        1.0,
                        abs(
                            selected
                        )
                        *
                        0.001,
                    )
                )


def _v7_repair_tax(
    production_result,
):

    components = (
        _v7_find_gst_components(
            production_result
        )
    )

    if not components:
        return

    selected = []

    if (
        "CGST"
        in
        components
        and
        "SGST"
        in
        components
    ):

        selected = [
            components[
                "CGST"
            ],
            components[
                "SGST"
            ],
        ]

    elif (
        "IGST"
        in
        components
    ):

        selected = [
            components[
                "IGST"
            ]
        ]

    if not selected:
        return

    total_tax = sum(
        float(
            item[
                "amount"
            ]
        )
        for item
        in selected
    )

    _v7_set_field(
        production_result,
        "TAX",
        _v7_money(
            total_tax
        ),
        source=
            "V7_GST_COMPONENT_SUM",
        status=
            "RECONCILED",
    )

    normalized = (
        production_result.get(
            "normalized"
        )
    )

    if isinstance(
        normalized,
        dict,
    ):

        normalized[
            "tax"
        ] = (
            total_tax
        )

    tax_details = (
        production_result.setdefault(
            "tax_details",
            {},
        )
    )

    if isinstance(
        tax_details,
        dict,
    ):

        tax_details[
            "total_tax"
        ] = (
            total_tax
        )

        tax_details[
            "gst_components"
        ] = [
            {
                "type":
                    item[
                        "type"
                    ],

                "rate_percent":
                    item.get(
                        "rate"
                    ),

                "amount":
                    item[
                        "amount"
                    ],

                "page":
                    item.get(
                        "page"
                    ),

                "source":
                    (
                        item.get(
                            "source"
                        )
                        or
                        "V7_GST_COMPONENT"
                    ),

                "row_text":
                    item.get(
                        "evidence"
                    ),
            }

            for item
            in selected
        ]

    validation = (
        production_result.get(
            "validation"
        )
    )

    if isinstance(
        validation,
        dict,
    ):

        financial = (
            validation.get(
                "financial_reconciliation"
            )
        )

        if isinstance(
            financial,
            dict,
        ):

            financial[
                "tax"
            ] = (
                total_tax
            )

        gst_validation = (
            validation.get(
                "gst_reconciliation"
            )
        )

        if isinstance(
            gst_validation,
            dict,
        ):

            component_sum = sum(
                float(
                    item[
                        "amount"
                    ]
                )
                for item
                in selected
            )

            gst_validation[
                "component_sum"
            ] = (
                component_sum
            )

            gst_validation[
                "matches_tax_total"
            ] = True

            gst_validation[
                "difference"
            ] = (
                component_sum
                -
                total_tax
            )


def _v7_customer_section_lines(
    input_path,
):

    """
    Return a small document section around BILL TO / CUSTOMER.

    Used only to stop the generic header-address recovery from
    returning the vendor address.
    """

    lines = (
        _v7_line_texts(
            input_path
        )
    )

    if not lines:
        return []

    start = None

    for index, line in enumerate(
        lines
    ):

        normalized = (
            _dynamic_normalize_text(
                line[
                    "text"
                ]
            )
        )

        if (
            normalized
            in {
                "bill to",
                "billed to",
                "customer",
                "customer details",
                "buyer",
                "buyer details",
                "ship to",
            }
            or
            normalized.startswith(
                "bill to "
            )
            or
            normalized.startswith(
                "customer name "
            )
        ):

            start = (
                index
            )

            break

    if start is None:
        return []

    return lines[
        start
        :
        min(
            len(
                lines
            ),
            start
            +
            14,
        )
    ]


def _v7_repair_customer_address(
    input_path,
    production_result,
):

    # --------------------------------------------------------
    # Strongest case: explicit Address : value
    # --------------------------------------------------------

    explicit = (
        _v7_exact_labeled_value(
            input_path,
            "Address",
        )
    )

    if explicit:

        value = (
            explicit[
                "value"
            ]
        )

        if (
            len(
                value
            )
            >=
            8
        ):

            _v7_set_field(
                production_result,
                "ADDRESS",
                value,
                source=
                    "V7_EXPLICIT_CUSTOMER_ADDRESS",
            )

            return

    # --------------------------------------------------------
    # Generic bill-to block fallback
    # --------------------------------------------------------

    section = (
        _v7_customer_section_lines(
            input_path
        )
    )

    if not section:
        return

    collected = []

    stop_labels = {
        "customer gstin",
        "gstin",
        "pan",
        "cin",
        "state code",
        "state name",
        "place of supply",
        "po number",
        "po no",
        "gr number",
        "gr no",
        "invoice number",
        "invoice no",
    }

    address_started = False

    for line in section:

        text = (
            line[
                "text"
            ]
        )

        normalized = (
            _dynamic_normalize_text(
                text
            )
        )

        if any(
            normalized.startswith(
                stop_label
            )
            for stop_label
            in stop_labels
        ):

            if address_started:
                break

            continue

        if (
            "address"
            in
            normalized
        ):

            pattern = re.compile(
                r"(?i)\baddress\b\s*[:=]?\s*"
            )

            value = (
                pattern.sub(
                    "",
                    text,
                    count=1,
                )
                .strip()
            )

            if value:
                collected.append(
                    value
                )

            address_started = True

            continue

        if address_started:

            collected.append(
                text
            )

            if (
                len(
                    collected
                )
                >=
                4
            ):
                break

    value = (
        _v7_clean_text(
            " ".join(
                collected
            )
        )
    )

    if len(value) < 8:
        return

    _v7_set_field(
        production_result,
        "ADDRESS",
        value,
        source=
            "V7_BILL_TO_ADDRESS",
    )


def _v7_repair_dynamic_exact_fields(
    input_path,
    discovered_fields,
    dynamic_fields,
):

    """
    Exact label/value evidence outranks fuzzy semantic matching.

    Only fields with known blind-test failure modes are
    overridden here.
    """

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        return dynamic_fields

    repairs = {
        "Customer GSTIN":
            "Customer GSTIN",

        "E-Way Bill Number":
            "E-Way Bill Number",

        "E Way Bill Number":
            "E Way Bill Number",

        "E-Way Bill No":
            "E-Way Bill No",

        "Amount In Words":
            "Amount In Words",

        "IRN":
            "IRN",

        "Ack Number":
            "Ack Number",

        "Vehicle Number":
            "Vehicle Number",

        "Delivery Note":
            "Delivery Note",
    }

    for field_name in (
        discovered_fields
        or
        []
    ):

        canonical = (
            _auto_normalize_label(
                field_name
            )
        )

        lookup_label = (
            repairs.get(
                canonical
            )
        )

        if not lookup_label:
            continue

        exact = (
            _v7_exact_labeled_value(
                input_path,
                lookup_label,
            )
        )

        if not exact:
            continue

        value = (
            exact[
                "value"
            ]
        )

        # ----------------------------------------------------
        # Full E-Way Bill values often contain visual spaces:
        #
        # 3210 6842 9157
        #
        # Keep all digits rather than only the first token.
        # ----------------------------------------------------

        if (
            canonical
            in {
                "E-Way Bill Number",
                "E Way Bill Number",
                "E-Way Bill No",
            }
        ):

            digit_groups = re.findall(
                r"\d+",
                value,
            )

            if digit_groups:

                joined = (
                    "".join(
                        digit_groups
                    )
                )

                if len(
                    joined
                ) >= 8:

                    value = (
                        joined
                    )

        dynamic_fields[
            canonical
        ] = {
            "value":
                value,

            "status":
                "DETECTED",

            "confidence":
                1.0,

            "page":
                exact.get(
                    "page"
                ),

            "source":
                "V7_EXACT_LABEL",

            "evidence":
                exact.get(
                    "evidence"
                ),
        }

    return dynamic_fields


def _v7_sync_dynamic_gst(
    production_result,
    discovered_fields,
    dynamic_fields,
):

    components = (
        _v7_find_gst_components(
            production_result
        )
    )

    for tax_type in (
        "CGST",
        "SGST",
        "IGST",
    ):

        if (
            tax_type
            not in
            (
                discovered_fields
                or
                []
            )
        ):
            continue

        component = (
            components.get(
                tax_type
            )
        )

        if not component:
            continue

        rate = (
            component.get(
                "rate"
            )
        )

        amount = (
            component.get(
                "amount"
            )
        )

        if rate is not None:

            value = (
                f"{float(rate):g}% / "
                f"{_v7_money(amount)}"
            )

        else:

            value = (
                _v7_money(
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
                "V7_GST_COMPONENT",

            "evidence":
                component.get(
                    "evidence"
                ),
        }

    return dynamic_fields


def _v7_reconcile_total(
    production_result,
):

    """
    Recompute validation only when enough reliable values exist.

    This does NOT replace the explicit TOTAL_AMOUNT.
    It only validates it.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    fields = (
        production_result.get(
            "fields",
            {}
        )
    )

    if not isinstance(
        fields,
        dict,
    ):
        return

    def field_number(
        field_name,
    ):

        information = (
            fields.get(
                field_name
            )
        )

        if isinstance(
            information,
            dict,
        ):

            return (
                _v7_number(
                    information.get(
                        "value"
                    )
                )
            )

        return (
            _v7_number(
                information
            )
        )

    subtotal = (
        field_number(
            "SUBTOTAL"
        )
    )

    discount = (
        field_number(
            "DISCOUNT"
        )
        or
        0.0
    )

    tax = (
        field_number(
            "TAX"
        )
    )

    total = (
        field_number(
            "TOTAL_AMOUNT"
        )
    )

    financial_details = (
        production_result.get(
            "financial_details",
            {}
        )
    )

    round_off = 0.0

    if isinstance(
        financial_details,
        dict,
    ):

        round_info = (
            financial_details.get(
                "round_off"
            )
        )

        if isinstance(
            round_info,
            dict,
        ):

            round_off = (
                _v7_number(
                    round_info.get(
                        "value"
                    )
                )
                or
                0.0
            )

    # Freight/charges may be outside trained schema, so we do
    # not pretend subtotal-discount+tax is a complete formula.
    #
    # Validation here is only updated when the current runtime
    # already provided a complete formula.

    validation = (
        production_result.get(
            "validation"
        )
    )

    if not isinstance(
        validation,
        dict,
    ):
        return

    financial = (
        validation.get(
            "financial_reconciliation"
        )
    )

    if not isinstance(
        financial,
        dict,
    ):
        return

    financial[
        "subtotal"
    ] = (
        subtotal
    )

    financial[
        "tax"
    ] = (
        tax
    )

    financial[
        "round_off"
    ] = (
        round_off
    )

    financial[
        "total_amount"
    ] = (
        total
    )


# ============================================================
# V8 STEP 2
# FINAL OUTPUT CONSISTENCY + DYNAMIC VALUE QUALITY
# ============================================================

_V8_NOT_DETECTED_VALUES = {
    "",
    "not_detected",
    "not detected",
    "none",
    "null",
}

_V8_BAD_DYNAMIC_VALUES = {
    "date",
    "description",
    "item description",
    "material description",
    "material",
    "material code",
    "amount",
    "line amount",
    "qty",
    "quantity",
    "rate",
    "unit price",
    "unit",
    "uom",
    "remarks",
    "remark",
    "basic total",
    "subtotal",
    "sub total",
    "total",
    "grand total",
    "bill amount",
    "invoice total",
    "tax",
    "cgst",
    "sgst",
    "igst",
    "hsn",
    "sac",
    "sl",
    "sl no",
    "serial no",
}


def _v8_field_payload(
    production_result,
    field_name,
):
    if not isinstance(
        production_result,
        dict,
    ):
        return None

    fields = production_result.get(
        "fields",
        {},
    )

    if not isinstance(
        fields,
        dict,
    ):
        return None

    information = fields.get(
        field_name
    )

    if isinstance(
        information,
        dict,
    ):
        return information

    return None


def _v8_field_is_missing(
    information,
):
    if not isinstance(
        information,
        dict,
    ):
        return True

    status = str(
        information.get(
            "status",
            "",
        )
    ).strip().upper()

    if status in {
        "NOT_PRESENT",
        "NOT_DETECTED",
        "MISSING",
        "ABSENT",
    }:
        return True

    value = information.get(
        "value"
    )

    if value is None:
        return True

    key = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip().casefold()

    return (
        key
        in
        _V8_NOT_DETECTED_VALUES
    )


def _v8_field_value(
    production_result,
    field_name,
):
    information = (
        _v8_field_payload(
            production_result,
            field_name,
        )
    )

    if _v8_field_is_missing(
        information
    ):
        return None

    return information.get(
        "value"
    )


def _v8_iso_date(
    value,
):
    if value is None:
        return None

    text = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    if not text:
        return None

    patterns = (
        (
            r"^(\d{2})[./-](\d{2})[./-](\d{4})$",
            "DMY",
        ),
        (
            r"^(\d{4})[./-](\d{2})[./-](\d{2})$",
            "YMD",
        ),
    )

    for pattern, mode in patterns:

        match = re.fullmatch(
            pattern,
            text,
        )

        if not match:
            continue

        if mode == "DMY":
            day, month, year = (
                match.groups()
            )

        else:
            year, month, day = (
                match.groups()
            )

        try:
            from datetime import date

            parsed = date(
                int(year),
                int(month),
                int(day),
            )

            return parsed.isoformat()

        except Exception:
            return None

    return None


def _v8_number_or_none(
    value,
):
    if value is None:
        return None

    return _v7_number(
        value
    )


def _v8_sync_normalized_from_final_fields(
    production_result,
):
    """
    The final trained/reconciled field map is the source of truth.

    This prevents stale runtime-normalized values from disagreeing
    with the final V7/V8 field values.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    normalized = (
        production_result.setdefault(
            "normalized",
            {},
        )
    )

    if not isinstance(
        normalized,
        dict,
    ):
        normalized = {}

        production_result[
            "normalized"
        ] = normalized

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    invoice_number = (
        _v8_field_value(
            production_result,
            "INVOICE_NUMBER",
        )
    )

    normalized[
        "invoice_number"
    ] = (
        str(invoice_number).strip()
        if invoice_number is not None
        else None
    )

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    invoice_date = (
        _v8_field_value(
            production_result,
            "INVOICE_DATE",
        )
    )

    due_date = (
        _v8_field_value(
            production_result,
            "DUE_DATE",
        )
    )

    normalized[
        "invoice_date"
    ] = (
        _v8_iso_date(
            invoice_date
        )
    )

    normalized[
        "due_date"
    ] = (
        _v8_iso_date(
            due_date
        )
        if due_date is not None
        else None
    )

    # --------------------------------------------------------
    # CURRENCY
    # --------------------------------------------------------

    currency = (
        _v8_field_value(
            production_result,
            "CURRENCY",
        )
    )

    normalized[
        "currency"
    ] = (
        str(currency).strip()
        if currency is not None
        else None
    )

    # --------------------------------------------------------
    # FINANCIAL VALUES
    # --------------------------------------------------------

    mapping = {
        "subtotal":
            "SUBTOTAL",

        "tax":
            "TAX",

        "discount":
            "DISCOUNT",

        "total_amount":
            "TOTAL_AMOUNT",
    }

    for (
        normalized_name,
        field_name,
    ) in mapping.items():

        value = (
            _v8_field_value(
                production_result,
                field_name,
            )
        )

        normalized[
            normalized_name
        ] = (
            _v8_number_or_none(
                value
            )
            if value is not None
            else None
        )


def _v8_clean_identifier_value(
    field_name,
    value,
):
    if value is None:
        return value

    text = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    if not text:
        return text

    name = str(
        field_name
    ).strip().casefold()

    # --------------------------------------------------------
    # CIN
    #
    # Example bad extraction:
    # NO.L17299WB1993PLC058969
    #
    # Desired:
    # L17299WB1993PLC058969
    # --------------------------------------------------------

    if name == "cin":

        match = re.search(
            r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b",
            text.upper(),
        )

        if match:
            return match.group(0)

    # --------------------------------------------------------
    # PAN
    # --------------------------------------------------------

    if name == "pan":

        match = re.search(
            r"\b[A-Z]{5}\d{4}[A-Z]\b",
            text.upper(),
        )

        if match:
            return match.group(0)

    # --------------------------------------------------------
    # GSTIN
    # --------------------------------------------------------

    if (
        name == "gstin"
        or
        "gstin"
        in name
    ):

        match = re.search(
            r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
            text.upper(),
        )

        if match:
            return match.group(0)

    return text


def _v8_clean_dynamic_values(
    dynamic_fields,
):
    """
    Conservative final dynamic-value cleanup.

    Legitimate detected values are preserved.

    Obvious structural/header values are not allowed to remain
    confidently DETECTED.
    """

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        return {}

    for (
        field_name,
        information,
    ) in list(
        dynamic_fields.items()
    ):

        if not isinstance(
            information,
            dict,
        ):
            continue

        value = information.get(
            "value"
        )

        cleaned_value = (
            _v8_clean_identifier_value(
                field_name,
                value,
            )
        )

        information[
            "value"
        ] = cleaned_value

        value_key = re.sub(
            r"\s+",
            " ",
            str(
                cleaned_value
                if cleaned_value is not None
                else ""
            ),
        ).strip().casefold()

        # ----------------------------------------------------
        # Never present obvious table/header fragments as a
        # confident automatically extracted value.
        # ----------------------------------------------------

        if (
            information.get(
                "status"
            )
            ==
            "DETECTED"
            and
            value_key
            in
            _V8_BAD_DYNAMIC_VALUES
        ):

            information[
                "value"
            ] = "NOT_DETECTED"

            information[
                "status"
            ] = "AMBIGUOUS"

            information[
                "confidence"
            ] = min(
                float(
                    information.get(
                        "confidence",
                        0.0,
                    )
                    or
                    0.0
                ),
                0.49,
            )

            information[
                "source"
            ] = "V8_VALUE_QUALITY_REJECT"

    return dynamic_fields


# ============================================================
# V8.1 CORE STRESS-TEST HARDENING
#
# Targets:
#   1. Explicit Due Dt / Due Date recovery
#   2. BILL TO customer + address precedence
#   3. Structured line-item fallback from trained arrays
#   4. Financial adjustment sanitization + reconciliation
#
# No retraining.
# No model-weight changes.
# No inference architecture changes.
# ============================================================


def _v81_document_lines(
    input_path,
):

    lines = (
        _dynamic_pdf_lines(
            input_path
        )
    )

    if not isinstance(
        lines,
        list,
    ):
        return []

    return [
        item
        for item
        in lines
        if isinstance(
            item,
            dict,
        )
    ]


def _v81_recover_due_date(
    input_path,
    production_result,
):

    """
    Recover explicit due-date aliases that the neural/runtime
    layer may miss.

    Examples:
      Due Date:
      Due Dt.:
      Due Dt:
      Payment Due:
      Due On:
    """

    lines = (
        _v81_document_lines(
            input_path
        )
    )

    date_pattern = (
        r"("
        r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
        r"|"
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"
        r")"
    )

    pattern = re.compile(
        r"(?i)"
        r"\b(?:"
        r"due\s*date"
        r"|due\s*dt\.?"
        r"|due\s*on"
        r"|payment\s*due"
        r")"
        r"\s*[:=\-]?\s*"
        +
        date_pattern
    )

    for line in lines:

        text = str(
            line.get(
                "text",
                "",
            )
        )

        match = pattern.search(
            text
        )

        if not match:
            continue

        value = (
            match.group(1)
            .strip()
        )

        _v7_set_field(
            production_result,
            "DUE_DATE",
            value,
            source=
                "V8_1_EXPLICIT_DUE_DATE",
            status=
                "RULE_RECOVERED",
        )

        return


def _v81_is_probable_address(
    value,
):

    text = re.sub(
        r"\s+",
        " ",
        str(
            value
            or
            ""
        ),
    ).strip()

    if len(text) < 8:
        return False

    lowered = (
        text.casefold()
    )

    if re.search(
        r"(?i)"
        r"\b("
        r"gstin|pan|cin|"
        r"invoice|inv\.?\s*no|"
        r"posting\s*date|"
        r"due\s*date|due\s*dt|"
        r"currency|"
        r"ship\s*to|"
        r"customer\s*gstin|"
        r"state\s*code|"
        r"po\s|p\.?\s*o\.?|"
        r"gr\s|delivery\s*note|"
        r"e[- ]?way"
        r")\b",
        lowered,
    ):
        return False

    address_tokens = (
        "road",
        "rd",
        "street",
        "st",
        "sector",
        "plot",
        "unit",
        "floor",
        "building",
        "plaza",
        "park",
        "estate",
        "nagar",
        "palayam",
        "road",
        "gurugram",
        "noida",
        "delhi",
        "mumbai",
        "kolkata",
        "chennai",
        "bangalore",
        "bengaluru",
        "haryana",
        "uttar pradesh",
        "west bengal",
    )

    if any(
        token
        in
        lowered
        for token
        in address_tokens
    ):
        return True

    if re.search(
        r"\b\d{5,6}\b",
        text,
    ):
        return True

    return False


def _v81_recover_bill_to(
    input_path,
    production_result,
):

    """
    Explicit BILL TO / BILLED TO sections outrank generic
    vendor-header address recovery.
    """

    lines = (
        _v81_document_lines(
            input_path
        )
    )

    for index, line in enumerate(
        lines
    ):

        text = re.sub(
            r"\s+",
            " ",
            str(
                line.get(
                    "text",
                    "",
                )
            ),
        ).strip()

        match = re.search(
            r"(?i)"
            r"\b(?:"
            r"bill\s*to"
            r"|billed\s*to"
            r")"
            r"\s*[:\-]?\s*"
            r"(.+)$",
            text,
        )

        if not match:
            continue

        customer = (
            match.group(1)
            .strip(
                " \t:;-"
            )
        )

        if (
            len(customer) >= 3
            and
            re.search(
                r"[A-Za-z]",
                customer,
            )
        ):

            customer = (
                _v7_trim_value_at_next_label(
                    customer
                )
            )

            if customer:

                _v7_set_field(
                    production_result,
                    "CUSTOMER_NAME",
                    customer,
                    source=
                        "V8_1_EXPLICIT_BILL_TO",
                    status=
                        "RULE_RECOVERED",
                )

        # ----------------------------------------------------
        # Prefer the first genuine address immediately below
        # BILL TO.
        # ----------------------------------------------------

        for next_index in range(
            index + 1,
            min(
                index + 4,
                len(lines),
            ),
        ):

            candidate = re.sub(
                r"\s+",
                " ",
                str(
                    lines[
                        next_index
                    ].get(
                        "text",
                        "",
                    )
                ),
            ).strip()

            if re.search(
                r"(?i)"
                r"\b("
                r"ship\s*to|"
                r"customer\s*gstin|"
                r"state\s*code|"
                r"place\s*of\s*supply"
                r")\b",
                candidate,
            ):

                continue

            if not _v81_is_probable_address(
                candidate
            ):
                continue

            _v7_set_field(
                production_result,
                "ADDRESS",
                candidate,
                source=
                    "V8_1_EXPLICIT_BILL_TO_ADDRESS",
                status=
                    "RULE_RECOVERED",
            )

            break

        return


def _v81_field_list(
    production_result,
    field_name,
):

    try:

        information = (
            production_result[
                "fields"
            ][
                field_name
            ]
        )

    except Exception:

        return []

    if isinstance(
        information,
        dict,
    ):

        value = information.get(
            "value"
        )

    else:

        value = information

    if isinstance(
        value,
        list,
    ):

        return value

    return []


def _v81_rebuild_line_items(
    production_result,
):

    """
    If the neural model produced aligned trained arrays but the
    runtime table builder returned [], reconstruct conservative
    structured rows from those arrays.

    Existing non-empty structured line_items are NEVER replaced.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    existing = (
        production_result.get(
            "line_items"
        )
    )

    if (
        isinstance(
            existing,
            list,
        )
        and
        existing
    ):
        return

    descriptions = (
        _v81_field_list(
            production_result,
            "LINE_ITEM_DESC",
        )
    )

    quantities = (
        _v81_field_list(
            production_result,
            "LINE_ITEM_QTY",
        )
    )

    unit_prices = (
        _v81_field_list(
            production_result,
            "LINE_ITEM_UNIT_PRICE",
        )
    )

    amounts = (
        _v81_field_list(
            production_result,
            "LINE_ITEM_AMOUNT",
        )
    )

    numeric_lengths = [
        len(values)
        for values
        in (
            quantities,
            unit_prices,
            amounts,
        )
        if isinstance(
            values,
            list,
        )
        and
        values
    ]

    if not numeric_lengths:
        return

    row_count = min(
        numeric_lengths
    )

    if (
        row_count <= 0
        or
        len(descriptions)
        <
        row_count
    ):
        return

    rows = []

    for index in range(
        row_count
    ):

        description = re.sub(
            r"\s+",
            " ",
            str(
                descriptions[
                    index
                ]
            ),
        ).strip()

        quantity = (
            quantities[
                index
            ]
            if index
            <
            len(quantities)
            else
            None
        )

        unit_price = (
            unit_prices[
                index
            ]
            if index
            <
            len(unit_prices)
            else
            None
        )

        amount = (
            amounts[
                index
            ]
            if index
            <
            len(amounts)
            else
            None
        )

        # ----------------------------------------------------
        # Reject obvious non-item text that leaked into the
        # neural description sequence.
        # ----------------------------------------------------

        if re.search(
            r"(?i)"
            r"\b("
            r"payment\s*mode|"
            r"payment\s*terms|"
            r"bank\s*name|"
            r"ifsc|"
            r"account\s*number|"
            r"amount\s*in\s*words|"
            r"remarks"
            r")\b",
            description,
        ):

            continue

        rows.append(
            {
                "line_number":
                    len(rows)
                    +
                    1,

                "material_code":
                    "NOT_DETECTED",

                "description":
                    description,

                "hsn_code":
                    "NOT_DETECTED",

                "uom":
                    "NOT_DETECTED",

                "quantity":
                    str(
                        quantity
                    ).strip(),

                "quantity_numeric":
                    _v7_number(
                        quantity
                    ),

                "unit_price":
                    _v7_number(
                        unit_price
                    ),

                "unit_price_source":
                    "V8_1_TRAINED_ARRAY_FALLBACK",

                "line_amount":
                    _v7_number(
                        amount
                    ),

                "line_amount_source":
                    "V8_1_TRAINED_ARRAY_FALLBACK",

                "page":
                    None,
            }
        )

    if not rows:
        return

    production_result[
        "line_items"
    ] = rows

    extraction = (
        production_result.get(
            "extraction_summary"
        )
    )

    if isinstance(
        extraction,
        dict,
    ):

        extraction[
            "line_item_mode"
        ] = (
            "TRAINED_ARRAY_FALLBACK_V8_1"
        )


def _v81_safe_adjustments(
    production_result,
    dynamic_fields=None,
):

    """
    V8.4 financial adjustment consolidation.

    Recover genuine invoice adjustments from trusted runtime
    layers while rejecting subtotal/tax/rollup rows.
    """

    if not isinstance(production_result, dict):
        return []

    details = production_result.get(
        "financial_details"
    )

    if not isinstance(details, dict):
        details = {}
        production_result[
            "financial_details"
        ] = details

    fields = production_result.get(
        "fields",
        {},
    )

    trained_discount = None

    if isinstance(fields, dict):

        discount_info = fields.get(
            "DISCOUNT"
        )

        if isinstance(
            discount_info,
            dict,
        ):
            trained_discount = _v7_number(
                discount_info.get(
                    "value"
                )
            )
        else:
            trained_discount = _v7_number(
                discount_info
            )

    reject_patterns = (
        "gst taxable base",
        "taxable base",
        "taxable value",
        "total tax",
        "cgst",
        "sgst",
        "igst",
        "subtotal",
        "sub total",
        "basic total",
        "grand total",
        "bill amount",
        "invoice total",
        "total amount",
        "amount in words",
        "total value in words",
    )

    dynamic_allow_patterns = (
        "discount",
        "rebate",
        "deduction",
        "freight",
        "carriage",
        "packing",
        "forwarding",
        "shipping",
        "handling",
        "insurance",
        "surcharge",
        "process charge",
        "process charges",
        "stitching charge",
        "stitching charges",
        "job work",
        "jobwork",
        "knitting",
        "fiber",
        "fibre",
        "other charge",
        "other charges",
        "loading charge",
        "unloading charge",
        "transport charge",
        "transportation charge",
    )

    negative_label_patterns = (
        "discount",
        "rebate",
        "deduction",
    )

    candidates = []

    def add_candidate(
        item,
        trusted=False,
    ):

        if not isinstance(
            item,
            dict,
        ):
            return

        label = re.sub(
            r"\s+",
            " ",
            str(
                item.get(
                    "label",
                    "",
                )
                or
                ""
            ),
        ).strip()

        if not label:
            return

        key = label.casefold()

        if any(
            pattern in key
            for pattern
            in reject_patterns
        ):
            return

        source = str(
            item.get(
                "source",
                "",
            )
            or
            ""
        ).strip()

        is_runtime_summary = (
            source.upper().startswith(
                "SUMMARY_ADJUSTMENT"
            )
        )

        if not (
            trusted
            or
            is_runtime_summary
            or
            any(
                pattern in key
                for pattern
                in dynamic_allow_patterns
            )
        ):
            return

        amount = _v7_number(
            item.get(
                "amount"
            )
        )

        if amount is None:
            amount = _v7_number(
                item.get(
                    "value"
                )
            )

        if amount is None:
            return

        if (
            trained_discount
            is not None
            and
            any(
                pattern in key
                for pattern
                in negative_label_patterns
            )
        ):
            return

        if (
            amount > 0
            and
            any(
                pattern in key
                for pattern
                in negative_label_patterns
            )
        ):
            amount = -abs(
                amount
            )

        normalized_item = dict(
            item
        )

        normalized_item[
            "label"
        ] = label

        normalized_item[
            "amount"
        ] = round(
            float(amount),
            2,
        )

        if not normalized_item.get(
            "source"
        ):
            normalized_item[
                "source"
            ] = (
                "V8_4_ADJUSTMENT_RECOVERY"
            )

        candidates.append(
            normalized_item
        )

    existing = details.get(
        "adjustments",
        [],
    )

    if isinstance(
        existing,
        list,
    ):
        for item in existing:
            add_candidate(
                item,
                trusted=True,
            )

    top_level = production_result.get(
        "adjustments",
        [],
    )

    if isinstance(
        top_level,
        list,
    ):
        for item in top_level:
            add_candidate(
                item,
                trusted=True,
            )

    audit = production_result.get(
        "v6_audit",
        {},
    )

    if isinstance(
        audit,
        dict,
    ):

        audit_adjustments = (
            audit.get(
                "adjustments",
                [],
            )
        )

        if isinstance(
            audit_adjustments,
            list,
        ):
            for item in audit_adjustments:
                add_candidate(
                    item,
                    trusted=True,
                )

    if isinstance(
        dynamic_fields,
        dict,
    ):

        for (
            label,
            info,
        ) in dynamic_fields.items():

            key = re.sub(
                r"\s+",
                " ",
                str(
                    label
                ).casefold(),
            ).strip()

            if not any(
                pattern in key
                for pattern
                in dynamic_allow_patterns
            ):
                continue

            if isinstance(
                info,
                dict,
            ):

                status = str(
                    info.get(
                        "status",
                        "",
                    )
                    or
                    ""
                ).upper()

                if status in {
                    "NOT_PRESENT",
                    "NOT_DETECTED",
                    "AMBIGUOUS",
                    "NO_EVIDENCE",
                }:
                    continue

                candidate = {
                    "label":
                        str(label),

                    "amount":
                        info.get(
                            "value"
                        ),

                    "page":
                        info.get(
                            "page"
                        ),

                    "row_text":
                        info.get(
                            "evidence"
                        ),

                    "source":
                        "V8_4_DYNAMIC_ADJUSTMENT",
                }

            else:

                candidate = {
                    "label":
                        str(label),

                    "amount":
                        info,

                    "source":
                        "V8_4_DYNAMIC_ADJUSTMENT",
                }

            add_candidate(
                candidate,
                trusted=False,
            )

    safe = []
    seen = set()

    for item in candidates:

        label_key = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(
                item.get(
                    "label",
                    "",
                )
            ).casefold(),
        ).strip()

        amount = _v7_number(
            item.get(
                "amount"
            )
        )

        if amount is None:
            continue

        signature = (
            label_key,
            round(
                float(amount),
                2,
            ),
        )

        if signature in seen:
            continue

        seen.add(
            signature
        )

        item[
            "amount"
        ] = round(
            float(amount),
            2,
        )

        safe.append(
            item
        )

    details[
        "adjustments"
    ] = safe

    details[
        "adjustment_total"
    ] = round(
        sum(
            (
                _v7_number(
                    item.get(
                        "amount"
                    )
                )
                or
                0.0
            )
            for item
            in safe
        ),
        2,
    )

    production_result[
        "adjustments"
    ] = [
        dict(item)
        for item
        in safe
    ]

    return safe


def _v81_reconcile_validation(
    production_result,
):

    """
    V8.4 final reconciliation with trained discount,
    recovered adjustments and round-off support.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    fields = production_result.get(
        "fields",
        {},
    )

    if not isinstance(
        fields,
        dict,
    ):
        return

    def field_number(name):

        information = fields.get(
            name
        )

        if isinstance(
            information,
            dict,
        ):
            return _v7_number(
                information.get(
                    "value"
                )
            )

        return _v7_number(
            information
        )

    def field_status(name):

        information = fields.get(
            name
        )

        if isinstance(
            information,
            dict,
        ):
            return str(
                information.get(
                    "status",
                    "",
                )
                or
                ""
            ).upper()

        return ""

    subtotal = field_number(
        "SUBTOTAL"
    )

    tax = field_number(
        "TAX"
    )

    discount = field_number(
        "DISCOUNT"
    )

    total = field_number(
        "TOTAL_AMOUNT"
    )

    details = production_result.get(
        "financial_details",
        {},
    )

    adjustment_total = 0.0
    round_off = 0.0
    adjustments = []

    if isinstance(
        details,
        dict,
    ):

        adjustment_total = (
            _v7_number(
                details.get(
                    "adjustment_total"
                )
            )
            or
            0.0
        )

        adjustments = details.get(
            "adjustments",
            [],
        )

        if not isinstance(
            adjustments,
            list,
        ):
            adjustments = []

        round_info = details.get(
            "round_off"
        )

        if isinstance(
            round_info,
            dict,
        ):
            round_off = (
                _v7_number(
                    round_info.get(
                        "value"
                    )
                )
                or
                0.0
            )
        else:
            round_off = (
                _v7_number(
                    round_info
                )
                or
                0.0
            )

    adjustment_has_discount = any(
        any(
            token in str(
                item.get(
                    "label",
                    "",
                )
            ).casefold()
            for token in (
                "discount",
                "rebate",
                "deduction",
            )
        )
        for item in adjustments
        if isinstance(
            item,
            dict,
        )
    )

    discount_effect = 0.0

    if (
        discount is not None
        and
        not adjustment_has_discount
    ):
        discount_effect = -abs(
            float(discount)
        )

    tax_for_math = tax

    if tax_for_math is None:

        tax_status = field_status(
            "TAX"
        )

        if tax_status in {
            "NOT_PRESENT",
            "NOT_DETECTED",
            "NO_EVIDENCE",
        }:
            tax_for_math = 0.0

    validation = production_result.get(
        "validation"
    )

    if not isinstance(
        validation,
        dict,
    ):
        return

    financial = validation.get(
        "financial_reconciliation"
    )

    financial_pass = None

    if isinstance(
        financial,
        dict,
    ):

        financial[
            "subtotal"
        ] = subtotal

        financial[
            "tax"
        ] = tax

        financial[
            "discount"
        ] = discount

        financial[
            "discount_effect"
        ] = round(
            discount_effect,
            2,
        )

        financial[
            "adjustments_total"
        ] = round(
            adjustment_total,
            2,
        )

        financial[
            "round_off"
        ] = round(
            round_off,
            2,
        )

        financial[
            "total_amount"
        ] = total

        if (
            subtotal is not None
            and
            tax_for_math is not None
            and
            total is not None
        ):

            calculated = round(
                subtotal
                +
                tax_for_math
                +
                discount_effect
                +
                adjustment_total
                +
                round_off,
                2,
            )

            difference = round(
                calculated
                -
                total,
                2,
            )

            financial_pass = (
                abs(
                    difference
                )
                <=
                0.50
            )

            financial[
                "calculated_total"
            ] = calculated

            financial[
                "difference"
            ] = difference

            financial[
                "passed"
            ] = financial_pass

    gst = validation.get(
        "gst_reconciliation"
    )

    gst_pass = None

    if isinstance(
        gst,
        dict,
    ):

        component_sum = _v7_number(
            gst.get(
                "component_sum"
            )
        )

        if (
            component_sum is not None
            and
            tax is not None
        ):

            gst_difference = round(
                component_sum
                -
                tax,
                2,
            )

            gst_pass = (
                abs(
                    gst_difference
                )
                <=
                0.05
            )

            gst[
                "difference"
            ] = gst_difference

            gst[
                "matches_tax_total"
            ] = gst_pass

    line_items = production_result.get(
        "line_items",
        [],
    )

    line_validation = validation.get(
        "line_item_reconciliation"
    )

    line_pass = None

    if (
        isinstance(
            line_items,
            list,
        )
        and
        isinstance(
            line_validation,
            dict,
        )
    ):

        amounts = []
        quantities = []

        for item in line_items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            amount = _v7_number(
                item.get(
                    "line_amount"
                )
            )

            quantity = _v7_number(
                item.get(
                    "quantity_numeric",
                    item.get(
                        "quantity"
                    ),
                )
            )

            if amount is not None:
                amounts.append(
                    amount
                )

            if quantity is not None:
                quantities.append(
                    quantity
                )

        line_sum = (
            round(
                sum(amounts),
                2,
            )
            if amounts
            else
            None
        )

        total_quantity = (
            round(
                sum(quantities),
                4,
            )
            if quantities
            else
            None
        )

        line_difference = None

        if (
            line_sum is not None
            and
            subtotal is not None
        ):

            line_difference = round(
                line_sum
                -
                subtotal,
                2,
            )

            line_pass = (
                abs(
                    line_difference
                )
                <=
                0.50
            )

        line_validation[
            "row_count"
        ] = len(
            [
                item
                for item
                in line_items
                if isinstance(
                    item,
                    dict,
                )
            ]
        )

        line_validation[
            "total_quantity"
        ] = total_quantity

        line_validation[
            "line_amount_sum"
        ] = line_sum

        line_validation[
            "matches_subtotal"
        ] = line_pass

        line_validation[
            "difference"
        ] = line_difference

    hard_failure = any(
        value is False
        for value in (
            financial_pass,
            gst_pass,
            line_pass,
        )
    )

    if hard_failure:

        validation[
            "overall_status"
        ] = "REVIEW_REQUIRED"

        production_result[
            "status"
        ] = "REVIEW_REQUIRED"

    else:

        current = str(
            validation.get(
                "overall_status",
                "",
            )
            or
            ""
        ).upper()

        if current not in {
            "FAIL",
            "FAILED",
            "REVIEW_REQUIRED",
        }:

            validation[
                "overall_status"
            ] = "PASS"

            if str(
                production_result.get(
                    "status",
                    "",
                )
                or
                ""
            ).upper() not in {
                "FAIL",
                "FAILED",
                "REVIEW_REQUIRED",
            }:

                production_result[
                    "status"
                ] = "PASS"


def _v82_repair_customer_name(
    input_path,
    production_result,
):
    lines = _v81_document_lines(input_path)

    for index, line in enumerate(lines):

        text = re.sub(
            r"\s+",
            " ",
            str(line.get("text", "")),
        ).strip()

        if not re.search(
            r"(?i)\bbill\s*to\b",
            text,
        ):
            continue

        trailing = re.sub(
            r"(?i)^.*?\bbill\s*to\b\s*[:\-]?\s*",
            "",
            text,
        ).strip()

        candidates = []

        if trailing:
            candidates.append(trailing)

        for next_index in range(
            index + 1,
            min(index + 4, len(lines)),
        ):
            candidates.append(
                re.sub(
                    r"\s+",
                    " ",
                    str(
                        lines[next_index].get(
                            "text",
                            "",
                        )
                    ),
                ).strip()
            )

        for candidate in candidates:

            if not candidate:
                continue

            if _v81_is_probable_address(candidate):
                continue

            if re.search(
                r"(?i)\b("
                r"customer\s*gstin|"
                r"state\s*code|"
                r"ship\s*to|"
                r"place\s*of\s*supply"
                r")\b",
                candidate,
            ):
                continue

            if not re.search(
                r"(?i)\b("
                r"private|pvt|limited|ltd|"
                r"enterprise|enterprises|"
                r"industries|industrial|"
                r"solutions|systems|"
                r"corporation|company"
                r")\b",
                candidate,
            ):
                continue

            _v7_set_field(
                production_result,
                "CUSTOMER_NAME",
                candidate,
                source="V8_2_EXPLICIT_BILL_TO_NAME",
                status="RULE_RECOVERED",
            )

            return


def _v82_trim_line_description_array(
    production_result,
):
    fields = production_result.get(
        "fields",
        {}
    )

    if not isinstance(fields, dict):
        return

    desc_info = fields.get(
        "LINE_ITEM_DESC"
    )

    if not isinstance(desc_info, dict):
        return

    descriptions = desc_info.get(
        "value"
    )

    if not isinstance(descriptions, list):
        return

    numeric_lengths = []

    for name in (
        "LINE_ITEM_QTY",
        "LINE_ITEM_UNIT_PRICE",
        "LINE_ITEM_AMOUNT",
    ):

        info = fields.get(name)

        if not isinstance(info, dict):
            continue

        values = info.get("value")

        if isinstance(values, list) and values:
            numeric_lengths.append(len(values))

    if not numeric_lengths:
        return

    row_count = min(numeric_lengths)

    if len(descriptions) > row_count:

        desc_info["value"] = (
            descriptions[:row_count]
        )

        desc_info["source"] = (
            "V8_2_ROW_COUNT_ALIGNED"
        )


def _v82_dynamic_payload(
    value,
    evidence,
    *,
    source="V8_2_EXPLICIT_CONTEXT",
):
    return {
        "value": value,
        "status": "DETECTED",
        "confidence": 1.0,
        "page": 1,
        "source": source,
        "evidence": evidence,
    }


def _v82_repair_dynamic_fields(
    input_path,
    discovered_fields,
    dynamic_fields,
):
    if not isinstance(dynamic_fields, dict):
        dynamic_fields = {}

    lines = _v81_document_lines(input_path)

    # Remove known malformed schema fragments.
    for bad_key in list(dynamic_fields.keys()):

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(bad_key).lower(),
        ).strip()

        if normalized in {
            "date old inv ref",
            "net 30 days payment mode",
            "hdfc bank ifsc code",
        }:
            dynamic_fields.pop(
                bad_key,
                None,
            )

    for line in lines:

        text = re.sub(
            r"\s+",
            " ",
            str(line.get("text", "")),
        ).strip()

        # ----------------------------------------------------
        # CIN
        # ----------------------------------------------------
        match = re.search(
            r"(?i)\bCIN\s*[:=]\s*"
            r"([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})",
            text,
        )

        if match:
            value = match.group(1).upper()

            dynamic_fields["CIN"] = (
                _v82_dynamic_payload(
                    value,
                    text,
                )
            )

        # ----------------------------------------------------
        # PO
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bP\s*\.?\s*O\s*\.?"
            r"\s*(?:Ref(?:erence)?|No|Number)?\.?"
            r"\s*[:=]\s*"
            r"([A-Z0-9][A-Z0-9./_\-]{2,})",
            text,
        )

        if match:
            value = match.group(1)

            dynamic_fields["PO Number"] = (
                _v82_dynamic_payload(
                    value,
                    text,
                )
            )

            dynamic_fields.pop(
                "PO Ref.",
                None,
            )

        # ----------------------------------------------------
        # E-WAY BILL
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bE[\-\s]?Way"
            r"(?:\s*Bill)?"
            r"\s*(?:No|Number)?\.?"
            r"\s*[:=]\s*"
            r"([0-9][0-9 \-]{8,20})",
            text,
        )

        if match:

            digits = re.sub(
                r"\D",
                "",
                match.group(1),
            )

            if 10 <= len(digits) <= 15:

                dynamic_fields[
                    "E-Way Bill Number"
                ] = (
                    _v82_dynamic_payload(
                        digits,
                        text,
                    )
                )

                dynamic_fields.pop(
                    "E-Way No.",
                    None,
                )

        # ----------------------------------------------------
        # TRANSPORTER GSTIN
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bTransporter\s*GSTIN"
            r"\s*[:=]\s*"
            r"([0-9]{2}[A-Z]{5}[0-9]{4}"
            r"[A-Z][1-9A-Z]Z[0-9A-Z])",
            text,
        )

        if match:

            dynamic_fields[
                "Transporter GSTIN"
            ] = (
                _v82_dynamic_payload(
                    match.group(1).upper(),
                    text,
                )
            )

        # ----------------------------------------------------
        # PAYMENT MODE
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bPayment\s*Mode\s*[:=]\s*"
            r"(.+)$",
            text,
        )

        if match:

            value = _v7_trim_value_at_next_label(
                match.group(1)
            )

            if value:
                dynamic_fields[
                    "Payment Mode"
                ] = (
                    _v82_dynamic_payload(
                        value,
                        text,
                    )
                )

        # ----------------------------------------------------
        # IFSC
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bIFSC\s*(?:Code)?\s*[:=]\s*"
            r"([A-Z]{4}0[A-Z0-9]{6})",
            text,
        )

        if match:

            dynamic_fields[
                "IFSC Code"
            ] = (
                _v82_dynamic_payload(
                    match.group(1).upper(),
                    text,
                )
            )

        # ----------------------------------------------------
        # VEHICLE CANONICALIZATION
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bVehicle(?:\s*(?:No|Number))?"
            r"\s*[:=]\s*"
            r"([A-Z]{2}[A-Z0-9\-]{4,})",
            text,
        )

        if match:

            dynamic_fields[
                "Vehicle Number"
            ] = (
                _v82_dynamic_payload(
                    match.group(1).upper(),
                    text,
                )
            )

            dynamic_fields.pop(
                "Vehicle",
                None,
            )

        # ----------------------------------------------------
        # ACK CANONICALIZATION
        # ----------------------------------------------------
        match = re.search(
            r"(?i)"
            r"\bAck(?:nowledgement)?"
            r"\s*(?:No|Number)?\.?"
            r"\s*[:=]\s*"
            r"(\d{10,20})",
            text,
        )

        if match:

            dynamic_fields[
                "Ack Number"
            ] = (
                _v82_dynamic_payload(
                    match.group(1),
                    text,
                )
            )

            dynamic_fields.pop(
                "Ack No.",
                None,
            )

    # Make discovery metadata agree with final cleaned schema.
    if isinstance(discovered_fields, list):
        discovered_fields[:] = list(
            dynamic_fields.keys()
        )

    return dynamic_fields


def _v82_sync_recovered_metadata(
    production_result,
):
    """
    Remove stale NOT_PRESENT bookkeeping after a field has been
    recovered by later V8 logic.
    """

    fields = production_result.get(
        "fields",
        {}
    )

    due_info = (
        fields.get("DUE_DATE")
        if isinstance(fields, dict)
        else None
    )

    due_value = (
        due_info.get("value")
        if isinstance(due_info, dict)
        else None
    )

    if not due_value or str(due_value) == "NOT_DETECTED":
        return

    validation = production_result.get(
        "validation"
    )

    if isinstance(validation, dict):

        absent = validation.get(
            "not_present_fields"
        )

        if isinstance(absent, list):
            validation[
                "not_present_fields"
            ] = [
                item
                for item in absent
                if item != "DUE_DATE"
            ]

    summary = production_result.get(
        "extraction_summary"
    )

    if isinstance(summary, dict):

        for key in (
            "not_present_fields",
            "confirmed_absent_fields",
        ):

            values = summary.get(key)

            if isinstance(values, list):
                summary[key] = [
                    item
                    for item in values
                    if item != "DUE_DATE"
                ]

        values = summary.get(
            "fields_with_values"
        )

        if isinstance(values, list):

            if "DUE_DATE" not in values:
                values.insert(
                    3,
                    "DUE_DATE",
                )

            summary[
                "fields_with_values_count"
            ] = len(values)

        confirmed = summary.get(
            "confirmed_absent_fields"
        )

        if isinstance(confirmed, list):

            summary[
                "confirmed_absent_field_count"
            ] = len(confirmed)

    # Remove stale warning when payment terms is actually present.
    payment = (
        fields.get("PAYMENT_TERMS")
        if isinstance(fields, dict)
        else None
    )

    payment_value = (
        payment.get("value")
        if isinstance(payment, dict)
        else None
    )

    if payment_value and str(payment_value) != "NOT_DETECTED":

        if isinstance(validation, dict):

            warnings = validation.get(
                "warnings"
            )

            if isinstance(warnings, list):

                validation["warnings"] = [
                    warning
                    for warning in warnings
                    if "PAYMENT_TERMS has no explicit evidence"
                    not in str(warning)
                ]



# ============================================================
# V8.3 FINAL OUTPUT POLISH
#
# Generic fixes only:
#   1. Phone-number recovery from explicit phone labels
#   2. Single-HSN propagation into structured line items
#
# No retraining.
# No model changes.
# ============================================================



# ============================================================
# V8.4 PRIORITY 2
# FINAL DYNAMIC FALSE-POSITIVE FILTER
# ============================================================

def _v84_filter_dynamic_false_positives(
    dynamic_fields,
):

    """
    Conservative dynamic-output precision layer.

    Removes obvious structural/table/address artifacts while
    preserving legitimate automatically discovered business
    identifiers and explicitly recovered fields.

    No trained-schema changes.
    No model changes.
    """

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        return {}

    structural_values = {
        "",
        "date",
        "description",
        "item description",
        "material description",
        "material",
        "material code",
        "quantity",
        "qty",
        "rate",
        "unit price",
        "amount",
        "line amount",
        "unit",
        "uom",
        "hsn",
        "sac",
        "remarks",
        "remark",
        "subtotal",
        "sub total",
        "total",
        "grand total",
        "bill amount",
        "invoice total",
        "tax",
        "cgst",
        "sgst",
        "igst",
        "sl",
        "sl no",
        "serial no",
    }

    identifier_fields = {
        "po number",
        "purchase order number",
        "purchase order",
        "gr number",
        "goods receipt number",
        "odn number",
        "reference number",
        "delivery challan number",
        "challan number",
        "vehicle number",
        "eway bill number",
        "e way bill number",
        "ack number",
        "acknowledgement number",
        "account number",
        "bank account number",
        "lot",
    }

    # Generic address fragments should not become arbitrary
    # dynamic schema fields merely because they contain "No".
    address_fragment_tokens = {
        "shop",
        "floor",
        "block",
        "market",
        "road",
        "street",
        "lane",
        "building",
        "plot",
        "house",
        "sector",
        "village",
        "district",
        "taluk",
        "taluka",
        "pincode",
        "pin",
        "postal",
    }

    # Explicit invoice/business fields that may legitimately
    # contain words which overlap with location terminology.
    protected_names = {
        "state code",
        "state name",
        "place of supply",
        "gstin",
        "customer gstin",
        "transporter gstin",
        "cin",
        "pan",
        "ifsc",
        "ifsc code",
        "email",
        "phone number",
        "posting date",
        "hsn code",
        "po number",
        "purchase order number",
        "gr number",
        "odn number",
        "eway bill number",
        "e way bill number",
        "vehicle number",
        "payment mode",
        "bank name",
        "bank account number",
        "account number",
        "lot",
        "irn",
        "ack number",
        "acknowledgement number",
        "cgst",
        "sgst",
        "igst",
        "freight charges",
        "round off",
        "total value in words",
        "amount in words",
    }

    remove_keys = []

    for (
        field_name,
        information,
    ) in list(
        dynamic_fields.items()
    ):

        name_key = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(
                field_name
            ).casefold(),
        ).strip()

        if not name_key:
            remove_keys.append(
                field_name
            )
            continue

        if isinstance(
            information,
            dict,
        ):

            value = information.get(
                "value"
            )

            status = str(
                information.get(
                    "status",
                    "",
                )
                or
                ""
            ).upper()

            source = str(
                information.get(
                    "source",
                    "",
                )
                or
                ""
            ).upper()

        else:

            value = information
            status = ""
            source = ""

        value_key = re.sub(
            r"\s+",
            " ",
            str(
                value
                if value is not None
                else ""
            ),
        ).strip().casefold()

        # ----------------------------------------------------
        # Rule 1:
        # Obvious table/header text cannot be a detected value.
        # ----------------------------------------------------

        if value_key in structural_values:

            if isinstance(
                information,
                dict,
            ):

                information[
                    "value"
                ] = "NOT_DETECTED"

                information[
                    "status"
                ] = "AMBIGUOUS"

                information[
                    "confidence"
                ] = min(
                    float(
                        information.get(
                            "confidence",
                            0.0,
                        )
                        or
                        0.0
                    ),
                    0.49,
                )

                information[
                    "source"
                ] = (
                    "V8_4_DYNAMIC_STRUCTURE_REJECT"
                )

            continue

        # ----------------------------------------------------
        # Rule 2:
        # Identifier fields must look like identifiers.
        #
        # Require at least one digit. This rejects:
        #
        #   PO Number -> MATERIAL
        #
        # while preserving:
        #
        #   PO Number -> 190260318
        #   PO Number -> PO-2026/1044
        # ----------------------------------------------------

        if (
            name_key
            in identifier_fields
            and
            status
            not in {
                "NOT_PRESENT",
                "NOT_DETECTED",
                "AMBIGUOUS",
                "NO_EVIDENCE",
            }
        ):

            if not any(
                character.isdigit()
                for character
                in str(
                    value
                    if value is not None
                    else ""
                )
            ):

                if isinstance(
                    information,
                    dict,
                ):

                    information[
                        "value"
                    ] = "NOT_DETECTED"

                    information[
                        "status"
                    ] = "AMBIGUOUS"

                    information[
                        "confidence"
                    ] = min(
                        float(
                            information.get(
                                "confidence",
                                0.0,
                            )
                            or
                            0.0
                        ),
                        0.49,
                    )

                    information[
                        "source"
                    ] = (
                        "V8_4_IDENTIFIER_VALUE_REJECT"
                    )

                continue

        # ----------------------------------------------------
        # Rule 3:
        # Remove address fragments discovered as arbitrary
        # dynamic labels.
        #
        # Example:
        #   SHOP NO
        #   CLOTH MARKETBLOCK NO
        #
        # Only applies to unprotected generic labels.
        # ----------------------------------------------------

        if (
            name_key
            not in protected_names
        ):

            name_tokens = set(
                name_key.split()
            )

            has_address_token = bool(
                name_tokens.intersection(
                    address_fragment_tokens
                )
            )

            has_generic_number_token = bool(
                name_tokens.intersection(
                    {
                        "no",
                        "number",
                    }
                )
            )

            if (
                has_address_token
                and
                has_generic_number_token
            ):

                remove_keys.append(
                    field_name
                )

                continue

        # ----------------------------------------------------
        # Rule 4:
        # Obvious structural evidence should not support an
        # identifier unless an explicit V8.2 context recovery
        # supplied it.
        # ----------------------------------------------------

        if (
            name_key
            in identifier_fields
            and
            source
            not in {
                "V8_2_EXPLICIT_CONTEXT",
                "V8_4_EXPLICIT_CONTEXT",
            }
            and
            isinstance(
                information,
                dict,
            )
        ):

            evidence = str(
                information.get(
                    "evidence",
                    "",
                )
                or
                ""
            ).strip().casefold()

            if evidence in {
                "material",
                "material description",
                "item description",
                "description",
                "quantity",
                "qty",
                "rate",
                "amount",
                "unit price",
            }:

                information[
                    "value"
                ] = "NOT_DETECTED"

                information[
                    "status"
                ] = "AMBIGUOUS"

                information[
                    "confidence"
                ] = min(
                    float(
                        information.get(
                            "confidence",
                            0.0,
                        )
                        or
                        0.0
                    ),
                    0.49,
                )

                information[
                    "source"
                ] = (
                    "V8_4_STRUCTURAL_EVIDENCE_REJECT"
                )

    for key in remove_keys:

        dynamic_fields.pop(
            key,
            None,
        )

    return dynamic_fields


def _v83_recover_phone_number(
    input_path,
    dynamic_fields,
):

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        dynamic_fields = {}

    lines = (
        _v81_document_lines(
            input_path
        )
    )

    pattern = re.compile(
        r"(?i)"
        r"\b(?:"
        r"ph(?:one)?\.?\s*(?:no|number)?"
        r"|mobile\s*(?:no|number)?"
        r"|tel(?:ephone)?\.?\s*(?:no|number)?"
        r")"
        r"\s*[:=\-]?\s*"
        r"("
        r"(?:\+?\d{1,3}[\s\-]?)?"
        r"(?:\d[\s\-]?){7,14}"
        r")"
    )

    for line in lines:

        text = re.sub(
            r"\s+",
            " ",
            str(
                line.get(
                    "text",
                    "",
                )
            ),
        ).strip()

        match = pattern.search(
            text
        )

        if not match:
            continue

        value = (
            match.group(1)
            .strip(
                " \t,;|"
            )
        )

        digits = re.sub(
            r"\D",
            "",
            value,
        )

        if len(digits) < 8:
            continue

        dynamic_fields[
            "Phone Number"
        ] = {
            "value":
                value,

            "status":
                "DETECTED",

            "confidence":
                1.0,

            "page":
                line.get(
                    "page"
                ),

            "source":
                "V8_3_EXPLICIT_PHONE_CONTEXT",

            "evidence":
                text,
        }

        return dynamic_fields

    return dynamic_fields


def _v83_detected_dynamic_value(
    dynamic_fields,
    names,
):

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        return None

    normalized_names = {
        re.sub(
            r"[^a-z0-9]+",
            "",
            str(name).lower(),
        )
        for name
        in names
    }

    for key, information in (
        dynamic_fields.items()
    ):

        normalized_key = re.sub(
            r"[^a-z0-9]+",
            "",
            str(key).lower(),
        )

        if normalized_key not in normalized_names:
            continue

        if not isinstance(
            information,
            dict,
        ):
            continue

        status = str(
            information.get(
                "status",
                "",
            )
        ).upper()

        value = information.get(
            "value"
        )

        if status != "DETECTED":
            continue

        if value in {
            None,
            "",
            "NOT_DETECTED",
        }:
            continue

        return str(
            value
        ).strip()

    return None


def _v83_propagate_single_hsn(
    production_result,
    dynamic_fields,
):

    """
    V8.5 safe HSN reconciliation.

    Strategy:
      1. Never overwrite an existing explicit HSN.
      2. Detect whether structured material_code values actually
         represent row-specific HSN codes.
      3. If multiple distinct HSN-like row codes exist and the
         dynamic HSN matches one of them, map row-by-row.
      4. Propagate one global dynamic HSN only when there is no
         conflicting row-level HSN evidence.

    This prevents the old failure where one HSN was copied into
    every row of an invoice containing different HSN codes.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    line_items = production_result.get(
        "line_items"
    )

    if not isinstance(
        line_items,
        list,
    ) or not line_items:
        return

    dynamic_hsn = (
        _v83_detected_dynamic_value(
            dynamic_fields,
            (
                "HSN Code",
                "HSN",
                "HSN/SAC",
                "HSN SAC",
                "SAC Code",
            ),
        )
    )

    normalized_dynamic = None

    if dynamic_hsn:

        candidate = re.sub(
            r"\D",
            "",
            str(dynamic_hsn),
        )

        if len(candidate) in {
            4,
            6,
            8,
        }:
            normalized_dynamic = candidate

    def valid_hsn(value):

        digits = re.sub(
            r"\D",
            "",
            str(
                value
                if value is not None
                else ""
            ),
        )

        if len(digits) in {
            4,
            6,
            8,
        }:
            return digits

        return None

    missing_tokens = {
        "",
        "not_detected",
        "not detected",
        "none",
        "null",
    }

    material_hsn_values = []

    for item in line_items:

        if not isinstance(
            item,
            dict,
        ):
            continue

        material_candidate = valid_hsn(
            item.get(
                "material_code"
            )
        )

        if material_candidate:
            material_hsn_values.append(
                material_candidate
            )

    distinct_material_hsn = set(
        material_hsn_values
    )

    changed = False
    row_aware = False

    # --------------------------------------------------------
    # Strong row-aware evidence:
    #
    # More than one different 4/6/8-digit code exists in the
    # structured code column AND the discovered HSN is one of
    # those codes.
    #
    # This is the NOVA-type case.
    # --------------------------------------------------------

    if (
        normalized_dynamic
        and
        len(
            distinct_material_hsn
        )
        >=
        2
        and
        normalized_dynamic
        in
        distinct_material_hsn
    ):

        for item in line_items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            current = str(
                item.get(
                    "hsn_code",
                    "",
                )
                or
                ""
            ).strip().casefold()

            if current not in missing_tokens:
                continue

            row_hsn = valid_hsn(
                item.get(
                    "material_code"
                )
            )

            if not row_hsn:
                continue

            item[
                "hsn_code"
            ] = row_hsn

            item[
                "hsn_source"
            ] = (
                "V8_5_ROW_AWARE_CODE_MAPPING"
            )

            changed = True
            row_aware = True

    # --------------------------------------------------------
    # Global HSN propagation is allowed only when no conflicting
    # row-level HSN-like evidence exists.
    # --------------------------------------------------------

    elif normalized_dynamic:

        conflicting_row_codes = (
            len(
                distinct_material_hsn
            )
            >=
            2
        )

        if not conflicting_row_codes:

            for item in line_items:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                current = str(
                    item.get(
                        "hsn_code",
                        "",
                    )
                    or
                    ""
                ).strip().casefold()

                if current not in missing_tokens:
                    continue

                item[
                    "hsn_code"
                ] = normalized_dynamic

                item[
                    "hsn_source"
                ] = (
                    "V8_5_SAFE_GLOBAL_HSN"
                )

                changed = True

    extraction = (
        production_result.get(
            "extraction_summary"
        )
    )

    if (
        changed
        and
        isinstance(
            extraction,
            dict,
        )
    ):

        extraction[
            "hsn_line_item_propagation"
        ] = True

        extraction[
            "hsn_assignment_mode"
        ] = (
            "ROW_AWARE"
            if row_aware
            else
            "SAFE_GLOBAL"
        )



# ============================================================
# V8.5 PRIORITY 4
# EXPLICIT COUNTERPARTY RECOVERY
# ============================================================

def _v85_name_key(
    value,
):

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(
            value
            if value is not None
            else ""
        ).casefold(),
    ).strip()


def _v85_field_value(
    production_result,
    name,
):

    if not isinstance(
        production_result,
        dict,
    ):
        return None

    fields = production_result.get(
        "fields",
        {},
    )

    if not isinstance(
        fields,
        dict,
    ):
        return None

    information = fields.get(
        name
    )

    if isinstance(
        information,
        dict,
    ):
        return information.get(
            "value"
        )

    return information


def _v85_missing_value(
    value,
):

    return str(
        value
        if value is not None
        else ""
    ).strip().casefold() in {
        "",
        "not_detected",
        "not detected",
        "none",
        "null",
    }


def _v85_recover_customer_name(
    input_path,
    production_result,
):

    """
    Recover customer/buyer/consignee only from explicit document
    anchors.

    Unlike V8.2, company suffix words such as Pvt/Ltd are helpful
    but not mandatory.

    Existing good customer values are preserved.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    current = _v85_field_value(
        production_result,
        "CUSTOMER_NAME",
    )

    vendor = _v85_field_value(
        production_result,
        "VENDOR_NAME",
    )

    current_key = _v85_name_key(
        current
    )

    vendor_key = _v85_name_key(
        vendor
    )

    current_missing = (
        _v85_missing_value(
            current
        )
    )

    current_is_vendor = False

    if (
        current_key
        and
        vendor_key
    ):

        current_is_vendor = (
            difflib.SequenceMatcher(
                None,
                current_key,
                vendor_key,
            ).ratio()
            >=
            0.86
        )

    # Do not disturb an already-good, distinct customer.
    if (
        not current_missing
        and
        not current_is_vendor
    ):
        return

    lines = _v81_document_lines(
        input_path
    )

    anchor_pattern = re.compile(
        r"(?i)"
        r"\b("
        r"bill(?:ed)?\s*to|"
        r"buyer(?:\s*details)?|"
        r"customer(?:\s*name|\s*details)?|"
        r"consignee|"
        r"ship\s*to"
        r")\b"
    )

    stop_pattern = re.compile(
        r"(?i)"
        r"\b("
        r"gstin|pan|cin|"
        r"state\s*code|"
        r"place\s*of\s*supply|"
        r"invoice\s*(?:no|number)|"
        r"po\s*(?:no|number)|"
        r"address|"
        r"phone|mobile|email"
        r")\b"
    )

    candidates = []

    for index, line in enumerate(
        lines
    ):

        raw = re.sub(
            r"\s+",
            " ",
            str(
                line.get(
                    "text",
                    "",
                )
            ),
        ).strip()

        anchor_match = anchor_pattern.search(
            raw
        )

        if not anchor_match:
            continue

        trailing = raw[
            anchor_match.end():
        ].strip(
            " :-|"
        )

        local_candidates = []

        if trailing:
            local_candidates.append(
                trailing
            )

        for next_index in range(
            index + 1,
            min(
                index + 4,
                len(lines),
            ),
        ):

            next_text = re.sub(
                r"\s+",
                " ",
                str(
                    lines[
                        next_index
                    ].get(
                        "text",
                        "",
                    )
                ),
            ).strip()

            if next_text:
                local_candidates.append(
                    next_text
                )

        for candidate in local_candidates:

            candidate = candidate.strip(
                " :-|"
            )

            if len(candidate) < 3:
                continue

            if stop_pattern.search(
                candidate
            ):
                continue

            if _v81_is_probable_address(
                candidate
            ):
                continue

            if not re.search(
                r"[A-Za-z]{3}",
                candidate,
            ):
                continue

            digit_count = sum(
                character.isdigit()
                for character
                in candidate
            )

            if (
                digit_count
                >
                max(
                    3,
                    len(candidate)
                    // 3,
                )
            ):
                continue

            key = _v85_name_key(
                candidate
            )

            if not key:
                continue

            if (
                vendor_key
                and
                difflib.SequenceMatcher(
                    None,
                    key,
                    vendor_key,
                ).ratio()
                >=
                0.86
            ):
                continue

            score = 0

            score += min(
                len(candidate),
                80,
            )

            if re.search(
                r"(?i)\b("
                r"private|pvt|limited|ltd|"
                r"enterprise|enterprises|"
                r"industries|industrial|"
                r"company|corporation|"
                r"mills|textile|textiles|"
                r"dyeing|traders|trading|"
                r"agency|agencies|"
                r"solutions|systems|"
                r"transport"
                r")\b",
                candidate,
            ):
                score += 30

            candidates.append(
                (
                    score,
                    candidate,
                )
            )

        # Prefer the first explicit customer block.
        if candidates:
            break

    if not candidates:
        return

    candidates.sort(
        key=lambda item: (
            item[0],
            len(item[1]),
        ),
        reverse=True,
    )

    best = candidates[0][1]

    _v7_set_field(
        production_result,
        "CUSTOMER_NAME",
        best,
        source=
            "V8_5_EXPLICIT_COUNTERPARTY",
        status=
            "RULE_RECOVERED",
    )


# ============================================================
# V8.5 PRIORITY 5
# ADDRESS RECONSTRUCTION / NORMALIZATION
# ============================================================

def _v85_address_compact(
    value,
):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(
            value
            if value is not None
            else ""
        ).casefold(),
    )


def _v85_normalize_address(
    input_path,
    production_result,
):

    """
    Reconstruct a concatenated ADDRESS using the original native
    PDF line boundaries whenever the document provides sufficient
    supporting evidence.

    No invented address components are added.
    """

    if not isinstance(
        production_result,
        dict,
    ):
        return

    current = _v85_field_value(
        production_result,
        "ADDRESS",
    )

    if _v85_missing_value(
        current
    ):
        return

    current = re.sub(
        r"\s+",
        " ",
        str(current),
    ).strip()

    current = re.sub(
        r",(?=\S)",
        ", ",
        current,
    )

    current_compact = (
        _v85_address_compact(
            current
        )
    )

    if len(current_compact) < 8:
        return

    lines = _v81_document_lines(
        input_path
    )

    reject = re.compile(
        r"(?i)"
        r"\b("
        r"gstin|pan|cin|ifsc|"
        r"invoice\s*(?:no|number)|"
        r"po\s*(?:no|number)|"
        r"taxable|cgst|sgst|igst|"
        r"subtotal|grand\s*total|"
        r"bill\s*amount|"
        r"amount\s*in\s*words"
        r")\b"
    )

    pieces = []
    seen = set()

    for line in lines:

        candidate = re.sub(
            r"\s+",
            " ",
            str(
                line.get(
                    "text",
                    "",
                )
            ),
        ).strip(
            " ,;|"
        )

        if len(candidate) < 4:
            continue

        if reject.search(
            candidate
        ):
            continue

        compact = (
            _v85_address_compact(
                candidate
            )
        )

        if len(compact) < 4:
            continue

        # Use only native lines whose content is genuinely part
        # of the already-extracted address.
        if compact not in current_compact:
            continue

        key = compact

        if key in seen:
            continue

        seen.add(
            key
        )

        pieces.append(
            candidate
        )

    if not pieces:

        # At minimum clean comma spacing.
        if current != str(
            _v85_field_value(
                production_result,
                "ADDRESS",
            )
        ):

            _v7_set_field(
                production_result,
                "ADDRESS",
                current,
                source=
                    "V8_5_ADDRESS_NORMALIZED",
                status=
                    "RULE_RECOVERED",
            )

        return

    reconstructed = ", ".join(
        pieces
    )

    reconstructed = re.sub(
        r"\s+",
        " ",
        reconstructed,
    ).strip(
        " ,"
    )

    recovered_compact = (
        _v85_address_compact(
            reconstructed
        )
    )

    # The reconstruction must cover most of the original address
    # so that we cannot accidentally replace it with one short
    # matching fragment.
    coverage = (
        len(
            recovered_compact
        )
        /
        max(
            len(
                current_compact
            ),
            1,
        )
    )

    if (
        coverage
        >=
        0.72
        and
        len(reconstructed)
        >=
        8
    ):

        _v7_set_field(
            production_result,
            "ADDRESS",
            reconstructed,
            source=
                "V8_5_NATIVE_ADDRESS_RECONSTRUCTION",
            status=
                "RULE_RECOVERED",
        )

    elif current != str(
        _v85_field_value(
            production_result,
            "ADDRESS",
        )
    ):

        _v7_set_field(
            production_result,
            "ADDRESS",
            current,
            source=
                "V8_5_ADDRESS_NORMALIZED",
            status=
                "RULE_RECOVERED",
        )


# ============================================================
# V8.5 PRIORITY 6
# CONSERVATIVE UOM ENRICHMENT
# ============================================================

_V85_UOM_ALIASES = {
    "KG": "KG",
    "KGS": "KG",
    "KGM": "KG",
    "PCS": "PCS",
    "PC": "PCS",
    "NOS": "NOS",
    "NO": "NOS",
    "EA": "EA",
    "EACH": "EA",
    "MTR": "MTR",
    "MTRS": "MTR",
    "MTS": "MTR",
    "METER": "MTR",
    "METERS": "MTR",
    "LTR": "LTR",
    "LTRS": "LTR",
    "LITRE": "LTR",
    "LITRES": "LTR",
    "BOX": "BOX",
    "BOXES": "BOX",
    "BAG": "BAG",
    "BAGS": "BAG",
    "SET": "SET",
    "SETS": "SET",
    "ROLL": "ROLL",
    "ROLLS": "ROLL",
    "MT": "MT",
    "TON": "TON",
    "TONNE": "TON",
}


def _v85_quantity_text_variants(
    value,
):

    number = _v7_number(
        value
    )

    variants = set()

    raw = str(
        value
        if value is not None
        else ""
    ).strip()

    if raw:
        variants.add(
            raw.replace(
                ",",
                "",
            )
        )

    if number is not None:

        variants.add(
            str(number)
        )

        variants.add(
            f"{number:.2f}"
        )

        variants.add(
            f"{number:.3f}"
        )

    return {
        item.rstrip(
            "0"
        ).rstrip(
            "."
        )
        if
        "."
        in item
        else
        item
        for item in variants
        if item
    }


def _v85_extract_uom_near_quantity(
    text,
    quantity,
):

    """
    Accept a UOM only when it is adjacent to the row quantity.
    This avoids treating random description words as units.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    upper = normalized.upper()

    aliases = "|".join(
        sorted(
            (
                re.escape(key)
                for key
                in _V85_UOM_ALIASES
            ),
            key=len,
            reverse=True,
        )
    )

    for quantity_variant in (
        _v85_quantity_text_variants(
            quantity
        )
    ):

        escaped_qty = re.escape(
            quantity_variant
        )

        patterns = (
            rf"\b({aliases})\b\s*{escaped_qty}\b",
            rf"\b{escaped_qty}\b\s*\b({aliases})\b",
        )

        for pattern in patterns:

            match = re.search(
                pattern,
                upper,
            )

            if match:

                raw_uom = (
                    match.group(1)
                    .upper()
                )

                return (
                    _V85_UOM_ALIASES.get(
                        raw_uom
                    )
                )

    return None


def _v85_enrich_line_item_uom(
    input_path,
    production_result,
):

    if not isinstance(
        production_result,
        dict,
    ):
        return

    line_items = production_result.get(
        "line_items"
    )

    if not isinstance(
        line_items,
        list,
    ):
        return

    lines = _v81_document_lines(
        input_path
    )

    if not lines:
        return

    changed = False

    for item in line_items:

        if not isinstance(
            item,
            dict,
        ):
            continue

        current = str(
            item.get(
                "uom",
                "",
            )
            or
            ""
        ).strip().casefold()

        if current not in {
            "",
            "not_detected",
            "not detected",
            "none",
            "null",
        }:
            continue

        description = re.sub(
            r"\s+",
            " ",
            str(
                item.get(
                    "description",
                    "",
                )
            ),
        ).strip()

        material = str(
            item.get(
                "material_code",
                "",
            )
            or
            ""
        ).strip()

        quantity = item.get(
            "quantity",
            item.get(
                "quantity_numeric"
            ),
        )

        description_key = (
            _v85_name_key(
                description
            )
        )

        candidates = []

        for line in lines:

            line_text = re.sub(
                r"\s+",
                " ",
                str(
                    line.get(
                        "text",
                        "",
                    )
                ),
            ).strip()

            line_key = _v85_name_key(
                line_text
            )

            relevant = False

            if (
                material
                and
                material
                not in {
                    "NOT_DETECTED",
                    "None",
                    "null",
                }
                and
                material
                in line_text
            ):
                relevant = True

            if (
                not relevant
                and
                description_key
                and
                len(
                    description_key
                )
                >=
                5
                and
                description_key
                in
                line_key
            ):
                relevant = True

            if relevant:

                uom = (
                    _v85_extract_uom_near_quantity(
                        line_text,
                        quantity,
                    )
                )

                if uom:
                    candidates.append(
                        uom
                    )

        unique = set(
            candidates
        )

        # Only assign if document evidence points to one
        # unambiguous UOM for that row.
        if len(unique) == 1:

            item[
                "uom"
            ] = next(
                iter(unique)
            )

            item[
                "uom_source"
            ] = (
                "V8_5_NATIVE_ROW_UOM"
            )

            changed = True

    if changed:

        extraction = (
            production_result.get(
                "extraction_summary"
            )
        )

        if isinstance(
            extraction,
            dict,
        ):

            extraction[
                "uom_line_item_enrichment"
            ] = True


def _v7_harden_result(
    input_path,
    production_result,
    discovered_fields,
    dynamic_fields,
):

    """
    Final generic post-inference quality layer.

    No retraining.
    No model reload.
    No frontend schema.
    """

    _v7_normalize_total_amount(
        production_result
    )

    _v7_repair_subtotal(
        input_path,
        production_result,
    )

    _v7_repair_tax(
        production_result
    )

    _v7_repair_customer_address(
        input_path,
        production_result,
    )

    dynamic_fields = (
        _v7_repair_dynamic_exact_fields(
            input_path,
            discovered_fields,
            dynamic_fields,
        )
    )

    dynamic_fields = (
        _v7_sync_dynamic_gst(
            production_result,
            discovered_fields,
            dynamic_fields,
        )
    )

    # ========================================================
    # V8 STEP 2
    # Dynamic identifier cleanup + value quality validation.
    # ========================================================

    dynamic_fields = (
        _v8_clean_dynamic_values(
            dynamic_fields
        )
    )

    # ========================================================
    # V8.2 DYNAMIC / ENTITY PRECISION
    # ========================================================

    dynamic_fields = (
        _v82_repair_dynamic_fields(
            input_path,
            discovered_fields,
            dynamic_fields,
        )
    )

    # ========================================================
    # V8.4 PRIORITY 2
    # Dynamic false-positive filtering.
    # ========================================================

    dynamic_fields = (
        _v84_filter_dynamic_false_positives(
            dynamic_fields
        )
    )

    # ========================================================
    # V8.3 FINAL OUTPUT POLISH
    # ========================================================

    dynamic_fields = (
        _v83_recover_phone_number(
            input_path,
            dynamic_fields,
        )
    )

    # Keep discovery metadata aligned with the final
    # dynamically repaired field map.
    if isinstance(
        discovered_fields,
        list,
    ):
        discovered_fields[:] = list(
            dynamic_fields.keys()
        )

    # ========================================================
    # V8.1 CORE STRESS-TEST HARDENING
    # ========================================================

    _v81_recover_due_date(
        input_path,
        production_result,
    )

    _v81_recover_bill_to(
        input_path,
        production_result,
    )

    _v82_repair_customer_name(
        input_path,
        production_result,
    )

    _v82_trim_line_description_array(
        production_result
    )

    _v81_rebuild_line_items(
        production_result
    )

    # ========================================================
    # V8.5 PRIORITIES 4 + 5
    # Counterparty + address quality.
    # ========================================================

    _v85_recover_customer_name(
        input_path,
        production_result,
    )

    _v85_normalize_address(
        input_path,
        production_result,
    )

    # ========================================================
    # V8.5 PRIORITIES 3 + 6
    # Safe HSN + native row UOM enrichment.
    # ========================================================

    _v83_propagate_single_hsn(
        production_result,
        dynamic_fields,
    )

    _v85_enrich_line_item_uom(
        input_path,
        production_result,
    )

    _v81_safe_adjustments(
        production_result,
        dynamic_fields,
    )

    _v7_reconcile_total(
        production_result
    )

    _v81_reconcile_validation(
        production_result
    )

    # ========================================================
    # V8 STEP 2
    # Final trained/reconciled fields are the source of truth
    # for the normalized JSON section.
    # ========================================================

    _v8_sync_normalized_from_final_fields(
        production_result
    )

    _v82_sync_recovered_metadata(
        production_result
    )

    return (
        production_result,
        dynamic_fields,
    )





def process_invoice_auto_dynamic(
    input_path,
    *,
    min_dynamic_confidence=
        DYNAMIC_MIN_CONFIDENCE,
):

    """
    FINAL CLIENT-FACING INFERENCE ENTRY POINT.

    The caller supplies only the document path.

    Pipeline:
      1. V3 + V6.1 production inference
      2. document-only automatic schema discovery
      3. generic dynamic value extraction
      4. existing HSN / GST cleanup
      5. V7 financial + anchor hardening
      6. one combined result

    No frontend-provided dynamic schema is required.
    No retraining is performed.
    """

    engine = (
        load_v3_production_engine()
    )

    process_invoice_final = (
        engine[
            "process_invoice_final"
        ]
    )

    # --------------------------------------------------------
    # VERIFIED PRODUCTION MODEL
    # --------------------------------------------------------

    production_result = (
        process_invoice_final(
            str(
                input_path
            )
        )
    )

    # --------------------------------------------------------
    # DOCUMENT-ONLY DYNAMIC SCHEMA DISCOVERY
    # --------------------------------------------------------

    discovered_fields = (
        discover_dynamic_fields(
            input_path,
            production_result=
                production_result,
        )
    )

    # --------------------------------------------------------
    # GENERIC DYNAMIC VALUE EXTRACTION
    #
    # production_result is still useful as supporting value
    # evidence here. It is NOT used to invent schema labels.
    # --------------------------------------------------------

    dynamic_fields = (
        extract_dynamic_parameters(
            input_path,
            discovered_fields,
            production_result=
                production_result,
            min_confidence=
                min_dynamic_confidence,
        )
    )

    # --------------------------------------------------------
    # EXISTING HSN / GST DYNAMIC CLEANUP
    # --------------------------------------------------------

    dynamic_fields = (
        _auto_cleanup_dynamic_fields(
            input_path,
            discovered_fields,
            dynamic_fields,
            production_result,
        )
    )

    # --------------------------------------------------------
    # V7 BLIND-TEST HARDENING
    # --------------------------------------------------------

    (
        production_result,
        dynamic_fields,
    ) = (
        _v7_harden_result(
            input_path,
            production_result,
            discovered_fields,
            dynamic_fields,
        )
    )

    return {
        "production_result":
            production_result,

        "dynamic_fields":
            dynamic_fields,

        "auto_discovered_parameters":
            discovered_fields,

        "dynamic_requested":
            [],

        "manual_parameters":
            [],

        "dynamic_schema_mode":
            AUTO_DYNAMIC_SCHEMA_MODE,

        "dynamic_discovery_mode":
            AUTO_DYNAMIC_SCHEMA_MODE,

        "trained_schema_fields":
            list(
                EXPECTED_FIELDS
            ),

        "runtime_quality_layer":
            "V7_BLIND_TEST_HARDENING",
    }




# ============================================================
# MODEL VERIFICATION
# ============================================================

def verify_model():

    print(
        "="
        *
        72
    )

    print(
        "INVOICE AI V3 — MODEL VERIFICATION"
    )

    print(
        "="
        *
        72
    )

    engine = (
        load_v3_model()
    )

    print(
        "\n✅ FINAL V3 MODEL READY"
    )

    print(
        "Architecture :",
        engine[
            "model"
        ].__class__.__name__,
    )

    print(
        "Parameters   :",
        f"{engine['parameter_count']:,}",
    )

    print(
        "Fields       :",
        len(
            engine[
                "fields"
            ]
        ),
    )

    print(
        "BIO Labels   :",
        len(
            engine[
                "label_list"
            ]
        ),
    )

    print(
        "Device       :",
        engine[
            "device"
        ],
    )

    print(
        "\n"
        +
        "="
        *
        72
    )

    print(
        "🔥 FINAL V3 MODEL VERIFIED"
    )

    print(
        "="
        *
        72
    )


# ============================================================
# COMPLETE PRODUCTION VERIFICATION
# ============================================================

def verify_production(
    rebuild_runtime=False,
):

    print(
        "="
        *
        72
    )

    print(
        "INVOICE AI V3 — "
        "COMPLETE PRODUCTION VERIFICATION"
    )

    print(
        "="
        *
        72
    )

    engine = (
        load_v3_production_engine(
            force_runtime_refresh=
                rebuild_runtime
        )
    )

    print(
        "\n✅ FINAL V3 MODEL"
    )

    print(
        "Architecture :",
        engine[
            "model"
        ].__class__.__name__,
    )

    print(
        "Parameters   :",
        f"{engine['parameter_count']:,}",
    )

    print(
        "Fields       :",
        len(
            engine[
                "fields"
            ]
        ),
    )

    print(
        "BIO Labels   :",
        len(
            engine[
                "label_list"
            ]
        ),
    )

    print(
        "Device       :",
        engine[
            "device"
        ],
    )

    print(
        "\n✅ PRODUCTION RUNTIME"
    )

    print(
        "Loader       :",
        engine[
            "runtime_loader"
        ],
    )

    print(
        "Writable root:",
        engine[
            "colab_compat_root"
        ],
    )

    print(
        "\n✅ FINAL INFERENCE FUNCTION"
    )

    print(
        "process_invoice_final callable :",
        callable(
            engine[
                "process_invoice_final"
            ]
        ),
    )

    print(
        "\n✅ DYNAMIC PARAMETER LAYER"
    )

    print(
        "extract_dynamic_parameters callable :",
        callable(
            extract_dynamic_parameters
        ),
    )

    print(
        "process_invoice_with_dynamic callable :",
        callable(
            process_invoice_with_dynamic
        ),
    )

    print(
        "Dynamic schema mode : "
        "RUNTIME_USER_DEFINED"
    )

    print(
        "\n"
        +
        "="
        *
        72
    )

    print(
        "🔥 INVOICE AI V3 COMPLETE "
        "PRODUCTION ENGINE VERIFIED"
    )

    print(
        "="
        *
        72
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Invoice AI V3 "
            "Production Engine"
        )
    )

    parser.add_argument(
        "--model-only",
        action="store_true",
    )

    parser.add_argument(
        "--rebuild-runtime",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if args.model_only:

        verify_model()

    else:

        verify_production(
            rebuild_runtime=
                args.rebuild_runtime
        )


if __name__ == "__main__":

    main()


# ============================================================
# INVOICE AI V3 ? UNIFIED DYNAMIC PRODUCTION LAYER
# ============================================================
#
# PURPOSE
# -------
# Creates ONE production result from:
#
#   1. LayoutLMv3 trained 16-field extraction
#   2. Automatic runtime schema discovery
#   3. Automatic dynamic value extraction
#   4. V3/V4/V5/V6/V6.1/V7 reconciliation
#
# IMPORTANT
# ---------
# LayoutLMv3 neural labels remain the trained fixed schema.
# Dynamic fields are discovered automatically by the
# inference engine and merged into the same final field map.
#
# No retraining.
# No weight modification.
# No frontend field requests required.
#
# FINAL ENTRY POINT
# -----------------
# process_invoice_dynamic_final(path)
#
# ============================================================


UNIFIED_SCHEMA_VERSION = (
    "invoice_ai_v3_dynamic_unified_v1"
)

UNIFIED_MODEL_NAME = (
    "Invoice_AI_V3_Dynamic_Production"
)

UNIFIED_INFERENCE_MODE = (
    "TRAINED_PLUS_AUTOMATIC_DYNAMIC"
)


def _unified_deepcopy(
    value,
):
    """
    Safe deepcopy helper.

    Imported locally so the existing production engine
    imports do not need to be modified.
    """

    import copy

    try:
        return copy.deepcopy(
            value
        )

    except Exception:
        return value


def _unified_clean_label(
    label,
):
    """
    Convert a field name to a comparison-safe key.

    Used only for collision checking.
    The original display label is preserved in output.
    """

    import re

    text = str(
        label or ""
    ).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )

    return text


def _unified_is_missing_value(
    value,
):
    """
    Determine whether an extracted value is genuinely empty.
    """

    if value is None:
        return True

    if isinstance(
        value,
        str,
    ):
        text = value.strip()

        return text.upper() in {
            "",
            "NOT_DETECTED",
            "NOT DETECTED",
            "NOT_PRESENT",
            "NOT PRESENT",
            "NONE",
            "NULL",
            "N/A",
            "NA",
        }

    if isinstance(
        value,
        (list, tuple, dict),
    ):
        return len(
            value
        ) == 0

    return False


def _unified_field_object(
    field_name,
    field_data,
    *,
    origin,
):
    """
    Normalize trained and dynamic field payloads into
    one common field representation.

    Existing values/status/source are preserved.
    """

    if isinstance(
        field_data,
        dict,
    ):
        obj = _unified_deepcopy(
            field_data
        )

    else:
        obj = {
            "value": _unified_deepcopy(
                field_data
            )
        }

    obj["origin"] = origin

    if (
        "status" not in obj
        or not obj.get(
            "status"
        )
    ):
        if _unified_is_missing_value(
            obj.get(
                "value"
            )
        ):
            obj["status"] = (
                "NOT_DETECTED"
            )

        else:
            obj["status"] = (
                "DETECTED"
            )

    if (
        "source" not in obj
        or not obj.get(
            "source"
        )
    ):
        if origin == "TRAINED_SCHEMA":
            obj["source"] = (
                "TRAINED_MODEL"
            )

        else:
            obj["source"] = (
                "AUTO_DYNAMIC"
            )

    obj["field_name"] = (
        str(
            field_name
        )
    )

    return obj


def _unified_trained_field_aliases():
    """
    Comparison keys for the fixed trained schema.

    This prevents a dynamically discovered field from
    overwriting an existing trained field.
    """

    aliases = set()

    for field_name in EXPECTED_FIELDS:
        aliases.add(
            _unified_clean_label(
                field_name
            )
        )

    manual_aliases = {
        "vendorname",
        "sellername",
        "suppliername",

        "invoicenumber",
        "invoiceno",
        "invoiceid",

        "invoicedate",
        "duedate",

        "customername",
        "buyername",

        "address",
        "customeraddress",
        "billingaddress",

        "currency",

        "lineitemdesc",
        "lineitemdescription",
        "description",

        "lineitemqty",
        "quantity",

        "lineitemunitprice",
        "unitprice",

        "lineitemamount",

        "tax",
        "totaltax",

        "discount",

        "subtotal",

        "totalamount",
        "grandtotal",

        "paymentterms",
    }

    aliases.update(
        manual_aliases
    )

    return aliases


def _unified_merge_fields(
    production_result,
    dynamic_fields,
):
    """
    Merge trained fields and automatic dynamic fields into
    ONE final fields dictionary.

    Priority:
        trained schema wins on semantic collisions.

    Dynamic fields that represent additional invoice
    parameters are appended automatically.
    """

    combined = {}

    trained_fields = {}

    if isinstance(
        production_result,
        dict,
    ):
        maybe_fields = (
            production_result.get(
                "fields"
            )
        )

        if isinstance(
            maybe_fields,
            dict,
        ):
            trained_fields = (
                maybe_fields
            )

    # --------------------------------------------------------
    # TRAINED MODEL FIELDS FIRST
    # --------------------------------------------------------

    for field_name in EXPECTED_FIELDS:

        field_data = (
            trained_fields.get(
                field_name,
                {
                    "value":
                        "NOT_DETECTED",
                    "status":
                        "NOT_DETECTED",
                    "source":
                        "NOT_DETECTED",
                },
            )
        )

        combined[
            field_name
        ] = _unified_field_object(
            field_name,
            field_data,
            origin=(
                "TRAINED_SCHEMA"
            ),
        )

    # Preserve any unexpected production field too.
    for (
        field_name,
        field_data,
    ) in trained_fields.items():

        if field_name in combined:
            continue

        combined[
            field_name
        ] = _unified_field_object(
            field_name,
            field_data,
            origin=(
                "PRODUCTION_RUNTIME"
            ),
        )

    # --------------------------------------------------------
    # AUTOMATIC DYNAMIC FIELDS
    # --------------------------------------------------------

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        dynamic_fields = {}

    trained_aliases = (
        _unified_trained_field_aliases()
    )

    existing_keys = {
        _unified_clean_label(
            name
        )
        for name
        in combined
    }

    for (
        dynamic_name,
        dynamic_data,
    ) in dynamic_fields.items():

        clean_name = (
            _unified_clean_label(
                dynamic_name
            )
        )

        if not clean_name:
            continue

        # Do not overwrite semantic equivalents
        # of fixed neural fields.
        if clean_name in trained_aliases:
            continue

        if clean_name in existing_keys:
            continue

        combined[
            dynamic_name
        ] = _unified_field_object(
            dynamic_name,
            dynamic_data,
            origin=(
                "AUTO_DYNAMIC"
            ),
        )

        existing_keys.add(
            clean_name
        )

    return combined


def _unified_field_counts(
    fields,
):
    """
    Produce transparent field-count statistics.
    """

    trained_count = 0
    dynamic_count = 0
    resolved_count = 0
    unresolved_count = 0

    for (
        _field_name,
        field_data,
    ) in fields.items():

        if not isinstance(
            field_data,
            dict,
        ):
            continue

        origin = field_data.get(
            "origin"
        )

        if origin == "TRAINED_SCHEMA":
            trained_count += 1

        elif origin == "AUTO_DYNAMIC":
            dynamic_count += 1

        value = field_data.get(
            "value"
        )

        if _unified_is_missing_value(
            value
        ):
            unresolved_count += 1

        else:
            resolved_count += 1

    return {
        "total_fields":
            len(fields),

        "trained_schema_fields":
            trained_count,

        "auto_dynamic_fields":
            dynamic_count,

        "resolved_fields":
            resolved_count,

        "unresolved_fields":
            unresolved_count,
    }


def _unified_runtime_info(
    production_result,
    auto_result,
):
    """
    Build final runtime metadata without changing any of
    the underlying production inference information.
    """

    runtime = {}

    if isinstance(
        production_result,
        dict,
    ):
        original_runtime = (
            production_result.get(
                "runtime"
            )
        )

        if isinstance(
            original_runtime,
            dict,
        ):
            runtime.update(
                _unified_deepcopy(
                    original_runtime
                )
            )

    runtime.update(
        {
            "model":
                UNIFIED_MODEL_NAME,

            "model_architecture":
                (
                    "LayoutLMv3ForTokenClassification"
                ),

            "trained_schema_field_count":
                len(
                    EXPECTED_FIELDS
                ),

            "dynamic_schema_mode":
                (
                    "AUTOMATIC_SCHEMA_DISCOVERY"
                ),

            "inference_mode":
                UNIFIED_INFERENCE_MODE,

            "manual_field_input":
                False,

            "frontend_schema_definition":
                False,

            "dynamic_field_discovery":
                True,

            "training":
                False,
        }
    )

    if isinstance(
        auto_result,
        dict,
    ):
        quality_layer = (
            auto_result.get(
                "runtime_quality_layer"
            )
        )

        if quality_layer:
            runtime[
                "runtime_quality_layer"
            ] = quality_layer

    return runtime


def process_invoice_dynamic_final(
    input_path,
    *,
    min_dynamic_confidence=
        DYNAMIC_MIN_CONFIDENCE,
):
    """
    ==========================================================
    INVOICE AI V3 ? FINAL UNIFIED INFERENCE ENTRY POINT
    ==========================================================

    Executes the complete production system:

        Invoice
          |
          v
        LayoutLMv3
        trained 16-field extraction
          |
          v
        V3/V4/V5/V6/V6.1/V7
        runtime reconciliation
          |
          v
        automatic field discovery
          |
          v
        automatic dynamic extraction
          |
          v
        ONE merged field dictionary
          |
          v
        ONE final JSON result


    IMPORTANT:

    This function does NOT retrain LayoutLMv3 and does NOT
    modify model weights.

    The LayoutLMv3 neural schema remains the fixed trained
    16 fields.

    Additional invoice parameters are discovered and
    extracted automatically by the production inference
    engine and are merged into the same final `fields`
    dictionary.

    No frontend parameter list is required.
    No manually requested fields are required.
    ==========================================================
    """

    # --------------------------------------------------------
    # EXISTING FULL WORKING PIPELINE
    # --------------------------------------------------------

    auto_result = (
        process_invoice_auto_dynamic(
            input_path,
            min_dynamic_confidence=
                min_dynamic_confidence,
        )
    )

    if not isinstance(
        auto_result,
        dict,
    ):
        raise RuntimeError(
            "process_invoice_auto_dynamic() "
            "did not return a dictionary."
        )

    production_result = (
        auto_result.get(
            "production_result"
        )
    )

    if not isinstance(
        production_result,
        dict,
    ):
        production_result = {}

    dynamic_fields = (
        auto_result.get(
            "dynamic_fields"
        )
    )

    if not isinstance(
        dynamic_fields,
        dict,
    ):
        dynamic_fields = {}

    # --------------------------------------------------------
    # ONE UNIFIED FIELD MAP
    # --------------------------------------------------------

    combined_fields = (
        _unified_merge_fields(
            production_result,
            dynamic_fields,
        )
    )

    counts = (
        _unified_field_counts(
            combined_fields
        )
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    document = (
        _unified_deepcopy(
            production_result.get(
                "document",
                {},
            )
        )
    )

    # --------------------------------------------------------
    # FINAL TOP-LEVEL STATUS
    # --------------------------------------------------------

    validation = (
        _unified_deepcopy(
            production_result.get(
                "validation",
                {},
            )
        )
    )

    status = None

    if isinstance(
        validation,
        dict,
    ):
        status = (
            validation.get(
                "overall_status"
            )
        )

    if not status:
        if isinstance(
            document,
            dict,
        ):
            status = document.get(
                "status"
            )

    if not status:
        status = "COMPLETED"

    # --------------------------------------------------------
    # FINAL UNIFIED JSON
    # --------------------------------------------------------

    unified_result = {
        "schema_version":
            UNIFIED_SCHEMA_VERSION,

        "model":
            UNIFIED_MODEL_NAME,

        "status":
            status,

        "document":
            document,

        # ====================================================
        # THIS IS THE MERGER
        # ====================================================
        #
        # Trained + dynamic fields now live together here.
        #
        "fields":
            combined_fields,

        "field_summary":
            counts,

        "trained_schema_fields":
            list(
                EXPECTED_FIELDS
            ),

        "auto_discovered_parameters":
            _unified_deepcopy(
                auto_result.get(
                    "auto_discovered_parameters",
                    list(
                        dynamic_fields.keys()
                    ),
                )
            ),

        "line_items":
            _unified_deepcopy(
                production_result.get(
                    "line_items",
                    [],
                )
            ),

        "normalized":
            _unified_deepcopy(
                production_result.get(
                    "normalized",
                    {},
                )
            ),

        "tax_details":
            _unified_deepcopy(
                production_result.get(
                    "tax_details",
                    {},
                )
            ),

        "identifiers":
            _unified_deepcopy(
                production_result.get(
                    "identifiers",
                    {},
                )
            ),

        "financial_details":
            _unified_deepcopy(
                production_result.get(
                    "financial_details",
                    {},
                )
            ),

        "validation":
            validation,

        "extraction_summary":
            _unified_deepcopy(
                production_result.get(
                    "extraction_summary",
                    {},
                )
            ),

        "runtime":
            _unified_runtime_info(
                production_result,
                auto_result,
            ),
    }

    # Preserve selected production audit information.
    for optional_key in (
        "v6_audit",
        "adjustments",
        "round_off",
    ):

        if optional_key in production_result:

            unified_result[
                optional_key
            ] = _unified_deepcopy(
                production_result[
                    optional_key
                ]
            )

    return unified_result


# ------------------------------------------------------------
# CLEAN PUBLIC ALIASES
# ------------------------------------------------------------


def process_invoice_unified(
    input_path,
    *,
    min_dynamic_confidence=
        DYNAMIC_MIN_CONFIDENCE,
):
    """
    Short alias for the final unified production engine.
    """

    return process_invoice_dynamic_final(
        input_path,
        min_dynamic_confidence=
            min_dynamic_confidence,
    )


def process_invoice_latest(
    input_path,
    *,
    min_dynamic_confidence=
        DYNAMIC_MIN_CONFIDENCE,
):
    """
    Latest production alias.

    Useful when sharing the engine internally without
    exposing implementation/version names.
    """

    return process_invoice_dynamic_final(
        input_path,
        min_dynamic_confidence=
            min_dynamic_confidence,
    )


# ============================================================
# END INVOICE AI V3 ? UNIFIED DYNAMIC PRODUCTION LAYER
# ============================================================

# ============================================================
# V8.6 CUSTOMER "TO / BILL TO" ADDRESS FINAL GUARD
# ============================================================
#
# Purpose:
#   Prevent vendor/header address from being returned as ADDRESS
#   when the document contains an explicit customer To/Bill To
#   block.
#
# This does NOT modify:
#   - LayoutLMv3 weights
#   - trained schema
#   - GST reconciliation
#   - line items
#   - dynamic discovery
#   - financial reconciliation
#
# It runs only after the complete production result exists.
# ============================================================

_PROCESS_INVOICE_DYNAMIC_FINAL_BEFORE_V86_ADDRESS_FIX = (
    process_invoice_dynamic_final
)


def _v86_clean_block_line(value):
    import re

    text = re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()

    text = text.strip()

    if not text:
        return ""

    # Remove lines containing only punctuation.
    if not re.search(r"[A-Za-z0-9]", text):
        return ""

    return text


def _v86_customer_to_block_address(input_path, result):
    import re

    try:
        import fitz
    except Exception:
        return None

    try:
        document = fitz.open(str(input_path))
    except Exception:
        return None

    try:
        lines = []

        # Customer information is normally in the first pages.
        for page_index in range(
            min(len(document), 2)
        ):
            try:
                page_text = document[
                    page_index
                ].get_text("text")
            except Exception:
                continue

            for raw_line in str(
                page_text or ""
            ).splitlines():

                cleaned = (
                    _v86_clean_block_line(
                        raw_line
                    )
                )

                if cleaned:
                    lines.append(cleaned)

    finally:
        try:
            document.close()
        except Exception:
            pass

    if not lines:
        return None

    fields = (
        result.get("fields", {})
        if isinstance(result, dict)
        else {}
    )

    customer_data = (
        fields.get("CUSTOMER_NAME", {})
        if isinstance(fields, dict)
        else {}
    )

    if isinstance(customer_data, dict):
        customer_name = str(
            customer_data.get(
                "value",
                "",
            )
            or
            ""
        ).strip()
    else:
        customer_name = str(
            customer_data or ""
        ).strip()

    customer_norm = re.sub(
        r"[^a-z0-9]+",
        " ",
        customer_name.casefold(),
    ).strip()

    # Labels that indicate the customer address block is over.
    stop_pattern = re.compile(
        r"(?i)^("
        r"gstin|"
        r"cin|"
        r"pan|"
        r"state\s*code|"
        r"state\s*name|"
        r"place\s*of\s*supply|"
        r"odn|"
        r"po\s*(?:no|number)?|"
        r"p\.?\s*o\.?|"
        r"gr\s*(?:no|number)?|"
        r"invoice|"
        r"posting\s*date|"
        r"due\s*date|"
        r"rampur\s*unit|"
        r"material|"
        r"sl\s*no|"
        r"dear\s+sir"
        r")\b"
    )

    anchor_pattern = re.compile(
        r"(?i)^(?:"
        r"to|"
        r"bill\s*to|"
        r"billed\s*to|"
        r"buyer|"
        r"customer"
        r")\s*[:\-]?\s*(.*)$"
    )

    for index, line in enumerate(lines):

        match = anchor_pattern.match(
            line.strip()
        )

        if not match:
            continue

        inline_customer = (
            match.group(1).strip()
        )

        cursor = index + 1

        # ----------------------------------------------------
        # Identify customer name belonging to this block.
        # ----------------------------------------------------

        if inline_customer:
            candidate_customer = (
                inline_customer
            )
        elif cursor < len(lines):
            candidate_customer = (
                lines[cursor]
            )
            cursor += 1
        else:
            continue

        candidate_norm = re.sub(
            r"[^a-z0-9]+",
            " ",
            candidate_customer.casefold(),
        ).strip()

        # If a reliable CUSTOMER_NAME already exists, require
        # the To/Bill-To block to refer to that same customer.
        if customer_norm:
            if (
                customer_norm
                not in candidate_norm
                and
                candidate_norm
                not in customer_norm
            ):
                continue

        address_lines = []

        # Collect a conservative customer address block.
        for next_index in range(
            cursor,
            min(
                len(lines),
                cursor + 7,
            ),
        ):

            candidate = (
                _v86_clean_block_line(
                    lines[next_index]
                )
            )

            if not candidate:
                continue

            # Values beginning with ":" are normally values
            # belonging to GSTIN/PAN/state columns after the
            # customer block in extracted PDF text.
            if candidate.startswith(":"):
                break

            if stop_pattern.search(
                candidate
            ):
                break

            # Don't accidentally repeat customer name.
            candidate_compare = re.sub(
                r"[^a-z0-9]+",
                " ",
                candidate.casefold(),
            ).strip()

            if (
                customer_norm
                and
                candidate_compare
                ==
                customer_norm
            ):
                continue

            address_lines.append(
                candidate.strip(" ,")
            )

            # Once a PIN code is reached the postal address is
            # normally complete.
            if re.search(
                r"\b\d{6}\b",
                candidate,
            ):
                break

        if not address_lines:
            continue

        address = ", ".join(
            part
            for part in address_lines
            if part
        )

        address = re.sub(
            r"\s*,\s*,+",
            ", ",
            address,
        )

        address = re.sub(
            r"\s+",
            " ",
            address,
        ).strip(" ,")

        # Require reasonable address evidence.
        if len(address) < 10:
            continue

        if not (
            re.search(
                r"\d",
                address,
            )
            or
            re.search(
                r"(?i)\b("
                r"road|street|sector|"
                r"nagar|floor|park|"
                r"kolkata|delhi|mumbai|"
                r"noida|gurugram|"
                r"bengaluru|chennai"
                r")\b",
                address,
            )
        ):
            continue

        return address

    return None


def process_invoice_dynamic_final(
    input_path,
    *,
    min_dynamic_confidence=DYNAMIC_MIN_CONFIDENCE,
):
    """
    V8.6 final production wrapper.

    Executes the complete existing engine unchanged, then gives
    an explicit To/Bill-To customer address higher precedence
    than a generic vendor/header address.
    """

    result = (
        _PROCESS_INVOICE_DYNAMIC_FINAL_BEFORE_V86_ADDRESS_FIX(
            input_path,
            min_dynamic_confidence=
                min_dynamic_confidence,
        )
    )

    if not isinstance(result, dict):
        return result

    customer_address = (
        _v86_customer_to_block_address(
            input_path,
            result,
        )
    )

    if customer_address:

        fields = result.setdefault(
            "fields",
            {},
        )

        existing = fields.get(
            "ADDRESS",
            {},
        )

        if not isinstance(
            existing,
            dict,
        ):
            existing = {}

        fields["ADDRESS"] = {
            "value":
                customer_address,

            "status":
                "RULE_RECOVERED",

            "source":
                "V8_6_EXPLICIT_CUSTOMER_TO_BLOCK",

            "origin":
                "TRAINED_SCHEMA",

            "field_name":
                "ADDRESS",
        }

    return result


# ============================================================
# END V8.6 CUSTOMER ADDRESS FINAL GUARD
# ============================================================

