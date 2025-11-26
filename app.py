"""Scientific Named Entity Explorer Streamlit application.

Quickstart:
    python3 install.py            # optional helper to create .venv and install deps
    source .venv/bin/activate     # on Windows use: .venv\\Scripts\\activate
    streamlit run app.py
"""

from __future__ import annotations

import sys, io, contextlib
import streamlit as st  # type: ignore
import streamlit.runtime.state.session_state as ss  # type: ignore

if hasattr(ss, "print"):
    ss.print = lambda *a, **kw: None

import importlib
import importlib.metadata as metadata
import importlib.util
import json
import logging
import os
import platform
import re
import subprocess
import time
import warnings
import hashlib
import urllib.request
import urllib.error
import builtins
import inspect
import math
from functools import wraps
from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC, Mapping
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

# Enable CPU fallback for Apple MPS backends to avoid runtime errors.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logging.getLogger().setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

DEBUG_MODE = False
_ORIGINAL_PRINT = builtins.print
_APP_FILE = Path(__file__).resolve()
_APP_ROOT = _APP_FILE.parent
_REMOVE = object()

# Pipeline defaults: allow empty abstracts, keep duplicates unless explicitly toggled.
ALLOW_EMPTY_ABSTRACTS = True
DEFAULT_MIN_ABSTRACT_LENGTH = 0
DEFAULT_INCLUDE_ALL_TYPES = True
ENABLE_DEDUPLICATION = False


def safe_to_str(obj: object) -> str:
    if isinstance(obj, (set, list, tuple)):
        return ", ".join(sorted(map(str, obj)))
    if isinstance(obj, dict):
        return ", ".join([f"{key}:{value}" for key, value in obj.items()])
    return str(obj)


def safe_str(obj: object) -> str:
    return safe_to_str(obj)

def sanitize_for_display(value: object) -> object:
    if isinstance(value, set):
        return [sanitize_for_display(item) for item in sorted(value, key=lambda x: safe_to_str(x))]
    if isinstance(value, dict):
        return {key: sanitize_for_display(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_display(item) for item in value]
    return value


def _is_pandas_object(value: Any) -> bool:
    pandas_module = globals().get("pd")
    return pandas_module is not None and isinstance(value, (pandas_module.DataFrame, pandas_module.Series))


def _clean_nested(value: Any) -> Any:
    if _is_pandas_object(value):
        return deep_clean(value)
    if value is None:
        return _REMOVE
    if isinstance(value, set):
        if not value:
            return _REMOVE
        cleaned_items: List[Any] = []
        for item in sorted(value, key=lambda entry: safe_to_str(entry)):
            cleaned_item = _clean_nested(item)
            if cleaned_item is _REMOVE:
                continue
            cleaned_items.append(cleaned_item)
        return cleaned_items if cleaned_items else _REMOVE
    if isinstance(value, dict):
        cleaned_dict: Dict[Any, Any] = {}
        for key, item in value.items():
            cleaned_item = _clean_nested(item)
            if cleaned_item is _REMOVE:
                continue
            cleaned_dict[key] = cleaned_item
        return cleaned_dict if cleaned_dict else _REMOVE
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            cleaned_item = _clean_nested(item)
            if cleaned_item is _REMOVE:
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list if cleaned_list else _REMOVE
    if isinstance(value, tuple):
        cleaned_tuple: List[Any] = []
        for item in value:
            cleaned_item = _clean_nested(item)
            if cleaned_item is _REMOVE:
                continue
            cleaned_tuple.append(cleaned_item)
        return tuple(cleaned_tuple) if cleaned_tuple else _REMOVE
    if isinstance(value, str):
        return value if value.strip() else _REMOVE
    if isinstance(value, bytes):
        return value if value else _REMOVE
    return value


def _should_keep(value: Any) -> bool:
    if value is None:
        return False
    if _is_pandas_object(value):
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _strip_placeholders(value: Any) -> Any:
    if value is _REMOVE:
        return None
    if _is_pandas_object(value):
        return value
    if isinstance(value, dict):
        cleaned: Dict[Any, Any] = {}
        for key, item in value.items():
            stripped = _strip_placeholders(item)
            if _should_keep(stripped):
                cleaned[key] = stripped
        return cleaned if cleaned else None
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            stripped = _strip_placeholders(item)
            if _should_keep(stripped):
                cleaned_list.append(stripped)
        return cleaned_list if cleaned_list else None
    if isinstance(value, tuple):
        stripped_items = [_strip_placeholders(item) for item in value]
        cleaned_tuple = tuple(item for item in stripped_items if _should_keep(item))
        return cleaned_tuple if cleaned_tuple else None
    return value


def _prepare_scalar_for_dataframe(value: Any) -> Any:
    if _is_pandas_object(value):
        return deep_clean(value)
    cleaned = _clean_nested(value)
    stripped = _strip_placeholders(cleaned)
    return stripped if _should_keep(stripped) else None


def deep_clean(obj: Any) -> Any:
    pandas_module = globals().get("pd")
    if pandas_module is not None:
        if isinstance(obj, pandas_module.DataFrame):
            if obj.empty:
                return obj.copy(deep=True)
            return obj.applymap(_prepare_scalar_for_dataframe)
        if isinstance(obj, pandas_module.Series):
            if obj.empty:
                return obj.copy(deep=True)
            return obj.map(_prepare_scalar_for_dataframe)
    cleaned = _clean_nested(obj)
    return _strip_placeholders(cleaned)


_STREAMLIT_RENDER_METHODS = [
    "write",
    "markdown",
    "dataframe",
    "table",
    "json",
    "info",
    "warning",
    "error",
    "success",
    "exception",
    "caption",
    "title",
    "header",
    "subheader",
    "text",
]


def _clean_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_kwargs: Dict[str, Any] = {}
    for key, value in kwargs.items():
        cleaned_value = deep_clean(value)
        if _should_keep(cleaned_value):
            cleaned_kwargs[key] = cleaned_value
    return cleaned_kwargs


def _wrap_render_function(func):
    if not callable(func) or getattr(func, "_deep_clean_wrapped", False):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        cleaned_args = []
        for arg in args:
            cleaned = deep_clean(arg)
            if _should_keep(cleaned):
                cleaned_args.append(cleaned)
        cleaned_kwargs = _clean_kwargs(kwargs)
        if not cleaned_args and not cleaned_kwargs:
            return None
        return func(*cleaned_args, **cleaned_kwargs)

    wrapper._deep_clean_wrapped = True
    return wrapper


def _patch_streamlit_renderers():
    target = st if "st" in globals() else None
    if not target:
        return
    try:
        from streamlit.delta_generator import DeltaGenerator  # type: ignore
    except Exception:
        DeltaGenerator = None  # type: ignore

    if DeltaGenerator is not None:
        for name in _STREAMLIT_RENDER_METHODS:
            method = getattr(DeltaGenerator, name, None)
            if method:
                setattr(DeltaGenerator, name, _wrap_render_function(method))

    for render_target in filter(None, [target, getattr(target, "sidebar", None)]):
        for name in _STREAMLIT_RENDER_METHODS:
            if hasattr(render_target, name):
                setattr(render_target, name, _wrap_render_function(getattr(render_target, name)))


_patch_streamlit_renderers()

def _is_app_frame(frame) -> bool:
    """Return True when the frame originates from this project."""
    try:
        frame_path = Path(frame.f_code.co_filename).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return frame_path == _APP_FILE or _APP_ROOT in frame_path.parents


def safe_print(*args, **kwargs):
    if _running_in_streamlit():
        _ORIGINAL_PRINT(*args, **kwargs)
        return
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    current = caller
    should_sanitize = False
    try:
        while current:
            if _is_app_frame(current):
                should_sanitize = True
                break
            current = current.f_back
    finally:
        del frame
        del caller
        del current
    if should_sanitize:
        sanitized_args = [safe_to_str(arg) for arg in args]
        _ORIGINAL_PRINT(*sanitized_args, **kwargs)
    else:
        _ORIGINAL_PRINT(*args, **kwargs)


def _running_in_streamlit() -> bool:
    """Return True when running under Streamlit."""
    if any("streamlit" in arg for arg in sys.argv):
        return True
    for name in sys.modules:
        if name == "streamlit" or name.startswith("streamlit."):
            return True
    return False

# ADDED: Execute functions while silencing stdout/stderr noise.
def silent_run(func: Callable, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return func(*args, **kwargs)


def debug_log(*args, **kwargs) -> None:
    if not DEBUG_MODE:
        return
    formatted_args = [safe_str(arg) for arg in args]
    formatted_kwargs = {key: (safe_str(value) if key not in {"file"} else value) for key, value in kwargs.items()}
    _ORIGINAL_PRINT(*formatted_args, **formatted_kwargs)

pd = None
st = None
AutoTokenizer = None
pipeline = None
torch = None

DEPENDENCIES: List[Tuple[str, str, str, bool]] = [
    ("torch", "torch", "2.0.0", True),
    ("transformers", "transformers", "4.30.0", True),
    ("streamlit", "streamlit", "1.25.0", True),
    ("pandas", "pandas", "1.3.0", True),
    ("sklearn", "scikit-learn", "1.0.0", True),
    ("sentence_transformers", "sentence-transformers", "2.2.2", True),
    ("openpyxl", "openpyxl", "3.0.0", False),
    ("xlrd", "xlrd", "2.0.1", False),
]

PIP_SUGGESTION = (
    "pip install torch transformers streamlit pandas scikit-learn sentence-transformers openpyxl xlrd"
)

ENVIRONMENT_READY_MESSAGE = "✅ Environment ready: All dependencies installed."


def _parse_version(version_str: str) -> Tuple:
    parts = re.split(r"[.+-]", version_str)
    parsed: List[Tuple[int, str]] = []
    for part in parts:
        if part.isdigit():
            parsed.append((int(part), ""))
        else:
            parsed.append((0, part))
    return tuple(parsed)


def ensure_dependencies(auto_install: bool = False) -> None:
    """Ensure dependencies are available, installing them if requested."""
    global pd, st, AutoTokenizer, pipeline, torch

    auto_install = bool(auto_install)
    modules: Dict[str, object] = {}

    def _probe() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        required_missing: List[Tuple[str, str]] = []
        optional_missing: List[Tuple[str, str]] = []
        modules.clear()
        for module_name, pip_name, _, required in DEPENDENCIES:
            try:
                modules[module_name] = importlib.import_module(module_name)
            except ImportError:
                target = required_missing if required else optional_missing
                target.append((module_name, pip_name))
        return required_missing, optional_missing

    required_missing, optional_missing = _probe()

    if (required_missing or optional_missing) and auto_install:
        packages = sorted({pip for _, pip in required_missing + optional_missing})
        if packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
            except Exception as install_err:  # pragma: no cover
                _ORIGINAL_PRINT(
                    safe_str(f"⚠️ Automatic dependency installation failed: {install_err}"),
                    file=sys.stderr,
                )
                if required_missing:
                    raise SystemExit(1)
            required_missing, optional_missing = _probe()

    if required_missing:
        missing_names = ", ".join(sorted({name for name, _ in required_missing}))
        message = (
            f"⚠️ Missing required library '{missing_names}'.\n"
            f"   Please install the required packages, e.g.: {PIP_SUGGESTION}"
        )
        _ORIGINAL_PRINT(safe_str(message), file=sys.stderr)
        raise SystemExit(1)

    if optional_missing:
        optional_names = ", ".join(sorted({name for name, _ in optional_missing}))
        _ORIGINAL_PRINT(
            safe_str(
                f"ℹ️ Optional libraries missing: {optional_names}. "
                f"Install them with: {PIP_SUGGESTION}"
            ),
            file=sys.stderr,
        )

    pandas_module = modules["pandas"]
    streamlit_module = modules["streamlit"]
    transformers_module = modules["transformers"]
    torch_module = modules.get("torch")

    pd = pandas_module
    st = streamlit_module
    torch = torch_module
    AutoTokenizer = getattr(transformers_module, "AutoTokenizer", None)
    pipeline_candidate = getattr(transformers_module, "pipeline", None)
    if AutoTokenizer is None or pipeline_candidate is None:
        _ORIGINAL_PRINT(
            safe_str(
                "⚠️ transformers library did not expose AutoTokenizer/pipeline as expected."
            ),
            file=sys.stderr,
        )
    pipeline = pipeline_candidate

    version_warnings: List[str] = []
    if st is not None:
        seen_warnings = st.session_state.get("seen_warnings")
        if seen_warnings is None:
            seen_warnings = set()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                st.session_state["seen_warnings"] = seen_warnings
    else:
        seen_warnings = getattr(ensure_dependencies, "_seen_warnings", set())
        ensure_dependencies._seen_warnings = seen_warnings
    for module_name, _, minimum, required in DEPENDENCIES:
        if not required and module_name not in modules:
            continue
        try:
            current = getattr(modules[module_name], "__version__", metadata.version(module_name))
        except (metadata.PackageNotFoundError, KeyError):  # pragma: no cover
            continue
        warning_text = (
            f"⚠️ {module_name} {current} detected. Upgrade to >= {minimum} for best results."
        )
        if _parse_version(current) < _parse_version(minimum) and warning_text not in seen_warnings:
            version_warnings.append(warning_text)
            seen_warnings.add(warning_text)

    if version_warnings:
        if st is not None:
            try:
                for warning in version_warnings:
                    st.sidebar.warning(warning)
            except Exception:
                for warning in version_warnings:
                    debug_log(warning)
        else:
            for warning in version_warnings:
                debug_log(warning)

    ensure_dependencies._environment_ready = True

    return None


_ = ensure_dependencies(auto_install=True)  # pragma: no cover

DEFAULT_DEVICE_PREFERENCE = "auto"

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    st_util = None

if TYPE_CHECKING:  # pragma: no cover
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.runtime.uploaded_file_manager import UploadedFile


st.set_page_config(
    page_title="Scientific NER Explorer",
    page_icon="📖",
    layout="wide",
)

if getattr(ensure_dependencies, "_environment_ready", False) and not getattr(
    ensure_dependencies, "_ready_banner_shown", False
):
    try:
        st.sidebar.success(ENVIRONMENT_READY_MESSAGE)
        ensure_dependencies._ready_banner_shown = True
    except Exception:  # pragma: no cover
        pass


APP_VERSION = "1.0.0"
GITHUB_URL = "https://github.com/k25063738/Named_entity_algorithm_project"
DEFAULT_MODEL = "pranav-s/PolymerNER"
OFFLINE_MODEL_DIR = Path("./models/allenai_scibert_scivocab_uncased")
DISABLED_MODELS: Dict[str, str] = {
    "WENGSYX/ChemBERTa-2-chemical-ner": "ChemBERTa-2 repository is unavailable for automatic download.",
}
SCIBERT_MODELS: Set[str] = {
    DEFAULT_MODEL,
    "turing-1/scibert-base-finetuned-ner",
}
MODEL_OPTIONS = {
    "PolymerNER": "pranav-s/PolymerNER",
    "SciBERT (AllenAI)": "allenai/scibert_scivocab_uncased",
    "finetuned SciBERT": "JonyC/scibert-NER-finetuned-improved"
}

USER_CACHE_DIR = Path(".ner_cache")
USER_ENTITIES_FILE = USER_CACHE_DIR / "user_entities.json"
USER_CANONICAL_FILE = USER_CACHE_DIR / "user_canonical.json"
USER_CANONICAL_MERGES_FILE = USER_CACHE_DIR / "user_canonical_merges.json"

REQUIRED_COLUMNS = {"Title", "Abstract"}
OPTIONAL_COLUMNS = {"DOI", "Year"}

MODEL_ID2LABEL: Dict[str, str] = {
    "0": "INORGANIC",
    "1": "MATERIAL_AMOUNT",
    "2": "MONOMER",
    "3": "O",
    "4": "ORGANIC",
    "5": "POLYMER",
    "6": "POLYMER_FAMILY",
    "7": "PROP_NAME",
    "8": "PROP_VALUE",
}

def _normalize_label_from_config(label: str) -> str:
    cleaned = safe_to_str(label).strip()
    cleaned = re.sub(r"^[BIES]-", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("-", "_").replace(" ", "_")
    return cleaned.upper()


ACTIVE_ID2LABEL: Dict[str, str] = {
    str(idx): _normalize_label_from_config(label) for idx, label in MODEL_ID2LABEL.items()
}
ACTIVE_LABEL2ID: Dict[str, int] = {label: int(idx) for idx, label in ACTIVE_ID2LABEL.items()}

ENTITY_TYPES: List[str] = list(ACTIVE_LABEL2ID.keys())

# ADDED: Supported custom entity types for user entries. Always synced with the active label schema.
CUSTOM_ENTITY_TYPES = sorted(set(ENTITY_TYPES + ["OTHER"]))

# === Visualization Colors ===
ENTITY_COLORS = {
    "INORGANIC": "#1E3A8A",
    "MATERIAL_AMOUNT": "#0EA5E9",
    "MONOMER": "#22C55E",
    "O": "#9CA3AF",
    "ORGANIC": "#008000",
    "POLYMER": "#EF4444",
    "POLYMER_FAMILY": "#9333EA",
    "PROP_NAME": "#F59E0B",
    "PROP_VALUE": "#2563EB",
}


def _ensure_entity_colors() -> None:
    palette = [
        "#1E3A8A",
        "#0EA5E9",
        "#22C55E",
        "#9CA3AF",
        "#FACC15",
        "#EF4444",
        "#9333EA",
        "#F59E0B",
        "#2563EB",
        "#4B5563",
    ]
    for idx, label in enumerate(ENTITY_TYPES):
        if label not in ENTITY_COLORS:
            ENTITY_COLORS[label] = palette[idx % len(palette)]


_ensure_entity_colors()

LABEL_PREFIX_PATTERN = re.compile(r"^LABEL[_-]?(?P<idx>\d+)$", re.IGNORECASE)

LEGACY_LABEL_ALIASES: Dict[str, str] = {
    "AMOUNT": "MATERIAL_AMOUNT",
    "CARDINAL": "PROP_VALUE",
    "CHEM": "POLYMER",
    "CHEMICAL": "POLYMER",
    "INORG": "INORGANIC",
    "MATERIAL": "POLYMER",
    "MATERIALS": "POLYMER",
    "METHOD": "O",
    "METHOD/PROCESS": "O",
    "MISC": "O",
    "ORG": "ORGANIC",
    "OTHER": "O",
    "PROCESS": "O",
    "PRODUCT": "POLYMER",
    "POLYMERFAMILY": "POLYMER_FAMILY",
    "PROPERTY": "PROP_NAME",
    "PROP": "PROP_NAME",
    "PROPNAME": "PROP_NAME",
    "PROPVALUE": "PROP_VALUE",
    "PROP-VALUE": "PROP_VALUE",
    "PROP_VALUE": "PROP_VALUE",
    "PROP-NAME": "PROP_NAME",
    "PROP_NAME": "PROP_NAME",
    "QUANTITY": "MATERIAL_AMOUNT",
    "UNIT": "PROP_VALUE",
    "VALUE": "PROP_VALUE",
}

UNMAPPED_LABELS: Set[str] = set()


def _refresh_entity_type_constants(label_set: Iterable[str]) -> None:
    """Sync label-derived globals after loading a model config."""
    global ENTITY_TYPES, CUSTOM_ENTITY_TYPES
    ENTITY_TYPES = list(label_set)
    CUSTOM_ENTITY_TYPES = sorted(set(ENTITY_TYPES + ["OTHER"]))
    _ensure_entity_colors()


def log_unmapped_label(label: str, context: str = "") -> None:
    if not label:
        return
    if label in UNMAPPED_LABELS:
        return
    UNMAPPED_LABELS.add(label)
    message = f"Unmapped entity label '{label}' encountered"
    if context:
        message = f"{message} ({context})"
    warnings.warn(message)
    try:
        logging.warning(message)
    except Exception:
        pass


def canonicalize_label(label: object, warn: bool = False) -> str:
    raw = safe_to_str(label).strip()
    if not raw:
        return "UNKNOWN"

    normalized = _normalize_label_from_config(raw)
    match = LABEL_PREFIX_PATTERN.match(normalized)
    if match:
        idx = match.group("idx")
        resolved = ACTIVE_ID2LABEL.get(idx) or MODEL_ID2LABEL.get(idx)
        if resolved:
            return _normalize_label_from_config(resolved)

    if normalized in ACTIVE_LABEL2ID:
        return normalized

    alias = LEGACY_LABEL_ALIASES.get(normalized) or LEGACY_LABEL_ALIASES.get(normalized.replace("-", "_"))
    if alias and alias in ACTIVE_LABEL2ID:
        return alias

    if warn:
        log_unmapped_label(raw, context="missing from model label set")
    return "UNKNOWN"


def apply_model_label_schema(
    id2label: Optional[Mapping[str, str]],
    label2id: Optional[Mapping[str, int]] = None,
) -> None:
    """Refresh active label schema from a model configuration."""
    global ACTIVE_ID2LABEL, ACTIVE_LABEL2ID

    if not id2label:
        return

    cleaned_id2label: Dict[str, str] = {}
    for key, value in id2label.items():
        cleaned_id2label[str(key)] = _normalize_label_from_config(value)

    ACTIVE_ID2LABEL = cleaned_id2label
    if label2id:
        cleaned_label2id: Dict[str, int] = {}
        for label, idx in label2id.items():
            normalized_label = _normalize_label_from_config(label)
            cleaned_label2id[normalized_label] = int(idx)
        ACTIVE_LABEL2ID = cleaned_label2id
    else:
        ACTIVE_LABEL2ID = {label: int(idx) for idx, label in ACTIVE_ID2LABEL.items()}

    _refresh_entity_type_constants(ACTIVE_LABEL2ID.keys())


def refresh_labels_from_pipeline(pipeline_obj: object) -> None:
    """Apply label schema from a loaded pipeline's config when available."""
    try:
        model = getattr(pipeline_obj, "model", None)
        config = getattr(model, "config", None)
        if config and getattr(config, "id2label", None):
            apply_model_label_schema(
                getattr(config, "id2label", None),
                getattr(config, "label2id", None),
            )
        else:
            apply_model_label_schema(MODEL_ID2LABEL)
    except Exception:
        apply_model_label_schema(MODEL_ID2LABEL)


# Ensure globals reflect the default schema at import time.
apply_model_label_schema(MODEL_ID2LABEL)



ABBREVIATION_PATTERN = re.compile(
    r"(?P<expansion>[A-Za-z][A-Za-z0-9\-/\s]+?)\s*\((?P<abbr>[A-Z0-9]{2,})\)"
)

DOMAIN_TOKENS = [
    "OPV",
    "OPVs",
    "OFET",
    "OFETs",
    "OECT",
    "OECTs",
    "OMIEC",
    "OMIECs",
    "perovskite",
    "Perovskite",
    "DFT",
    "dft",
    "PCE",
    "pce",
    "bandgap",
    "Bandgap",
    "mobility",
    "Mobility",
    "non-fullerene",
    "Non-fullerene",
    "side-chain",
    "Side-chain",
]

ACRONYM_SET = {
    "opv",
    "ofet",
    "oect",
    "omiec",
    "bhj",
    "dssc",
    "perovskite",
}
ACRONYM_UPPER = {abbr.upper() for abbr in ACRONYM_SET}
PROTECTED_TERMS = {
    "opv",
    "ofet",
    "oect",
    "omiec",
    "bhj",
    "dssc",
    "perovskite",
    "pksc",
}

ACRONYM_CANONICAL_MAP: Dict[str, Set[str]] = {
    "OPV": {
        "opv",
        "organic photovoltaic",
        "organic photovoltaics",
        "organic photovoltaic cell",
        "organic photovoltaic cells",
        "organic solar cell",
        "organic solar cells",
    },
    "OFET": {
        "ofet",
        "organic field effect transistor",
        "organic field-effect transistor",
        "organic field effect transistors",
        "organic field-effect transistors",
    },
    "OECT": {
        "oect",
        "organic electrochemical transistor",
        "organic electrochemical transistors",
    },
    "BHJ": {
        "bhj",
        "bulk heterojunction",
        "bulk heterojunctions",
    },
    "DSSC": {
        "dssc",
        "dye-sensitized solar cell",
        "dye sensitized solar cell",
        "dye-sensitized solar cells",
        "dye sensitized solar cells",
    },
    "PKSC": {
        "pksc",
        "perovskite solar cell",
        "perovskite solar cells",
    },
}

ACRONYM_LOOKUP: Dict[str, str] = {
    phrase.strip().lower(): canonical
    for canonical, variants in ACRONYM_CANONICAL_MAP.items()
    for phrase in variants
}

DOMAIN_ALIAS_SUBSTRINGS: Dict[str, str] = {
    "organic photovoltaic": "opv",
    "organic field effect transistor": "ofet",
    "organic field-effect transistor": "ofet",
    "organic electrochemical transistor": "oect",
    "organic mixed ionic electronic conductor": "omiec",
    "mixed ionic electronic conductor": "omiec",
    "bulk heterojunction": "bhj",
}

DOMAIN_TERM_CATEGORIES: Dict[str, str] = {
    "opv": "ORGANIC",
    "opvs": "ORGANIC",
    "organic photovoltaic": "ORGANIC",
    "organic photovoltaics": "ORGANIC",
    "organic photovoltaic cells": "ORGANIC",
    "ofet": "ORGANIC",
    "ofets": "ORGANIC",
    "organic field-effect transistor": "ORGANIC",
    "organic field effect transistor": "ORGANIC",
    "organic field-effect transistors": "ORGANIC",
    "organic field effect transistors": "ORGANIC",
    "oect": "ORGANIC",
    "oects": "ORGANIC",
    "organic electrochemical transistor": "ORGANIC",
    "organic electrochemical transistors": "ORGANIC",
    "omiec": "ORGANIC",
    "omiecs": "ORGANIC",
    "perovskite": "INORGANIC",
    "non-fullerene acceptor": "ORGANIC",
    "non fullerene acceptor": "ORGANIC",
    "non-fullerene acceptors": "ORGANIC",
    "non fullerene acceptors": "ORGANIC",
    "dft": "O",
    "density functional theory": "O",
    "side-chain": "POLYMER",
    "side chain": "POLYMER",
    "side-chain engineering": "POLYMER",
    "side chain engineering": "POLYMER",
    "computational materials science": "O",
    "band alignment": "PROP_NAME",
    "bandgap": "PROP_NAME",
    "band gap": "PROP_NAME",
    "pce": "PROP_NAME",
    "power conversion efficiency": "PROP_NAME",
    "mobility": "PROP_NAME",
}

DOMAIN_MULTIWORD_TERMS = {
    term: category for term, category in DOMAIN_TERM_CATEGORIES.items() if " " in term or "-" in term
}


MATERIAL_KEYWORDS = {
    "poly",
    "polymer",
    "film",
    "oxide",
    "perovskite",
    "omiec",
    "oect",
    "ofet",
    "opv",
    "semiconductor",
    "device",
    "thin film",
}

PROPERTY_KEYWORDS = {
    "efficiency",
    "mobility",
    "conductivity",
    "stability",
    "bandgap",
    "lifetime",
    "selectivity",
    "hardness",
    "temperature",
    "yield",
    "current density",
    "open-circuit voltage",
    "voc",
    "fill factor",
    "absorption",
    "emission",
    "thickness",
    "porosity",
    "density",
}

METHOD_KEYWORDS = {
    "annealing",
    "anneal",
    "spin-coating",
    "spin",
    "solution processing",
    "measure",
    "measurement",
    "sputtering",
    "deposition",
    "fabrication",
    "fabricate",
    "printing",
    "casting",
    "exfoliation",
    "synthesis",
    "chemical vapor deposition",
    "catalysis",
    "etching",
    "templating",
    "doping",
    "functionalization",
    "grinding",
    "milling",
}

UNIT_KEYWORDS = {
    "%",
    "v",
    "kv",
    "mv",
    "ma",
    "pa",
    "kpa",
    "mpa",
    "gpa",
    "w",
    "kw",
    "mw",
    "gw",
    "hz",
    "khz",
    "mhz",
    "ghz",
    "s/cm",
    "s/m",
    "cm2/vs",
    "cm\u00b2/v\u00b7s",
    "ohm",
    "\u03a9",
    "g/cm3",
    "mg/ml",
    "mah/g",
    "mah g-1",
    "cd/m2",
    "cm-1",
    "nm",
    "mm",
    "cm",
    "pm",
    "um",
    "lm",
    "\u00b0c",
    "ev",
    "kj",
    "hz",
    "khz",
    "mhz",
    "ghz",
    "s-1",
}

UNIT_TERMS = {term.upper() for term in UNIT_KEYWORDS}

TITLE_ALIASES = {
    "title",
    "article title",
    "paper title",
    "document title",
    "publication title",
}

ABSTRACT_ALIASES = {
    "abstract",
    "abstract text",
    "summary",
    "description",
    "abstracttext",
}


GETTING_STARTED_MARKDOWN = f"""
### Getting Started
1. Prepare an Excel file (`.xls` or `.xlsx`) with **Title** and **Abstract** columns.
2. Click **Browse files** to upload it from your computer.
3. Pick a pretrained model and confidence threshold from the sidebar.
4. Press **Run NER extraction** to process your abstracts.
5. Use the filters, search, and downloads to explore the results.

Need more detail? [Open the README]({GITHUB_URL}#readme).
"""


def show_getting_started_help():
    """Show onboarding instructions using the best available UI affordance."""
    dialog = getattr(st, "dialog", None)
    if callable(dialog):

        @dialog("Getting Started", width="large")
        def _dialog() -> None:
            st.markdown(GETTING_STARTED_MARKDOWN)

        _dialog()
    else:
        st.session_state["show_inline_help"] = True

VALUE_PATTERN = re.compile(
    r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s?(?:/|per)?\s?[A-Za-z\u00b7\u00b0\u03a9%0-9^\/\-\*]+)?$"
)
# ADDED: Extra patterns for value/unit recognition.
SIMPLE_VALUE_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?$")
VALUE_WITH_UNIT_PATTERN = re.compile(
    r"^[+-]?\d+(?:\.\d+)?\s?(?:%|ppm|ppb|m|cm|mm|nm|pm|km|g|kg|mg|ug|µg|A|mA|V|W|kW|J|K|°C|°F|s|ms|µs|ns|Hz|kHz|MHz|GHz|Pa|kPa|MPa|bar|mol|M|nM|µM)(?:[\-\/][A-Za-z]+)?$",
    flags=re.IGNORECASE,
)
UNIT_ONLY_PATTERN = re.compile(r"^[A-Za-z\u00b7\u00b0\u03a9µµ/\\-]{1,10}$")

MATERIAL_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9\-\+\(\)\/\.\u00b7\u00b0]{2,}$")


# === VALUE–UNIT LINKING LOGIC ======================================================
def _legacy_link_values_with_units(entity_df: pd.DataFrame, max_distance: int = 30) -> pd.DataFrame:
    """Legacy helper retained for backward compatibility (no longer used)."""
    return entity_df

# Legacy implementation retained below for historical reference. Unreachable due to
# early return above and kept only to aid future diff reviews.
    pandas_module = globals().get("pd")
    if pandas_module is None:
        if DEBUG_MODE:
            debug_log("[ValueUnitLinker] pandas unavailable; skipping value–unit linking.")
        return entity_df
    if entity_df is None:
        if DEBUG_MODE:
            debug_log("[ValueUnitLinker] Received None entity dataframe; skipping.")
        return entity_df

    df = entity_df.copy()
    if df.empty:
        if DEBUG_MODE:
            debug_log("[ValueUnitLinker] Empty dataframe; nothing to link.")
        if "LinkedUnit" not in df.columns:
            df["LinkedUnit"] = pandas_module.Series([], dtype="object")
        if "Measurement" not in df.columns and "entity" in df.columns:
            df["Measurement"] = pandas_module.Series([], dtype="object")
        return df

    if "entity" not in df.columns:
        if DEBUG_MODE:
            debug_log("[ValueUnitLinker] Missing 'entity' column; aborting linking.")
        return df

    type_column = next((col for col in ("EntityType", "entity_type", "Type") if col in df.columns), None)
    if type_column is None:
        if DEBUG_MODE:
            debug_log("[ValueUnitLinker] Missing entity type column; cannot classify VALUE/UNIT rows.")
        df["LinkedUnit"] = df.get("LinkedUnit", pandas_module.Series([None] * len(df), index=df.index))
        df["Measurement"] = df.get("Measurement", df["entity"].astype(str).str.strip())
        return df

    if "start" not in df.columns or "end" not in df.columns:
        if DEBUG_MODE:
            debug_log("[ValueUnitLinker] Missing character offsets; unable to compute distances.")
        df["LinkedUnit"] = df.get("LinkedUnit", pandas_module.Series([None] * len(df), index=df.index))
        df["Measurement"] = df.get("Measurement", df["entity"].astype(str).str.strip())
        return df

    df["LinkedUnit"] = pandas_module.Series([None] * len(df), index=df.index)
    df["Measurement"] = df["entity"].astype(str).str.strip()

    numeric_starts = pandas_module.to_numeric(df["start"], errors="coerce")
    numeric_ends = pandas_module.to_numeric(df["end"], errors="coerce")
    df["_start_num"] = numeric_starts
    df["_end_num"] = numeric_ends
    df["_type_upper"] = df[type_column].astype(str).str.upper()

    max_gap = max(0, int(max_distance or 0))

    total_values = int((df["_type_upper"] == "VALUE").sum())
    total_units = int((df["_type_upper"] == "UNIT").sum())
    linked_pairs = 0
    skipped_offsets = 0
    skipped_no_units = 0
    processed_papers: Set[int] = set()
    linked_papers: Set[int] = set()

    def _log_debug(message: str) -> None:
        if DEBUG_MODE:
            debug_log(f"[ValueUnitLinker] {message}")

    _log_debug(
        f"Starting value–unit linking: rows={len(df)}, values={total_values}, "
        f"units={total_units}, max_distance={max_gap}"
    )

    def _is_valid_unit(text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned or len(cleaned) > 20:
            return False
        if UNIT_ONLY_PATTERN.match(cleaned):
            return True
        if cleaned.upper() in UNIT_TERMS:
            return True
        return bool(re.fullmatch(r"[A-Za-z0-9µμ%°·^/\\\-\.\s]+", cleaned))

    for paper_id, group_df in df.groupby("paper_id", sort=False):
        processed_papers.add(paper_id)
        value_rows = group_df[group_df["_type_upper"] == "VALUE"]
        unit_rows = group_df[group_df["_type_upper"] == "UNIT"]
        if value_rows.empty or unit_rows.empty:
            skipped_no_units += 1
            _log_debug(
                f"Paper {paper_id}: skipped linking (values={len(value_rows)}, units={len(unit_rows)})."
            )
            continue

        paper_linked = 0
        paper_skipped_offsets = 0

        for value_idx, value_row in value_rows.iterrows():
            v_start = value_row["_start_num"]
            v_end = value_row["_end_num"]
            if pandas_module.isna(v_start) or pandas_module.isna(v_end):
                paper_skipped_offsets += 1
                continue
            try:
                v_start_int = int(v_start)
                v_end_int = int(v_end)
            except (TypeError, ValueError):
                paper_skipped_offsets += 1
                continue

            best_choice: Optional[Tuple[int, int, int, int]] = None
            best_unit_text: Optional[str] = None
            value_text = str(value_row["entity"]).strip()

            for _, unit_row in unit_rows.iterrows():
                unit_text = str(unit_row["entity"]).strip()
                if not _is_valid_unit(unit_text):
                    continue
                u_start = unit_row["_start_num"]
                u_end = unit_row["_end_num"]
                if pandas_module.isna(u_start) or pandas_module.isna(u_end):
                    continue
                try:
                    u_start_int = int(u_start)
                    u_end_int = int(u_end)
                except (TypeError, ValueError):
                    continue

                if u_start_int >= v_end_int:
                    direction = 0  # unit follows value
                    gap = u_start_int - v_end_int
                elif u_end_int <= v_start_int:
                    direction = 1  # unit precedes value
                    gap = v_start_int - u_end_int
                else:
                    direction = 0
                    gap = 0

                if gap > max_gap:
                    continue

                proximity = abs((u_start_int + u_end_int) // 2 - (v_start_int + v_end_int) // 2)
                tie_breaker = abs(u_start_int - v_end_int)
                candidate = (direction, gap, proximity, tie_breaker)

                if best_choice is None or candidate < best_choice:
                    best_choice = candidate
                    best_unit_text = unit_text

            if best_unit_text:
                df.at[value_idx, "LinkedUnit"] = best_unit_text
                df.at[value_idx, "Measurement"] = f"{value_text} {best_unit_text}".strip()
                linked_pairs += 1
                paper_linked += 1

        if paper_skipped_offsets:
            skipped_offsets += paper_skipped_offsets
        if paper_linked:
            linked_papers.add(paper_id)

        _log_debug(
            f"Paper {paper_id}: values={len(value_rows)}, units={len(unit_rows)}, "
            f"linked={paper_linked}, skipped_offsets={paper_skipped_offsets}"
        )

    df = df.drop(columns=["_start_num", "_end_num", "_type_upper"], errors="ignore")

    # Only update/append LinkedUnit & Measurement columns; preserve everything else verbatim.
    for column in ("LinkedUnit", "Measurement"):
        if column in df.columns:
            if column in entity_df.columns:
                entity_df[column] = df[column]
            else:
                entity_df[column] = df[column]

    summary_message = (
        f"Linked {linked_pairs} VALUE–UNIT pairs across {len(linked_papers)} papers "
        f"(values observed: {total_values}, units observed: {total_units}, "
        f"papers processed: {len(processed_papers)}, skipped papers: {skipped_no_units}, "
        f"skipped offsets: {skipped_offsets}). Canonical mappings preserved."
    )

    _log_debug("Completed value–unit linking.")
    _log_debug(summary_message)

    if st is not None:
        try:
            st.caption(f"🔗 {summary_message}")
        except Exception:
            pass
    else:
        safe_print(summary_message)

    return entity_df


@st.cache_resource(show_spinner="Loading NER model...")
def load_ner_pipeline(
    model_name: str,
    token_signature: Optional[str],
    device_preference: str,
) -> Tuple[Optional[object], str]:
    """Load a Hugging Face NER pipeline with detailed logging and safe fallbacks."""

    _ = token_signature  # retained for cache signature consistency
    del device_preference  # unused but part of cached signature

    def emit(level: str, message: str) -> None:
        formatted = safe_str(message)
        if st is None:
            debug_log(f"{level.upper()}: {formatted}")
            return
        try:
            if level == "info":
                st.info(formatted)
            elif level == "success":
                st.success(formatted)
            elif level == "warning":
                st.warning(formatted)
            else:
                st.error(formatted)
        except Exception:
            debug_log(f"{level.upper()}: {formatted}")

    if pipeline is None:  # pragma: no cover
        emit(
            "error",
            "❌ transformers.pipeline is unavailable. Install dependencies and restart the app.",
        )
        dummy = DummyPipeline()
        emit("error", "⚠️ Running in Dummy mode — no real NER model is active.")
        return dummy, "local-dummy"

    token = os.environ.get("HUGGINGFACE_TOKEN") or None
    offline = is_offline()

    torch_module = torch
    torch_version = getattr(torch_module, "__version__", "unknown") if torch_module is not None else "unknown"

    def _parse_major_minor(version: str) -> Tuple[int, int]:
        numeric = re.split(r"[^\d]+", version)
        digits: List[int] = [int(part) for part in numeric if part.isdigit()]
        while len(digits) < 2:
            digits.append(0)
        return digits[0], digits[1]

    torch_ready_for_scibert = True
    if torch_module is not None:
        major, minor = _parse_major_minor(torch_version)
        if (major, minor) < (2, 6):
            torch_ready_for_scibert = False
            emit(
                "warning",
                f"⚠️ Torch {torch_version} detected. SciBERT requires torch >= 2.6.0. "
                "Upgrade with `pip install --upgrade torch` for best results.",
            )
    else:  # pragma: no cover
        torch_ready_for_scibert = False
        emit("warning", "⚠️ Torch library is unavailable; NER models may fail to load.")

    if torch_module is not None:
        try:
            mps_backend = getattr(torch_module.backends, "mps", None)
            if mps_backend is not None and hasattr(mps_backend, "is_available") and mps_backend.is_available():
                fallback_flag = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
                if fallback_flag == "1":
                    emit("info", "ℹ️ Apple Silicon detected — MPS fallback to CPU is enabled.")
                else:
                    emit(
                        "warning",
                        "⚠️ Apple Silicon detected — enable CPU fallback by setting PYTORCH_ENABLE_MPS_FALLBACK=1.",
                    )
        except Exception as mps_exc:  # pragma: no cover
            debug_log("MPS availability check failed:", mps_exc)

    token_kwargs: Dict[str, object] = {"token": token} if token else {}
    if not token:
        emit("warning", "⚠️ Hugging Face token not detected — loading only public models.")

    attempts: List[Tuple[str, str, Dict[str, object]]] = []
    seen: Set[str] = set()

    def add_attempt(identifier: Optional[str], label: str, extra: Optional[Dict[str, object]] = None) -> None:
        if not identifier:
            return
        if identifier in DISABLED_MODELS:
            emit(
                "warning",
                f"⚠️ Skipping {label} — {DISABLED_MODELS[identifier]}",
            )
            return
        if not torch_ready_for_scibert and identifier in SCIBERT_MODELS:
            emit(
                "warning",
                f"⚠️ Skipping {label} — requires torch >= 2.6.0 (detected {torch_version}).",
            )
            return
        if identifier in seen:
            return
        seen.add(identifier)
        attempts.append((identifier, label, extra or {}))

    add_attempt(model_name, model_name)
    if model_name != DEFAULT_MODEL:
        add_attempt(DEFAULT_MODEL, DEFAULT_MODEL)

    for local_path, label in LOCAL_MODEL_CANDIDATES:
        candidate = Path(local_path)
        if candidate.exists():
            add_attempt(str(candidate.resolve()), label)

    if OFFLINE_MODEL_DIR.exists():
        add_attempt(str(OFFLINE_MODEL_DIR.resolve()), DEFAULT_MODEL)

    add_attempt("dslim/bert-base-NER", "dslim/bert-base-NER", token_kwargs)
    add_attempt("allenai/scibert_scivocab_uncased", "allenai/scibert_scivocab_uncased", token_kwargs)
    add_attempt("Jean-Baptiste/roberta-large-ner-english", "Jean-Baptiste/roberta-large-ner-english", token_kwargs)

    if offline:
        emit(
            "warning",
            "⚠️ Offline mode detected — attempting to use local or cached models first.",
        )

    for identifier, resolved_label, extra_kwargs in attempts:
        try:
            emit("info", f"🔍 Attempting to load model: {resolved_label}")
            kwargs: Dict[str, object] = {
                "model": identifier,
                "aggregation_strategy": "simple",
            }
            kwargs.update(extra_kwargs)
            ner = pipeline("token-classification", **kwargs)

            tokenizer = getattr(ner, "tokenizer", None)
            if tokenizer is not None and not getattr(tokenizer, "is_fast", False):
                try:
                    fast_kwargs: Dict[str, object] = {"use_fast": True}
                    fast_kwargs.update(extra_kwargs)
                    fast_tokenizer = AutoTokenizer.from_pretrained(identifier, **fast_kwargs)
                    ner.tokenizer = fast_tokenizer
                except Exception as fast_exc:  # pragma: no cover
                    emit(
                        "warning",
                        f"⚠️ Could not load fast tokenizer for {resolved_label}: "
                        f"{type(fast_exc).__name__} - {fast_exc}",
                    )

            extend_tokenizer_with_domain_terms(ner)
            refresh_labels_from_pipeline(ner)
            emit("success", f"✅ Model loaded successfully: {resolved_label}")
            return ner, resolved_label
        except Exception as load_exc:
            emit(
                "error",
                f"❌ Failed to load model {resolved_label}: {type(load_exc).__name__} - {load_exc}",
            )
            continue

    emit("warning", "⚠️ All model load attempts failed — using DummyPipeline fallback.")
    apply_model_label_schema(MODEL_ID2LABEL)
    dummy = DummyPipeline()
    emit("error", "⚠️ Running in Dummy mode — no real NER model is active.")
    return dummy, "local-dummy"


def normalize_text(value: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def flatten_iterable(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd is not None and pd.isna(value):
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = ""
    if isinstance(value, str):
        text = value.strip()
        if text in {"set()", "[]", "{}", "()"}:
            return ""
        return text
    if isinstance(value, Mapping):
        items: List[str] = []
        for key, val in value.items():
            key_clean = flatten_iterable(key)
            val_clean = flatten_iterable(val)
            if key_clean and val_clean:
                items.append(f"{key_clean}:{val_clean}")
            elif val_clean:
                items.append(val_clean)
            elif key_clean:
                items.append(key_clean)
        items = [item for item in items if item]
        items.sort(key=lambda entry: entry.lower())
        return ", ".join(items)
    if isinstance(value, (set, list, tuple)):
        flattened = [flatten_iterable(item) for item in value]
        flattened = [item for item in flattened if item]
        flattened.sort(key=lambda entry: entry.lower())
        return ", ".join(flattened)
    if isinstance(value, IterableABC) and not isinstance(value, (str, bytes)):
        flattened = [flatten_iterable(item) for item in value]
        flattened = [item for item in flattened if item]
        flattened.sort(key=lambda entry: entry.lower())
        return ", ".join(flattened)
    return str(value).strip()


def deep_sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sanitized = df.copy()
    numeric_candidates = {"confidence", "Confidence", "score", "Score"}
    for column in sanitized.columns:
        sanitized[column] = sanitized[column].apply(flatten_iterable)
    for column in sanitized.columns:
        if column in numeric_candidates:
            sanitized[column] = pd.to_numeric(sanitized[column], errors="coerce")
    for column in sanitized.columns:
        if column not in numeric_candidates:
            sanitized[column] = sanitized[column].astype(str)
    return sanitized


def sanitize_entities_output(obj):
    if isinstance(obj, pd.DataFrame):
        return deep_sanitize_dataframe(obj)
    return flatten_iterable(obj)


SESSION_USER_ENTITIES_KEY = "custom_user_entities"
SESSION_USER_CANONICAL_KEY = "custom_user_canonical"
SESSION_CANONICAL_MERGES_KEY = "custom_canonical_merges"


def _ensure_user_cache_dir() -> None:
    try:
        USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _clean_term(value: object) -> str:
    return flatten_iterable(value).strip()


def _deduplicate_terms(terms: Iterable[object]) -> List[str]:
    seen: Set[str] = set()
    cleaned: List[str] = []
    for term in terms:
        text = _clean_term(term)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


# ADDED: Normalize custom entity entries with type metadata.
def _normalize_user_entity_entry(entry: object) -> Optional[Dict[str, str]]:
    term = ""
    entity_type = "O"
    if isinstance(entry, Mapping):
        term = _clean_term(entry.get("term"))
        entity_type = canonicalize_label(entry.get("type", "O"), warn=True)
    else:
        term = _clean_term(entry)
    if not term:
        return None
    if entity_type not in ENTITY_TYPES:
        entity_type = "O"
    return {"term": term, "type": entity_type}


# FIXED: Persist custom entities as term/type records.
def load_user_entities_cache() -> List[Dict[str, str]]:
    _ensure_user_cache_dir()
    if not USER_ENTITIES_FILE.exists():
        return []
    try:
        data = json.loads(USER_ENTITIES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = data.get("user_entities", [])
    if not isinstance(entries, IterableABC) or isinstance(entries, (str, bytes)):
        return []
    normalized_map: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for entry in entries:
        normalized = _normalize_user_entity_entry(entry)
        if not normalized:
            continue
        key = normalized["term"].lower()
        if key not in normalized_map:
            order.append(key)
        normalized_map[key] = normalized
    return [normalized_map[key] for key in order]


# FIXED: Store custom entity types alongside terms.
def save_user_entities_cache(terms: Iterable[object]) -> List[Dict[str, str]]:
    normalized_map: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for entry in terms:
        normalized = _normalize_user_entity_entry(entry)
        if not normalized:
            continue
        key = normalized["term"].lower()
        if key not in normalized_map:
            order.append(key)
        normalized_map[key] = normalized
    cleaned = [normalized_map[key] for key in order]
    payload = {"user_entities": cleaned}
    _ensure_user_cache_dir()
    try:
        USER_ENTITIES_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return cleaned


def load_user_canonical_cache() -> Dict[str, List[str]]:
    _ensure_user_cache_dir()
    if not USER_CANONICAL_FILE.exists():
        return {}
    try:
        data = json.loads(USER_CANONICAL_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_map = data.get("user_canonical", {})
    if not isinstance(raw_map, Mapping):
        return {}
    cleaned: Dict[str, List[str]] = {}
    for canonical, variants in raw_map.items():
        canonical_term = _clean_term(canonical)
        if not canonical_term:
            continue
        if isinstance(variants, IterableABC) and not isinstance(variants, (str, bytes)):
            cleaned_variants = _deduplicate_terms(variants)
        else:
            cleaned_variants = []
        cleaned[canonical_term] = cleaned_variants
    return cleaned


def save_user_canonical_cache(mapping: Mapping[object, Iterable[object]]) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for canonical, variants in mapping.items():
        canonical_term = _clean_term(canonical)
        if not canonical_term:
            continue
        cleaned_variants = _deduplicate_terms(variants)
        cleaned[canonical_term] = cleaned_variants
    payload = {"user_canonical": cleaned}
    _ensure_user_cache_dir()
    try:
        USER_CANONICAL_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return cleaned


def get_user_entities_state() -> List[Dict[str, str]]:
    if SESSION_USER_ENTITIES_KEY not in st.session_state:
        st.session_state[SESSION_USER_ENTITIES_KEY] = load_user_entities_cache()
    return list(st.session_state[SESSION_USER_ENTITIES_KEY])


def update_user_entities_state(terms: Iterable[object]) -> List[Dict[str, str]]:
    cleaned = save_user_entities_cache(terms)
    st.session_state[SESSION_USER_ENTITIES_KEY] = cleaned
    return list(cleaned)


def reset_user_entities_state() -> List[Dict[str, str]]:
    cleaned = save_user_entities_cache([])
    st.session_state[SESSION_USER_ENTITIES_KEY] = cleaned
    return list(cleaned)


def get_user_canonical_state() -> Dict[str, List[str]]:
    if SESSION_USER_CANONICAL_KEY not in st.session_state:
        st.session_state[SESSION_USER_CANONICAL_KEY] = load_user_canonical_cache()
    return dict(st.session_state[SESSION_USER_CANONICAL_KEY])


def update_user_canonical_state(mapping: Mapping[object, Iterable[object]]) -> Dict[str, List[str]]:
    cleaned = save_user_canonical_cache(mapping)
    st.session_state[SESSION_USER_CANONICAL_KEY] = cleaned
    return dict(cleaned)


def reset_user_canonical_state() -> Dict[str, List[str]]:
    cleaned = save_user_canonical_cache({})
    st.session_state[SESSION_USER_CANONICAL_KEY] = cleaned
    return dict(cleaned)


def load_user_canonical_merges_cache() -> Dict[str, str]:
    _ensure_user_cache_dir()
    if not USER_CANONICAL_MERGES_FILE.exists():
        return {}
    try:
        data = json.loads(USER_CANONICAL_MERGES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_map = data.get("merged_canonicals", {})
    if not isinstance(raw_map, Mapping):
        return {}
    cleaned: Dict[str, str] = {}
    for source, target in raw_map.items():
        source_clean = _clean_term(source)
        target_clean = _clean_term(target)
        if not source_clean or not target_clean:
            continue
        if source_clean.lower() == target_clean.lower():
            continue
        cleaned[source_clean] = target_clean
    return cleaned


def save_user_canonical_merges_cache(mapping: Mapping[object, object]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for source, target in mapping.items():
        source_clean = _clean_term(source)
        target_clean = _clean_term(target)
        if not source_clean or not target_clean:
            continue
        if source_clean.lower() == target_clean.lower():
            continue
        cleaned[source_clean] = target_clean
    payload = {"merged_canonicals": cleaned}
    _ensure_user_cache_dir()
    try:
        USER_CANONICAL_MERGES_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return cleaned


def get_user_canonical_merges_state() -> Dict[str, str]:
    if SESSION_CANONICAL_MERGES_KEY not in st.session_state:
        st.session_state[SESSION_CANONICAL_MERGES_KEY] = load_user_canonical_merges_cache()
    return dict(st.session_state[SESSION_CANONICAL_MERGES_KEY])


def update_user_canonical_merges_state(mapping: Mapping[object, object]) -> Dict[str, str]:
    cleaned = save_user_canonical_merges_cache(mapping)
    st.session_state[SESSION_CANONICAL_MERGES_KEY] = cleaned
    return dict(cleaned)


def reset_user_canonical_merges_state() -> Dict[str, str]:
    cleaned = save_user_canonical_merges_cache({})
    st.session_state[SESSION_CANONICAL_MERGES_KEY] = cleaned
    return dict(cleaned)

# ADDED: Guarantee entity schema consistency before merges or filtering.
def ensure_entity_schema(df: pd.DataFrame, context: str = "entity") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    created: List[str] = []
    if "entity" not in working.columns:
        working["entity"] = ""
        created.append("entity")
    if "entity_type" not in working.columns:
        working["entity_type"] = "UNKNOWN"
        created.append("entity_type")
    if "entity_norm" not in working.columns:
        base_series = working.get("entity", pd.Series([""] * len(working), index=working.index))
        working["entity_norm"] = base_series.astype(str).str.lower()
        created.append("entity_norm")
    if "canonical" not in working.columns:
        source_series = None
        for candidate in ("entity_norm", "entity"):
            if candidate in working.columns:
                source_series = working[candidate].astype(str)
                break
        if source_series is None:
            source_series = pd.Series([""] * len(working), index=working.index)
        working["canonical"] = source_series
        created.append("canonical")
    if "Canonical" not in working.columns:
        working["Canonical"] = working["canonical"].astype(str).str.upper()
        created.append("Canonical")
    for column in ("entity", "entity_norm", "canonical", "Canonical", "entity_type"):
        working[column] = working[column].astype(str)
    if created:
        message = f"Adjusted missing columns {', '.join(sorted(created))} in {context} dataset."
        if st is not None:
            st.warning(message)
        else:
            safe_print(message)
    return working


def apply_user_canonical_overrides(
    entity_df: pd.DataFrame, user_canonical: Optional[Mapping[str, Iterable[str]]]
) -> pd.DataFrame:
    entity_df = ensure_entity_schema(entity_df, context="user canonical override")
    if entity_df.empty or not user_canonical or "entity" not in entity_df.columns:
        return entity_df
    df = entity_df.copy()
    type_column = next((col for col in ("EntityType", "entity_type", "Type") if col in df.columns), None)
    skip_types = {"VALUE", "PROP_VALUE", "MATERIAL_AMOUNT"}
    if "canonical" not in df.columns:
        if "entity_norm" in df.columns:
            df["canonical"] = df["entity_norm"].astype(str)
        else:
            df["canonical"] = df["entity"].astype(str)
    if "Canonical" not in df.columns:
        df["Canonical"] = df["canonical"].astype(str).str.upper()
    entity_lower = df["entity"].astype(str).str.lower()
    for canonical, variants in user_canonical.items():
        canonical_term = _clean_term(canonical)
        if not canonical_term:
            continue
        keys = {canonical_term.lower()}
        for variant in variants:
            variant_term = _clean_term(variant)
            if variant_term:
                keys.add(variant_term.lower())
        if not keys:
            continue
        mask = entity_lower.isin(keys)
        if type_column:
            mask = mask & ~df[type_column].astype(str).str.upper().isin(skip_types)
        if not mask.any():
            continue
        df.loc[mask, "canonical"] = canonical_term
        df.loc[mask, "Canonical"] = canonical_term.upper()
    return df


def _build_canonical_merge_lookup(mapping: Mapping[str, str]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for source, target in mapping.items():
        source_clean = _clean_term(source)
        target_clean = _clean_term(target)
        if not source_clean or not target_clean:
            continue
        if source_clean.lower() == target_clean.lower():
            continue
        cleaned[source_clean.lower()] = target_clean
    resolved: Dict[str, str] = {}
    for source_lower, initial_target in cleaned.items():
        current = initial_target
        seen: Set[str] = {source_lower}
        while current and current.lower() in cleaned and current.lower() not in seen:
            seen.add(current.lower())
            current = cleaned[current.lower()]
        if current:
            resolved[source_lower] = current
    return resolved


def apply_canonical_merge_map(entity_df: pd.DataFrame, merge_map: Mapping[str, str]) -> pd.DataFrame:
    entity_df = ensure_entity_schema(entity_df, context="canonical merge")
    if entity_df.empty or not merge_map:
        return entity_df
    type_column = next((col for col in ("EntityType", "entity_type", "Type") if col in entity_df.columns), None)
    skip_types = {"VALUE", "PROP_VALUE", "MATERIAL_AMOUNT"}
    lookup = _build_canonical_merge_lookup(merge_map)
    if not lookup:
        return entity_df
    df = entity_df.copy()
    def resolve(value: object) -> str:
        text = _clean_term(value)
        if not text:
            return ""
        key = text.lower()
        target = lookup.get(key)
        return target if target else text
    type_mask = (
        ~df[type_column].astype(str).str.upper().isin(skip_types)
        if type_column else pd.Series([True] * len(df))
    )
    if "canonical" in df.columns:
        df.loc[type_mask, "canonical"] = df.loc[type_mask, "canonical"].apply(resolve)
    if "Canonical" in df.columns:
        df.loc[type_mask, "Canonical"] = df.loc[type_mask, "canonical"].astype(str).str.upper()
    if "entity_norm" in df.columns:
        df.loc[type_mask, "entity_norm"] = df.loc[type_mask, "entity_norm"].apply(
            lambda val: normalize_entity(resolve(val))
        )
    return df


def _merge_value_list(value: object, lookup: Mapping[str, str], normalize: bool = False, uppercase: bool = False) -> str:
    items = _split_entities(value)
    if not items:
        return ""
    merged: List[str] = []
    for item in items:
        key = item.lower()
        target = lookup.get(key)
        replacement = target if target else item
        if normalize:
            replacement = normalize_entity(replacement)
        merged.append(replacement)
    unique: List[str] = []
    seen_keys: Set[str] = set()
    for entry in merged:
        entry_clean = _clean_term(entry)
        if not entry_clean:
            continue
        key = entry_clean.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(entry_clean.upper() if uppercase else entry_clean)
    return ", ".join(unique)


def apply_canonical_merges_to_processed_df(
    processed_df: pd.DataFrame, merge_map: Mapping[str, str]
) -> pd.DataFrame:
    if processed_df.empty or not merge_map:
        return processed_df
    lookup = _build_canonical_merge_lookup(merge_map)
    if not lookup:
        return processed_df
    df = processed_df.copy()
    normalized_columns = {f"{etype} Normalized" for etype in ENTITY_TYPES} | {"All Entities Normalized"}
    uppercase_columns = set(ENTITY_TYPES) | {"All Entities"}
    for column in df.columns:
        if column in normalized_columns:
            df[column] = df[column].apply(lambda val: _merge_value_list(val, lookup, normalize=True))
        elif column in uppercase_columns:
            df[column] = df[column].apply(lambda val: _merge_value_list(val, lookup, uppercase=True))
    return df


# ADDED: Filter entity DataFrame by selected types.
def filter_by_entity_type(entity_df: pd.DataFrame, selected_types: Iterable[str]) -> pd.DataFrame:
    if entity_df.empty:
        return entity_df
    allowed = {str(t) for t in selected_types if str(t).strip()}
    full_allowed = set(ENTITY_TYPES) | set(CUSTOM_ENTITY_TYPES)
    if allowed:
        full_allowed |= allowed
    filtered = entity_df[
        entity_df["entity_type"].isin(full_allowed) | entity_df.get("is_user_entity", False)
    ].copy()
    return filtered


# ADDED: Apply exclusions to entity and processed datasets.
def apply_exclusions(
    entity_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    excluded_canonical: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    if not excluded_canonical or entity_df.empty:
        return entity_df, processed_df, None
    excluded_upper = excluded_canonical.upper()
    excluded_ids: Set[int] = set()
    if {"Canonical", "paper_id"}.issubset(entity_df.columns):
        mask = entity_df["Canonical"].astype(str).str.upper() == excluded_upper
        excluded_ids = set(entity_df.loc[mask, "paper_id"].astype(int).tolist())
        entity_df = entity_df[~mask].copy()
    if not processed_df.empty and excluded_ids and "paper_id" in processed_df.columns:
        processed_df = processed_df[
            ~processed_df["paper_id"].astype(int).isin(excluded_ids)
        ].copy()
    return entity_df, processed_df, excluded_canonical


# ADDED: Display a concise exclusion summary.
def render_exclusion_summary(excluded_canonical: Optional[str]) -> None:
    if not excluded_canonical:
        return
    st.info(safe_to_str(f"Excluded canonical entity: {excluded_canonical}"))


# ADDED: Merge selected variants within a canonical group.
def merge_selected_variants(
    canonical_map: Dict[str, List[str]],
    selected_group: str,
    chosen_variants: Iterable[str],
) -> Dict[str, List[str]]:
    updated = {key: list(values) for key, values in canonical_map.items()}
    canonical_key = _clean_term(selected_group)
    if not canonical_key or canonical_key not in updated:
        return updated
    additions = [_clean_term(variant) for variant in chosen_variants]
    additions = [variant for variant in additions if variant]
    if not additions:
        return updated
    updated[canonical_key] = _deduplicate_terms(updated.get(canonical_key, []) + additions)
    return updated


# ADDED: Remove selected variants from a canonical group.
def delete_selected_variants(
    canonical_map: Dict[str, List[str]],
    selected_group: str,
    chosen_variants: Iterable[str],
) -> Dict[str, List[str]]:
    updated = {key: list(values) for key, values in canonical_map.items()}
    canonical_key = _clean_term(selected_group)
    if not canonical_key or canonical_key not in updated:
        return updated
    removals = {_clean_term(variant) for variant in chosen_variants if _clean_term(variant)}
    if not removals:
        return updated
    remaining = [variant for variant in updated[canonical_key] if _clean_term(variant) not in removals]
    updated[canonical_key] = remaining
    return updated


# ADDED: Inject user-defined entities with explicit types.
def inject_user_entities(
    entity_df: pd.DataFrame,
    paper_text_map: Mapping[int, object],
    user_terms: Iterable[object],
) -> Tuple[pd.DataFrame, Dict[int, List[Dict[str, str]]]]:
    def _normalize_match_strings(value: object) -> Tuple[str, str]:
        base = _clean_term(value).lower()
        if not base:
            return "", ""
        spaced = re.sub(r"[\s\-\u2010-\u2015]+", " ", base).strip()
        compact = re.sub(r"[\s\-\u2010-\u2015]+", "", base)
        return spaced, compact

    normalized_map: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for entry in user_terms or []:
        normalized = _normalize_user_entity_entry(entry)
        if not normalized:
            continue
        key = normalized["term"].lower()
        if key not in normalized_map:
            order.append(key)
        normalized_map[key] = normalized
    cleaned_terms = [normalized_map[key] for key in order]
    if not cleaned_terms or not paper_text_map:
        return entity_df, {}
    if entity_df.empty:
        base_columns = [
            "paper_id",
            "entity",
            "entity_norm",
            "entity_type",
            "confidence",
            "is_unknown",
            "is_user_entity",
            "start",
            "end",
        ]
        working = pd.DataFrame(columns=base_columns)
    else:
        working = entity_df.copy()
    if "is_user_entity" not in working.columns:
        working["is_user_entity"] = False
    existing: Set[Tuple[int, str, Optional[int]]] = set()
    if not working.empty and "paper_id" in working.columns and "entity" in working.columns:
        for _, row in working.iterrows():
            try:
                pid = int(row.get("paper_id"))
            except Exception:
                continue
            entity_value = _clean_term(row.get("entity"))
            norm_spaced, norm_compact = _normalize_match_strings(entity_value)
            normalized_key = norm_compact or norm_spaced
            if not normalized_key:
                continue
            start_val = row.get("start")
            try:
                start_val = int(start_val) if start_val is not None and str(start_val).strip() != "" else None
            except Exception:
                start_val = None
            existing.add((pid, normalized_key, start_val))
    additions: List[Dict[str, object]] = []
    per_paper: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for raw_pid, text in paper_text_map.items():
        try:
            paper_id = int(raw_pid)
        except Exception:
            continue
        # FIXED: user-defined entities now match dashed/compact variants and show up downstream
        haystack = _clean_term(text).lower()
        if not haystack:
            continue
        haystack = re.sub(r"[\u2010-\u2015]", "-", haystack)
        for entry in cleaned_terms:
            term_text = entry["term"]
            term_spaced, term_compact = _normalize_match_strings(term_text)
            normalized_key = term_compact or term_spaced
            if not normalized_key:
                continue
            tokens = [tok for tok in term_spaced.split(" ") if tok]
            if not tokens:
                continue
            separator = r"[\s\-\u2010-\u2015]*"
            pattern_body = separator.join(re.escape(tok) for tok in tokens)
            pattern = re.compile(rf"(?<!\w){pattern_body}(?:es|s)?(?!\w)", flags=re.IGNORECASE)
            # FIXED: user entities now include start/end offsets for relationship extraction.
            for match in pattern.finditer(haystack):
                start_idx = match.start()
                end_idx = match.end()
                key = (paper_id, normalized_key, start_idx)
                if key in existing:
                    if not any(
                        _normalize_match_strings(item["term"])[1] == normalized_key
                        for item in per_paper[paper_id]
                    ):
                        per_paper[paper_id].append({"term": term_text, "type": entry["type"]})
                    continue
                entity_type_value = canonicalize_label(entry["type"]) or "O"
                if entity_type_value not in ENTITY_TYPES:
                    entity_type_value = "O"
                additions.append(
                    {
                        "paper_id": paper_id,
                        "entity": term_text,
                        "entity_norm": normalize_entity(term_text),
                        "entity_type": entity_type_value,
                        "confidence": 1.0,
                        "is_unknown": False,
                        "is_user_entity": True,
                        "start": start_idx,
                        "end": end_idx,
                    }
                )
                existing.add(key)
                per_paper[paper_id].append({"term": term_text, "type": entity_type_value})
    if not additions:
        return working, per_paper
    addition_df = pd.DataFrame(additions)
    working = pd.concat([working, addition_df], ignore_index=True, sort=False)
    return working, per_paper


def augment_processed_with_user_entities(
    processed_df: pd.DataFrame,
    user_entities_by_paper: Mapping[int, List[Dict[str, str]]],
) -> pd.DataFrame:
    if processed_df.empty or not user_entities_by_paper:
        return processed_df
    working = processed_df.copy()
    for column in ("All Entities", "All Entities Normalized"):
        if column not in working.columns:
            working[column] = ""
    for idx in working.index:
        try:
            paper_id = int(working.at[idx, "paper_id"])
        except Exception:
            continue
        additions = user_entities_by_paper.get(paper_id, [])
        if not additions:
            continue
        for entry in additions:
            term = entry.get("term")
            entity_type = canonicalize_label(entry.get("type", "O"))
            if not term:
                continue
            if entity_type not in ENTITY_TYPES:
                entity_type = "O"
            normalized_col = f"{entity_type} Normalized"
            if entity_type not in working.columns:
                working[entity_type] = ""
            if normalized_col not in working.columns:
                working[normalized_col] = ""
            current_terms = _split_entities(working.at[idx, entity_type])
            if term not in current_terms:
                current_terms.append(term)
            working.at[idx, entity_type] = ", ".join(sorted(set(current_terms), key=str.lower))
            normalized_term = normalize_entity(term)
            current_norm = _split_entities(working.at[idx, normalized_col])
            if normalized_term not in current_norm:
                current_norm.append(normalized_term)
            working.at[idx, normalized_col] = ", ".join(sorted(set(current_norm), key=str.lower))
            all_terms = _split_entities(working.at[idx, "All Entities"])
            if term not in all_terms:
                all_terms.append(term)
            working.at[idx, "All Entities"] = ", ".join(sorted(set(all_terms), key=str.lower))
            all_norm = _split_entities(working.at[idx, "All Entities Normalized"])
            if normalized_term not in all_norm:
                all_norm.append(normalized_term)
            working.at[idx, "All Entities Normalized"] = ", ".join(
                sorted(set(all_norm), key=str.lower)
            )
    return working


def refresh_active_results_with_user_settings() -> None:
    bundle = st.session_state.get("processed_bundle")
    if not bundle:
        return
    processed_df = pd.DataFrame(bundle.get("processed_df", pd.DataFrame()))
    entity_df = pd.DataFrame(bundle.get("entity_df", pd.DataFrame()))
    if processed_df.empty and entity_df.empty:
        return
    paper_text_map: Dict[int, object] = {}
    if not processed_df.empty and {"paper_id", "Abstract"}.issubset(processed_df.columns):
        paper_text_map = {
            int(row.paper_id): row.Abstract
            for row in processed_df.itertuples()
            if pd.notna(row.paper_id)
        }
    user_entities = get_user_entities_state()
    entity_df, user_entities_map = inject_user_entities(entity_df, paper_text_map, user_entities)
    processed_df = augment_processed_with_user_entities(processed_df, user_entities_map)
    user_canonical = get_user_canonical_state()
    entity_df = apply_user_canonical_overrides(entity_df, user_canonical)
    merge_map = get_user_canonical_merges_state()
    entity_df = apply_canonical_merge_map(entity_df, merge_map)
    processed_df = apply_canonical_merges_to_processed_df(processed_df, merge_map)
    selection = st.session_state.get("exclude_entity_filter_main")
    selection = selection if selection and selection != "None" else None
    entity_df, processed_df, _ = apply_exclusions(
        entity_df,
        processed_df,
        selection,
    )
    processed_df = sanitize_processed_dataframe(
        processed_df,
        allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
        drop_duplicates=ENABLE_DEDUPLICATION,
    )
    entity_df = sanitize_entity_dataframe(entity_df)
    entity_df = filter_false_units(entity_df)
    entity_df = suppress_lonely_units(entity_df)
    entity_df = trim_and_filter_entities(entity_df)
    linking_context = sanitize_linking_context(bundle.get("linking_context"))
    refreshed_bundle = dict(bundle)
    refreshed_bundle["processed_df"] = processed_df
    refreshed_bundle["entity_df"] = entity_df
    refreshed_bundle["linking_context"] = linking_context
    st.session_state["processed_bundle"] = refreshed_bundle
    cache_key = st.session_state.get("active_cache_key")
    if cache_key:
        cache_entry = dict(get_results_cache().get(cache_key, {}))
        if cache_entry:
            cache_entry["processed_df"] = processed_df
            cache_entry["entity_df"] = entity_df
            cache_entry["linking_context"] = linking_context
            get_results_cache()[cache_key] = cache_entry


def session_available_entities() -> List[str]:
    bundle = st.session_state.get("processed_bundle")
    if not bundle:
        return []
    entity_frame = bundle.get("entity_df", pd.DataFrame())
    entity_df = pd.DataFrame(entity_frame)
    if entity_df.empty or "entity" not in entity_df.columns:
        return []
    values = [
        _clean_term(value)
        for value in entity_df["entity"].tolist()
    ]
    cleaned = _deduplicate_terms(values)
    cleaned.sort(key=str.lower)
    return cleaned


def session_available_canonicals() -> List[str]:
    bundle = st.session_state.get("processed_bundle")
    values: List[str] = []
    if bundle:
        entity_frame = bundle.get("entity_df", pd.DataFrame())
        entity_df = pd.DataFrame(entity_frame)
        if not entity_df.empty and "Canonical" in entity_df.columns:
            values.extend([_clean_term(val) for val in entity_df["Canonical"].tolist()])
    user_canonical = get_user_canonical_state()
    for canonical, variants in user_canonical.items():
        canonical_clean = _clean_term(canonical)
        if canonical_clean:
            values.append(canonical_clean)
        for variant in variants:
            variant_clean = _clean_term(variant)
            if variant_clean:
                values.append(variant_clean)
    merge_map = get_user_canonical_merges_state()
    for source, target in merge_map.items():
        source_clean = _clean_term(source)
        target_clean = _clean_term(target)
        if source_clean:
            values.append(source_clean)
        if target_clean:
            values.append(target_clean)
    cleaned = _deduplicate_terms(values)
    cleaned.sort(key=str.lower)
    return cleaned


def sanitize_entity_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sanitized = sanitize_entities_output(df.copy())
    user_rows = pd.DataFrame()
    if "is_user_entity" in sanitized.columns:
        sanitized["is_user_entity"] = (
            sanitized["is_user_entity"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        user_rows = sanitized[sanitized["is_user_entity"] == True].copy()
        sanitized = sanitized[sanitized["is_user_entity"] != True].copy()
    if "entity" in sanitized.columns:
        sanitized = sanitized[sanitized["entity"].str.strip() != ""]
    if "canonical_display" in sanitized.columns:
        sanitized = sanitized.drop(columns=["canonical_display"])
    if "entity_type" in sanitized.columns and "Type" not in sanitized.columns:
        sanitized["Type"] = sanitized["entity_type"].astype(str)
    for column in ("entity", "entity_norm", "entity_type", "canonical", "Canonical", "Type"):
        if column in sanitized.columns:
            sanitized[column] = sanitized[column].astype(str)
    if "is_user_entity" in sanitized.columns:
        sanitized["is_user_entity"] = sanitized["is_user_entity"].fillna(False).astype(bool)
    type_column = next((col for col in ("Type", "entity_type", "EntityType") if col in sanitized.columns), None)
    if "entity" in sanitized.columns and "canonical" not in sanitized.columns:
        if "entity_norm" in sanitized.columns:
            sanitized["canonical"] = sanitized["entity_norm"].astype(str)
        else:
            sanitized["canonical"] = sanitized["entity"].astype(str)
    if "canonical" in sanitized.columns and "Canonical" not in sanitized.columns:
        sanitized["Canonical"] = sanitized["canonical"].astype(str).str.upper()
    if type_column:
        sanitized[type_column] = sanitized[type_column].apply(lambda value: canonicalize_label(value, warn=True))
        sanitized["entity_type"] = sanitized[type_column]
        sanitized["Type"] = sanitized[type_column]
        fallback_mask = sanitized[type_column].astype(str).str.upper().isin({"PROP_VALUE", "MATERIAL_AMOUNT"})
        if "entity" in sanitized.columns and "canonical" in sanitized.columns:
            sanitized.loc[fallback_mask, "canonical"] = sanitized.loc[fallback_mask, "entity"].astype(str)
        if "entity" in sanitized.columns and "Canonical" in sanitized.columns:
            sanitized.loc[fallback_mask, "Canonical"] = sanitized.loc[fallback_mask, "entity"].astype(str)
    if not user_rows.empty:
        sanitized = pd.concat([sanitized, user_rows], ignore_index=True, sort=False)
        # FIXED: ensure user-injected entities retain canonical/type fields through sanitization
        if "canonical" in sanitized.columns:
            if "entity_norm" in sanitized.columns:
                sanitized["canonical"] = sanitized["canonical"].fillna(sanitized["entity_norm"])
            sanitized["canonical"] = sanitized["canonical"].fillna(sanitized["entity"])
        if "Canonical" in sanitized.columns and "canonical" in sanitized.columns:
            sanitized["Canonical"] = sanitized["Canonical"].fillna(sanitized["canonical"].astype(str).str.upper())
        if "Type" in sanitized.columns and "entity_type" in sanitized.columns:
            sanitized["Type"] = sanitized["Type"].fillna(sanitized["entity_type"])
    sanitized = sanitized.reset_index(drop=True)
    if ENABLE_DEDUPLICATION:
        sanitized = sanitized.drop_duplicates().reset_index(drop=True)
    preferred_order = [col for col in ["paper_id", "entity", "Canonical", "Type", "confidence"] if col in sanitized.columns]
    remaining = [col for col in sanitized.columns if col not in preferred_order]
    if preferred_order:
        sanitized = sanitized[preferred_order + remaining]
    cleaned = deep_clean(sanitized)
    pandas_module = globals().get("pd")
    if pandas_module is not None and isinstance(cleaned, pandas_module.DataFrame):
        return cleaned
    return sanitized


# === VALUE–UNIT LINKING LOGIC ======================================================


def build_property_value_unit_table(
    entity_df: pd.DataFrame,
    max_value_gap: int = 40,
    max_property_gap: int = 80,
) -> pd.DataFrame:
    """Return a simplified PROPERTY–VALUE table including value/context details."""
    pandas_module = globals().get("pd")
    columns = ["Property", "Value", "Context", "Source"]
    if pandas_module is None:
        if DEBUG_MODE:
            debug_log("[PropertyValueUnit] pandas unavailable; returning empty table.")
        return []
    if entity_df is None or entity_df.empty:
        if DEBUG_MODE:
            debug_log("[PropertyValueUnit] No entities available; returning empty table.")
        return pandas_module.DataFrame(columns=columns)
    if "entity" not in entity_df.columns:
        if DEBUG_MODE:
            debug_log("[PropertyValueUnit] Missing 'entity' column; aborting.")
        return pandas_module.DataFrame(columns=columns)

    type_column = next((col for col in ("EntityType", "entity_type", "Type") if col in entity_df.columns), None)
    if type_column is None:
        if DEBUG_MODE:
            debug_log("[PropertyValueUnit] Missing entity type column; aborting.")
        return pandas_module.DataFrame(columns=columns)
    if "paper_id" not in entity_df.columns:
        return pandas_module.DataFrame(columns=columns)

    df = entity_df.copy()
    df[type_column] = df[type_column].apply(canonicalize_label)
    if df.empty:
        return pandas_module.DataFrame(columns=columns)

    def _context_for(row: pd.Series) -> Optional[str]:
        for key in ("context", "Context", "sentence", "Sentence", "source_sentence"):
            if key in row and isinstance(row.get(key), str) and row.get(key).strip():
                return str(row.get(key)).strip()
        return None

    rows: List[Dict[str, object]] = []
    for paper_id, group_df in df.groupby("paper_id", sort=False):
        prop_rows = group_df[group_df[type_column] == "PROP_NAME"]
        value_rows = group_df[group_df[type_column].isin({"PROP_VALUE", "MATERIAL_AMOUNT"})]

        for _, val_row in value_rows.iterrows():
            val_text = str(val_row.get("entity", "")).strip()
            context_text = _context_for(val_row)
            best_property = None

            if not prop_rows.empty and "start" in val_row and "start" in prop_rows.columns:
                try:
                    v_start = int(pandas_module.to_numeric(val_row.get("start"), errors="coerce"))
                except Exception:
                    v_start = None
                if v_start is not None and not pandas_module.isna(v_start):
                    prop_rows["_start_num"] = pandas_module.to_numeric(prop_rows["start"], errors="coerce")
                    prop_rows["_start_num"] = prop_rows["_start_num"].fillna(prop_rows["_start_num"].max())
                    prop_rows["_dist"] = (prop_rows["_start_num"] - v_start).abs()
                    nearest = prop_rows.sort_values("_dist").head(1)
                    if not nearest.empty:
                        prop_row = nearest.iloc[0]
                        best_property = str(prop_row.get("canonical", prop_row.get("entity", ""))).strip() or None

            rows.append(
                {
                    "Property": best_property,
                    "Value": val_text if val_text else None,
                    "Context": context_text,
                    "Source": paper_id,
                }
            )

        if value_rows.empty:
            for _, prop_row in prop_rows.iterrows():
                rows.append(
                    {
                        "Property": str(prop_row.get("canonical", prop_row.get("entity", ""))).strip(),
                        "Value": None,
                        "Context": _context_for(prop_row),
                        "Source": paper_id,
                    }
                )

    return pandas_module.DataFrame(rows, columns=columns)


def trim_and_filter_entities(
    entity_df: pd.DataFrame,
    max_tokens: int = 6,
    max_chars: int = 80,
    remove_verbs: bool = True,
) -> pd.DataFrame:
    """Trim or remove entities that are too long or contain verbs."""

    pandas_module = globals().get("pd")
    if pandas_module is None or entity_df is None or entity_df.empty:
        return entity_df

    if "entity" not in entity_df.columns:
        return entity_df

    df = entity_df.copy()
    if "is_user_entity" in df.columns:
        user_rows = df[df["is_user_entity"] == True].copy()
        df = df[df["is_user_entity"] != True].copy()
    else:
        user_rows = pd.DataFrame()
    trailing_stopwords = {
        "and",
        "with",
        "was",
        "were",
        "is",
        "are",
        "the",
        "of",
        "for",
        "to",
        "on",
        "in",
        ",",
        "&",
        "and/or",
    }
    fallback_verbs = {"is", "are", "was", "were", "be", "been", "being"}

    nlp = None
    if remove_verbs:
        try:
            import spacy  # type: ignore

            nlp = getattr(trim_and_filter_entities, "_nlp", None)
            if nlp is None:
                try:
                    nlp = spacy.load("en_core_web_sm")
                except (OSError, ImportError):
                    nlp = None
                trim_and_filter_entities._nlp = nlp
        except ImportError:
            nlp = None

    removed_indices: List[int] = []
    trimmed_count = 0

    for idx, row in df.iterrows():
        if bool(row.get("is_user_entity", False)):
            continue
        raw_entity = str(row.get("entity", ""))
        if not raw_entity.strip():
            removed_indices.append(idx)
            continue

        token_list = raw_entity.split()
        if len(token_list) > max_tokens or len(raw_entity.strip()) > max_chars:
            removed_indices.append(idx)
            continue

        tokens = list(token_list)
        while tokens:
            candidate = tokens[-1].rstrip(",.;:")
            if candidate.lower() in trailing_stopwords:
                tokens.pop()
            else:
                break

        trimmed_entity = " ".join(tokens).strip(",.;: ")
        if not trimmed_entity:
            removed_indices.append(idx)
            continue

        if trimmed_entity != raw_entity:
            df.at[idx, "entity"] = trimmed_entity
            trimmed_count += 1

        contains_verb = False
        if remove_verbs:
            if nlp is not None:
                doc = nlp(trimmed_entity)
                contains_verb = any(token.pos_.startswith("VERB") for token in doc)
            else:
                lower_tokens = {tok.lower() for tok in tokens}
                contains_verb = bool(lower_tokens & fallback_verbs)

        if contains_verb:
            removed_indices.append(idx)

    if removed_indices:
        df = df.drop(index=removed_indices)

    if not user_rows.empty:
        df = pd.concat([df, user_rows], ignore_index=True, sort=False)

    removed_count = len(removed_indices) + trimmed_count
    if st is not None and removed_count:
        try:
            st.info(f"🧹 Trimmed {removed_count} overlong or invalid entities.")
        except Exception:
            pass

    return df.reset_index(drop=True)


PROPERTY_HINTS = {
    "mobility",
    "efficiency",
    "conductivity",
    "bandgap",
    "pce",
    "selectivity",
    "yield",
}

UNIT_PATTERN = re.compile(r"[%°A-Za-zµμΩ/·0-9^+-]+$", re.IGNORECASE)



def filter_false_units(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through: unit-like reclassification disabled."""
    return entity_df


def suppress_lonely_units(entity_df: pd.DataFrame, distance_threshold: int = 50) -> pd.DataFrame:
    """Remove UNIT entities that are not near any VALUE entities (within distance_threshold)."""

    pandas_module = globals().get("pd")
    if pandas_module is None or entity_df is None or entity_df.empty:
        return entity_df

    type_column = next((col for col in ("EntityType", "entity_type", "Type") if col in entity_df.columns), None)
    if type_column is None or "paper_id" not in entity_df.columns or "start" not in entity_df.columns:
        return entity_df

    df = entity_df.copy()
    df[type_column] = df[type_column].apply(canonicalize_label)
    df["start"] = pandas_module.to_numeric(df["start"], errors="coerce")

    drop_indices: Set[int] = set()

    for pid, group in df.groupby("paper_id"):
        group = group.dropna(subset=["start"])
        if group.empty:
            continue

        values = group[group[type_column] == "PROP_VALUE"]
        units = values[values["entity"].astype(str).str.match(UNIT_ONLY_PATTERN)]

        if values.empty or units.empty:
            continue

        for idx, unit in units.iterrows():
            u_start = unit["start"]
            if pandas_module.isna(u_start):
                drop_indices.add(idx)
                continue
            distances = (values["start"] - u_start).abs()
            if (distances < distance_threshold).any():
                continue
            drop_indices.add(idx)

    if drop_indices:
        removed_count = len(drop_indices)
        df = df.drop(index=list(drop_indices), errors="ignore")
        try:
            st.info(f"🧹 Removed {removed_count} unit-like entities with no nearby PROP_VALUE.")
        except Exception:
            print(f"Removed {removed_count} unit-like entities with no nearby PROP_VALUE.")

    return df


def sanitize_processed_dataframe(
    df: pd.DataFrame,
    allow_empty_abstracts: bool = ALLOW_EMPTY_ABSTRACTS,
    drop_duplicates: bool = ENABLE_DEDUPLICATION,
    stats: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    pandas_module = globals().get("pd")
    if df.empty:
        if stats is not None:
            stats["processed_initial"] = 0
        return df.copy()

    if stats is not None:
        stats["processed_initial"] = len(df)

    sanitized = sanitize_entities_output(df.copy())
    if "Title" in sanitized.columns:
        sanitized["Title"] = sanitized["Title"].astype(str)
    if "Abstract" in sanitized.columns:
        sanitized["Abstract"] = sanitized["Abstract"].astype(str)

    key_columns = [col for col in ("Title", "Abstract") if col in sanitized.columns]
    empty_rows_mask: Optional[pd.Series] = None
    if key_columns:
        for col in key_columns:
            series = sanitized[col]
            if series.dtype == object:
                col_mask = series.fillna("").astype(str).str.strip().eq("")
            else:
                col_mask = series.isna()
            empty_rows_mask = col_mask if empty_rows_mask is None else (empty_rows_mask & col_mask)
        if empty_rows_mask is not None:
            empty_count = int(empty_rows_mask.sum())
            if stats is not None:
                stats["processed_empty_rows"] = empty_count
            if not allow_empty_abstracts and empty_count:
                sanitized = sanitized.loc[~empty_rows_mask].copy()
                if stats is not None:
                    stats["processed_removed_empty"] = empty_count

    if drop_duplicates:
        before = len(sanitized)
        sanitized = sanitized.drop_duplicates().reset_index(drop=True)
        removed_dup = before - len(sanitized)
        if stats is not None:
            stats["processed_removed_duplicates"] = removed_dup
    else:
        sanitized = sanitized.reset_index(drop=True)

    cleaned = deep_clean(sanitized)
    if pandas_module is not None and isinstance(cleaned, pandas_module.DataFrame):
        return cleaned
    return sanitized


def _split_entities(value: object) -> List[str]:
    sanitized = sanitize_entities_output(value)
    if not sanitized:
        return []
    parts = [item.strip() for item in sanitized.split(",") if item.strip()]
    unique: List[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return unique


def sanitize_linking_context(context: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(context, dict):
        return {
            "strategy": "skipped",
            "canonical_map": {},
            "alias_groups": {},
            "canonical_display": {},
            "entity_count": 0,
        }

    sanitized_context: Dict[str, object] = {}

    strategy = context.get("strategy")
    sanitized_context["strategy"] = sanitize_entities_output(strategy) if strategy else "skipped"

    canonical_map_raw = context.get("canonical_map", {})
    canonical_map_clean: Dict[str, str] = {}
    if isinstance(canonical_map_raw, Mapping):
        for key, value in canonical_map_raw.items():
            key_clean = sanitize_entities_output(key)
            value_clean = sanitize_entities_output(value)
            if key_clean and value_clean:
                canonical_map_clean[key_clean] = value_clean
    sanitized_context["canonical_map"] = canonical_map_clean

    alias_groups_raw = context.get("alias_groups", {})
    alias_groups_clean: Dict[str, List[str]] = {}
    if isinstance(alias_groups_raw, Mapping):
        for key, values in alias_groups_raw.items():
            key_clean = sanitize_entities_output(key)
            members = []
            if isinstance(values, IterableABC):
                for val in values:
                    member = sanitize_entities_output(val)
                    if member and member not in members:
                        members.append(member)
            if key_clean:
                alias_groups_clean[key_clean] = sorted(members)
    sanitized_context["alias_groups"] = alias_groups_clean

    canonical_display_raw = context.get("canonical_display", {})
    canonical_display_clean: Dict[str, str] = {}
    if isinstance(canonical_display_raw, Mapping):
        for key, value in canonical_display_raw.items():
            key_clean = sanitize_entities_output(key)
            value_clean = sanitize_entities_output(value)
            if key_clean and value_clean:
                canonical_display_clean[key_clean] = value_clean
    sanitized_context["canonical_display"] = canonical_display_clean

    entity_count = context.get("entity_count", 0)
    try:
        sanitized_context["entity_count"] = int(entity_count)
    except Exception:
        sanitized_context["entity_count"] = 0

    error_message = context.get("error")
    if error_message:
        sanitized_context["error"] = sanitize_entities_output(error_message)

    cleaned_context = deep_clean(sanitized_context)
    if not _should_keep(cleaned_context):
        return {}
    return cleaned_context


def notify_once(key: str, level: str, message: str) -> None:
    if st is None:
        return
    if st.session_state.get(key):
        return
    st.session_state[key] = True
    # FIXED: set() display issue – normalize any message before emitting via Streamlit.
    message_text = safe_to_str(message)
    try:
        if level == "sidebar_info":
            st.sidebar.info(message_text)
        elif level == "sidebar_warning":
            st.sidebar.warning(message_text)
        elif level == "warning":
            st.warning(message_text)
        elif level == "error":
            st.error(message_text)
        else:
            st.info(message_text)
    except Exception:
        pass


def is_offline() -> bool:
    if st is None:
        return False
    cached = st.session_state.get("offline_mode")
    if cached is not None:
        return cached
    env_flags = [
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("TRANSFORMERS_OFFLINE"),
    ]

    def _truthy(value: Optional[str]) -> bool:
        if value is None:
            return False
        return value.strip().lower() not in {"", "0", "false", "no", "off"}

    forced_offline = next((flag for flag in env_flags if _truthy(flag)), None)
    if forced_offline:
        st.session_state["offline_mode"] = True
        notify_once("offline_info_shown", "sidebar_info", "⚠️ Offline mode: using local models only.")
        return True
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        offline = False
    except Exception as exc:
        offline = False
        notify_once(
            "offline_check_failed",
            "warning",
            f"⚠️ Could not verify Hugging Face connectivity ({type(exc).__name__}). Assuming online mode.",
        )
        debug_log("Hugging Face connectivity check failed:", exc)
    st.session_state["offline_mode"] = offline
    return offline


class DummyPipeline:
    def __call__(self, *args, **kwargs):
        return []


# Define empty local candidates list to prevent NameError; users may append paths at runtime.
LOCAL_MODEL_CANDIDATES: List[Tuple[str, str]] = []


def _normalize_phrase(text: str) -> str:
    cleaned = sanitize_entities_output(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def merge_acronym_variants(entity_df: pd.DataFrame) -> pd.DataFrame:
    if entity_df.empty:
        return entity_df
    if "entity" not in entity_df.columns or "entity_type" not in entity_df.columns:
        return entity_df

    df = sanitize_entities_output(entity_df.copy())
    if "entity_norm" not in df.columns:
        df["entity_norm"] = df["entity"].str.lower()
    if "Canonical" not in df.columns:
        df["Canonical"] = ""

    updates: Dict[int, str] = {}
    for idx, row in df.iterrows():
        entity = row.get("entity", "")
        entity_type = flatten_iterable(row.get("entity_type", "")).upper()
        normalized = _normalize_phrase(entity)
        if not normalized:
            continue
        canonical = ACRONYM_LOOKUP.get(normalized)
        if not canonical:
            continue
        updates[idx] = canonical

    if not updates:
        df = df.reset_index(drop=True)
        return sanitize_entities_output(df)

    for idx, canonical in updates.items():
        df.at[idx, "Canonical"] = canonical
        df.at[idx, "entity_norm"] = canonical.lower()

    if ENABLE_DEDUPLICATION:
        df = df.drop_duplicates(subset=[col for col in df.columns if col != "confidence"], keep="first")
    return sanitize_entities_output(df.reset_index(drop=True))


def preprocess_abstract(text: str) -> Tuple[str, Dict[str, str]]:
    if not text:
        return "", {}

    abbreviation_map: Dict[str, str] = {}

    def replacer(match: re.Match) -> str:
        expansion = normalize_text(match.group("expansion"))
        abbreviation = match.group("abbr").strip()
        abbreviation_map[abbreviation] = expansion
        return f"{expansion} {abbreviation}"

    cleaned = ABBREVIATION_PATTERN.sub(replacer, text)
    cleaned = re.sub(r"\(([A-Z0-9]{2,})\)", r"\1", cleaned)
    return cleaned, abbreviation_map


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 512,
    overlap: int = 50,
) -> List[Dict[str, object]]:
    if not text:
        return [{"text": "", "start_char": 0, "start_token": 0, "end_token": 0}]

    overlap = max(0, min(overlap, max_tokens // 4))
    inner_max = max_tokens - 2  # reserve for CLS/SEP

    try:
        tokens = tokenizer.tokenize(text)
    except Exception:
        words = text.split()
        chunks: List[Dict[str, object]] = []
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(len(words), start_idx + inner_max)
            chunk_words = words[start_idx:end_idx]
            chunk_text_str = " ".join(chunk_words)
            if chunks:
                prev = chunks[-1]
                search_start = prev["start_char"] + len(prev["text"])
            else:
                search_start = 0
            start_char = text.find(chunk_text_str, search_start)
            if start_char < 0:
                start_char = sum(len(w) + 1 for w in words[:start_idx])
            chunks.append(
                {
                    "text": chunk_text_str,
                    "start_char": start_char,
                    "start_token": start_idx,
                    "end_token": end_idx,
                }
            )
            if end_idx == len(words):
                break
            start_idx = max(end_idx - overlap, start_idx + 1)
        return chunks

    if len(tokens) <= inner_max:
        return [{"text": text, "start_char": 0, "start_token": 0, "end_token": len(tokens)}]

    offsets_encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = offsets_encoding.get("offset_mapping", [])
    chunks: List[Dict[str, object]] = []

    start_token_idx = 0
    total_tokens = len(tokens)
    while start_token_idx < total_tokens:
        end_token_idx = min(total_tokens, start_token_idx + inner_max)
        chunk_tokens = tokens[start_token_idx:end_token_idx]
        chunk_text_str = tokenizer.convert_tokens_to_string(chunk_tokens)
        if not chunk_text_str:
            chunk_text_str = text[offsets[start_token_idx][0] : offsets[end_token_idx - 1][1]]

        while True:
            adjusted_ids = tokenizer(chunk_text_str, add_special_tokens=True)["input_ids"]
            if len(adjusted_ids) <= max_tokens:
                break
            # truncate tokens until within limit
            if end_token_idx - start_token_idx <= 1:
                chunk_tokens = [tokens[start_token_idx]]
                chunk_text_str = tokenizer.convert_tokens_to_string(chunk_tokens)
                end_token_idx = start_token_idx + 1
                break
            end_token_idx -= 1
            chunk_tokens = tokens[start_token_idx:end_token_idx]
            chunk_text_str = tokenizer.convert_tokens_to_string(chunk_tokens)

        chunk_offsets = offsets[start_token_idx:end_token_idx]
        start_char = chunk_offsets[0][0]
        end_char = chunk_offsets[-1][1]
        chunks.append(
            {
                "text": chunk_text_str,
                "start_char": start_char,
                "start_token": start_token_idx,
                "end_token": end_token_idx,
            }
        )

        if end_token_idx == total_tokens:
            break
        start_token_idx = max(end_token_idx - overlap, start_token_idx + 1)

    return chunks


def canonicalize_entity(entity: object) -> str:
    if entity is None:
        return ""
    if isinstance(entity, bytes):
        try:
            entity = entity.decode("utf-8", errors="ignore")
        except Exception:
            entity = str(entity)
    if not isinstance(entity, str):
        entity = str(entity)
    return entity.strip()


def normalize_entity(entity: str) -> str:
    """Retain entity text with punctuation/symbols; only trim surrounding whitespace."""
    return canonicalize_entity(entity)


def extract_abbrev_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    if not text:
        return pairs
    for match in ABBREVIATION_PATTERN.finditer(text):
        abbr_norm = normalize_entity(match.group("abbr"))
        expansion_norm = normalize_entity(match.group("expansion"))
        if abbr_norm and expansion_norm:
            pairs.setdefault(abbr_norm, expansion_norm)
    return pairs


def _domain_alias(norm: str) -> Optional[str]:
    if not norm:
        return None
    for needle, alias in DOMAIN_ALIAS_SUBSTRINGS.items():
        if needle in norm:
            return alias
    return None


def _format_canonical(value: str) -> str:
    if not value:
        return "unknown"
    if len(value) <= 4:
        return value.upper()
    return value.title()


def _can_merge_terms(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return False
    min_len = min(len(a), len(b))
    max_len = max(len(a), len(b))
    if min_len <= 2 and max_len > min_len:
        return False
    return True


def _adaptive_threshold(term: str, base: float) -> float:
    cleaned = (term or "").replace(" ", "")
    length = len(cleaned)
    if length <= 3:
        return max(0.95, base)
    if length <= 4:
        return max(0.92, base)
    if length <= 6:
        return max(0.88, base)
    return max(base, 0.82)


def _types_compatible(types_a: Set[str], types_b: Set[str]) -> bool:
    if not types_a and not types_b:
        return True
    clean_a = {t for t in types_a if t and t != "UNKNOWN"}
    clean_b = {t for t in types_b if t and t != "UNKNOWN"}
    if clean_a and clean_b:
        return not clean_a.isdisjoint(clean_b)
    if not clean_a and not clean_b:
        return True
    return False


def _collect_abbreviation_links(
    abbreviation_registry: Optional[Dict[int, Dict[str, str]]],
    paper_texts: Dict[int, str],
) -> Set[Tuple[str, str]]:
    links: Set[Tuple[str, str]] = set()
    if abbreviation_registry:
        for registry in abbreviation_registry.values():
            for abbr, expansion in registry.items():
                abbr_norm = normalize_entity(abbr)
                expansion_norm = normalize_entity(expansion)
                if abbr_norm and expansion_norm:
                    links.add((abbr_norm, expansion_norm))
    for text in paper_texts.values():
        for abbr_norm, expansion_norm in extract_abbrev_pairs(text).items():
            if abbr_norm and expansion_norm:
                links.add((abbr_norm, expansion_norm))
    return links


@st.cache_data(show_spinner=False)
def _compute_tfidf_similarity(norms: Tuple[str, ...]) -> Dict[str, List[Tuple[str, float]]]:
    if not norms:
        return {}
    if TfidfVectorizer is None or cosine_similarity is None:
        raise ImportError("scikit-learn is required for TF-IDF based entity linking.")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(norms)
    similarity = cosine_similarity(matrix)
    lookup: Dict[str, List[Tuple[str, float]]] = {}
    for idx, term in enumerate(norms):
        scores: List[Tuple[str, float]] = []
        row = similarity[idx]
        for jdx, score in enumerate(row):
            if idx == jdx:
                continue
            if score <= 0:
                continue
            scores.append((norms[jdx], float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        lookup[term] = scores
    return lookup


def _get_sentence_embedder() -> Optional["SentenceTransformer"]:
    if SentenceTransformer is None:
        return None
    embedder = getattr(_get_sentence_embedder, "_cached", None)
    if embedder is not None:
        return embedder
    try:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _get_sentence_embedder._cached = embedder
        return embedder
    except Exception:  # pragma: no cover
        return None


@st.cache_data(show_spinner=False)
def _compute_semantic_similarity(norms: Tuple[str, ...]) -> Dict[str, List[Tuple[str, float]]]:
    if not norms:
        return {}
    embedder = _get_sentence_embedder()
    if embedder is None or st_util is None:
        raise ImportError("sentence-transformers is required for semantic entity linking.")
    embeddings = embedder.encode(
        list(norms),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sim_matrix = st_util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    lookup: Dict[str, List[Tuple[str, float]]] = {}
    for idx, term in enumerate(norms):
        scores: List[Tuple[str, float]] = []
        row = sim_matrix[idx]
        for jdx, score in enumerate(row):
            if idx == jdx:
                continue
            if float(score) <= 0:
                continue
            scores.append((norms[jdx], float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        lookup[term] = scores
    return lookup


def _build_cluster_maps(
    norms: Iterable[str],
    similarity_lookup: Dict[str, List[Tuple[str, float]]],
    base_threshold: float,
    abbreviation_links: Set[Tuple[str, str]],
    freq_map: Dict[str, int],
    type_lookup: Dict[str, Set[str]],
    protected_terms: Set[str],
) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    all_terms: Set[str] = set(norms)
    for source, target in abbreviation_links:
        if source:
            all_terms.add(source)
        if target:
            all_terms.add(target)
    domain_alias_pairs: Dict[str, str] = {}
    for term in list(all_terms):
        alias = _domain_alias(term)
        if alias and alias != term:
            domain_alias_pairs[term] = alias
            all_terms.add(alias)

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for term in all_terms:
        adjacency[term]  # initialize
        type_lookup.setdefault(term, {"UNKNOWN"})

    for source, target in abbreviation_links:
        if source and target:
            if source in protected_terms and target in protected_terms and source != target:
                continue
            types_source = type_lookup.get(source, {"UNKNOWN"})
            types_target = type_lookup.get(target, {"UNKNOWN"})
            if not _types_compatible(types_source, types_target):
                continue
            combined_types = (types_source | types_target) or {"UNKNOWN"}
            type_lookup.setdefault(source, set()).update(combined_types)
            type_lookup.setdefault(target, set()).update(combined_types)
            adjacency[source].add(target)
            adjacency[target].add(source)

    for source, target in list(domain_alias_pairs.items()):
        if source in protected_terms or target in protected_terms:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
        types_source = type_lookup.get(source, {"UNKNOWN"})
        types_target = type_lookup.get(target, {"UNKNOWN"})
        if target not in type_lookup:
            type_lookup[target] = set(types_source)
        if source not in type_lookup:
            type_lookup[source] = set(types_target)

    for term, neighbors in similarity_lookup.items():
        for other, score in neighbors:
            if term in protected_terms and other in protected_terms and term != other:
                continue
            adaptive_threshold = min(
                _adaptive_threshold(term, base_threshold),
                _adaptive_threshold(other, base_threshold),
            )
            if score < adaptive_threshold:
                continue
            if not _can_merge_terms(term, other):
                continue
            types_term = type_lookup.get(term, {"UNKNOWN"})
            types_other = type_lookup.get(other, {"UNKNOWN"})
            if not _types_compatible(types_term, types_other):
                continue
            adjacency[term].add(other)
            adjacency[other].add(term)

    visited: Set[str] = set()
    canonical_map: Dict[str, str] = {}
    alias_groups: Dict[str, Set[str]] = defaultdict(set)

    def _score(term: str) -> Tuple[int, int, int, str]:
        frequency = freq_map.get(term, 0)
        acronym_rank = 0 if term in ACRONYM_SET else 1
        return (acronym_rank, -frequency, len(term), term)

    for term in sorted(all_terms):
        if not term or term in visited:
            continue
        stack = [term]
        component: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency[current])
        canonical = min(component, key=_score)
        for member in component:
            canonical_map[member] = canonical
        alias_groups[canonical].update(component)

    return canonical_map, alias_groups


def _prepare_linking_context(
    df: pd.DataFrame,
    canonical_map: Dict[str, str],
    alias_groups: Dict[str, Set[str]],
    strategy: str,
) -> Dict[str, object]:
    canonical_display = {canonical: canonical.upper() for canonical in alias_groups.keys()}
    context = {
        "strategy": strategy,
        "canonical_map": canonical_map,
        "alias_groups": {key: sorted(values) for key, values in alias_groups.items()},
        "canonical_display": canonical_display,
        "entity_count": len(df),
    }
    return sanitize_linking_context(context)


def link_entities_fast(
    entity_df: pd.DataFrame,
    paper_texts: Dict[int, str],
    abbreviation_registry: Optional[Dict[int, Dict[str, str]]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    user_canonical: Optional[Mapping[str, Iterable[str]]] = None,
) -> Tuple[pd.DataFrame, int, Dict[str, object]]:
    sanitized_input = sanitize_entity_dataframe(entity_df)
    if sanitized_input.empty:
        df = sanitized_input.copy()
        if "norm" not in df.columns:
            df["norm"] = []
        if "canonical" not in df.columns:
            df["canonical"] = []
        if "Canonical" not in df.columns:
            df["Canonical"] = []
        return df, 0, {"strategy": "tfidf", "canonical_map": {}, "alias_groups": {}}

    if progress_callback:
        progress_callback(0.05)

    skip_types = {"PROP_VALUE", "MATERIAL_AMOUNT"}
    df = sanitized_input.copy()
    excluded = df[df["entity_type"].astype(str).str.upper().isin(skip_types)].copy()
    df = df[~df["entity_type"].astype(str).str.upper().isin(skip_types)].copy()
    df["norm"] = df["entity"].apply(normalize_entity)
    freq_map = Counter(df["norm"])
    unique_norms = tuple(sorted({norm for norm in df["norm"] if norm}))
    type_lookup: Dict[str, Set[str]] = (
        df.groupby("norm")["entity_type"]
        .apply(lambda values: {str(v).upper() for v in values if str(v).strip()})
        .to_dict()
    )

    abbreviation_links = _collect_abbreviation_links(abbreviation_registry, paper_texts)

    if progress_callback:
        progress_callback(0.25)

    similarity_lookup = _compute_tfidf_similarity(unique_norms)

    if progress_callback:
        progress_callback(0.6)

    base_threshold = 0.80
    if len(df) > 0:
        try:
            base_threshold = 0.80 + min(0.05, 0.02 * math.log10(max(len(df) / 1000, 1e-6)))
        except Exception:
            base_threshold = 0.80

    canonical_map, alias_groups = _build_cluster_maps(
        unique_norms,
        similarity_lookup,
        base_threshold=base_threshold,
        abbreviation_links=abbreviation_links,
        freq_map=freq_map,
        type_lookup=type_lookup,
        protected_terms=PROTECTED_TERMS,
    )

    df["canonical"] = df["norm"].map(lambda n: canonical_map.get(n, n) or n)
    previous_canonical = df["Canonical"].astype(str) if "Canonical" in df.columns else None
    df["Canonical"] = df["canonical"].apply(lambda val: val.upper() if isinstance(val, str) and val else "")
    if previous_canonical is not None:
        mask = previous_canonical.str.strip() != ""
        df.loc[mask, "Canonical"] = previous_canonical[mask]
    merged_count = int((df["canonical"] != df["norm"]).sum())

    if progress_callback:
        progress_callback(0.95)

    context = _prepare_linking_context(df, canonical_map, alias_groups, strategy="tfidf")

    if progress_callback:
        progress_callback(1.0)

    df = apply_user_canonical_overrides(df, user_canonical)
    df = sanitize_entity_dataframe(df)
    if not excluded.empty:
        if "norm" not in excluded.columns:
            excluded["norm"] = excluded["entity"].apply(normalize_entity)
        if "canonical" not in excluded.columns:
            excluded["canonical"] = excluded["norm"]
        if "Canonical" not in excluded.columns:
            excluded["Canonical"] = excluded["canonical"].apply(lambda val: str(val).strip().upper())
        df = pd.concat([df, excluded], ignore_index=True, sort=False)
    context = sanitize_linking_context(context)
    return df, merged_count, context


def link_entities_semantic(
    entity_df: pd.DataFrame,
    paper_texts: Dict[int, str],
    abbreviation_registry: Optional[Dict[int, Dict[str, str]]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    user_canonical: Optional[Mapping[str, Iterable[str]]] = None,
) -> Tuple[pd.DataFrame, int, Dict[str, object]]:
    sanitized_input = sanitize_entity_dataframe(entity_df)
    if sanitized_input.empty:
        df = sanitized_input.copy()
        if "norm" not in df.columns:
            df["norm"] = []
        if "canonical" not in df.columns:
            df["canonical"] = []
        if "Canonical" not in df.columns:
            df["Canonical"] = []
        return df, 0, {"strategy": "semantic", "canonical_map": {}, "alias_groups": {}}

    if progress_callback:
        progress_callback(0.05)

    skip_types = {"PROP_VALUE", "MATERIAL_AMOUNT"}
    df = sanitized_input.copy()
    excluded = df[df["entity_type"].astype(str).str.upper().isin(skip_types)].copy()
    df = df[~df["entity_type"].astype(str).str.upper().isin(skip_types)].copy()
    df["norm"] = df["entity"].apply(normalize_entity)
    freq_map = Counter(df["norm"])
    unique_norms = tuple(sorted({norm for norm in df["norm"] if norm}))
    type_lookup: Dict[str, Set[str]] = (
        df.groupby("norm")["entity_type"]
        .apply(lambda values: {str(v).upper() for v in values if str(v).strip()})
        .to_dict()
    )

    abbreviation_links = _collect_abbreviation_links(abbreviation_registry, paper_texts)

    if progress_callback:
        progress_callback(0.25)

    similarity_lookup = _compute_semantic_similarity(unique_norms)

    if progress_callback:
        progress_callback(0.6)

    base_threshold = 0.80
    if len(df) > 0:
        try:
            base_threshold = 0.80 + min(0.05, 0.02 * math.log10(max(len(df) / 1000, 1e-6)))
        except Exception:
            base_threshold = 0.80

    canonical_map, alias_groups = _build_cluster_maps(
        unique_norms,
        similarity_lookup,
        base_threshold=base_threshold,
        abbreviation_links=abbreviation_links,
        freq_map=freq_map,
        type_lookup=type_lookup,
        protected_terms=PROTECTED_TERMS,
    )

    df["canonical"] = df["norm"].map(lambda n: canonical_map.get(n, n) or n)
    previous_canonical = df["Canonical"].astype(str) if "Canonical" in df.columns else None
    df["Canonical"] = df["canonical"].apply(lambda val: val.upper() if isinstance(val, str) and val else "")
    if previous_canonical is not None:
        mask = previous_canonical.str.strip() != ""
        df.loc[mask, "Canonical"] = previous_canonical[mask]
    merged_count = int((df["canonical"] != df["norm"]).sum())

    context = _prepare_linking_context(df, canonical_map, alias_groups, strategy="semantic")

    if progress_callback:
        progress_callback(1.0)

    df = apply_user_canonical_overrides(df, user_canonical)
    df = sanitize_entity_dataframe(df)
    if not excluded.empty:
        if "norm" not in excluded.columns:
            excluded["norm"] = excluded["entity"].apply(normalize_entity)
        if "canonical" not in excluded.columns:
            excluded["canonical"] = excluded["norm"]
        if "Canonical" not in excluded.columns:
            excluded["Canonical"] = excluded["canonical"].apply(lambda val: str(val).strip().upper())
        df = pd.concat([df, excluded], ignore_index=True, sort=False)
    context = sanitize_linking_context(context)
    return df, merged_count, context


def domain_category_for(token: str) -> Optional[str]:
    if not token:
        return None
    key = token.lower().strip()
    if key in DOMAIN_TERM_CATEGORIES:
        mapped = canonicalize_label(DOMAIN_TERM_CATEGORIES[key])
        return mapped if mapped in ACTIVE_LABEL2ID else None
    key_spaced = key.replace("-", " ")
    if key_spaced in DOMAIN_TERM_CATEGORIES:
        mapped = canonicalize_label(DOMAIN_TERM_CATEGORIES[key_spaced])
        return mapped if mapped in ACTIVE_LABEL2ID else None
    return None


def extend_tokenizer_with_domain_terms(ner) -> None:
    tokenizer = getattr(ner, "tokenizer", None)
    model = getattr(ner, "model", None)
    if tokenizer is None or model is None:
        return
    existing_tokens: Set[str] = set(getattr(tokenizer, "_ner_added_tokens", []))
    vocab_source = getattr(tokenizer, "vocab", None)
    if vocab_source is None and hasattr(tokenizer, "get_vocab"):
        vocab_source = tokenizer.get_vocab()
    if isinstance(vocab_source, dict):
        vocab = set(vocab_source.keys())
    else:
        vocab = set()
    tokens_to_add: List[str] = []
    for token in DOMAIN_TOKENS:
        key_lower = token.lower()
        if token in vocab or key_lower in vocab or token in existing_tokens:
            continue
        tokens_to_add.append(token)
    if tokens_to_add:
        num_added = tokenizer.add_tokens(tokens_to_add)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
        tokenizer._ner_added_tokens = list(existing_tokens.union(tokens_to_add))


def map_entity_type(entity_group: str, text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "UNKNOWN"

    lower = normalized.lower()
    label_guess = canonicalize_label(entity_group)

    domain_hint = domain_category_for(normalized)
    if domain_hint:
        return domain_hint

    if label_guess in ACTIVE_LABEL2ID:
        return label_guess

    if SIMPLE_VALUE_PATTERN.match(normalized) or VALUE_WITH_UNIT_PATTERN.match(normalized) or VALUE_PATTERN.match(normalized):
        return "PROP_VALUE"

    if UNIT_ONLY_PATTERN.match(normalized) or normalized.upper() in UNIT_TERMS:
        return "PROP_VALUE"

    if "family" in lower:
        return "POLYMER_FAMILY"

    if "monomer" in lower:
        return "MONOMER"

    if any(keyword in lower for keyword in PROPERTY_KEYWORDS):
        return "PROP_NAME"

    if any(keyword in lower for keyword in MATERIAL_KEYWORDS):
        return "POLYMER"

    if MATERIAL_PATTERN.match(normalized):
        inorganic_clues = {"oxide", "perovskite", "sulfide", "nitride"}
        if any(clue in lower for clue in inorganic_clues):
            return "INORGANIC"
        return "ORGANIC"

    return "UNKNOWN"


def format_chip(text: str, entity_type: str) -> str:
    color = ENTITY_COLORS.get(entity_type, "#4B5563")
    return (
        f"<span style='display:inline-block;padding:2px 6px;margin:2px;"
        f"border-radius:12px;background:{color};color:#FFFFFF;font-size:12px;'>"
        f"{entity_type}: {text}</span>"
    )


def render_entity_chips(entity_map: Dict[str, List[str]]) -> str:
    entity_map_clean = deep_clean(entity_map) if entity_map else {}
    if not isinstance(entity_map_clean, dict):
        return ""
    chips: List[str] = []
    for entity_type in ENTITY_TYPES:
        values = entity_map_clean.get(entity_type, [])
        for value in values:
            chips.append(format_chip(value, entity_type))
    return " ".join(chips)


def build_records(
    df: pd.DataFrame,
    skip_filters: bool = True,
    min_abstract_length: int = DEFAULT_MIN_ABSTRACT_LENGTH,
    include_all_types: bool = DEFAULT_INCLUDE_ALL_TYPES,
    deduplicate: bool = ENABLE_DEDUPLICATION,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    pandas_module = globals().get("pd")
    working = df.reset_index(drop=True) if not df.empty else df.copy()
    total_rows = len(working)
    stats: Dict[str, int] = {
        "total_rows": total_rows,
        "missing_abstract_rows": 0,
        "short_abstract_rows": 0,
        "type_mismatches": 0,
        "duplicate_candidates": 0,
        "removed_rows": 0,
        "removed_missing_abstract": 0,
        "removed_by_length": 0,
        "removed_by_type": 0,
        "duplicates_removed": 0,
    }

    records: List[Dict[str, str]] = []
    seen_dedup_keys: Set[Tuple[Any, Any, Any]] = set()

    for idx, row in working.iterrows():
        raw_id = row.get("paper_id")
        paper_id = None
        if raw_id is not None and (pandas_module is None or not (pandas_module.isna(raw_id))):
            try:
                paper_id = int(raw_id)
            except Exception:
                try:
                    paper_id = int(float(raw_id))
                except Exception:
                    paper_id = None
        if paper_id is None:
            paper_id = idx + 1

        title_value = row.get("Title", "")
        abstract_raw = row.get("Abstract", "")
        if pandas_module is not None and isinstance(abstract_raw, pandas_module.Series):
            abstract_text = abstract_raw.astype(str).tolist()
        else:
            abstract_text = "" if (pandas_module is not None and pandas_module.isna(abstract_raw)) else str(abstract_raw)
        abstract_text = abstract_text if isinstance(abstract_text, str) else str(abstract_text)
        abstract_stripped = abstract_text.strip()

        if not abstract_stripped:
            stats["missing_abstract_rows"] += 1
        if len(abstract_stripped) < max(min_abstract_length, 0):
            stats["short_abstract_rows"] += 1

        type_value = row.get("Type", row.get("entity_type", ""))
        type_str = normalize_text(type_value).upper()
        if type_str and type_str not in ENTITY_TYPES:
            stats["type_mismatches"] += 1

        dedup_key = (
            paper_id,
            normalize_text(title_value),
            abstract_stripped.lower(),
        )
        if deduplicate:
            if dedup_key in seen_dedup_keys:
                stats["duplicate_candidates"] += 1
                stats["duplicates_removed"] += 1
                stats["removed_rows"] += 1
                continue
            seen_dedup_keys.add(dedup_key)
        else:
            if dedup_key in seen_dedup_keys:
                stats["duplicate_candidates"] += 1
            else:
                seen_dedup_keys.add(dedup_key)

        if not skip_filters:
            if not abstract_stripped:
                stats["removed_rows"] += 1
                stats["removed_missing_abstract"] += 1
                continue
            if min_abstract_length and len(abstract_stripped) < min_abstract_length:
                stats["removed_rows"] += 1
                stats["removed_by_length"] += 1
                continue
            if not include_all_types and type_str and type_str not in ENTITY_TYPES:
                stats["removed_rows"] += 1
                stats["removed_by_type"] += 1
                continue

        data: Dict[str, str] = {"paper_id": paper_id}
        for column in working.columns:
            if column == "paper_id":
                continue
            raw_value = row.get(column, "")
            data[column] = normalize_text(raw_value)
        if "Abstract" not in data:
            data["Abstract"] = abstract_text
        records.append(data)

    stats["included_rows"] = len(records)
    stats.pop("_next_id", None)
    return records, stats


def process_abstracts(
    records: List[Dict[str, str]],
    model_name: str,
    confidence_threshold: float,
    device_preference: str,
    progress: Optional["DeltaGenerator"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, int, int]:
    token_raw = os.environ.get("HUGGINGFACE_TOKEN", "")
    token_signature = hashlib.sha256(token_raw.encode("utf-8")).hexdigest() if token_raw else None
    ner, resolved_model = load_ner_pipeline(model_name, token_signature, device_preference)
    if ner is None or not resolved_model:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            0,
            {},
            {},
        )

    ner_callable = ner
    augmented_rows: List[Dict[str, object]] = []
    entity_rows: List[Dict[str, object]] = []

    total_records = len(records)
    if progress is not None:
        progress.progress(0.0, text="Starting NER extraction...")

    abbreviation_registry: Dict[int, Dict[str, str]] = {}
    abbreviation_originals: Dict[int, Dict[str, str]] = {}
    prepared_records: List[Dict[str, str]] = []
    abbreviation_tokens: Set[str] = set()
    paper_text_map: Dict[int, str] = {}

    for index, record in enumerate(records, start=1):
        row_dict = dict(record)
        paper_id = int(row_dict.get("paper_id") or index)
        original_abstract = row_dict.get("Abstract", "")
        paper_text_map[paper_id] = original_abstract
        abstract = original_abstract
        cleaned_abstract, abbr_map = preprocess_abstract(abstract)
        processed_map = {abbr.lower().strip(): expansion for abbr, expansion in abbr_map.items()}
        originals_map = {abbr.lower().strip(): abbr for abbr in abbr_map.keys()}
        abbreviation_registry[paper_id] = processed_map
        abbreviation_originals[paper_id] = originals_map
        abbreviation_tokens.update(abbr_map.keys())
        row_dict["Abstract"] = cleaned_abstract
        prepared_records.append(row_dict)

    tokenizer = getattr(ner, "tokenizer", None)
    model = getattr(ner, "model", None)
    torch_module = torch
    if model is not None and hasattr(model, "eval"):
        try:
            model.eval()
        except Exception:
            pass
    if tokenizer is not None and model is not None and abbreviation_tokens:
        existing_tokens: Set[str] = set(getattr(tokenizer, "_ner_added_tokens", []))
        vocab = set()
        if hasattr(tokenizer, "get_vocab"):
            vocab = set(tokenizer.get_vocab().keys())
        new_tokens = [abbr for abbr in abbreviation_tokens if abbr not in vocab and abbr.lower() not in vocab and abbr not in existing_tokens]
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
            tokenizer._ner_added_tokens = list(existing_tokens.union(new_tokens))

    long_text_count = 0

    for index, row_dict in enumerate(prepared_records, start=1):
        paper_id = int(row_dict.get("paper_id") or index)
        abstract = row_dict.get("Abstract", "")

        entity_set: Dict[str, set] = {entity_type: set() for entity_type in ENTITY_TYPES}
        entity_map: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}
        normalized_map: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}
        all_entities: Set[str] = set()
        all_entities_normalized: Set[str] = set()

        abbreviation_type_hints: Dict[str, str] = {}
        for abbr_norm, expansion in abbreviation_registry.get(paper_id, {}).items():
            hinted_type = map_entity_type(entity_group="", text=expansion)
            if hinted_type == "UNKNOWN":
                hinted_type = map_entity_type(entity_group="", text=abbr_norm.upper())
            abbreviation_type_hints[abbr_norm] = hinted_type

        if abstract and ner_callable is not None:
            chunk_spec = chunk_text(abstract, tokenizer) if tokenizer is not None else [{"text": abstract, "start_char": 0, "start_token": 0, "end_token": len(abstract.split())}]
            if len(chunk_spec) > 1:
                long_text_count += 1
            seen_spans: Set[Tuple[int, int, str]] = set()
            no_grad_ctx = torch_module.no_grad() if torch_module is not None else contextlib.nullcontext()
            with no_grad_ctx:
                for chunk in chunk_spec:
                    chunk_text_str = chunk.get("text", "")
                    chunk_start_char = int(chunk.get("start_char", 0))
                    global_start = 0
                    global_end = 0
                    if tokenizer is not None:
                        input_ids = tokenizer(chunk_text_str, add_special_tokens=True)["input_ids"]
                        if len(input_ids) > 512:
                            chunk_text_str = tokenizer.convert_tokens_to_string(
                                tokenizer.tokenize(chunk_text_str)[:510]
                            )
                            input_ids = tokenizer(chunk_text_str, add_special_tokens=True)["input_ids"]
                            if len(input_ids) > 512:
                                continue
                    predictions = ner_callable(chunk_text_str) or []
                    for prediction in predictions:
                        scores_raw = prediction.get("scores")
                        if scores_raw and torch_module is not None:
                            try:
                                tensor_scores = torch_module.tensor(scores_raw, dtype=torch_module.float32)
                                probs = torch_module.nn.functional.softmax(tensor_scores, dim=-1)
                                prediction["score"] = float(probs.max().item())
                            except Exception:
                                pass
                    for prediction in predictions:
                        start_local = int(prediction.get("start", 0))
                        end_local = int(prediction.get("end", 0))
                        global_start = chunk_start_char + start_local
                        global_end = chunk_start_char + end_local
                        if global_start >= global_end:
                            continue
                        word = normalize_text(abstract[global_start:global_end])
                        if not word:
                            continue
                        span_key = (global_start, global_end, word.lower())
                        if span_key in seen_spans:
                            continue
                        seen_spans.add(span_key)

                        score = float(prediction.get("score", 0.0)) if isinstance(prediction, dict) else 0.0
                        candidate_type = map_entity_type(prediction.get("entity_group"), word)
                        normalized_key = word.lower().strip()

                        domain_hint = domain_category_for(normalized_key) or domain_category_for(word)
                        if not domain_hint and normalized_key.replace("-", " ") != normalized_key:
                            domain_hint = domain_category_for(normalized_key.replace("-", " "))
                        if domain_hint and (score < 0.7 or candidate_type in {"UNKNOWN", "O"}):
                            candidate_type = domain_hint
                            score = max(score, max(confidence_threshold + 0.05, 0.75))

                        abbr_hint = abbreviation_type_hints.get(normalized_key)
                        if abbr_hint and (
                            candidate_type in {"UNKNOWN", "O"} or score < 0.7
                        ):
                            candidate_type = abbr_hint
                            score = max(score, max(confidence_threshold + 0.05, 0.75))

                        if candidate_type not in ENTITY_TYPES:
                            candidate_type = "UNKNOWN"

                        if score < confidence_threshold:
                            continue
                        entity_type = candidate_type if candidate_type in ENTITY_TYPES else "UNKNOWN"

                        if entity_type not in entity_set:
                            entity_set[entity_type] = set()
                            entity_map[entity_type] = []
                            normalized_map[entity_type] = []
                        if normalized_key in entity_set[entity_type]:
                            continue
                        entity_set[entity_type].add(normalized_key)
                        entity_map[entity_type].append(word)
                        normalized_map[entity_type].append(normalized_key)
                        all_entities.add(word)
                        all_entities_normalized.add(normalized_key)
                        entity_rows.append(
                            {
                                "paper_id": paper_id,
                                "entity": word,
                                "entity_norm": normalized_key,
                                "entity_type": entity_type,
                                "confidence": round(score, 4),
                                "is_unknown": entity_type == "UNKNOWN",
                                "start": global_start,
                                "end": global_end,
                            }
                        )

        existing_norms = {row["entity_norm"] for row in entity_rows if row["paper_id"] == paper_id}
        for abbr_norm, expansion in abbreviation_registry.get(paper_id, {}).items():
            if abbr_norm in existing_norms:
                continue
            abbr_text = abbreviation_originals.get(paper_id, {}).get(abbr_norm, abbr_norm.upper())
            candidate_type = abbreviation_type_hints.get(abbr_norm, "UNKNOWN")
            entity_type = candidate_type if candidate_type in ENTITY_TYPES else "UNKNOWN"
            if entity_type not in entity_set:
                entity_set[entity_type] = set()
                entity_map[entity_type] = []
                normalized_map[entity_type] = []
            if abbr_norm in entity_set[entity_type]:
                continue
            entity_set[entity_type].add(abbr_norm)
            entity_map[entity_type].append(abbr_text)
            normalized_map[entity_type].append(abbr_norm)
            all_entities.add(abbr_text)
            all_entities_normalized.add(abbr_norm)
            entity_rows.append(
                {
                    "paper_id": paper_id,
                    "entity": abbr_text,
                    "entity_norm": abbr_norm,
                    "entity_type": entity_type,
                    "confidence": round(confidence_threshold, 4),
                    "is_unknown": entity_type == "UNKNOWN",
                    "start": None,
                    "end": None,
                }
            )
            existing_norms.add(abbr_norm)

        abstract_lower = abstract.lower()
        for term, domain_type in DOMAIN_MULTIWORD_TERMS.items():
            if term not in abstract_lower:
                continue
            target_type = domain_type if domain_type in ENTITY_TYPES else "UNKNOWN"
            if target_type not in entity_set:
                entity_set[target_type] = set()
                entity_map[target_type] = []
                normalized_map[target_type] = []
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            match = pattern.search(abstract)
            if not match:
                continue
            display_text = match.group(0)
            norm_value = display_text.lower().strip()
            if norm_value in entity_set[target_type] or norm_value in existing_norms:
                continue
            entity_set[target_type].add(norm_value)
            entity_map[target_type].append(display_text)
            normalized_map[target_type].append(norm_value)
            all_entities.add(display_text)
            all_entities_normalized.add(norm_value)
            synthetic_conf = max(confidence_threshold, 0.85)
            entity_rows.append(
                {
                    "paper_id": paper_id,
                    "entity": display_text,
                    "entity_norm": norm_value,
                    "entity_type": target_type,
                    "confidence": round(synthetic_conf, 4),
                    "is_unknown": target_type == "UNKNOWN",
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            existing_norms.add(norm_value)

        augmented_row = dict(row_dict)
        augmented_row["paper_id"] = paper_id
        for entity_type in ENTITY_TYPES:
            augmented_row[entity_type] = entity_map[entity_type]
            augmented_row[f"{entity_type} Normalized"] = normalized_map[entity_type]
            augmented_row["All Entities"] = sorted(all_entities)
            augmented_row["All Entities Normalized"] = sorted(all_entities_normalized)
            augmented_row["Entity Map"] = entity_map
            augmented_rows.append(augmented_row)

        if torch_module is not None and hasattr(torch_module, "cuda") and getattr(torch_module.cuda, "is_available", lambda: False)():
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                pass

        if progress is not None:
            progress.progress(
                min(index / max(total_records, 1), 1.0),
                text=f"Processing abstracts ({index}/{total_records})",
            )

    augmented_df = pd.DataFrame(augmented_rows)
    entity_df = pd.DataFrame(entity_rows)
    if not augmented_df.empty:
        augmented_df["paper_id"] = augmented_df["paper_id"].astype(int)
    if not entity_df.empty:
        entity_df["paper_id"] = entity_df["paper_id"].astype(int)
        entity_df = entity_df.sort_values(["paper_id", "entity_type", "entity"]).reset_index(drop=True)
        entity_df = merge_acronym_variants(entity_df)
        char_mask = entity_df["entity"].astype(str).str.fullmatch(r"\s*[A-Za-z0-9]\s*")
        keep_mask = ~char_mask | entity_df.get("is_user_entity", False)
        entity_df = entity_df[keep_mask].reset_index(drop=True)
    augmented_df = sanitize_processed_dataframe(
        augmented_df,
        allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
        drop_duplicates=ENABLE_DEDUPLICATION,
    )
    entity_df = sanitize_entity_dataframe(entity_df)
    entity_df = filter_false_units(entity_df)
    entity_df = suppress_lonely_units(entity_df)
    entity_df = trim_and_filter_entities(entity_df)
    merge_map_active = get_user_canonical_merges_state()
    entity_df = apply_canonical_merge_map(entity_df, merge_map_active)
    augmented_df = apply_canonical_merges_to_processed_df(augmented_df, merge_map_active)
    augmented_df = sanitize_processed_dataframe(
        augmented_df,
        allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
        drop_duplicates=ENABLE_DEDUPLICATION,
    )
    entity_df = sanitize_entity_dataframe(entity_df)
    entity_df = trim_and_filter_entities(entity_df)

    if progress is not None:
        if ner is None and total_records:
            progress.progress(1.0, text="NER model unavailable")
        else:
            final_message = (
                "NER extraction complete" if total_records else "No abstracts found to process"
            )
            progress.progress(1.0, text=final_message)

    return augmented_df, entity_df, resolved_model, long_text_count, abbreviation_registry, paper_text_map

def filter_dataframe(
    df: pd.DataFrame,
    entity_df: pd.DataFrame,
    selected_types: Iterable[str],
    selected_entities: Iterable[str],
    search_terms: List[str],
) -> pd.DataFrame:
    working_df = df.copy()
    if not working_df.empty:
        working_df["paper_id"] = working_df["paper_id"].astype(int)

    allowed_types = set(selected_types) if selected_types else set(ENTITY_TYPES) | set(CUSTOM_ENTITY_TYPES)
    if not allowed_types:
        allowed_types = set(ENTITY_TYPES) | set(CUSTOM_ENTITY_TYPES)

    working_entity_df = entity_df.copy()
    if "canonical" not in working_entity_df.columns and not working_entity_df.empty:
        working_entity_df["canonical"] = working_entity_df.get("entity_norm", working_entity_df.get("entity", "")).apply(normalize_entity)
    if "Canonical" not in working_entity_df.columns and not working_entity_df.empty:
        working_entity_df["Canonical"] = working_entity_df["canonical"].apply(lambda val: str(val).strip().upper())
    if not working_entity_df.empty:
        working_entity_df["paper_id"] = working_entity_df["paper_id"].astype(int)
        allowed_entity_subset = working_entity_df[
            working_entity_df["entity_type"].isin(allowed_types)
        ]
        if allowed_types != set(ENTITY_TYPES):
            allowed_paper_ids = set(allowed_entity_subset["paper_id"])
            working_df = working_df[working_df["paper_id"].isin(allowed_paper_ids)]

        if selected_entities:
            normalized_selection = {entity.strip().upper() for entity in selected_entities if entity.strip()}
            canonical_series = allowed_entity_subset["Canonical"].astype(str).str.upper()
            matching_rows = set(
                allowed_entity_subset[canonical_series.isin(normalized_selection)]["paper_id"]
            )
            if matching_rows:
                working_df = working_df[working_df["paper_id"].isin(matching_rows)]
            else:
                working_df = working_df.iloc[0:0]
    elif selected_entities:
        # No entity data but entity filters applied -> no results
        working_df = working_df.iloc[0:0]

    if search_terms:
        lowered_terms = [term.lower() for term in search_terms]

        def matches_search(row: pd.Series) -> bool:
            entity_values_raw = row.get("All Entities", [])
            entity_values = _split_entities(entity_values_raw)
            haystacks = [
                row.get("Title", "").lower(),
                row.get("Abstract", "").lower(),
                " ".join(entity_values).lower(),
            ]
            return all(any(term in hay for hay in haystacks) for term in lowered_terms)

        working_df = working_df[working_df.apply(matches_search, axis=1)]

    return working_df.reset_index(drop=True)


def filter_papers_by_entities(
    entity_df: pd.DataFrame,
    required_entities: Optional[Iterable[str]] = None,
    excluded_entities: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    pandas_module = globals().get("pd")
    if entity_df.empty:
        st.info("0 of 0 papers match current filters (mode: all).")
        return entity_df.copy()

    df = entity_df.copy()
    if "Canonical" not in df.columns:
        df["Canonical"] = df.get("canonical", df.get("entity", "")).apply(lambda val: str(val).strip().upper())
    else:
        df["Canonical"] = df["Canonical"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["paper_id", "Canonical"])
    if not df.empty:
        df["paper_id"] = df["paper_id"].astype(int)

    required_set = {
        str(value).strip().upper()
        for value in (required_entities or [])
        if str(value).strip()
    }
    exclude_set = {
        str(value).strip().upper()
        for value in (excluded_entities or [])
        if str(value).strip()
    }

    mode = "all"
    grouped = df.groupby("paper_id")["Canonical"].agg(lambda vals: set(map(str, vals)))

    def has_required(canonicals: Set[str]) -> bool:
        return required_set.issubset(canonicals)

    def has_excluded(canonicals: Set[str]) -> bool:
        return bool(canonicals.intersection(exclude_set))

    match_mask = grouped.apply(lambda canonicals: has_required(canonicals) and not has_excluded(canonicals))
    if required_set:
        mode = "include"
    if exclude_set:
        mode = "exclude" if mode == "all" else "include+exclude"

    matching_ids = grouped.index[match_mask].tolist()
    total_papers = grouped.index.size
    matched_count = len(matching_ids)

    st.info(
        f"{matched_count} of {total_papers} papers match current entity filters (mode: {mode})."
    )

    filtered_entities = df[df["paper_id"].isin(matching_ids)].copy()
    if pandas_module is not None and "Canonical" in filtered_entities.columns:
        filtered_entities["Canonical"] = filtered_entities["Canonical"].astype(str)
    return filtered_entities.reset_index(drop=True)


def download_links(filtered_df: pd.DataFrame, entity_df: pd.DataFrame):
    filtered_df = sanitize_entities_output(filtered_df.drop(columns=["Entity Map"], errors="ignore"))
    entity_df = sanitize_entities_output(entity_df.drop(columns=["Entity Map"], errors="ignore"))

    csv_buffer = filtered_df.to_csv(index=False)
    json_buffer = filtered_df.to_json(
        orient="records", indent=2
    )

    left, right, extra = st.columns(3)
    with left:
        st.download_button(
            "Download CSV",
            data=csv_buffer.encode("utf-8"),
            file_name="ner_results.csv",
            mime="text/csv",
        )
    with right:
        st.download_button(
            "Download JSON",
            data=json_buffer.encode("utf-8"),
            file_name="ner_results.json",
            mime="application/json",
        )
    with extra:
        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                filtered_df.drop(columns=["Entity Map"], errors="ignore").to_excel(
                    writer, index=False, sheet_name="Papers"
                )
                if not entity_df.empty:
                    entity_df.to_excel(writer, index=False, sheet_name="Entities")
            st.download_button(
                "Download Excel",
                data=excel_buffer.getvalue(),
                file_name="ner_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as err:
            st.warning(f"Excel export unavailable: {err}")


def display_summary(entity_df: pd.DataFrame, show_canonical: bool = True):
    st.subheader("Entity Highlights")
    type_column = next((col for col in ("entity_type", "EntityType", "Type") if col in entity_df.columns), None)
    working_entities = entity_df.copy()
    skip_types = {"PROP_VALUE", "MATERIAL_AMOUNT"}
    if type_column:
        working_entities = working_entities[~working_entities[type_column].astype(str).str.upper().isin(skip_types)]

    summary_types = [etype for etype in ENTITY_TYPES if etype not in {"UNKNOWN", "O"}]
    if not summary_types:
        summary_types = ENTITY_TYPES
    metrics = st.columns(len(summary_types))
    for col, entity_type in zip(metrics, summary_types):
        count = int((working_entities["entity_type"] == entity_type).sum()) if not working_entities.empty else 0
        col.metric(entity_type, f"{count}")

    if working_entities.empty:
        st.info("No entities detected at the specified confidence threshold.")
        return

    material_priority = ("POLYMER", "POLYMER_FAMILY", "ORGANIC", "INORGANIC")
    primary_label = next(
        (label for label in material_priority if label in summary_types and not working_entities[working_entities["entity_type"] == label].empty),
        None,
    )
    if not primary_label:
        st.info("No material-like entities found to summarize.")
        return

    user_canonical = get_user_canonical_state()
    merge_map = get_user_canonical_merges_state()
    working_entities = apply_user_canonical_overrides(working_entities, user_canonical)
    working_entities = apply_canonical_merge_map(working_entities, merge_map)

    material_entities = working_entities[working_entities["entity_type"] == primary_label]
    if material_entities.empty:
        st.info(f"No {primary_label} entities found to summarize.")
        return

    canonical_series = material_entities["canonical"] if "canonical" in material_entities.columns else material_entities.get("entity_norm", material_entities.get("entity", ""))
    canonical_counter: Counter = Counter()
    representative: Dict[str, str] = {}

    for canonical_val in canonical_series:
        canonical_key = str(canonical_val).lower()
        canonical_counter[canonical_key] += 1
        representative.setdefault(canonical_key, str(canonical_val))

    top_materials = canonical_counter.most_common(15)
    if not top_materials:
        return

    labels = [representative[key] for key, _ in top_materials]
    counts = [count for _, count in top_materials]
    summary_df = pd.DataFrame({"Material": labels, "Count": counts})

    chart_col, table_col = st.columns((2, 1))
    with chart_col:
        st.bar_chart(summary_df.set_index("Material"))
    with table_col:
        working = working_entities.copy()
        if type_column is None:
            working["entity_type"] = ""
            type_column = "entity_type"

        if "canonical" in working.columns:
            working["Canonical Entity"] = working["canonical"]
        elif "entity_norm" in working.columns:
            working["Canonical Entity"] = working["entity_norm"]
        else:
            working["Canonical Entity"] = working.get("entity", "")
        working["Entity Type"] = working[type_column]

        frequency_rows: list[dict[str, object]] = []
        for (canonical_value, entity_type_value), group_df in working.groupby(
            ["Canonical Entity", "Entity Type"], dropna=False
        ):
            frequency_rows.append(
                {
                    "Canonical Entity": canonical_value,
                    "Entity Type": entity_type_value,
                    "Frequency": int(len(group_df)),
                }
            )

        frequency_df = pd.DataFrame(frequency_rows)
        column_order = ["Canonical Entity", "Entity Type", "Frequency"]
        if not show_canonical and "Canonical Entity" in frequency_df.columns:
            frequency_df = frequency_df.drop(columns=["Canonical Entity"])
            column_order = ["Entity Type", "Frequency"]
        frequency_df = frequency_df[column_order] if not frequency_df.empty else frequency_df
        st.dataframe(sanitize_entities_output(frequency_df), use_container_width=True)

def display_results(
    processed_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    model_name: str,
    confidence_threshold: float,
    show_uncertain: bool,
    linking_context: Optional[Dict[str, object]] = None,
):
    st.header("Analysis Results")
    if model_name:
        st.caption(f"Model in use: {model_name}")

    context = linking_context or st.session_state.get("linking_context", {}) or {}
    strategy_label = context.get("strategy")
    if strategy_label:
        strategy_descriptions = {
            "tfidf": "TF-IDF char n-grams (3–5) with cosine threshold ≥ 0.85",
            "semantic": "Semantic embeddings (all-MiniLM-L6-v2) with community threshold ≥ 0.80",
            "skipped": "Entity linking skipped for this run.",
            "error": "Entity linking unavailable — showing normalized entities only.",
        }
        description = strategy_descriptions.get(strategy_label, str(strategy_label))
        st.caption(f"Entity linking mode: {description}")

    search_input = st.text_input(
        "Search across titles, abstracts, and entities (use commas for multiple terms)",
        placeholder="e.g. perovskite, efficiency",
    )
    search_terms = [term.strip() for term in search_input.split(",") if term.strip()]

    processed_df = sanitize_entities_output(processed_df.copy())
    entity_df = sanitize_entities_output(entity_df.copy())
    st.session_state.setdefault("base_processed_df", processed_df.copy(deep=True))
    st.session_state.setdefault("base_entity_df", entity_df.copy(deep=True))
    processed_df = st.session_state["base_processed_df"].copy(deep=True)
    entity_df = st.session_state["base_entity_df"].copy(deep=True)
    pandas_module = globals().get("pd")
    pvu_df = st.session_state.get("pvu_df")
    if pvu_df is None and pandas_module is not None:
        pvu_df = pandas_module.DataFrame(
            columns=["paper_id", "PropertyEntity", "ValueEntity", "UnitEntity", "Measurement"]
        )
    total_papers_original = len(processed_df)
    total_entities_original = len(entity_df)
    entity_df = ensure_entity_schema(entity_df, context="analysis results")
    if not processed_df.empty and "paper_id" in processed_df.columns:
        processed_df["paper_id"] = processed_df["paper_id"].astype(int)
    if not entity_df.empty and "paper_id" in entity_df.columns:
        entity_df["paper_id"] = entity_df["paper_id"].astype(int)

    if entity_df.empty:
        uncertain_mask = pd.Series(dtype=bool)
        confident_entity_df = entity_df
        hidden_count = 0
    else:
        user_mask = entity_df.get("is_user_entity", False).astype(bool)
        # FIXED: allow custom user entities to bypass confidence/UNKNOWN filtering for display tables
        uncertain_mask = (~user_mask) & (
            (entity_df["confidence"] < confidence_threshold)
            | (entity_df["entity_type"] == "UNKNOWN")
        )
        if show_uncertain:
            confident_entity_df = entity_df
            hidden_count = 0
        else:
            confident_entity_df = entity_df[~uncertain_mask].copy()
            hidden_count = int(uncertain_mask.sum())

    if hidden_count > 0:
        st.caption(
            f"⚠️ Low-confidence or unknown entities hidden (score < {confidence_threshold:.2f})."
        )

    st.markdown("**Filter by entity types**")
    unique_types = (
        sorted(confident_entity_df["entity_type"].unique())
        if not confident_entity_df.empty
        else []
    )
    if unique_types:
        default_types = [
            etype
            for etype in st.session_state.get("selected_entity_types_state", unique_types)
            if etype in unique_types
        ] or unique_types
        selected_entity_types = st.multiselect(
            "Select entity types to include",
            options=unique_types,
            default=default_types,
            key="entity_type_filter",
        )
        if not selected_entity_types:
            st.warning("No entity types selected; results will be empty.")
        selected_types = selected_entity_types or unique_types
        st.session_state["selected_entity_types_state"] = selected_types
    else:
        st.info("No entity types available for filtering.")
        selected_types = []
    selected_types_for_filter = (
        selected_types
        if selected_types
        else (unique_types if unique_types else ENTITY_TYPES)
    )
    confident_entity_df = filter_by_entity_type(confident_entity_df, selected_types_for_filter)

    if not processed_df.empty and not confident_entity_df.empty and "paper_id" in processed_df.columns:
        allowed_ids = confident_entity_df["paper_id"].astype(int).unique().tolist()
        processed_df = processed_df[
            processed_df["paper_id"].astype(int).isin(allowed_ids)
        ].copy()
    elif not processed_df.empty:
        processed_df = processed_df.iloc[0:0].copy()

    # FIXED: Added session_state initialization for exclude_entity_filter
    if not processed_df.empty and "paper_id" in processed_df.columns:
        processed_df["paper_id"] = processed_df["paper_id"].astype(int)
    if not confident_entity_df.empty and "paper_id" in confident_entity_df.columns:
        confident_entity_df["paper_id"] = confident_entity_df["paper_id"].astype(int)

    # FIXED: ensured variable initialization before use
    filtered_entity_df = confident_entity_df.copy()
    entity_df = filtered_entity_df
    if "Canonical" not in filtered_entity_df.columns:
        filtered_entity_df["Canonical"] = filtered_entity_df.get("canonical", filtered_entity_df.get("entity", "")).apply(lambda val: str(val).strip().upper())

    canonical_options = sorted(
        {
            str(val).strip().upper()
            for val in filtered_entity_df["Canonical"].dropna().tolist()
            if str(val).strip()
        }
    )

    include_selection = st.multiselect(
        "Include papers that contain all of these entities:",
        options=canonical_options,
        key="canonical_include_filter",
        placeholder="Optional: limit results to selected canonicals",
    )
    exclude_selection = st.multiselect(
        "Exclude papers that contain any of these entities:",
        options=canonical_options,
        key="canonical_exclude_filter",
        placeholder="Optional: remove selected canonicals",
    )

    filter_mode = "all"
    filtered_entity_display = filter_papers_by_entities(
        filtered_entity_df,
        include_selection,
        exclude_selection,
    )
    allowed_paper_ids = (
        set(filtered_entity_display["paper_id"].astype(int).tolist())
        if "paper_id" in filtered_entity_display.columns
        else set()
    )
    if allowed_paper_ids:
        processed_display = processed_df[processed_df["paper_id"].astype(int).isin(allowed_paper_ids)].copy()
    else:
        processed_display = processed_df.iloc[0:0].copy()
        if include_selection:
            filter_mode = "include"
        if exclude_selection:
            filter_mode = "exclude" if filter_mode == "all" else "include+exclude"

    available_papers_after_filters = len(processed_display)
    selected_entities = set()

    filtered_df = filter_dataframe(
        processed_display,
        filtered_entity_display,
        selected_types_for_filter,
        selected_entities,
        search_terms,
    )

    matched_papers = (
        filtered_df["paper_id"].unique().tolist() if "paper_id" in filtered_df.columns else []
    )
    matched_paper_count = len(matched_papers)
    st.caption(
        safe_to_str(
            f"{matched_paper_count} of {available_papers_after_filters} papers remain after type/search filters (mode: {filter_mode})."
        )
    )
    st.caption(
        safe_to_str(
            f"Include filters: {len(include_selection)} · Exclude filters: {len(exclude_selection)} · Total canonicals available: {len(canonical_options)}"
        )
    )
    visible_entities = (
        filtered_entity_display[filtered_entity_display["paper_id"].isin(matched_papers)].copy()
        if not filtered_entity_display.empty and matched_papers
        else pd.DataFrame()
    )
    if not visible_entities.empty:
        if "canonical" not in visible_entities.columns:
            visible_entities["canonical"] = visible_entities.get("entity_norm", visible_entities.get("entity", "")).apply(normalize_entity)
        visible_entities["Canonical"] = visible_entities.get("Canonical", visible_entities["canonical"].apply(lambda val: str(val).strip().upper()))
    st.caption(
        safe_to_str(
            f"{len(filtered_entity_display)} of {total_entities_original} entities visible; {matched_paper_count} papers matched."
        )
    )
    st.dataframe(sanitize_entities_output(filtered_df.head()), use_container_width=True)

    if filtered_entity_display.empty:
        st.info(
            "No entities detected. Try lowering the confidence threshold or switching to a domain-specific model."
        )
    elif not visible_entities.empty:
        display_summary(visible_entities, show_canonical=True)

    st.subheader("📊 Extracted Property–Value–Unit Relationships")
    if (
        pandas_module is not None
        and isinstance(pvu_df, pandas_module.DataFrame)
        and not pvu_df.empty
    ):
        st.dataframe(pvu_df, use_container_width=True)
        st.info(
            f"✅ Found {len(pvu_df)} property/value records "
            f"across {pvu_df['Source'].nunique()} papers."
        )
    else:
        st.warning("No Property–Value–Unit triples detected.")

    paper_entity_lookup: Dict[int, Dict[str, List[str]]] = {}
    for paper_id, group in visible_entities.groupby("paper_id"):
        type_map: Dict[str, List[str]] = {}
        for entity_type, subset in group.groupby("entity_type"):
            source_series = subset["Canonical"] if "Canonical" in subset.columns else subset.get("entity", [])
            values = sorted({str(val).strip() for val in source_series if str(val).strip()})
            if not values and "entity" in subset.columns:
                values = sorted({str(val).strip() for val in subset["entity"] if str(val).strip()})
            if values:
                normalized_values = deep_clean(values)
                if not _should_keep(normalized_values):
                    continue
                if isinstance(normalized_values, list):
                    type_map[entity_type] = normalized_values
                elif normalized_values is not None:
                    type_map[entity_type] = [normalized_values]
        paper_entity_lookup[paper_id] = type_map

    def joined_entities(paper_id: int, entity_type: Optional[str] = None) -> str:
        type_map = paper_entity_lookup.get(paper_id, {})
        if entity_type:
            return ", ".join(type_map.get(entity_type, []))
        all_values: List[str] = []
        for values in type_map.values():
            all_values.extend(values)
        return ", ".join(sorted(set(all_values), key=str.lower))

    st.subheader("Interactive Table")
    display_df = filtered_df.drop(columns=["Entity Map"], errors="ignore").copy()
    if "paper_id" in display_df.columns:
        for entity_type in ENTITY_TYPES:
            display_df[entity_type] = display_df["paper_id"].map(
                lambda pid: joined_entities(pid, entity_type)
            )
        display_df["All Entities"] = display_df["paper_id"].map(joined_entities)
    st.dataframe(sanitize_entities_output(display_df), use_container_width=True)

    st.subheader("Detected Entities")
    if visible_entities.empty:
        if filtered_entity_df.empty:
            st.info("No entities available to display.")
        else:
            st.info("No entities match your current filter.")
    else:
        detailed_entities = visible_entities.copy()
        if "Canonical_Entity" not in detailed_entities.columns:
            if "Canonical" in detailed_entities.columns:
                detailed_entities["Canonical_Entity"] = detailed_entities["Canonical"]
            elif "entity" in detailed_entities.columns:
                detailed_entities["Canonical_Entity"] = detailed_entities["entity"]
        if "entity" in detailed_entities.columns and "Entity_Text" not in detailed_entities.columns:
            detailed_entities["Entity_Text"] = detailed_entities["entity"]
        if "entity_type" in detailed_entities.columns and "Entity_Type" not in detailed_entities.columns:
            detailed_entities["Entity_Type"] = detailed_entities["entity_type"]
        if {"Canonical_Entity", "Entity_Text", "Entity_Type"}.issubset(detailed_entities.columns):
            detailed_entities["Canonical_Entity"] = detailed_entities.apply(
                lambda row: row["Entity_Text"]
                if row["Entity_Type"] in ["PROP_VALUE", "MATERIAL_AMOUNT"]
                and (
                    row["Canonical_Entity"] is None
                    or str(row["Canonical_Entity"]).strip() == ""
                    or str(row["Canonical_Entity"]).lower() == "none"
                )
                else row["Canonical_Entity"],
                axis=1,
            )
        display_cols = [
            col
            for col in [
                "paper_id",
                "entity",
                "Canonical",
                "entity_type",
                "confidence",
                "Canonical_Entity",
            ]
            if col in detailed_entities.columns
        ]
        st.dataframe(
            sanitize_entities_output(detailed_entities[display_cols]),
            use_container_width=True,
        )

    if filtered_df.empty:
        download_links(display_df, visible_entities)
        return

    st.subheader("Detailed View")
    detail_df = filtered_df.copy()
    for subset_cols in (["DOI", "Title", "Abstract"], ["paper_id", "Title", "Abstract"], ["paper_id"]):
        existing = [col for col in subset_cols if col in detail_df.columns]
        if existing:
            detail_df = detail_df.drop_duplicates(subset=existing, keep="first")
            break
    max_rows = min(len(detail_df), 50)
    if max_rows == 0:
        st.info("Adjust filters or lower the confidence threshold to view results.")
        return

    if max_rows == 1:
        row_limit = 1
    else:
        row_limit = st.slider(
            "How many papers to render below?",
            min_value=1,
            max_value=max_rows,
            value=min(10, max_rows),
            step=1,
        )

    for _, row in detail_df.head(row_limit).iterrows():
        with st.container():
            title = row.get("Title") or "Untitled"
            st.markdown(safe_to_str(f"### {title}"))
            meta_bits: List[str] = []
            year = row.get("Year")
            if year:
                meta_bits.append(f"**Year:** {year}")
            doi = row.get("DOI")
            if doi:
                meta_bits.append(f"**DOI:** {doi}")
            if meta_bits:
                st.markdown(safe_to_str(" | ".join(meta_bits)))
            paper_entities_raw = paper_entity_lookup.get(row.get("paper_id"), {})
            paper_entities = deep_clean(paper_entities_raw) if paper_entities_raw else {}
            # --- Fix PROP_VALUE / MATERIAL_AMOUNT canonical None issue safely ---
            if isinstance(paper_entities, dict):
                fixed_entities = {}
                for label, entity_value in paper_entities.items():
                    if label in ["PROP_VALUE", "MATERIAL_AMOUNT"]:
                        if entity_value is None or str(entity_value).strip().lower() in ["", "none", "nan"]:
                            fixed_entities[label] = label
                        else:
                            fixed_entities[label] = entity_value
                    else:
                        fixed_entities[label] = entity_value
                paper_entities = fixed_entities
            if not isinstance(paper_entities, dict) or not _should_keep(paper_entities):
                paper_entities = {}
            chips_html = render_entity_chips(paper_entities)
            if chips_html:
                st.markdown(safe_to_str(chips_html), unsafe_allow_html=True)
            abstract = row.get("Abstract", "")
            if abstract:
                abstract_normalized = deep_clean(abstract)
                if not _should_keep(abstract_normalized):
                    continue
                if isinstance(abstract_normalized, str):
                    st.write(safe_to_str(abstract_normalized))
                else:
                    st.write(abstract_normalized)
        st.divider()

    download_links(display_df, visible_entities)


def reset_state_on_file_change(uploaded_name: str, file_hash: str):
    stored_name = st.session_state.get("uploaded_file_name")
    stored_hash = st.session_state.get("uploaded_file_hash")
    if uploaded_name != stored_name or file_hash != stored_hash:
        st.session_state["uploaded_file_name"] = uploaded_name
        st.session_state["uploaded_file_hash"] = file_hash
        st.session_state.pop("processed_bundle", None)
        st.session_state.pop("processing_signature", None)
        st.session_state.pop("last_elapsed", None)


def reset_state_on_option_change(model_name: str, confidence: float, cache_key: str):
    signature = (model_name, round(confidence, 3))
    stored_signature = st.session_state.get("processing_signature")
    if signature != stored_signature:
        st.session_state["processing_signature"] = signature
        if not use_cached_results(cache_key):
            st.session_state.pop("processed_bundle", None)
            st.session_state.pop("last_elapsed", None)


def clear_results():
    """Clear cached processing results and restart the app state."""
    for key in (
        "processed_bundle",
        "uploaded_file_name",
        "uploaded_file_hash",
        "processing_signature",
        "last_elapsed",
        "chunked_count",
        "linked_count",
        "linking_context",
        "active_cache_key",
        "pvu_df",
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("results_cache", None)
    st.experimental_rerun()


def hide_inline_help():
    """Collapse the inline help panel."""
    st.session_state["show_inline_help"] = False


def safe_read_excel(file_name: str, data: bytes) -> pd.DataFrame:
    """Read Excel files with engine preference and clear messaging."""
    extension = Path(file_name).suffix.lower()

    if not data:
        raise ValueError("Uploaded file is empty.")

    engine_preference: List[Optional[str]]
    if extension == ".xls":
        engine_preference = ["xlrd", None]
    elif extension == ".xlsx":
        engine_preference = ["openpyxl", None]
    else:
        engine_preference = [None]

    last_error: Optional[Exception] = None
    for engine in engine_preference:
        buffer = io.BytesIO(data)
        try:
            if engine:
                return pd.read_excel(buffer, engine=engine)
            return pd.read_excel(buffer)
        except ImportError as err:
            last_error = err
            continue
        except ValueError as err:
            raise ValueError(str(err)) from err
        except Exception as err:  # pragma: no cover
            last_error = err
            continue

    if last_error:
        raise last_error
    raise ValueError(
        "Unable to read the Excel file with the available engines. "
        "Install Excel support libraries and retry."
    )


def standardize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
    """Rename common aliases to required column names and report missing fields."""
    rename_map: Dict[str, str] = {}
    lower_map = {col.lower().strip(): col for col in df.columns}

    if "Title" not in df.columns:
        for alias in TITLE_ALIASES:
            original = lower_map.get(alias)
            if original:
                rename_map[original] = "Title"
                break

    if "Abstract" not in df.columns:
        for alias in ABSTRACT_ALIASES:
            original = lower_map.get(alias)
            if original:
                rename_map[original] = "Abstract"
                break

    standardized = df.rename(columns=rename_map)
    missing = {column for column in REQUIRED_COLUMNS if column not in standardized.columns}
    return standardized, missing


def load_and_merge_excel_files(uploaded_files: List["UploadedFile"]) -> Tuple[pd.DataFrame, str, str]:
    frames: List[pd.DataFrame] = []
    column_signature: Optional[Tuple[str, ...]] = None
    combined_hash = hashlib.sha256()
    file_names: List[str] = []

    for index, uploaded in enumerate(uploaded_files, start=1):
        file_name = uploaded.name or f"file_{index}"
        data = uploaded.read() or b""
        uploaded.seek(0)
        combined_hash.update(file_name.encode("utf-8"))
        combined_hash.update(data)
        frame = safe_read_excel(file_name, data)
        signature = tuple(col.strip().lower() for col in frame.columns)
        if column_signature is None:
            column_signature = signature
        elif signature != column_signature:
            raise ValueError("❌ Uploaded files must have the same structure.")
        frames.append(frame)
        file_names.append(file_name)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if ENABLE_DEDUPLICATION:
        merged = merged.drop_duplicates().reset_index(drop=True)
    merged = sanitize_entities_output(merged)
    aggregated_name = "|".join(file_names)
    aggregated_hash = combined_hash.hexdigest()
    return merged, aggregated_name, aggregated_hash


def make_cache_key(file_hash: str, model_name: str, confidence: float) -> str:
    return f"{file_hash}|{model_name}|{confidence:.3f}"


def get_results_cache() -> Dict[str, Dict[str, object]]:
    return st.session_state.setdefault("results_cache", {})


def use_cached_results(cache_key: str) -> bool:
    cache_entry = get_results_cache().get(cache_key)
    if cache_entry:
        processed_clean = sanitize_processed_dataframe(
            cache_entry.get("processed_df", pd.DataFrame()),
            allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
            drop_duplicates=ENABLE_DEDUPLICATION,
        )
        entity_clean = sanitize_entity_dataframe(cache_entry.get("entity_df", pd.DataFrame()))
        user_entities_active = get_user_entities_state()
        user_canonical_active = get_user_canonical_state()
        paper_text_map: Dict[int, object] = {}
        if not processed_clean.empty and {"paper_id", "Abstract"}.issubset(processed_clean.columns):
            paper_text_map = {
                int(row.paper_id): row.Abstract
                for row in processed_clean.itertuples()
                if pd.notna(row.paper_id)
            }
        entity_clean, user_entities_map = inject_user_entities(
            entity_clean, paper_text_map, user_entities_active
        )
        processed_clean = augment_processed_with_user_entities(processed_clean, user_entities_map)
        entity_clean = apply_user_canonical_overrides(entity_clean, user_canonical_active)
        merge_map_active = get_user_canonical_merges_state()
        entity_clean = apply_canonical_merge_map(entity_clean, merge_map_active)
        processed_clean = apply_canonical_merges_to_processed_df(processed_clean, merge_map_active)
        selection = st.session_state.get("exclude_entity_filter_main")
        selection = selection if selection and selection != "None" else None
        entity_clean, processed_clean, _ = apply_exclusions(
            entity_clean,
            processed_clean,
            selection,
        )
        processed_clean = sanitize_processed_dataframe(
            processed_clean,
            allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
            drop_duplicates=ENABLE_DEDUPLICATION,
        )
        entity_clean = sanitize_entity_dataframe(entity_clean)
        sanitized_context = sanitize_linking_context(cache_entry.get("linking_context"))
        sanitized_entry = dict(cache_entry)
        pvu_table = build_property_value_unit_table(entity_clean)
        pandas_module = globals().get("pd")
        sanitized_entry["processed_df"] = processed_clean
        sanitized_entry["entity_df"] = entity_clean
        sanitized_entry["linking_context"] = sanitized_context
        sanitized_entry["pvu_df"] = (
            pvu_table.copy(deep=True)
            if pandas_module is not None and isinstance(pvu_table, pandas_module.DataFrame)
            else pvu_table
        )
        sanitized_entry = deep_clean(sanitized_entry)
        if not _should_keep(sanitized_entry):
            sanitized_entry = {}
        get_results_cache()[cache_key] = sanitized_entry
        st.session_state["processed_bundle"] = sanitized_entry
        st.session_state["last_elapsed"] = sanitized_entry.get("elapsed")
        st.session_state["chunked_count"] = sanitized_entry.get("chunk_count", 0)
        st.session_state["linked_count"] = sanitized_entry.get("linked_count", 0)
        st.session_state["linking_context"] = sanitized_context
        st.session_state["active_cache_key"] = sanitized_entry.get("cache_key", cache_key)
        st.session_state["pvu_df"] = sanitized_entry.get("pvu_df", pvu_table)
        return True
    return False


def cache_results(
    cache_key: str,
    processed_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    resolved_model: str,
    elapsed: float,
    chunk_count: int,
    linked_count: int,
    linking_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    entity_df = apply_user_canonical_overrides(entity_df, get_user_canonical_state())
    merge_map_active = get_user_canonical_merges_state()
    entity_df = apply_canonical_merge_map(entity_df, merge_map_active)
    processed_df = apply_canonical_merges_to_processed_df(processed_df, merge_map_active)
    selection = st.session_state.get("exclude_entity_filter_main")
    selection = selection if selection and selection != "None" else None
    entity_df, processed_df, _ = apply_exclusions(
        entity_df,
        processed_df,
        selection,
    )
    processed_clean = sanitize_processed_dataframe(
        processed_df,
        allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
        drop_duplicates=ENABLE_DEDUPLICATION,
    )
    entity_clean = sanitize_entity_dataframe(entity_df)
    entity_clean = trim_and_filter_entities(entity_clean)
    pvu_table = build_property_value_unit_table(entity_clean)
    context_payload = sanitize_linking_context(linking_context)
    processed_clean = deep_clean(processed_clean)
    entity_clean = deep_clean(entity_clean)
    context_payload = deep_clean(context_payload)
    pandas_module = globals().get("pd")
    if not _is_pandas_object(processed_clean) and pandas_module is not None:
        processed_clean = processed_clean if _should_keep(processed_clean) else pandas_module.DataFrame()
    if not _is_pandas_object(entity_clean) and pandas_module is not None:
        entity_clean = entity_clean if _should_keep(entity_clean) else pandas_module.DataFrame()
    if not _should_keep(context_payload):
        context_payload = {}
    pandas_module = globals().get("pd")
    entry = {
        "processed_df": processed_clean.copy(deep=True) if pandas_module is not None and isinstance(processed_clean, pandas_module.DataFrame) else processed_clean,
        "entity_df": entity_clean.copy(deep=True) if pandas_module is not None and isinstance(entity_clean, pandas_module.DataFrame) else entity_clean,
        "resolved_model": resolved_model,
        "elapsed": elapsed,
        "chunk_count": chunk_count,
        "linked_count": linked_count,
        "linking_context": context_payload,
        "cache_key": cache_key,
        "pvu_df": pvu_table.copy(deep=True) if pandas_module is not None and isinstance(pvu_table, pandas_module.DataFrame) else pvu_table,
    }
    get_results_cache()[cache_key] = entry
    st.session_state["processed_bundle"] = entry
    st.session_state["last_elapsed"] = elapsed
    st.session_state["chunked_count"] = chunk_count
    st.session_state["linked_count"] = linked_count
    st.session_state["linking_context"] = context_payload
    st.session_state["active_cache_key"] = cache_key
    st.session_state["pvu_df"] = entry["pvu_df"]
    return entry


def display_footer():
    """Render a footer with metadata."""
    st.markdown(
        f"---\n"
        f"<small>Scientific Named Entity Explorer v{APP_VERSION} · "
        f"<a href=\"{GITHUB_URL}\" target=\"_blank\">View on GitHub</a></small>",
        unsafe_allow_html=True,
    )


def main():
    if pipeline is None:  # pragma: no cover
        st.error(
            "Transformers library is unavailable. Install dependencies with "
            "`pip install -r requirements.txt` and restart the app."
        )
        display_footer()
        st.stop()

    device_preference = st.session_state.get("device_preference", DEFAULT_DEVICE_PREFERENCE)
    if device_preference not in {"auto"}:
        device_preference = "auto"

    # ADDED: Initialize session state defaults before creating widgets.
    session_defaults = {
        "user_entity_selection": [],
        "selected_entity_types_state": [],
        "canonical_include_filter": [],
        "canonical_exclude_filter": [],
        "exclude_entity_filter_main": None,
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value

    semantic_available = SentenceTransformer is not None and st_util is not None
    if "use_semantic_linking" not in st.session_state:
            st.session_state["use_semantic_linking"] = semantic_available
    elif not semantic_available and st.session_state.get("use_semantic_linking"):
        st.session_state["use_semantic_linking"] = False

    if "show_inline_help" not in st.session_state:
        st.session_state["show_inline_help"] = False

    st.title("Scientific Named Entity Explorer")
    st.markdown(
        "Upload an Excel file containing research abstracts, run pretrained NER models, "
        "and interactively explore extracted entities."
    )
    st.markdown(
        "<style>[data-testid='stMarkdown'] span[data-ner-artifact='set()'],"
        "[data-testid='stMarkdown'] span[data-ner-artifact='{}']{display:none!important;}</style>",
        unsafe_allow_html=True,
    )

    get_user_entities_state()
    get_user_canonical_state()
    get_user_canonical_merges_state()

    header_cols = st.columns([1, 4])
    with header_cols[0]:
        if st.button("Getting Started"):
            show_getting_started_help()
    with header_cols[1]:
        st.caption("Runs fully local with cached Hugging Face models.")

    if st.session_state.get("show_inline_help"):
        with st.expander("Getting Started", expanded=True):
            st.markdown(GETTING_STARTED_MARKDOWN)
            st.button(
                "Hide Getting Started",
                key="hide_help_button",
                on_click=hide_inline_help,
            )

    with st.sidebar:
        st.header("Configuration")
        config_box = st.container(border=True)
        with config_box:
            st.subheader("Model & Threshold", divider="gray")
            model_label = st.selectbox("NER model", list(MODEL_OPTIONS.keys()))
            model_name = MODEL_OPTIONS[model_label]
            confidence_threshold = st.slider(
                "Minimum confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
            )
            run_linking = st.checkbox(
                "Run Entity Linking",
                value=st.session_state.get("run_linking", True),
            )
            use_semantic_default = st.session_state.get("use_semantic_linking", semantic_available)
            use_semantic_linking = st.checkbox(
                "Use Semantic Linking (slower, higher accuracy)",
                value=use_semantic_default,
                help="Toggle to switch from TF-IDF cosine similarity to semantic sentence embeddings.",
            )
            if not semantic_available and use_semantic_linking:
                st.caption("Semantic linking model unavailable; TF-IDF similarity will be used instead.")
            device_preference = "auto"
            st.markdown(
                "[Model selection tips](https://huggingface.co/models?pipeline_tag=token-classification)",
                help="Browse available NER checkpoints on Hugging Face.",
            )
            st.session_state["run_linking"] = run_linking
            st.session_state["use_semantic_linking"] = use_semantic_linking
            st.session_state["show_canonical_entities"] = True

        with st.sidebar.expander("Custom Entities", expanded=False):
            st.caption("Add vocabulary that should always appear in the results.")
            user_entities_current = get_user_entities_state()
            if "user_entity_selection" not in st.session_state:
                st.session_state["user_entity_selection"] = []

            with st.form("custom_entity_form", clear_on_submit=True):
                new_entity_term = st.text_input(
                    "Entity term",
                    key="custom_entity_term_input",
                    placeholder="e.g. self-assembled monolayer",
                )
                new_entity_type = st.selectbox(
                    "Entity type",
                    CUSTOM_ENTITY_TYPES,
                    key="custom_entity_type_input",
                )
                submitted_new_entity = st.form_submit_button("Add custom entity")

            entities_changed = False
            if submitted_new_entity:
                candidate = _clean_term(new_entity_term)
                selected_type = safe_to_str(new_entity_type).upper()
                if not candidate:
                    st.warning("Enter a non-empty term before adding.")
                else:
                    candidate_key = candidate.lower()
                    updated_entities: List[Dict[str, str]] = []
                    replaced = False
                    for entry in user_entities_current:
                        if entry["term"].lower() == candidate_key:
                            if entry["type"] != selected_type:
                                updated_entities.append({"term": entry["term"], "type": selected_type})
                                replaced = True
                            else:
                                updated_entities.append(entry)
                                replaced = True
                        else:
                            updated_entities.append(entry)
                    if not replaced:
                        updated_entities.append({"term": candidate, "type": selected_type})
                        st.success(
                            safe_to_str(f"Added '{candidate}' as {selected_type}.")
                        )
                    else:
                        st.info(
                            safe_to_str(f"Updated '{candidate}' to type {selected_type}.")
                        )
                    user_entities_current = update_user_entities_state(updated_entities)
                    entities_changed = True
                    st.session_state["user_entity_selection"].clear()
            existing_terms = [entry["term"] for entry in user_entities_current]
            existing_selection = st.multiselect(
                "Existing terms",
                options=existing_terms,
                default=list(st.session_state.get("user_entity_selection", [])),
                key="user_entity_selection",
                help="Select terms to delete.",
            )
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Delete selected", key="user_entity_delete") and existing_selection:
                    updated = [
                        entry for entry in user_entities_current if entry["term"] not in existing_selection
                    ]
                    user_entities_current = update_user_entities_state(updated)
                    entities_changed = True
                    st.session_state["user_entity_selection"][:] = [
                        value
                        for value in st.session_state.get("user_entity_selection", [])
                        if value not in existing_selection
                    ]
            with action_cols[1]:
                if st.button("Reset terms", key="user_entity_reset"):
                    user_entities_current = reset_user_entities_state()
                    entities_changed = True
                    st.session_state["user_entity_selection"].clear()
            if user_entities_current:
                user_entities_df = pd.DataFrame(user_entities_current)
                st.dataframe(
                    sanitize_entities_output(user_entities_df[["term", "type"]]),
                    use_container_width=True,
                )
            else:
                st.info("No custom entities defined yet.")
            st.caption(f"{len(user_entities_current)} custom term(s) saved.")
            if entities_changed:
                refresh_active_results_with_user_settings()

        with st.sidebar.expander("Canonical Merging", expanded=False):
            st.caption("Manually merge detected entities into a shared canonical label.")
            user_canonical_map = get_user_canonical_state()
            canonical_changed = False
            if user_canonical_map:
                canonical_rows = [
                    {"Canonical": canonical, "Variants": ", ".join(variants)}
                    for canonical, variants in user_canonical_map.items()
                ]
                st.dataframe(
                    sanitize_entities_output(pd.DataFrame(canonical_rows)),
                    use_container_width=True,
                )
            else:
                st.info("No canonical merges defined yet.")

            available_entities = session_available_entities()
            canonical_name_input = st.text_input(
                "Canonical label",
                key="user_canonical_name",
                placeholder="e.g. OPV",
            )
            variant_selection = st.multiselect(
                "Variants to merge",
                options=available_entities,
                key="user_canonical_variants",
                help="Pick one or more variants to merge under the canonical label.",
            )
            if st.button("Add Variants", key="user_canonical_merge"):
                canonical_candidate = _clean_term(canonical_name_input)
                selected_variants = _deduplicate_terms(variant_selection)
                if not canonical_candidate:
                    st.warning("Canonical label cannot be empty.")
                elif not selected_variants:
                    st.warning("Select at least one variant to merge.")
                else:
                    updated_map = {key: list(values) for key, values in user_canonical_map.items()}
                    lookup = {key.lower(): key for key in updated_map.keys()}
                    canonical_key = lookup.get(canonical_candidate.lower(), canonical_candidate)
                    combined_variants = updated_map.get(canonical_key, []) + selected_variants
                    updated_map[canonical_key] = _deduplicate_terms(combined_variants)
                    user_canonical_map = update_user_canonical_state(updated_map)
                    merge_map = get_user_canonical_merges_state()
                    merge_updates = dict(merge_map)
                    for variant in selected_variants:
                        variant_clean = _clean_term(variant)
                        if variant_clean:
                            merge_updates[variant_clean] = canonical_key
                    update_user_canonical_merges_state(merge_updates)
                    canonical_changed = True
                    st.success(
                        safe_to_str(
                            f"Attached {', '.join(sorted(selected_variants))} to {canonical_key}"
                        )
                    )

            if user_canonical_map:
                st.markdown("**Manage Canonical Variants**")
                selected_group = st.selectbox(
                    "Select canonical group",
                    sorted(user_canonical_map.keys()),
                    key="canonical_variant_group",
                )
                variants = user_canonical_map.get(selected_group, [])
                chosen_variants = st.multiselect(
                    "Variants to merge or delete",
                    variants,
                    key=f"canonical_variant_choices_{selected_group}",
                )
                variant_cols = st.columns(3)
                with variant_cols[0]:
                    if st.button("Merge Selected", key="canonical_merge_selected"):
                        if not chosen_variants:
                            st.warning("Select variants before merging.")
                        else:
                            updated_map = merge_selected_variants(
                                user_canonical_map, selected_group, chosen_variants
                            )
                            user_canonical_map = update_user_canonical_state(updated_map)
                            merge_map = get_user_canonical_merges_state()
                            merge_updates = dict(merge_map)
                            for variant in chosen_variants:
                                variant_clean = _clean_term(variant)
                                if variant_clean:
                                    merge_updates[variant_clean] = _clean_term(selected_group)
                            update_user_canonical_merges_state(merge_updates)
                            canonical_changed = True
                            st.success(
                                safe_to_str(
                                    f"Merged {', '.join(sorted(chosen_variants))} into {selected_group}"
                                )
                            )
                with variant_cols[1]:
                    if st.button("Delete Selected", key="canonical_delete_selected"):
                        if not chosen_variants:
                            st.warning("Select variants before deleting.")
                        else:
                            updated_map = delete_selected_variants(
                                user_canonical_map, selected_group, chosen_variants
                            )
                            user_canonical_map = update_user_canonical_state(updated_map)
                            merge_map = get_user_canonical_merges_state()
                            merge_updates = {
                                key: val
                                for key, val in merge_map.items()
                                if _clean_term(key) not in {_clean_term(v) for v in chosen_variants}
                            }
                            update_user_canonical_merges_state(merge_updates)
                            canonical_changed = True
                            st.success(
                                safe_to_str(
                                    f"Deleted {', '.join(sorted(chosen_variants))} from {selected_group}"
                                )
                            )
                with variant_cols[2]:
                    if st.button("Reset Canonical", key="canonical_reset_variants"):
                        reset_user_canonical_state()
                        reset_user_canonical_merges_state()
                        user_canonical_map = get_user_canonical_state()
                        canonical_changed = True
                        st.success("Canonical mappings reset.")

                st.markdown("**Merge Canonical Labels**")
                available_canonicals = session_available_canonicals()
                merge_sources = st.multiselect(
                    "Canonicals to merge",
                    options=available_canonicals,
                    key="user_canonical_merge_sources",
                    help="Select canonical names that should be unified.",
                )
                merge_target_options = merge_sources if merge_sources else available_canonicals
                merge_target = st.selectbox(
                    "Canonical to keep",
                    options=merge_target_options if merge_target_options else ["No canonicals available"],
                    key="user_canonical_merge_target",
                    disabled=len(merge_target_options) == 0,
                )
                if merge_target == "No canonicals available":
                    merge_target = ""
                if st.button("Merge canonical entities", key="user_canonical_merge_button"):
                    if len(merge_sources) < 2:
                        st.warning("Select at least two canonical entities to merge.")
                    elif not merge_target or merge_target not in merge_target_options:
                        st.warning("Choose the canonical entity to keep after merging.")
                    else:
                        target_clean = _clean_term(merge_target)
                        working_sources = [
                            _clean_term(source)
                            for source in merge_sources
                            if _clean_term(source) and _clean_term(source).lower() != target_clean.lower()
                        ]
                        if not working_sources:
                            st.info("No additional canonicals selected to merge.")
                        else:
                            merge_map = get_user_canonical_merges_state()
                            updated_merge_map = dict(merge_map)
                            for source in working_sources:
                                updated_merge_map[source] = target_clean
                            update_user_canonical_merges_state(updated_merge_map)
                            updated_map = {key: list(values) for key, values in user_canonical_map.items()}
                            for source in working_sources:
                                variants = updated_map.pop(source, [])
                                updated_map.setdefault(target_clean, [])
                                updated_map[target_clean] = _deduplicate_terms(
                                    updated_map[target_clean] + variants + [source]
                                )
                            user_canonical_map = update_user_canonical_state(updated_map)
                            canonical_changed = True
                            st.success(
                                safe_to_str(
                                    f"Merged {', '.join(sorted(working_sources))} into {target_clean}"
                                )
                            )

            st.caption(f"{len(user_canonical_map)} canonical group(s) saved.")
            if canonical_changed:
                refresh_active_results_with_user_settings()

        st.divider()
        st.markdown(
            f"Need setup help? Run `python3 install.py` or review the "
            f"[README]({GITHUB_URL}#readme)."
        )
    st.session_state["device_preference"] = device_preference

    uploaded_files = st.file_uploader(
        "Upload Excel file(s) (.xls or .xlsx)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload an Excel file to begin.")
        display_footer()
        return

    try:
        merged_frame, uploaded_name, file_hash = load_and_merge_excel_files(uploaded_files)
    except ValueError as err:
        st.error(str(err))
        display_footer()
        return
    except Exception as err:  # pragma: no cover
        st.error("Unexpected error while reading the uploaded files.")
        st.exception(err)
        display_footer()
        return

    reset_state_on_file_change(uploaded_name, file_hash)

    dataframe, missing_columns = standardize_dataframe(merged_frame)
    if missing_columns:
        st.error(
            "The uploaded file is missing required columns. "
            f"Ensure it contains: {', '.join(sorted(REQUIRED_COLUMNS))}."
        )
        display_footer()
        return

    file_count = len(uploaded_files)
    if ENABLE_DEDUPLICATION:
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    dataframe = dataframe.copy()
    if "paper_id" not in dataframe.columns:
        dataframe.insert(0, "paper_id", range(1, len(dataframe) + 1))

    if dataframe["Abstract"].dropna().empty:
        st.error(
            "The Abstract column is present but empty. Provide abstracts "
            "to run entity extraction."
        )
        display_footer()
        return

    column_list = ", ".join(dataframe.columns.tolist())
    total_rows = len(dataframe)
    if file_count > 1:
        st.success(f"✅ Merged {file_count} files, total {total_rows} rows with columns: {column_list}.")
    else:
        st.success(f"Loaded {total_rows} records with columns: {column_list}.")

    preview_columns = list(REQUIRED_COLUMNS | OPTIONAL_COLUMNS)
    preview = dataframe.loc[:, [col for col in preview_columns if col in dataframe.columns]].head(5)
    st.dataframe(sanitize_entities_output(preview), use_container_width=True)

    cache_key = make_cache_key(file_hash, model_name, confidence_threshold)
    reset_state_on_option_change(model_name, confidence_threshold, cache_key)

    action_cols = st.columns([1, 1, 4])
    with action_cols[0]:
        run_button = st.button("Run NER extraction", type="primary", use_container_width=True)
    with action_cols[1]:
        if st.button("Clear Results", type="secondary", use_container_width=True):
            clear_results()

    if not dataframe.empty and dataframe["Abstract"].isna().all():
        st.warning("All abstracts are NaN. Please verify the spreadsheet content.")

    if not run_button:
        use_cached_results(cache_key)

    chunk_count_cached = st.session_state.get("chunked_count", 0)
    linked_count_cached = st.session_state.get("linked_count", 0)
    chunk_count = chunk_count_cached
    linked_count = linked_count_cached

    if run_button:
        min_len_used = DEFAULT_MIN_ABSTRACT_LENGTH
        include_all_types = DEFAULT_INCLUDE_ALL_TYPES
        records, record_stats = build_records(
            dataframe,
            skip_filters=True,
            min_abstract_length=min_len_used,
            include_all_types=include_all_types,
            deduplicate=ENABLE_DEDUPLICATION,
        )

        summary_parts: List[str] = []
        if record_stats.get("missing_abstract_rows"):
            summary_parts.append(f"missing abstracts: {record_stats['missing_abstract_rows']}")
        if record_stats.get("short_abstract_rows") and min_len_used > 0:
            summary_parts.append(
                f"short abstracts (<{min_len_used} chars): {record_stats['short_abstract_rows']}"
            )
        if record_stats.get("duplicate_candidates"):
            summary_parts.append(f"duplicate candidates: {record_stats['duplicate_candidates']}")
        if record_stats.get("type_mismatches"):
            summary_parts.append(f"non-standard types: {record_stats['type_mismatches']}")

        summary_suffix = f" ({'; '.join(summary_parts)})" if summary_parts else ""
        st.info(
            f"Prepared {record_stats['included_rows']} of {record_stats['total_rows']} papers for NER{summary_suffix}."
        )
        if record_stats.get("removed_rows"):
            st.warning(
                f"Rows filtered out: {record_stats['removed_rows']} (missing abstracts removed: {record_stats['removed_missing_abstract']},"
                f" short abstracts removed: {record_stats['removed_by_length']}, type filters removed: {record_stats['removed_by_type']},"
                f" duplicates removed: {record_stats['duplicates_removed']})."
            )

        if not records:
            st.warning("No rows found in the uploaded spreadsheet.")
            display_footer()
            return

        progress_bar = st.progress(0, text="Initializing NER pipeline...")
        start_time = time.perf_counter()
        (
            processed_df,
            entity_df,
            resolved_model,
            chunk_count,
            abbreviation_registry,
            paper_text_map,
        ) = process_abstracts(
            records,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device_preference=device_preference,
            progress=progress_bar,
        )
        elapsed = time.perf_counter() - start_time
        progress_bar.empty()

        if not resolved_model:
            st.error(
                "❌ NER model could not be loaded. Please connect to the internet or set your Hugging Face token."
            )
            st.warning("⚠️ Entity extraction unavailable (no model loaded).")
            st.session_state.pop("processed_bundle", None)
            display_footer()
            return

        linking_context: Dict[str, object] = {}
        linking_progress: Optional["DeltaGenerator"] = None

        fallback_used = resolved_model and resolved_model != model_name
        if fallback_used and entity_df.empty:
            st.warning("No entities detected — fallback model may not support scientific domain terms.")

        user_entities_active = get_user_entities_state()
        user_canonical_active = get_user_canonical_state()
        entity_df, user_entities_map = inject_user_entities(entity_df, paper_text_map, user_entities_active)
        processed_df = augment_processed_with_user_entities(processed_df, user_entities_map)
        entity_df = sanitize_entity_dataframe(entity_df)
        entity_df = trim_and_filter_entities(entity_df)
        processed_df = sanitize_processed_dataframe(
            processed_df,
            allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
            drop_duplicates=ENABLE_DEDUPLICATION,
        )

        def _progress_callback(value: float) -> None:
            if linking_progress is not None:
                linking_progress.progress(min(max(value, 0.0), 1.0))

        if run_linking and not entity_df.empty:
            linking_progress = st.progress(0, text="Linking related entities...")
            semantic_active = use_semantic_linking and semantic_available
            if semantic_active:
                linking_callable = link_entities_semantic
            else:
                linking_callable = link_entities_fast
                if use_semantic_linking and not semantic_available:
                    st.info("SentenceTransformer model unavailable; using TF-IDF entity linking.")
            try:
                entity_df, linked_count, linking_context = silent_run(
                    linking_callable,
                    entity_df,
                    paper_text_map,
                    abbreviation_registry,
                    progress_callback=_progress_callback,
                    user_canonical=user_canonical_active,
                )
            except ImportError as err:
                if use_semantic_linking:
                    st.warning(
                        "Semantic linking requires the 'sentence-transformers' package. "
                        "Falling back to TF-IDF mode."
                    )
                    linking_progress = st.progress(0, text="Linking related entities (TF-IDF fallback)...")
                    entity_df, linked_count, linking_context = silent_run(
                        link_entities_fast,
                        entity_df,
                        paper_text_map,
                        abbreviation_registry,
                        progress_callback=_progress_callback,
                        user_canonical=user_canonical_active,
                    )
                else:
                    st.error("Entity linking dependencies are missing. Showing unlinked entities.")
                    st.exception(err)
                    linked_count = 0
                    if "norm" not in entity_df.columns:
                        entity_df["norm"] = entity_df["entity"].apply(normalize_entity)
                    entity_df["canonical"] = entity_df["norm"]
                    entity_df["Canonical"] = entity_df["canonical"].apply(lambda val: str(val).strip().upper())
                    linking_context = {
                        "strategy": "error",
                        "canonical_map": {},
                        "alias_groups": {},
                        "error": str(err),
                    }
                    linking_context = sanitize_linking_context(linking_context)
                    entity_df = apply_user_canonical_overrides(entity_df, user_canonical_active)
            except Exception as err:
                st.error("Entity linking failed. Showing unlinked entities instead.")
                st.exception(err)
                linked_count = 0
                if "norm" not in entity_df.columns:
                    entity_df["norm"] = entity_df["entity"].apply(normalize_entity)
                entity_df["canonical"] = entity_df["norm"]
                entity_df["Canonical"] = entity_df["canonical"].apply(lambda val: str(val).strip().upper())
                linking_context = {
                    "strategy": "error",
                    "canonical_map": {},
                    "alias_groups": {},
                    "error": str(err),
                }
                linking_context = sanitize_linking_context(linking_context)
                entity_df = apply_user_canonical_overrides(entity_df, user_canonical_active)
            finally:
                if linking_progress is not None:
                    linking_progress.empty()
            merge_map_active = get_user_canonical_merges_state()
            entity_df = apply_canonical_merge_map(entity_df, merge_map_active)
            processed_df = apply_canonical_merges_to_processed_df(processed_df, merge_map_active)
            entity_df = sanitize_entity_dataframe(entity_df)
            
            entity_df = trim_and_filter_entities(entity_df)
            processed_df = sanitize_processed_dataframe(
                processed_df,
                allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
                drop_duplicates=ENABLE_DEDUPLICATION,
            )

            strategy_label = linking_context.get("strategy") if isinstance(linking_context, dict) else None
            if strategy_label == "semantic":
                st.success("✅ Entity linking completed in high-accuracy mode.")
                summary_total = len(entity_df)
                summary_unique = (
                    entity_df["Canonical"].nunique(dropna=True)
                    if "Canonical" in entity_df.columns
                    else entity_df.get("canonical", entity_df.get("norm", pd.Series(dtype=object))).nunique(dropna=True)
                )
                paper_total = (
                    entity_df["paper_id"].nunique(dropna=True)
                    if "paper_id" in entity_df.columns
                    else 0
                )
                st.info(f"Processed {summary_total} entities across {paper_total} papers.")
                st.caption(f"{summary_total} entities processed · {summary_unique} canonical terms · {linked_count} variants merged.")
            elif strategy_label == "tfidf":
                st.success("✅ Entity linking completed in high-accuracy mode.")
                if use_semantic_linking and semantic_available:
                    st.info("Using High-Accuracy Entity Linking (TF-IDF fallback)")
                elif not semantic_available:
                    st.info("Using High-Accuracy Entity Linking (TF-IDF mode)")
                summary_total = len(entity_df)
                summary_unique = (
                    entity_df["Canonical"].nunique(dropna=True)
                    if "Canonical" in entity_df.columns
                    else entity_df.get("canonical", entity_df.get("norm", pd.Series(dtype=object))).nunique(dropna=True)
                )
                paper_total = (
                    entity_df["paper_id"].nunique(dropna=True)
                    if "paper_id" in entity_df.columns
                    else 0
                )
                st.info(f"Processed {summary_total} entities across {paper_total} papers.")
                st.caption(f"{summary_total} entities processed · {summary_unique} canonical terms · {linked_count} variants merged.")
        else:
            if not entity_df.empty:
                if "norm" not in entity_df.columns:
                    entity_df["norm"] = entity_df["entity"].apply(normalize_entity)
                entity_df["canonical"] = entity_df["norm"]
                entity_df["Canonical"] = entity_df["canonical"].apply(lambda val: str(val).strip().upper())
            linked_count = 0
            linking_context = {
                "strategy": "skipped",
                "canonical_map": {},
                "alias_groups": {},
            }
            linking_context = sanitize_linking_context(linking_context)
            entity_df = apply_user_canonical_overrides(entity_df, user_canonical_active)
            merge_map_active = get_user_canonical_merges_state()
            entity_df = apply_canonical_merge_map(entity_df, merge_map_active)
            processed_df = apply_canonical_merges_to_processed_df(processed_df, merge_map_active)
            entity_df = sanitize_entity_dataframe(entity_df)
            processed_df = sanitize_processed_dataframe(
                processed_df,
                allow_empty_abstracts=ALLOW_EMPTY_ABSTRACTS,
                drop_duplicates=ENABLE_DEDUPLICATION,
            )

        resolved_to_store = resolved_model or "Unavailable"
        pvu_df = build_property_value_unit_table(entity_df)
        st.session_state["pvu_df"] = pvu_df
        cache_results(
            cache_key,
            processed_df,
            entity_df,
            resolved_to_store,
            elapsed,
            chunk_count,
            linked_count,
            linking_context,
        )
        st.success(
            f"Completed NER for {len(records)} abstracts in {elapsed:.2f} seconds "
            f"using {resolved_to_store}."
        )
        if entity_df.empty:
            st.info(
                "No entities detected. Try lowering the confidence threshold or switching to a domain-specific model."
            )
        if chunk_count > 0:
            st.sidebar.info(
                f"Chunking long abstracts into {chunk_count} sub-parts (≤512 tokens each)."
            )
        if linked_count > 0:
            st.sidebar.info(f"Linked {linked_count} variant entities into canonical forms.")
    else:
        chunk_count = st.session_state.get("chunked_count", chunk_count_cached)
        linked_count = st.session_state.get("linked_count", linked_count_cached)

    bundle = st.session_state.get("processed_bundle")
    if not bundle:
        st.warning("Click 'Run NER extraction' to process the uploaded file.")
        display_footer()
        return

    if "pvu_df" in bundle:
        st.session_state["pvu_df"] = bundle["pvu_df"]

    chunk_count = bundle.get("chunk_count", chunk_count)
    linked_count = bundle.get("linked_count", linked_count)
    if chunk_count and not run_button:
        st.sidebar.info(
            f"Chunking long abstracts into {chunk_count} sub-parts (≤512 tokens each)."
        )
    linking_context_bundle = bundle.get("linking_context", {})
    st.session_state["linking_context"] = linking_context_bundle
    if not run_button and run_linking:
        strategy_label_cached = linking_context_bundle.get("strategy")
        if strategy_label_cached == "semantic":
            st.success("✅ Entity linking completed in high-accuracy mode.")
            cached_entities = bundle.get("entity_df", pd.DataFrame())
            if not cached_entities.empty:
                summary_total = len(cached_entities)
                summary_unique = (
                    cached_entities["Canonical"].nunique(dropna=True)
                    if "Canonical" in cached_entities.columns
                    else cached_entities.get("canonical", cached_entities.get("norm", pd.Series(dtype=object))).nunique(dropna=True)
                )
                cached_linked = bundle.get("linked_count", 0)
                paper_total = (
                    cached_entities["paper_id"].nunique(dropna=True)
                    if "paper_id" in cached_entities.columns
                    else 0
                )
                st.info(f"Processed {summary_total} entities across {paper_total} papers.")
                st.caption(f"{summary_total} entities processed · {summary_unique} canonical terms · {cached_linked} variants merged.")
        elif strategy_label_cached == "tfidf":
            st.success("✅ Entity linking completed in high-accuracy mode.")
            if st.session_state.get("use_semantic_linking") and semantic_available:
                st.info("Using High-Accuracy Entity Linking (TF-IDF fallback)")
            elif not semantic_available:
                st.info("Using High-Accuracy Entity Linking (TF-IDF mode)")
            cached_entities = bundle.get("entity_df", pd.DataFrame())
            if not cached_entities.empty:
                summary_total = len(cached_entities)
                summary_unique = (
                    cached_entities["Canonical"].nunique(dropna=True)
                    if "Canonical" in cached_entities.columns
                    else cached_entities.get("canonical", cached_entities.get("norm", pd.Series(dtype=object))).nunique(dropna=True)
                )
                cached_linked = bundle.get("linked_count", 0)
                paper_total = (
                    cached_entities["paper_id"].nunique(dropna=True)
                    if "paper_id" in cached_entities.columns
                    else 0
                )
                st.info(f"Processed {summary_total} entities across {paper_total} papers.")
                st.caption(f"{summary_total} entities processed · {summary_unique} canonical terms · {cached_linked} variants merged.")
    if linked_count and not run_button and linking_context_bundle.get("strategy") not in {"skipped", "error"}:
        st.sidebar.info(f"Linked {linked_count} variant entities into canonical forms.")

    if st.session_state.get("last_elapsed"):
        st.caption(
            f"Last run completed in {st.session_state['last_elapsed']:.2f} seconds."
        )

    resolved_model = bundle.get("resolved_model", model_name)
    display_results(
        sanitize_entities_output(bundle["processed_df"].copy()),
        sanitize_entities_output(bundle["entity_df"].copy()),
        resolved_model,
        confidence_threshold,
        False,
        linking_context_bundle,
    )
    display_footer()


if __name__ == "__main__":
    main()
