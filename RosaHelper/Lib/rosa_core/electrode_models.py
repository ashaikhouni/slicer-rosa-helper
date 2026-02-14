"""Electrode model library loading and light validation."""

import json
from pathlib import Path


_REQUIRED_MODEL_KEYS = {
    "id",
    "type",
    "contact_count",
    "contact_length_mm",
    "diameter_mm",
    "total_exploration_length_mm",
    "contact_center_offsets_from_tip_mm",
}


def default_electrode_library_path():
    """Return default bundled DIXI D08 electrode library path."""
    here = Path(__file__).resolve()
    return here.parents[2] / "Resources" / "electrodes" / "dixi_d08_electrodes.json"


def load_electrode_library(path=None):
    """Load electrode model library JSON and return validated dictionary."""
    lib_path = Path(path) if path else default_electrode_library_path()
    data = json.loads(lib_path.read_text(encoding="utf-8"))
    validate_electrode_library(data)
    return data


def validate_electrode_library(data):
    """Validate required top-level and per-model fields.

    Raises
    ------
    ValueError
        If schema is missing required keys or contains inconsistent lengths.
    """
    if not isinstance(data, dict):
        raise ValueError("Electrode library must be a JSON object")
    if "models" not in data or not isinstance(data["models"], list):
        raise ValueError("Electrode library must include a 'models' list")

    seen_ids = set()
    for model in data["models"]:
        missing = sorted(k for k in _REQUIRED_MODEL_KEYS if k not in model)
        if missing:
            raise ValueError(f"Model missing required fields: {missing}")

        model_id = model["id"]
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model id: {model_id}")
        seen_ids.add(model_id)

        offsets = model["contact_center_offsets_from_tip_mm"]
        if len(offsets) != int(model["contact_count"]):
            raise ValueError(
                f"Model {model_id} has contact_count={model['contact_count']} "
                f"but {len(offsets)} center offsets"
            )

        if any(offsets[i] >= offsets[i + 1] for i in range(len(offsets) - 1)):
            raise ValueError(f"Model {model_id} center offsets must be strictly increasing")


def model_map(data):
    """Return dictionary mapping model id to model definition."""
    return {m["id"]: m for m in data["models"]}
