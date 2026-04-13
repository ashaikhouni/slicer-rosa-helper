"""Phase B: trajectory reconstruction (WIP stub).

This file is a placeholder between commits. The previous template-matching
implementation was removed along with its config fields; the redesigned
implementation (axis refinement, colinear reabsorption, metal-mask
extension, lateral-HU bone↔brain interface, rejection gates) is pending
and tracked in ``docs/PHASE_B_REDESIGN.md``.

The stub passes proposals through unchanged so the rest of the pipeline
and Slicer UI keep working while the real implementation is written. The
regression-test baselines are expected to drift while this stub is live
and will be re-locked once the real implementation lands.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def filter_models_by_family(
    library: dict[str, Any],
    families: tuple[str, ...],
    min_contacts: int = 0,
) -> list[dict[str, Any]]:
    """Return library models whose id starts with one of the family
    prefixes (case-insensitive). The ``min_contacts`` parameter is kept
    for call-site compatibility; enforcement will move into the new
    implementation.
    """
    if not families:
        models = list(library.get("models") or [])
    else:
        fams = tuple(f.upper().rstrip("-") + "-" for f in families)
        models = [
            m for m in (library.get("models") or [])
            if any(str(m.get("id", "")).upper().startswith(f) for f in fams)
        ]
    if int(min_contacts) > 0:
        models = [m for m in models if int(m.get("contact_count", 0)) >= int(min_contacts)]
    return models


def run_model_fit_group(
    proposals: list[dict[str, Any]],
    arr_kji: np.ndarray,
    ras_to_ijk_fn,
    head_distance_map_kji: np.ndarray,
    library_models: list[dict[str, Any]],
    cfg,
    *,
    metal_mask_kji: np.ndarray | None = None,
    hull_mask_kji: np.ndarray | None = None,
    support_atoms: list[dict[str, Any]] | None = None,
    blob_sample_points_ras: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pass proposals through unchanged (redesign pending).

    Keyword-only parameters ``metal_mask_kji``, ``hull_mask_kji``,
    ``support_atoms``, and ``blob_sample_points_ras`` are placeholders for
    the redesigned implementation — they will be consumed by the axis
    refinement, reabsorption, and extension steps in the new
    ``deep_core_axis_reconstruction`` helper module.
    """
    accepted_props: list[dict[str, Any]] = []
    for prop in proposals or []:
        new_prop = dict(prop)
        new_prop["model_fit_passed"] = True
        new_prop["model_fit_stub"] = True
        accepted_props.append(new_prop)

    return {
        "accepted_proposals": accepted_props,
        "stats": {
            "input_count": int(len(proposals or [])),
            "accepted_count": int(len(accepted_props)),
            "rejected_unassigned": 0,
            "stub": True,
        },
    }
