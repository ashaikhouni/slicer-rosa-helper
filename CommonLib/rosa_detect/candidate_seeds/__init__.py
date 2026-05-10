"""``rosa_detect.candidate_seeds`` — candidate-seed generation.

Per the user's 2026-05-09 unification (handoff_v3_production_lift_2026-05-09.md):
"Trajectory detection collapses to candidate seed generation — emit candidate
seeds, no scoring. All confidence comes from `place_seed` running per
candidate."

This package owns that responsibility. ``generate_candidate_seeds`` runs the
existing v1 stage1 walker + bolt anchoring (still living in
``contact_pitch_v1_fit.py`` for now; will be moved here tier-by-tier in
Sessions 3-4) and returns a list of ``Seed`` objects ready to feed into
``rosa_core.contact_placement.run_two_pass``.

Importantly, the seeds carry forward v1's seeder_label — that's what the
compound score's ``s_seeder`` term reads — so notebook parity is preserved.
The deprecation of ``confidence_label`` in the public API happens at the
``DetectedTrajectory`` level (Session 4), not here.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rosa_core.placement_modes import Seed


def generate_candidate_seeds(
    features: dict[str, Any],
    *,
    bolts: list[dict[str, Any]] | None = None,
    pitch_strategy: str | None = None,
    suggestion_vendors: Sequence[str] | None = None,
    progress_logger=None,
) -> list[Seed]:
    """Run v1 stage1 + bolt anchoring; return candidate seeds.

    The seed names are auto-assigned (``"CAND-001"``, ``"CAND-002"``, ...) —
    callers that need named outputs (mode 3) match by geometry/model after
    placement scoring.

    Args:
        features: feature dict from
            ``rosa_detect.guided_fit_engine.compute_features``.
        bolts: optional precomputed bolt CCs. Currently unused — v1's stage1
            re-extracts internally; passing them here is a forward-compat
            placeholder for when stage1 accepts them as input (Session 3+).
        pitch_strategy: ``"dixi"`` / ``"pmt_35"`` / ``"auto"`` / ``None``.
        suggestion_vendors: optional vendor allowlist.
        progress_logger: callable forwarded to v1's stage1 progress reporter.

    Returns:
        list[Seed] — one per emitted v1 trajectory, in v1's emission order.
        Endpoints are ``start_ras`` (skull-shallow) → ``end_ras`` (deep tip).
        ``model_id`` is left None so the placer's library matcher picks it.
        ``seeder_label`` and ``seeder_confidence`` carry v1's score.

    Notes:
        * No bidirectional retry — v1's stage1 emits seeds in canonical
          shallow→deep order. The placer's stage A respects that.
        * The function does NOT compute features or extract bolts — those
          inputs come from the caller (typically via
          ``rosa_core.volume_loader.load_features_and_bolts``).
    """
    # Lazy import — avoids pulling cpfit (and its numpy/scipy deps) into
    # callers that only need the package's docstring or constants.
    from rosa_detect import contact_pitch_v1_fit as cpfit

    img = features["img"]
    i2r = np.asarray(features["ijk_to_ras_mat"])
    r2i = np.asarray(features["ras_to_ijk_mat"])

    trajectories, _features = cpfit.run_two_stage_detection(
        img, i2r, r2i,
        return_features=True,
        progress_logger=progress_logger,
        suggestion_vendors=suggestion_vendors,
        pitch_strategy=pitch_strategy,
    )

    seeds: list[Seed] = []
    for idx, t in enumerate(trajectories or []):
        seeds.append(Seed(
            name=f"CAND-{idx + 1:03d}",
            start_ras=np.asarray(t["start_ras"], dtype=float),
            end_ras=np.asarray(t["end_ras"], dtype=float),
            model_id=None,
            seeder_label=str(t.get("confidence_label") or ""),
            seeder_confidence=float(t.get("confidence") or 0.0),
            seeder_model=t.get("electrode_model"),
        ))
    return seeds


__all__ = ["generate_candidate_seeds"]
