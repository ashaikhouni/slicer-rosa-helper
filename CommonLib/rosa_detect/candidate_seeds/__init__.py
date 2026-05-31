"""``rosa_detect.candidate_seeds`` — candidate-seed generation.

Per the user's 2026-05-09 unification (handoff_v3_production_lift_2026-05-09.md):
"Trajectory detection collapses to candidate seed generation — emit candidate
seeds, no scoring. All confidence comes from `place_seed` running per
candidate."

This package owns that responsibility. ``generate_candidate_seeds`` runs the
v1 stage1 walker + bolt anchoring (orchestrated by
``orchestrator.run_two_stage_detection``) and returns a list of ``Seed``
objects ready to feed into the snap-flow placement engine
(``rosa_core.placement_modes.place_seeg`` → ``snap_fit_to_ctxs``).

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
    # Lazy import — avoids pulling the orchestrator (and its numpy /
    # scipy / SimpleITK deps) into callers that only need the
    # package's docstring or constants.
    from rosa_detect.candidate_seeds.orchestrator import run_two_stage_detection

    img = features["img"]
    i2r = np.asarray(features["ijk_to_ras_mat"])
    r2i = np.asarray(features["ras_to_ijk_mat"])

    trajectories, _features = run_two_stage_detection(
        img, i2r, r2i,
        return_features=True,
        progress_logger=progress_logger,
        pitch_strategy=pitch_strategy,
        # Reuse the caller's already-computed shared-selector intracranial
        # (gfe.compute_features) so the full detect→place pipeline builds the
        # mask once — no redundant SynthStrip pass. Detection ignores it
        # (head-distance/hull drive emission); it rides along for parity.
        brain_mask=features.get("intracranial"),
    )

    # ``seeder_model`` is left None — the detection stage no longer
    # classifies an electrode model (canonical picker is the shared
    # ``rosa_core.model_pick.pick_electrode_model``, reached via
    # ``stage_d_pick.pick_model`` during placement).
    seeds: list[Seed] = []
    for idx, t in enumerate(trajectories or []):
        seeds.append(Seed(
            name=f"CAND-{idx + 1:03d}",
            start_ras=np.asarray(t["start_ras"], dtype=float),
            end_ras=np.asarray(t["end_ras"], dtype=float),
            model_id=None,
            seeder_label=str(t.get("confidence_label") or ""),
            seeder_confidence=float(t.get("confidence") or 0.0),
            seeder_model=None,
        ))
    return seeds


__all__ = ["generate_candidate_seeds"]
