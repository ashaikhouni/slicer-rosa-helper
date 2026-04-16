"""Deep Core v2: bolt-first SEEG detection pipeline.

Stages:
  1. ``mask`` — reuse ``_run_mask`` from v1.
  2. ``bolt_detection`` — reuse ``_run_bolt_detection`` from v1.
  3. ``bolt_fit`` — new. For each bolt, call into
     ``deep_core_v2_fit.fit_bolt_trajectory`` which either:
       * walks the axis inward through a loose-threshold metal mask
         (``model_fit.v2_fit_mode = "two_threshold"``, default), or
       * builds an intensity profile in a 5 mm disc perpendicular to
         the axis and fits library electrode contact patterns to the
         detected peaks (``model_fit.v2_fit_mode = "intensity_peaks"``).

v2 drops Phase A proposal generation, the support stage's atom work,
and the annulus rejection step. Every trajectory is anchored to a
RANSAC bolt with a precise axis. v1 remains the default pipeline.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..contracts import DetectionContext, DetectionResult
from .deep_core_v1 import DeepCoreV1Pipeline, _parse_deep_core_config


class DeepCoreV2Pipeline(DeepCoreV1Pipeline):
    """Bolt-first detection pipeline. See module docstring."""

    pipeline_id = "deep_core_v2"
    pipeline_version = "2.0.0"

    @staticmethod
    def _dedup_v2_trajectories(
        trajectories, *, angle_deg=15.0, perp_mm=8.0,
    ):
        """Remove near-collinear duplicate trajectories, keeping the longer one."""
        kept = []
        for t in trajectories:
            s = np.asarray(t["start_ras"], dtype=float)
            e = np.asarray(t["end_ras"], dtype=float)
            d = e - s
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                continue
            axis = d / length
            is_dup = False
            for ki, k in enumerate(kept):
                ks = np.asarray(k["start_ras"], dtype=float)
                ke = np.asarray(k["end_ras"], dtype=float)
                kd = ke - ks
                klen = float(np.linalg.norm(kd))
                if klen < 1e-6:
                    continue
                kaxis = kd / klen
                cos = abs(float(np.dot(axis, kaxis)))
                ang = float(np.degrees(np.arccos(min(1.0, cos))))
                if ang > angle_deg:
                    continue
                mid = 0.5 * (s + e)
                v = mid - ks
                along = float(np.dot(v, kaxis))
                perp = v - along * kaxis
                pd = float(np.linalg.norm(perp))
                # Reject if candidate midpoint projects outside the kept
                # segment (with margin).  Prevents bilateral electrodes
                # on opposite sides of the head from being merged.
                margin = 0.5 * klen
                if along < -margin or along > klen + margin:
                    continue
                if pd <= perp_mm:
                    is_dup = True
                    if length > klen:
                        kept[ki] = t
                    break
            if not is_dup:
                kept.append(t)
        return kept

    def _run_bolt_fit(self, ctx, mask, bolt_output, cfg) -> dict[str, Any]:
        from postop_ct_localization import deep_core_v2_fit as v2fit
        from postop_ct_localization.deep_core_model_fit import filter_models_by_family

        mfg = cfg.model_fit
        library = self._get_electrode_library()
        models = filter_models_by_family(library, tuple(mfg.families))

        arr_kji = np.asarray(ctx["arr_kji"], dtype=float)
        ras_to_ijk_fn = ctx["ras_to_ijk_fn"]
        head_distance_map_kji = mask["head_distance_map_kji"]

        result = v2fit.run_bolt_fit_group(
            bolts=list((bolt_output or {}).get("candidates") or []),
            arr_kji=arr_kji,
            head_distance_map_kji=head_distance_map_kji,
            ras_to_ijk_fn=ras_to_ijk_fn,
            ijk_kji_to_ras_fn=ctx.get("ijk_kji_to_ras_fn"),
            library_models=models,
            cfg=mfg,
        )
        return {
            "proposals": list(result["accepted_proposals"]),
            "model_fit_stats": dict(result.get("stats") or {}),
        }

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diag = self.diagnostics(result)
        cfg = _parse_deep_core_config(self._config(ctx))
        cfg = cfg.with_updates({"bolt.enabled": True})

        try:
            mask = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="mask",
                fn=lambda: self._run_mask(ctx, cfg),
            )

            bolt = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="bolt_detection",
                fn=lambda: self._run_bolt_detection(ctx, mask, cfg),
            )

            proposals: dict[str, Any] = {"proposals": []}
            if cfg.model_fit.enabled:
                proposals = self.run_stage(
                    ctx=ctx, result=result, diagnostics=diag,
                    stage_name="bolt_fit",
                    fn=lambda: self._run_bolt_fit(ctx, mask, bolt, cfg),
                )

            min_span = float(getattr(cfg.model_fit, "min_intracranial_span_mm", 15.0))
            all_trajs = self._proposals_to_trajectories(proposals)
            span_filtered = [
                t for t in all_trajs
                if float(t.get("intracranial_span_mm", t.get("span_mm", 0))) >= min_span
            ]
            result["trajectories"] = self._dedup_v2_trajectories(span_filtered)

            diag.set_count("proposal_count", len(result["trajectories"]))
            diag.set_count("atom_count", 0)
            diag.set_count(
                "bolt_candidate_count",
                int(len((bolt or {}).get("candidates") or [])),
            )

            self._last_mask_output = mask
            self._last_bolt_output = bolt
            self._last_proposal_payload = proposals

        except Exception as exc:
            self.fail(
                ctx=ctx, result=result, diagnostics=diag,
                stage="unknown", exc=exc,
            )
            if not hasattr(self, "_last_proposal_payload"):
                self._last_proposal_payload = {"proposals": []}

        return self.finalize(result, diag, t_start)

    def run_debug(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diag = self.diagnostics(result)
        cfg = _parse_deep_core_config(self._config(ctx))
        cfg = cfg.with_updates({"bolt.enabled": True})

        try:
            mask = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="mask",
                fn=lambda: self._run_mask(ctx, cfg),
            )
            bolt = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="bolt_detection",
                fn=lambda: self._run_bolt_detection(ctx, mask, cfg),
            )
            self._last_mask_output = mask
            self._last_bolt_output = bolt
            diag.set_count("atom_count", 0)
            diag.set_count(
                "bolt_candidate_count",
                int(len((bolt or {}).get("candidates") or [])),
            )
        except Exception as exc:
            self.fail(
                ctx=ctx, result=result, diagnostics=diag,
                stage="unknown", exc=exc,
            )

        return self.finalize(result, diag, t_start)
