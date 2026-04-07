"""Adapter pipeline for the existing shank_core voxel/ransac detector."""

from __future__ import annotations

import time

from ..adapters import run_blob_ransac_v1
from ..artifacts import add_artifact, write_standard_artifacts
from ..contracts import DetectionContext, DetectionResult
from .base import BaseDetectionPipeline


class BlobRansacV1Pipeline(BaseDetectionPipeline):
    """Bridge wrapper around the current production shank_core voxel pipeline."""

    pipeline_id = "blob_ransac_v1"
    display_name = "Voxel RANSAC v1"
    pipeline_version = "1.0.0"

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)
        config = self._config(ctx)

        try:
            adapter_out = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="legacy_adapter",
                fn=lambda: run_blob_ransac_v1(ctx, config),
            )
            # Preserve the full legacy payload for Slicer mapping code that still
            # consumes mask arrays and detailed line diagnostics.
            ctx_extras = ctx.get("extras")
            if isinstance(ctx_extras, dict):
                ctx_extras["legacy_result"] = adapter_out.get("raw")

            trajectories_raw = adapter_out.get("trajectories")
            contacts_raw = adapter_out.get("contacts")
            result["trajectories"] = list(trajectories_raw) if trajectories_raw is not None else []
            result["contacts"] = list(contacts_raw) if contacts_raw is not None else []

            diagnostics_raw = adapter_out.get("diagnostics")
            diagnostics_map = diagnostics_raw if isinstance(diagnostics_raw, dict) else {}
            for key, value in diagnostics_map.items():
                if key.endswith("_ms"):
                    diagnostics.set_timing(key, float(value))
                else:
                    diagnostics.set_count(key, int(value))

            diagnostics.set_extra("candidate_mode", "voxel")
            diagnostics.note("using legacy shank_core adapter")

            writer = self.get_artifact_writer(ctx, result)
            blobs_raw = adapter_out.get("blobs")
            blobs = list(blobs_raw) if blobs_raw is not None else []
            artifacts = write_standard_artifacts(
                writer,
                result,
                blobs=blobs,
                pipeline_payload={
                    "pipeline_id": self.pipeline_id,
                    "pipeline_version": self.pipeline_version,
                    "raw_summary": diagnostics_map,
                },
            )
            result["artifacts"].extend(artifacts)

            if bool(config.get("write_raw_diagnostics", False)):
                raw_path = writer.write_json("legacy_blob_ransac_raw.json", adapter_out.get("raw", {}))
                add_artifact(
                    result["artifacts"],
                    kind="legacy_raw_json",
                    path=raw_path,
                    description="Raw payload returned by shank_core",
                    stage="legacy_adapter",
                )

            result["meta"].setdefault("extras", {})
            result["meta"]["extras"]["legacy_adapter"] = {
                "line_count": len(result["trajectories"]),
                "blob_count": len(blobs),
            }
        except Exception as exc:
            self.fail(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage=str(getattr(exc, "stage", "pipeline")),
                exc=exc,
            )

        return self.finalize(result, diagnostics, t_start)
