"""Scaffold pipeline for blob-based EM detection."""

from __future__ import annotations

import time
from typing import Any

from ..artifacts import write_standard_artifacts
from ..contracts import BlobRecord, ContactRecord, DetectionContext, DetectionResult, ShankModel
from .base import BaseDetectionPipeline


class _NoOpGating:
    def compute(self, ctx: DetectionContext, state: dict[str, Any]) -> dict[str, Any]:
        return {}


class _NoOpBlobExtractor:
    def extract(self, ctx: DetectionContext, state: dict[str, Any]) -> list[BlobRecord]:
        return []


class _NoOpBlobScorer:
    def score(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[BlobRecord]:
        return blobs


class _NoOpInitializer:
    def initialize(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[ShankModel]:
        return []


class _NoOpRefiner:
    def refine(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> tuple[list[ShankModel], dict[str, Any]]:
        return models, {"iterations": 0}


class _NoOpSelector:
    def select(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> list[ShankModel]:
        return models


class _NoOpContactDetector:
    def detect(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> list[ContactRecord]:
        return []


class BlobEMV2Pipeline(BaseDetectionPipeline):
    """Second-generation blob detector scaffold with EM-style stage contract."""

    pipeline_id = "blob_em_v2"
    display_name = "Blob EM v2"
    scaffold = True
    pipeline_version = "0.1.0"

    default_components = {
        "gating": _NoOpGating(),
        "blob_extractor": _NoOpBlobExtractor(),
        "blob_scorer": _NoOpBlobScorer(),
        "initializer": _NoOpInitializer(),
        "shank_refiner": _NoOpRefiner(),
        "model_selector": _NoOpSelector(),
        "contact_detector": _NoOpContactDetector(),
    }

    def run(self, ctx: DetectionContext) -> DetectionResult:
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diagnostics = self.diagnostics(result)

        state: dict[str, Any] = {}
        blobs: list[BlobRecord] = []
        models: list[ShankModel] = []
        contacts: list[ContactRecord] = []

        try:
            gating = self.resolve_component(ctx, "gating")
            extractor = self.resolve_component(ctx, "blob_extractor")
            scorer = self.resolve_component(ctx, "blob_scorer")
            initializer = self.resolve_component(ctx, "initializer")
            refiner = self.resolve_component(ctx, "shank_refiner")
            selector = self.resolve_component(ctx, "model_selector")
            contact_detector = self.resolve_component(ctx, "contact_detector")

            state["masks"] = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="gating",
                fn=lambda: gating.compute(ctx, state),
            )
            blobs = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="blob_extraction",
                fn=lambda: extractor.extract(ctx, state),
            )
            diagnostics.set_count("blob_count_total", len(blobs))

            blobs = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="blob_scoring",
                fn=lambda: scorer.score(ctx, blobs, state),
            )
            diagnostics.set_count("blob_count_scored", len(blobs))

            models = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="seed_initialization",
                fn=lambda: initializer.initialize(ctx, blobs, state),
            )
            diagnostics.set_count("seed_count", len(models))

            models, refine_payload = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="em_refinement",
                fn=lambda: refiner.refine(ctx, blobs, models, state),
            )
            state["refine_payload"] = dict(refine_payload or {})
            diagnostics.set_extra("em_refinement", state["refine_payload"])

            models = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="model_selection",
                fn=lambda: selector.select(ctx, blobs, models, state),
            )
            diagnostics.set_count("final_shank_count", len(models))

            contacts = self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage_name="contact_detection",
                fn=lambda: contact_detector.detect(ctx, blobs, models, state),
            )
            diagnostics.set_count("final_contact_count", len(contacts))

            result["trajectories"] = [
                {
                    "name": m.shank_id,
                    "source": self.pipeline_id,
                    "model_kind": m.kind,
                    "params": m.params,
                    "support": m.support,
                    "assigned_blob_ids": list(m.assigned_blob_ids),
                }
                for m in models
            ]
            result["contacts"] = [
                {
                    "trajectory_name": c.shank_id,
                    "label": c.contact_id,
                    "position_ras": list(c.position_ras),
                    "confidence": float(c.confidence),
                }
                for c in contacts
            ]

            result["warnings"].append("blob_em_v2 scaffold mode: algorithm body not implemented yet")
            result["meta"].setdefault("extras", {})
            result["meta"]["extras"]["scaffold_mode"] = True

            writer = self.get_artifact_writer(ctx, result)
            blob_rows = [
                {
                    "blob_id": b.blob_id,
                    "x": b.centroid_ras[0],
                    "y": b.centroid_ras[1],
                    "z": b.centroid_ras[2],
                    "voxels": b.voxel_count,
                    "peak_hu": b.peak_hu,
                }
                for b in blobs
            ]
            artifacts = write_standard_artifacts(
                writer,
                result,
                blobs=blob_rows,
                pipeline_payload={
                    "pipeline_id": self.pipeline_id,
                    "pipeline_version": self.pipeline_version,
                    "state_keys": sorted(state.keys()),
                },
            )
            result["artifacts"].extend(artifacts)
        except Exception as exc:
            self.fail(
                ctx=ctx,
                result=result,
                diagnostics=diagnostics,
                stage=str(getattr(exc, "stage", "pipeline")),
                exc=exc,
            )

        return self.finalize(result, diagnostics, t_start)
