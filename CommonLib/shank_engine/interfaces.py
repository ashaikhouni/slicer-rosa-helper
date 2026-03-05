"""Interfaces for pluggable SEEG detection pipelines and components."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import BlobRecord, ContactRecord, DetectionContext, DetectionResult, ShankModel


class DetectionPipeline(Protocol):
    """Top-level pipeline contract shared by all detection implementations."""

    pipeline_id: str
    pipeline_version: str

    def run(self, ctx: DetectionContext) -> DetectionResult:
        """Run detection and return canonical result payload."""


class GatingComponent(Protocol):
    """Computes masks/depth maps used by downstream stages."""

    def compute(self, ctx: DetectionContext, state: dict[str, Any]) -> dict[str, Any]:
        """Return mask/depth payload."""


class BlobExtractor(Protocol):
    """Extracts blob observations from gating outputs."""

    def extract(self, ctx: DetectionContext, state: dict[str, Any]) -> list[BlobRecord]:
        """Return blob observations."""


class BlobScorer(Protocol):
    """Assigns soft evidence scores to blobs."""

    def score(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[BlobRecord]:
        """Return scored blobs."""


class Initializer(Protocol):
    """Proposes initial shank hypotheses."""

    def initialize(self, ctx: DetectionContext, blobs: list[BlobRecord], state: dict[str, Any]) -> list[ShankModel]:
        """Return initial shank models."""


class ShankRefiner(Protocol):
    """Refines assignments and model parameters (RANSAC/EM/etc.)."""

    def refine(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> tuple[list[ShankModel], dict[str, Any]]:
        """Return refined models and refinement diagnostics payload."""


class ModelSelector(Protocol):
    """Prunes/merges/splits models to produce final shanks."""

    def select(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> list[ShankModel]:
        """Return final selected models."""


class ContactDetector(Protocol):
    """Produces contact centers from selected models."""

    def detect(
        self,
        ctx: DetectionContext,
        blobs: list[BlobRecord],
        models: list[ShankModel],
        state: dict[str, Any],
    ) -> list[ContactRecord]:
        """Return contact detections."""


class ArtifactWriter(Protocol):
    """Output hook for persisting diagnostics/artifacts."""

    def write_json(self, name: str, payload: Any) -> str:
        """Write JSON payload and return absolute output path."""

    def write_text(self, name: str, text: str) -> str:
        """Write text payload and return absolute output path."""

    def write_bytes(self, name: str, data: bytes) -> str:
        """Write raw bytes and return absolute output path."""

    def write_csv_rows(self, name: str, header: list[str], rows: list[list[Any]]) -> str:
        """Write CSV rows and return absolute output path."""
