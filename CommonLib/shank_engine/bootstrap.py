"""Bootstrap helpers for built-in shank detection registry entries."""

from __future__ import annotations

from .registry import PipelineRegistry


def register_builtin_pipelines(registry: PipelineRegistry) -> None:
    """Register built-in pipelines in deterministic order.

    Imports are lazy so shank_engine contracts/tests can load without pulling
    optional heavy dependencies until pipeline registration is requested.
    """

    from .pipelines.blob_ransac_v1 import BlobRansacV1Pipeline
    from .pipelines.blob_em_v2 import BlobEMV2Pipeline
    from .pipelines.contact_pitch_v1 import ContactPitchV1Pipeline
    from .pipelines.deep_core_v1 import DeepCoreV1Pipeline
    from .pipelines.deep_core_v2 import DeepCoreV2Pipeline

    registry.register_pipeline("blob_ransac_v1", BlobRansacV1Pipeline, overwrite=True)
    registry.register_pipeline("blob_em_v2", BlobEMV2Pipeline, overwrite=True)
    registry.register_pipeline("deep_core_v1", DeepCoreV1Pipeline, overwrite=True)
    registry.register_pipeline("deep_core_v2", DeepCoreV2Pipeline, overwrite=True)
    registry.register_pipeline("contact_pitch_v1", ContactPitchV1Pipeline, overwrite=True)
