"""Bootstrap helpers for built-in shank detection registry entries."""

from __future__ import annotations

from .registry import PipelineRegistry


def register_builtin_pipelines(registry: PipelineRegistry) -> None:
    """Register built-in pipelines in deterministic order.

    Imports are lazy so shank_engine contracts/tests can load without pulling
    optional heavy dependencies until pipeline registration is requested.
    """

    from .pipelines.blob_em_v2 import BlobEMV2Pipeline
    from .pipelines.blob_consensus_v1 import BlobConsensusV1Pipeline
    from .pipelines.hybrid_bead_string_v1 import HybridBeadStringV1Pipeline
    from .pipelines.blob_persistence_v1 import BlobPersistenceV1Pipeline
    from .pipelines.blob_persistence_v2 import BlobPersistenceV2Pipeline
    from .pipelines.blob_ransac_v1 import BlobRansacV1Pipeline
    from .pipelines.de_novo_hypothesis_select_v1 import DeNovoHypothesisSelectV1Pipeline
    from .pipelines.de_novo_seed_extend_v2 import DeNovoSeedExtendV2Pipeline
    from .pipelines.shank_axis_v1 import ShankAxisV1Pipeline
    from .pipelines.shank_cluster_v1 import ShankClusterV1Pipeline
    from .pipelines.shank_graph_v1 import ShankGraphV1Pipeline
    from .pipelines.shank_hypothesis_v1 import ShankHypothesisV1Pipeline
    from .pipelines.shank_stitch_v1 import ShankStitchV1Pipeline
    from .pipelines.shank_grow_v1 import ShankGrowV1Pipeline

    registry.register_pipeline("blob_ransac_v1", BlobRansacV1Pipeline, overwrite=True)
    registry.register_pipeline("blob_em_v2", BlobEMV2Pipeline, overwrite=True)
    registry.register_pipeline("blob_consensus_v1", BlobConsensusV1Pipeline, overwrite=True)
    registry.register_pipeline("hybrid_bead_string_v1", HybridBeadStringV1Pipeline, overwrite=True)
    registry.register_pipeline("blob_persistence_v1", BlobPersistenceV1Pipeline, overwrite=True)
    registry.register_pipeline("blob_persistence_v2", BlobPersistenceV2Pipeline, overwrite=True)
    registry.register_pipeline("de_novo_hypothesis_select_v1", DeNovoHypothesisSelectV1Pipeline, overwrite=True)
    registry.register_pipeline("de_novo_seed_extend_v2", DeNovoSeedExtendV2Pipeline, overwrite=True)
    registry.register_pipeline("shank_axis_v1", ShankAxisV1Pipeline, overwrite=True)
    registry.register_pipeline("shank_cluster_v1", ShankClusterV1Pipeline, overwrite=True)
    registry.register_pipeline("shank_graph_v1", ShankGraphV1Pipeline, overwrite=True)
    registry.register_pipeline("shank_hypothesis_v1", ShankHypothesisV1Pipeline, overwrite=True)
    registry.register_pipeline("shank_stitch_v1", ShankStitchV1Pipeline, overwrite=True)
    registry.register_pipeline("shank_grow_v1", ShankGrowV1Pipeline, overwrite=True)
