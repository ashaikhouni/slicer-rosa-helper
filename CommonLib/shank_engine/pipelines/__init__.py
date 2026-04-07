"""Built-in detection pipelines."""

from .blob_em_v2 import BlobEMV2Pipeline
from .blob_consensus_v1 import BlobConsensusV1Pipeline
from .hybrid_bead_string_v1 import HybridBeadStringV1Pipeline
from .blob_persistence_v1 import BlobPersistenceV1Pipeline
from .blob_persistence_v2 import BlobPersistenceV2Pipeline
from .blob_ransac_v1 import BlobRansacV1Pipeline
from .de_novo_hypothesis_select_v1 import DeNovoHypothesisSelectV1Pipeline
from .de_novo_seed_extend_v2 import DeNovoSeedExtendV2Pipeline
from .shank_axis_v1 import ShankAxisV1Pipeline
from .shank_cluster_v1 import ShankClusterV1Pipeline
from .shank_graph_v1 import ShankGraphV1Pipeline
from .shank_hypothesis_v1 import ShankHypothesisV1Pipeline
from .shank_stitch_v1 import ShankStitchV1Pipeline
from .shank_grow_v1 import ShankGrowV1Pipeline

__all__ = [
    "BlobRansacV1Pipeline",
    "BlobEMV2Pipeline",
    "BlobConsensusV1Pipeline",
    "HybridBeadStringV1Pipeline",
    "BlobPersistenceV1Pipeline",
    "BlobPersistenceV2Pipeline",
    "DeNovoHypothesisSelectV1Pipeline",
    "DeNovoSeedExtendV2Pipeline",
    "ShankAxisV1Pipeline",
    "ShankClusterV1Pipeline",
    "ShankGraphV1Pipeline",
    "ShankHypothesisV1Pipeline",
    "ShankStitchV1Pipeline",
    "ShankGrowV1Pipeline",
]
