"""Built-in detection pipelines."""

from .blob_em_v2 import BlobEMV2Pipeline
from .blob_ransac_v1 import BlobRansacV1Pipeline
from .deep_core_v1 import DeepCoreV1Pipeline
from .deep_core_v2 import DeepCoreV2Pipeline

__all__ = [
    "BlobRansacV1Pipeline",
    "BlobEMV2Pipeline",
    "DeepCoreV1Pipeline",
    "DeepCoreV2Pipeline",
]
