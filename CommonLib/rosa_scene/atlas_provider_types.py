"""Typed atlas-provider contracts used by atlas labeling services."""

from __future__ import annotations

from typing import Protocol, Sequence, TypedDict


class AtlasSampleResult(TypedDict):
    """Standardized per-contact atlas sample result."""

    source: str
    label: str
    label_value: int
    distance_to_voxel_mm: float
    distance_to_centroid_mm: float
    native_ras: list[float]


class AtlasProvider(Protocol):
    """Minimal provider contract for atlas contact labeling."""

    source_id: str
    display_name: str

    def is_ready(self) -> bool:
        """Return True when provider can sample contact labels."""

    def sample_contact(self, point_world_ras: Sequence[float]) -> AtlasSampleResult | None:
        """Sample nearest label for a contact point in world RAS space."""

