"""Typed stage outputs and composed stage classes for Deep Core.

This module replaces the mixin-based pipeline with composition.  Each
stage is a standalone class that receives its dependencies through the
constructor and returns a typed dataclass result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from shank_core.blob_candidates import build_blob_labelmap, extract_blob_candidates
from shank_core.masking import compute_head_distance_map_kji, largest_component_binary

from .deep_core_annulus import AnnulusSampler
from .deep_core_config import (
    DeepCoreAnnulusConfig,
    DeepCoreConfig,
    DeepCoreInternalConfig,
    DeepCoreMaskConfig,
    DeepCoreProposalConfig,
    DeepCoreSupportConfig,
    deep_core_default_config,
)


# ---------------------------------------------------------------------------
# Typed stage outputs
# ---------------------------------------------------------------------------

@dataclass
class MaskStageOutput:
    """Result from the mask-building stage."""

    hull_mask_kji: np.ndarray
    deep_core_mask_kji: np.ndarray
    metal_mask_kji: np.ndarray
    metal_grown_mask_kji: np.ndarray
    deep_seed_raw_mask_kji: np.ndarray
    deep_seed_mask_kji: np.ndarray
    head_distance_map_kji: np.ndarray
    smoothed_hull_kji: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    # -- backward-compat helpers -------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """Return the legacy dict representation."""
        return {
            "hull_mask_kji": self.hull_mask_kji,
            "deep_core_mask_kji": self.deep_core_mask_kji,
            "metal_mask_kji": self.metal_mask_kji,
            "metal_grown_mask_kji": self.metal_grown_mask_kji,
            "deep_seed_raw_mask_kji": self.deep_seed_raw_mask_kji,
            "deep_seed_mask_kji": self.deep_seed_mask_kji,
            "head_distance_map_kji": self.head_distance_map_kji,
            "smoothed_hull_kji": self.smoothed_hull_kji,
            "stats": dict(self.stats),
        }


@dataclass
class SupportStageOutput:
    """Result from the support-atom extraction stage."""

    mask: MaskStageOutput
    support_atoms: list[dict[str, Any]]
    blob_sample_points_ras: np.ndarray
    blob_sample_blob_ids: np.ndarray
    blob_sample_atom_ids: np.ndarray
    blob_axes_ras_by_id: dict[int, Any]
    blob_elongation_by_id: dict[int, float]
    blob_class_by_id: dict[int, str]
    blob_parent_blob_ids_by_id: dict[int, int] = field(default_factory=dict)
    blob_labelmap_kji: np.ndarray | None = None
    blob_centroids_all_ras: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    blob_centroids_kept_ras: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    blob_centroids_rejected_ras: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    line_blob_sample_points_ras: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    contact_blob_sample_points_ras: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    complex_blob_sample_points_ras: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )
    complex_blob_chain_rows: list[dict[str, Any]] = field(default_factory=list)
    contact_chain_rows: list[dict[str, Any]] = field(default_factory=list)
    contact_chain_debug_rows: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors (used by proposal stages) -------------------

    @property
    def head_distance_map_kji(self) -> np.ndarray:
        return self.mask.head_distance_map_kji

    @property
    def metal_grown_mask_kji(self) -> np.ndarray:
        return self.mask.metal_grown_mask_kji

    # -- dict-like access for proposal methods -----------------------------
    #
    # The proposal stage methods read support data via
    #     support_payload.get("blob_sample_points_ras")
    # By implementing ``get()`` and ``combined_payload`` here the typed
    # object can be passed directly, avoiding a full dict copy.

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "support_atoms": "support_atoms",
        "blob_sample_points_ras": "blob_sample_points_ras",
        "blob_sample_blob_ids": "blob_sample_blob_ids",
        "blob_sample_atom_ids": "blob_sample_atom_ids",
        "blob_axes_ras_by_id": "blob_axes_ras_by_id",
        "blob_elongation_by_id": "blob_elongation_by_id",
        "blob_class_by_id": "blob_class_by_id",
        "blob_parent_blob_ids_by_id": "blob_parent_blob_ids_by_id",
        "blob_labelmap_kji": "blob_labelmap_kji",
        "blob_centroids_all_ras": "blob_centroids_all_ras",
        "blob_centroids_kept_ras": "blob_centroids_kept_ras",
        "blob_centroids_rejected_ras": "blob_centroids_rejected_ras",
        "line_blob_sample_points_ras": "line_blob_sample_points_ras",
        "contact_blob_sample_points_ras": "contact_blob_sample_points_ras",
        "complex_blob_sample_points_ras": "complex_blob_sample_points_ras",
        "complex_blob_chain_rows": "complex_blob_chain_rows",
        "contact_chain_rows": "contact_chain_rows",
        "contact_chain_debug_rows": "contact_chain_debug_rows",
        "stats": "stats",
        # Keys that live on the mask sub-object:
        "head_distance_map_kji": "head_distance_map_kji",
        "metal_grown_mask_kji": "metal_grown_mask_kji",
        "hull_mask_kji": "_mask_hull_mask_kji",
        "deep_core_mask_kji": "_mask_deep_core_mask_kji",
        "metal_mask_kji": "_mask_metal_mask_kji",
        "deep_seed_raw_mask_kji": "_mask_deep_seed_raw_mask_kji",
        "deep_seed_mask_kji": "_mask_deep_seed_mask_kji",
        "smoothed_hull_kji": "_mask_smoothed_hull_kji",
    }

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like lookup so proposal methods can read fields directly."""
        attr = self._KEY_MAP.get(key)
        if attr is None:
            return default
        if attr.startswith("_mask_"):
            return getattr(self.mask, attr[6:], default)
        return getattr(self, attr, default)

    @property
    def combined_payload(self):
        """Return *self* — the proposal methods call
        ``getattr(support_result, "combined_payload", ...)`` and then
        ``.get("key")`` on the result.  Returning self satisfies both."""
        return self

    # -- full materialisation (only for legacy widget code) ----------------

    def to_payload(self) -> dict[str, Any]:
        """Return the legacy dict with keys from both mask and support."""
        out = self.mask.to_payload()
        out.update({
            "support_atoms": list(self.support_atoms),
            "blob_sample_points_ras": self.blob_sample_points_ras,
            "blob_sample_blob_ids": self.blob_sample_blob_ids,
            "blob_sample_atom_ids": self.blob_sample_atom_ids,
            "blob_axes_ras_by_id": dict(self.blob_axes_ras_by_id),
            "blob_elongation_by_id": dict(self.blob_elongation_by_id),
            "blob_class_by_id": dict(self.blob_class_by_id),
            "blob_parent_blob_ids_by_id": dict(self.blob_parent_blob_ids_by_id),
            "blob_labelmap_kji": self.blob_labelmap_kji,
            "blob_centroids_all_ras": self.blob_centroids_all_ras,
            "blob_centroids_kept_ras": self.blob_centroids_kept_ras,
            "blob_centroids_rejected_ras": self.blob_centroids_rejected_ras,
            "line_blob_sample_points_ras": self.line_blob_sample_points_ras,
            "contact_blob_sample_points_ras": self.contact_blob_sample_points_ras,
            "complex_blob_sample_points_ras": self.complex_blob_sample_points_ras,
            "complex_blob_chain_rows": list(self.complex_blob_chain_rows),
            "contact_chain_rows": list(self.contact_chain_rows),
            "contact_chain_debug_rows": list(self.contact_chain_debug_rows),
            "stats": dict(self.stats),
        })
        return out


@dataclass
class ProposalStageOutput:
    """Result from the proposal generation stage."""

    support: SupportStageOutput
    proposals: list[dict[str, Any]]
    candidate_count: int = 0
    token_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return the legacy dict for the proposal stage."""
        return {
            "proposals": list(self.proposals),
            "candidate_count": self.candidate_count,
            "token_count": self.token_count,
        }


# ---------------------------------------------------------------------------
# MaskStage
# ---------------------------------------------------------------------------

class MaskStage:
    """Build hull, deep-core, and metal masks from a CT volume."""

    def __init__(self, volume_accessor):
        self._vol = volume_accessor

    def run(
        self,
        volume_node: Any,
        config: DeepCoreMaskConfig | None = None,
    ) -> MaskStageOutput:
        if sitk is None:
            raise RuntimeError("SimpleITK is required for deep-core mask stage.")
        cfg = config if config is not None else DeepCoreMaskConfig()

        arr_kji = np.asarray(self._vol.array_kji(volume_node), dtype=np.float32)
        spacing_xyz = self._vol.spacing_xyz(volume_node)

        arr_clip = np.asarray(arr_kji, dtype=np.float32)
        if np.isfinite(float(cfg.hull_clip_hu)):
            arr_clip = np.minimum(arr_clip, float(cfg.hull_clip_hu))
        hull_img = sitk.GetImageFromArray(arr_clip.astype(np.float32))
        hull_img.SetSpacing(
            (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
        )
        if float(cfg.hull_sigma_mm) > 0.0:
            hull_img = sitk.SmoothingRecursiveGaussian(
                hull_img, float(cfg.hull_sigma_mm)
            )
        smoothed_hull_kji = sitk.GetArrayFromImage(hull_img).astype(np.float32)

        binary = sitk.BinaryThreshold(
            hull_img,
            lowerThreshold=float(cfg.hull_threshold_hu),
            upperThreshold=float(
                max(
                    float(np.nanmax(smoothed_hull_kji)),
                    float(cfg.hull_threshold_hu) + 1.0,
                )
            ),
            insideValue=1,
            outsideValue=0,
        )
        if int(cfg.hull_open_vox) > 0:
            binary = sitk.BinaryMorphologicalOpening(
                binary, [int(cfg.hull_open_vox)] * 3, sitk.sitkBall
            )
        hull_lcc = largest_component_binary(binary)
        if hull_lcc is None:
            raise RuntimeError(
                "Failed to build a non-air hull mask from the selected CT."
            )
        if int(cfg.hull_close_vox) > 0:
            hull_lcc = sitk.BinaryMorphologicalClosing(
                hull_lcc, [int(cfg.hull_close_vox)] * 3, sitk.sitkBall
            )
        hull_mask_kji = sitk.GetArrayFromImage(hull_lcc).astype(bool)
        head_distance_map_kji = np.asarray(
            compute_head_distance_map_kji(
                hull_mask_kji, spacing_xyz=spacing_xyz
            ),
            dtype=np.float32,
        )
        deep_core_mask_kji = np.logical_and(
            hull_mask_kji,
            head_distance_map_kji >= float(max(0.0, cfg.deep_core_shrink_mm)),
        )

        metal_mask_kji = np.asarray(
            arr_kji >= float(cfg.metal_threshold_hu), dtype=bool
        )
        metal_grown_mask_kji = metal_mask_kji.copy()
        if int(cfg.metal_grow_vox) > 0:
            metal_img = sitk.GetImageFromArray(metal_mask_kji.astype(np.uint8))
            metal_grown_mask_kji = sitk.GetArrayFromImage(
                sitk.BinaryDilate(
                    metal_img, [int(cfg.metal_grow_vox)] * 3, sitk.sitkBall
                )
            ).astype(bool)

        deep_seed_raw_mask_kji = np.logical_and(metal_mask_kji, deep_core_mask_kji)
        deep_seed_mask_kji = np.logical_and(metal_grown_mask_kji, deep_core_mask_kji)

        stats = {
            "hull_voxels": int(np.count_nonzero(hull_mask_kji)),
            "deep_core_voxels": int(np.count_nonzero(deep_core_mask_kji)),
            "metal_voxels": int(np.count_nonzero(metal_mask_kji)),
            "metal_grown_voxels": int(np.count_nonzero(metal_grown_mask_kji)),
            "deep_seed_raw_voxels": int(np.count_nonzero(deep_seed_raw_mask_kji)),
            "deep_seed_voxels": int(np.count_nonzero(deep_seed_mask_kji)),
            "hull_threshold_hu": float(cfg.hull_threshold_hu),
            "hull_clip_hu": float(cfg.hull_clip_hu),
            "hull_sigma_mm": float(cfg.hull_sigma_mm),
            "hull_open_vox": int(cfg.hull_open_vox),
            "hull_close_vox": int(cfg.hull_close_vox),
            "deep_core_shrink_mm": float(cfg.deep_core_shrink_mm),
            "metal_threshold_hu": float(cfg.metal_threshold_hu),
            "metal_grow_vox": int(cfg.metal_grow_vox),
        }
        return MaskStageOutput(
            hull_mask_kji=hull_mask_kji,
            deep_core_mask_kji=deep_core_mask_kji,
            metal_mask_kji=metal_mask_kji,
            metal_grown_mask_kji=metal_grown_mask_kji,
            deep_seed_raw_mask_kji=deep_seed_raw_mask_kji,
            deep_seed_mask_kji=deep_seed_mask_kji,
            head_distance_map_kji=head_distance_map_kji,
            smoothed_hull_kji=smoothed_hull_kji,
            stats=stats,
        )


# ---------------------------------------------------------------------------
# SupportStage
# ---------------------------------------------------------------------------

def _make_support_stage_class():
    """Build the SupportStage class with mixin bases.

    Done in a factory so the mixin imports stay local and the module-level
    namespace is not polluted.
    """
    from .deep_core_annulus import DeepCoreAnnulusMixin
    from .deep_core_atoms import DeepCoreAtomBuilderMixin
    from .deep_core_candidate_inputs import DeepCoreCandidateInputMixin
    from .deep_core_complex_blob import DeepCoreComplexBlobMixin
    from .deep_core_contact_chain import DeepCoreContactChainMixin

    class SupportStage(
        DeepCoreAtomBuilderMixin,
        DeepCoreCandidateInputMixin,
        DeepCoreContactChainMixin,
        DeepCoreComplexBlobMixin,
        DeepCoreAnnulusMixin,
    ):
        """Extract support atoms from deep-core metal blobs."""

        def __init__(self, volume_accessor, annulus: AnnulusSampler):
            self._vol = volume_accessor
            self._annulus = annulus

        def _ijk_kji_to_ras_points(self, volume_node, ijk_kji):
            return self._vol.ijk_kji_to_ras_points(volume_node, ijk_kji)

        def run(
            self,
            volume_node: Any,
            mask: MaskStageOutput,
            support_config: DeepCoreSupportConfig | None = None,
            annulus_config: DeepCoreAnnulusConfig | None = None,
            internal_config: DeepCoreInternalConfig | None = None,
            show_support_diagnostics: bool = True,
        ) -> SupportStageOutput:
            defaults = deep_core_default_config()
            support_cfg = support_config or defaults.support
            annulus_cfg = annulus_config or defaults.annulus
            internal_cfg = internal_config or defaults.internal

            arr_kji = np.asarray(self._vol.array_kji(volume_node), dtype=np.float32)
            _ijk_kji_to_ras = lambda idx: self._vol.ijk_kji_to_ras_points(volume_node, idx)

            raw_blob_result = extract_blob_candidates(
                metal_mask_kji=mask.deep_seed_raw_mask_kji,
                arr_kji=arr_kji,
                depth_map_kji=mask.head_distance_map_kji,
                ijk_kji_to_ras_fn=_ijk_kji_to_ras,
            )
            grown_blob_result = extract_blob_candidates(
                metal_mask_kji=mask.deep_seed_mask_kji,
                arr_kji=arr_kji,
                depth_map_kji=mask.head_distance_map_kji,
                ijk_kji_to_ras_fn=_ijk_kji_to_ras,
            )
            sample_payload = self._build_support_atom_payload(
                volume_node=volume_node,
                labels_kji=raw_blob_result.get("labels_kji"),
                blobs=list(raw_blob_result.get("blobs") or []),
                support_spacing_mm=float(support_cfg.support_spacing_mm),
                component_min_elongation=float(support_cfg.component_min_elongation),
                line_atom_diameter_max_mm=float(support_cfg.line_atom_diameter_max_mm),
                line_atom_min_span_mm=float(support_cfg.line_atom_min_span_mm),
                line_atom_min_pca_dominance=float(support_cfg.line_atom_min_pca_dominance),
                contact_component_diameter_max_mm=float(support_cfg.contact_component_diameter_max_mm),
                support_cube_size_mm=float(support_cfg.support_cube_size_mm),
                head_distance_map_kji=mask.head_distance_map_kji,
                annulus_config=annulus_cfg,
                internal_config=internal_cfg,
            )

            raw_blobs = [
                dict(blob or {})
                for blob in list(raw_blob_result.get("blobs") or [])
            ]
            support_atoms = list(sample_payload.get("support_atoms") or [])
            kept_parent_blob_ids = {
                int(a.get("parent_blob_id", -1))
                for a in support_atoms
                if int(dict(a or {}).get("parent_blob_id", -1)) > 0
            }

            blob_labelmap_kji = (
                build_blob_labelmap(raw_blob_result.get("labels_kji"))
                if show_support_diagnostics
                else None
            )

            blob_centroids_all_ras = np.asarray(
                [b.get("centroid_ras") for b in raw_blobs if b.get("centroid_ras") is not None],
                dtype=float,
            ).reshape(-1, 3)
            blob_centroids_kept_ras = np.asarray(
                [b.get("centroid_ras") for b in raw_blobs
                 if b.get("centroid_ras") is not None and int(b.get("blob_id", -1)) in kept_parent_blob_ids],
                dtype=float,
            ).reshape(-1, 3)
            blob_centroids_rejected_ras = np.asarray(
                [b.get("centroid_ras") for b in raw_blobs
                 if b.get("centroid_ras") is not None and int(b.get("blob_id", -1)) not in kept_parent_blob_ids],
                dtype=float,
            ).reshape(-1, 3)

            blob_axes_ras_by_id = {
                int(k): [float(v) for v in np.asarray(val, dtype=float).reshape(3).tolist()]
                for k, val in dict(sample_payload.get("axes_ras_by_id") or {}).items()
                if int(k) > 0 and np.asarray(val, dtype=float).reshape(-1).size == 3
            }
            blob_elongation_by_id = {
                int(k): float(v)
                for k, v in dict(sample_payload.get("elongation_by_id") or {}).items()
                if int(k) > 0
            }
            blob_parent_blob_ids_by_id = {
                int(k): int(v)
                for k, v in dict(sample_payload.get("parent_blob_ids_by_id") or {}).items()
                if int(k) > 0
            }

            blob_sample_points_ras = np.asarray(sample_payload.get("points_ras"), dtype=float).reshape(-1, 3)
            blob_sample_blob_ids = np.asarray(sample_payload.get("blob_ids"), dtype=np.int32).reshape(-1)
            blob_sample_atom_ids = np.asarray(sample_payload.get("atom_ids"), dtype=np.int32).reshape(-1)

            stats = dict(mask.stats)
            stats.update({
                "support_spacing_mm": float(support_cfg.support_spacing_mm),
                "component_min_elongation": float(support_cfg.component_min_elongation),
                "line_atom_diameter_max_mm": float(support_cfg.line_atom_diameter_max_mm),
                "line_atom_min_span_mm": float(support_cfg.line_atom_min_span_mm),
                "line_atom_min_pca_dominance": float(support_cfg.line_atom_min_pca_dominance),
                "contact_component_diameter_max_mm": float(support_cfg.contact_component_diameter_max_mm),
                "support_cube_size_mm": float(support_cfg.support_cube_size_mm),
                "deep_seed_sampled_blob_count": int(len(blob_elongation_by_id)),
                "deep_seed_atom_count": int(len(support_atoms)),
                "deep_seed_raw_blob_count": int(raw_blob_result.get("blob_count_total", 0)),
                "deep_seed_grown_blob_count": int(grown_blob_result.get("blob_count_total", 0)),
                "deep_seed_sample_count": int(blob_sample_points_ras.shape[0]),
                "deep_seed_line_blob_token_count": int(np.asarray(sample_payload.get("line_blob_points_ras"), dtype=float).reshape(-1, 3).shape[0]),
                "deep_seed_contact_blob_token_count": int(np.asarray(sample_payload.get("contact_blob_points_ras"), dtype=float).reshape(-1, 3).shape[0]),
                "deep_seed_complex_blob_token_count": int(np.asarray(sample_payload.get("complex_blob_points_ras"), dtype=float).reshape(-1, 3).shape[0]),
                "deep_seed_complex_blob_chain_row_count": int(len(list(sample_payload.get("complex_blob_chain_rows") or []))),
                "deep_seed_contact_chain_row_count": int(len(list(sample_payload.get("contact_chain_rows") or []))),
                "deep_seed_contact_chain_debug_row_count": int(len(list(sample_payload.get("contact_chain_debug_rows") or []))),
            })

            return SupportStageOutput(
                mask=mask,
                support_atoms=support_atoms,
                blob_sample_points_ras=blob_sample_points_ras,
                blob_sample_blob_ids=blob_sample_blob_ids,
                blob_sample_atom_ids=blob_sample_atom_ids,
                blob_axes_ras_by_id=blob_axes_ras_by_id,
                blob_elongation_by_id=blob_elongation_by_id,
                blob_class_by_id=dict(sample_payload.get("blob_class_by_id") or {}),
                blob_parent_blob_ids_by_id=blob_parent_blob_ids_by_id,
                blob_labelmap_kji=blob_labelmap_kji,
                blob_centroids_all_ras=blob_centroids_all_ras,
                blob_centroids_kept_ras=blob_centroids_kept_ras,
                blob_centroids_rejected_ras=blob_centroids_rejected_ras,
                line_blob_sample_points_ras=np.asarray(sample_payload.get("line_blob_points_ras"), dtype=float).reshape(-1, 3),
                contact_blob_sample_points_ras=np.asarray(sample_payload.get("contact_blob_points_ras"), dtype=float).reshape(-1, 3),
                complex_blob_sample_points_ras=np.asarray(sample_payload.get("complex_blob_points_ras"), dtype=float).reshape(-1, 3),
                complex_blob_chain_rows=list(sample_payload.get("complex_blob_chain_rows") or []),
                contact_chain_rows=list(sample_payload.get("contact_chain_rows") or []),
                contact_chain_debug_rows=list(sample_payload.get("contact_chain_debug_rows") or []),
                stats=stats,
            )

    return SupportStage


SupportStage = _make_support_stage_class()


# ---------------------------------------------------------------------------
# ProposalStage
# ---------------------------------------------------------------------------

def _make_proposal_stage_class():
    """Build the ProposalStage class with mixin bases."""
    from .deep_core_proposals import DeepCoreProposalLogicMixin

    class ProposalStage(DeepCoreProposalLogicMixin):
        """Generate, filter, extend, and finalize trajectory proposals."""

        def __init__(self, volume_accessor, annulus: AnnulusSampler):
            self._vol = volume_accessor
            self._annulus = annulus

        def run(
            self,
            volume_node: Any,
            support: SupportStageOutput,
            config: DeepCoreConfig | None = None,
        ) -> ProposalStageOutput:
            cfg = config or deep_core_default_config()

            # Pass the typed SupportStageOutput directly — it satisfies
            # the .get() / .combined_payload interface the mixin methods
            # expect, avoiding a full dict materialisation.
            raw_payload = self._build_deep_core_raw_proposals_stage(
                volume_node=volume_node,
                support_result=support,
                mask_config=cfg.mask,
                support_config=cfg.support,
                proposal_config=cfg.proposal,
                annulus_config=cfg.annulus,
                internal_config=cfg.internal,
            )
            pre_ext_payload = self._apply_pre_extension_annulus_rejection_stage(
                volume_node=volume_node,
                support_result=support,
                proposal_payload=raw_payload,
                annulus_config=cfg.annulus,
            )
            ext_payload = self._extend_deep_core_proposals_stage(
                volume_node=volume_node,
                support_result=support,
                proposal_payload=pre_ext_payload,
                mask_config=cfg.mask,
                support_config=cfg.support,
                proposal_config=cfg.proposal,
                annulus_config=cfg.annulus,
                internal_config=cfg.internal,
            )
            final_payload = self._apply_final_deep_core_rejection_stage(
                volume_node=volume_node,
                support_result=support,
                proposal_payload=ext_payload,
                mask_config=cfg.mask,
                proposal_config=cfg.proposal,
            )

            return ProposalStageOutput(
                support=support,
                proposals=list(final_payload.get("proposals") or []),
                candidate_count=int(final_payload.get("candidate_count", 0)),
                token_count=int(final_payload.get("token_count", 0)),
            )

    return ProposalStage


ProposalStage = _make_proposal_stage_class()
