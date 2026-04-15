"""Configuration and UI metadata for the Deep Core trajectory predictor.

Deep Core now uses an explicit staged pipeline. This module is the single
source of truth for:
- default parameter values
- which parameters are user-facing vs internal
- UI labels/ranges/suffixes for exposed controls

The algorithm code should reference these dataclasses rather than embedding
policy constants inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any


def _ui_meta(
    label: str,
    *,
    minimum: float | int,
    maximum: float | int,
    decimals: int = 0,
    suffix: str = "",
    advanced: bool = False,
    description: str = "",
    control: str = "double",
) -> dict[str, Any]:
    """Attach UI metadata to a dataclass field.

    The widget reads this metadata so defaults/ranges are not hardcoded in UI
    construction code.
    """

    return {
        "ui": True,
        "label": str(label),
        "minimum": minimum,
        "maximum": maximum,
        "decimals": int(decimals),
        "suffix": str(suffix),
        "advanced": bool(advanced),
        "description": str(description),
        "control": str(control),
    }


@dataclass(frozen=True)
class DeepCoreMaskConfig:
    """Parameters used to build the deep-core metal support masks."""

    hull_threshold_hu: float = field(
        default=-500.0,
        metadata=_ui_meta(
            "Hull threshold",
            minimum=-1200.0,
            maximum=500.0,
            decimals=1,
            suffix=" HU",
            description="Lower threshold used to define the non-air hull.",
        ),
    )
    hull_clip_hu: float = field(
        default=1200.0,
        metadata=_ui_meta(
            "Hull clip",
            minimum=100.0,
            maximum=4000.0,
            decimals=1,
            suffix=" HU",
            description="Intensity clip applied before hull smoothing.",
        ),
    )
    hull_sigma_mm: float = field(
        default=4.0,
        metadata=_ui_meta(
            "Hull Gaussian",
            minimum=0.0,
            maximum=20.0,
            decimals=2,
            suffix=" mm",
            description="Gaussian smoothing used before hull thresholding.",
        ),
    )
    hull_open_vox: int = field(
        default=7,
        metadata=_ui_meta(
            "Hull opening",
            minimum=0,
            maximum=30,
            suffix=" vox",
            control="int",
            description="Morphological opening applied to the hull mask.",
        ),
    )
    hull_close_vox: int = field(
        default=0,
        metadata=_ui_meta(
            "Hull closing",
            minimum=0,
            maximum=30,
            suffix=" vox",
            control="int",
            description="Morphological closing applied to the hull mask.",
        ),
    )
    deep_core_shrink_mm: float = field(
        default=20.0,
        metadata=_ui_meta(
            "Deep-core shrink",
            minimum=0.0,
            maximum=40.0,
            decimals=2,
            suffix=" mm",
            description="Minimum hull distance required for deep-core support (axes only).",
        ),
    )
    metal_threshold_hu: float = field(
        default=1900.0,
        metadata=_ui_meta(
            "Metal threshold",
            minimum=-1200.0,
            maximum=4000.0,
            decimals=1,
            suffix=" HU",
            description="Threshold used to define metal support voxels.",
        ),
    )
    bolt_metal_threshold_hu: float = field(
        default=2000.0,
        metadata=_ui_meta(
            "Bolt metal threshold",
            minimum=1000.0,
            maximum=5000.0,
            decimals=1,
            suffix=" HU",
            description=("Higher threshold used to isolate saturation-bright bolt"
                         " metal for the bolt RANSAC stage."),
        ),
    )
    metal_grow_vox: int = field(
        default=1,
        metadata=_ui_meta(
            "Metal grow",
            minimum=0,
            maximum=6,
            suffix=" vox",
            control="int",
            description="Binary dilation applied to the metal mask.",
        ),
    )


@dataclass(frozen=True)
class DeepCoreSupportConfig:
    """Parameters controlling blob sampling and support-atom extraction."""

    support_spacing_mm: float = field(
        default=2.5,
        metadata=_ui_meta(
            "Support spacing",
            minimum=0.5,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            description="Nominal spacing used when fitting support atoms.",
        ),
    )
    component_min_elongation: float = field(
        default=4.0,
        metadata=_ui_meta(
            "Component min elongation",
            minimum=1.0,
            maximum=50.0,
            decimals=2,
            description="Minimum elongation for a blob to be considered line-like.",
        ),
    )
    line_atom_diameter_max_mm: float = field(
        default=2.0,
        metadata=_ui_meta(
            "Line atom max diameter",
            minimum=0.2,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            description="Maximum allowed diameter for direct line-blob atoms.",
        ),
    )
    line_atom_min_span_mm: float = field(
        default=10.0,
        metadata=_ui_meta(
            "Line atom min span",
            minimum=1.0,
            maximum=50.0,
            decimals=2,
            suffix=" mm",
            description="Minimum span required for a direct line-blob atom.",
        ),
    )
    line_atom_min_pca_dominance: float = field(
        default=6.0,
        metadata=_ui_meta(
            "Line atom PCA dominance",
            minimum=1.0,
            maximum=50.0,
            decimals=2,
            description="Minimum PCA dominance required for direct line blobs.",
        ),
    )
    contact_component_diameter_max_mm: float = field(
        default=10.0,
        metadata=_ui_meta(
            "Contact component max diameter",
            minimum=0.5,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            description="Maximum blob diameter treated as a compact contact blob.",
        ),
    )
    support_cube_size_mm: float = field(
        default=5.0,
        metadata=_ui_meta(
            "Support voxel size",
            minimum=1.0,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            description="Cube size used to voxelize complex blobs into support nodes.",
        ),
    )


@dataclass(frozen=True)
class DeepCoreProposalConfig:
    """Proposal building, extension, and final pruning parameters."""

    neighbor_max_mm: float = 8.0
    neighbor_min_mm: float = 1.0
    neighbor_axis_angle_deg: float = 30.0
    grow_turn_angle_deg: float = 30.0
    inlier_radius_mm: float = 2.0
    axial_bin_mm: float = 2.5
    min_span_mm: float = 16.0
    min_inlier_count: int = 5
    min_chain_tokens: int = 5
    max_seed_edges: int = 1200
    max_output_proposals: int | None = None
    guided_threshold_hu: float | None = None
    guided_head_mask_threshold_hu: float | None = None
    guided_roi_radius_mm: float = 5.0
    guided_max_angle_deg: float = 12.0
    guided_max_depth_shift_mm: float = 2.0
    guided_fit_mode: str = "deep_anchor_v2"
    guided_max_residual_mm: float = 2.0
    short_float_reject_span_mm: float = field(default=20.0)
    short_float_edge_tol_mm: float = field(default=2.5)
    outward_support_check_span_mm: float = field(
        default=45.0,
        metadata=_ui_meta(
            "Outward support check span",
            minimum=5.0,
            maximum=120.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Short proposals longer than this skip the outward-support rejector.",
        ),
    )
    outward_support_radius_mm: float = field(
        default=2.5,
        metadata=_ui_meta(
            "Outward support radius",
            minimum=0.5,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Lateral corridor used when searching for outward support.",
        ),
    )
    outward_support_search_mm: float = field(
        default=20.0,
        metadata=_ui_meta(
            "Outward support search",
            minimum=2.0,
            maximum=60.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Maximum axial distance searched beyond the shallow endpoint.",
        ),
    )
    outward_support_min_extension_mm: float = field(
        default=6.0,
        metadata=_ui_meta(
            "Outward support min extension",
            minimum=0.0,
            maximum=40.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Minimum outward support span required for short proposals.",
        ),
    )
    outward_support_min_depth_gain_mm: float = field(
        default=4.0,
        metadata=_ui_meta(
            "Outward support min depth gain",
            minimum=0.0,
            maximum=40.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Minimum head-depth gain required in outward support.",
        ),
    )


@dataclass(frozen=True)
class DeepCoreAnnulusConfig:
    """Annulus sampling and bone-adjacency rejection parameters."""

    pre_extension_annulus_reject_percentile: float = field(
        default=80.0,
        metadata=_ui_meta(
            "Pre-extension annulus percentile",
            minimum=0.0,
            maximum=100.0,
            decimals=1,
            suffix=" pct",
            description="Percentile above which pre-extension annulus support is treated as bone-adjacent.",
        ),
    )
    annulus_reference_upper_hu: float = 2500.0
    annulus_flag_min_samples: int = 120
    line_annulus_inner_mm: float = 2.5
    line_annulus_outer_mm: float = 3.5
    cross_section_annulus_inner_mm: float = field(
        default=3.0,
        metadata=_ui_meta(
            "Annulus inner radius",
            minimum=0.5,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Inner radius of the cross-section annulus used for profile and seed analysis.",
        ),
    )
    cross_section_annulus_outer_mm: float = field(
        default=4.0,
        metadata=_ui_meta(
            "Annulus outer radius",
            minimum=0.5,
            maximum=12.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Outer radius of the cross-section annulus used for profile and seed analysis.",
        ),
    )
    annulus_radial_steps: int = 2
    annulus_angular_samples: int = 12
    profile_axial_step_mm: float = 1.0
    profile_min_samples: int = 8
    profile_remaining_brain_fraction_req: float = field(
        default=0.80,
        metadata=_ui_meta(
            "Profile remaining brain fraction",
            minimum=0.0,
            maximum=1.0,
            decimals=2,
            advanced=True,
            description="Required brain-like fraction after removing one allowed endpoint non-brain run.",
        ),
    )
    profile_simple_middle_nonbrain_max_run_mm: float = field(
        default=1.0,
        metadata=_ui_meta(
            "Simple middle non-brain max run",
            minimum=0.0,
            maximum=20.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Maximum allowed middle non-brain run for single-emission blobs.",
        ),
    )
    profile_multi_middle_nonbrain_max_run_mm: float = field(
        default=3.0,
        metadata=_ui_meta(
            "Multi middle non-brain max run",
            minimum=0.0,
            maximum=20.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Maximum allowed middle non-brain run for multi-emission blobs.",
        ),
    )
    profile_endpoint_margin_mm: float = field(
        default=2.0,
        metadata=_ui_meta(
            "Profile endpoint margin",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Endpoint margin used when deciding whether a non-brain run counts as middle contamination.",
        ),
    )


@dataclass(frozen=True)
class DeepCoreInternalConfig:
    """Internal policy constants that shape the current algorithm.

    These remain centralized and documented, but are intentionally not exposed in
    the normal UI because they are implementation tuning knobs rather than
    routine user controls.
    """

    complex_blob_min_coverage: float = 0.90
    complex_blob_max_gap_bins: int = 1
    complex_blob_candidate_guard_scale: float = 1.1
    complex_blob_axis_guard_scale: float = 1.2
    complex_blob_target_rms_mm: float = 3.0
    complex_blob_min_dominant_bin_coverage: float = 0.75
    complex_blob_max_nondominant_run_bins: int = 1
    complex_blob_max_route_count: int = 6
    complex_blob_max_seed_count: int = 16
    complex_seed_shortlist_min: int = 8
    complex_seed_shortlist_multiplier: int = 3
    complex_seed_rank_good_percentile: float = 75.0
    complex_seed_rank_preferred_percentile: float = 85.0
    complex_seed_rank_tolerated_percentile: float = 92.0
    complex_seed_preferred_annulus_percentile: float = field(
        default=85.0,
        metadata=_ui_meta(
            "Seed annulus preferred percentile",
            minimum=0.0,
            maximum=100.0,
            decimals=1,
            suffix=" pct",
            advanced=True,
            description="Complex-blob seed shortlist preference threshold for annulus percentile.",
        ),
    )
    complex_seed_min_head_depth_mm: float = field(
        default=2.0,
        metadata=_ui_meta(
            "Seed minimum head depth",
            minimum=0.0,
            maximum=40.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description="Complex-blob seeds below this head depth are deprioritized.",
        ),
    )
    extension_bin_mm: float = 1.0
    extension_max_gap_bins: int = 2
    extension_support_radius_mm: float = 2.5
    extension_outward_search_mm: float = 120.0
    extension_inward_search_mm: float = 40.0
    extension_thick_run_bins: int = 3
    overlap_angle_deg_thresh: float = 8.0
    overlap_lateral_mm_thresh: float = 3.0


@dataclass(frozen=True)
class DeepCoreModelFitConfig:
    """Phase B: trajectory reconstruction from candidate proposals.

    The model_fit stage takes proposals from candidate generation (seeded
    in the clean deep_core region) and reconstructs each one into a
    trajectory by: (1) refining the axis via PCA line-fit on the
    proposal's atom point cloud, (2) reabsorbing any dropped colinear
    atoms from the full atom pool, (3) extending the line along the raw
    metal mask past the deep_core shrink boundary, (4) detecting bolt
    onset via metal cross-section widening, (5) classifying tissue at
    lateral offsets from the axis (air / brain / bone / metal) to locate
    the bone↔brain interface, (6) applying rejection gates, and (7)
    resolving group conflicts across accepted trajectories.

    The library of electrode models acts as a recognizer (soft span
    gate), not a generator. Contact placement is handled by a later
    stage, not here.
    """

    enabled: bool = field(
        default=True,
        metadata=_ui_meta(
            "Enable Phase B",
            minimum=0,
            maximum=1,
            control="bool",
            description="Run the trajectory reconstruction stage after annulus rejection.",
        ),
    )
    use_bolt_detection: bool = field(
        default=False,
        metadata=_ui_meta(
            "Use bolt detection",
            minimum=0,
            maximum=1,
            control="bool",
            description=(
                "When enabled, bolt candidates from the bolt_detection stage"
                " are converted into synthetic proposals (bolt_bridged or"
                " bolt_only) and merged with the atoms-only proposals coming"
                " from Phase A. Bolt-sourced proposals get priority in the"
                " claim-based assignment."
            ),
        ),
    )
    bolt_endpoint_offset_mm: float = field(
        default=8.0,
        metadata=_ui_meta(
            "Bolt endpoint offset",
            minimum=0.0,
            maximum=30.0,
            decimals=1,
            suffix=" mm",
            advanced=True,
            description=(
                "For bolt-seeded proposals, the shallow intracranial endpoint"
                " is placed this far inward of the bolt center along the"
                " fit axis. Replaces the per-subject head_distance threshold"
                " calibration."
            ),
        ),
    )
    bolt_bridge_radial_tol_mm: float = field(
        default=2.5,
        metadata=_ui_meta(
            "Bolt bridge radial tol",
            minimum=0.5,
            maximum=6.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description=(
                "Maximum perpendicular distance from the bolt axis for a"
                " support atom to be considered colinear and reabsorbed into"
                " a bolt-bridged proposal."
            ),
        ),
    )
    families: tuple[str, ...] = field(
        default=("DIXI",),
        metadata={"ui": False},
    )

    # --- Axis refinement -------------------------------------------------
    axis_fit_max_residual_mm: float = field(
        default=1.8,
        metadata=_ui_meta(
            "Axis fit max residual",
            minimum=0.1,
            maximum=5.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Reject proposal if PCA line-fit median radial residual exceeds this.",
        ),
    )

    # --- Colinear atom reabsorption --------------------------------------
    reabsorb_radial_tol_mm: float = field(
        default=1.5,
        metadata=_ui_meta(
            "Reabsorb radial tolerance",
            minimum=0.1,
            maximum=5.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Max perpendicular distance from refined axis for an atom to be reabsorbed.",
        ),
    )
    reabsorb_angle_tol_deg: float = field(
        default=5.0,
        metadata=_ui_meta(
            "Reabsorb angle tolerance",
            minimum=0.0,
            maximum=30.0,
            decimals=1,
            suffix=" deg",
            advanced=True,
            description="Max angular deviation between reabsorbed atom axis and refined proposal axis.",
        ),
    )
    reabsorb_axial_window_mm: float = field(
        default=5.0,
        metadata=_ui_meta(
            "Reabsorb axial window",
            minimum=0.0,
            maximum=30.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Max axial distance beyond existing cluster extent for atom reabsorption.",
        ),
    )

    # --- Metal-mask extension --------------------------------------------
    extension_step_mm: float = field(
        default=0.5,
        metadata=_ui_meta(
            "Extension step",
            minimum=0.1,
            maximum=5.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Axial step size when walking the axis outward to extend via metal mask.",
        ),
    )
    extension_tube_radius_mm: float = field(
        default=1.5,
        metadata=_ui_meta(
            "Extension tube radius",
            minimum=0.1,
            maximum=5.0,
            decimals=2,
            suffix=" mm",
            advanced=True,
            description="Perpendicular sampling tube radius used to detect metal during extension.",
        ),
    )
    extension_max_gap_mm: float = field(
        default=3.0,
        metadata=_ui_meta(
            "Extension max gap",
            minimum=0.0,
            maximum=20.0,
            decimals=2,
            suffix=" mm",
            description="Dark-interval tolerance allowed inside a continuing extension run.",
        ),
    )
    extension_termination_gap_mm: float = field(
        default=10.0,
        metadata=_ui_meta(
            "Extension termination gap",
            minimum=0.0,
            maximum=20.0,
            decimals=2,
            suffix=" mm",
            description="Sustained empty run that terminates extension on a given end.",
        ),
    )
    intracranial_exit_head_distance_mm: float = field(
        default=15.0,
        metadata=_ui_meta(
            "Intracranial exit head distance",
            minimum=0.0,
            maximum=30.0,
            decimals=2,
            suffix=" mm",
            description="Head-distance threshold marking the shallow intracranial end (burr hole / dura / cortex exit).",
        ),
    )
    scalp_exit_detect_head_distance_mm: float = field(
        default=5.0,
        metadata=_ui_meta(
            "Scalp-exit detect head distance",
            minimum=0.0,
            maximum=20.0,
            decimals=2,
            suffix=" mm",
            description="A metal-mask walk is considered to have reached the scalp if its sampled head_distance dropped below this threshold (rescues sparse-mask bolts).",
        ),
    )
    extension_head_distance_floor_mm: float = field(
        default=-1.0,
        metadata=_ui_meta(
            "Extension head-distance floor",
            minimum=-20.0,
            maximum=5.0,
            decimals=2,
            suffix=" mm",
            description="Extension stops when head_distance at the axis drops below this (crossing past the scalp into external air).",
        ),
    )

    # --- Lateral-HU tissue classification (for bone↔brain interface) -----
    lateral_hu_ring_radius_mm: float = field(
        default=3.5,
        metadata=_ui_meta(
            "Lateral-HU ring radius",
            minimum=1.0,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            description="Perpendicular offset where HU is sampled to classify tissue around the axis.",
        ),
    )
    lateral_hu_ring_samples: int = field(
        default=8,
        metadata=_ui_meta(
            "Lateral-HU ring samples",
            minimum=3,
            maximum=32,
            control="int",
            advanced=True,
            description="Number of equally spaced points on the lateral HU ring.",
        ),
    )
    hu_air_max: float = field(
        default=-500.0,
        metadata=_ui_meta(
            "HU air max",
            minimum=-1200.0,
            maximum=0.0,
            decimals=1,
            suffix=" HU",
            advanced=True,
            description="Upper HU bound for the air class (sinus, external).",
        ),
    )
    hu_brain_max: float = field(
        default=150.0,
        metadata=_ui_meta(
            "HU brain max",
            minimum=0.0,
            maximum=500.0,
            decimals=1,
            suffix=" HU",
            description="Upper HU bound for brain / soft-tissue / CSF class.",
        ),
    )
    hu_bone_max: float = field(
        default=1800.0,
        metadata=_ui_meta(
            "HU bone max",
            minimum=200.0,
            maximum=4000.0,
            decimals=1,
            suffix=" HU",
            description="Upper HU bound for bone class (above this is metal-contaminated).",
        ),
    )
    hu_classification_smoothing: int = field(
        default=3,
        metadata=_ui_meta(
            "HU classification smoothing",
            minimum=1,
            maximum=11,
            control="int",
            advanced=True,
            description="Consecutive-sample agreement window used when smoothing the tissue class walk.",
        ),
    )

    # --- Rejection gates -------------------------------------------------
    in_brain_min_depth_mm: float = field(
        default=10.0,
        metadata=_ui_meta(
            "In-brain min depth",
            minimum=0.0,
            maximum=50.0,
            decimals=2,
            suffix=" mm",
            description="Minimum hull distance (distance from air) for a sample to be treated as deep-in-brain.",
        ),
    )
    min_intracranial_span_mm: float = field(
        default=15.0,
        metadata=_ui_meta(
            "Minimum intracranial span",
            minimum=0.0,
            maximum=80.0,
            decimals=2,
            suffix=" mm",
            description="Minimum contiguous brain-surrounded length required to accept a trajectory.",
        ),
    )
    library_span_tolerance_mm: float = field(
        default=5.0,
        metadata=_ui_meta(
            "Library span tolerance",
            minimum=0.0,
            maximum=30.0,
            decimals=2,
            suffix=" mm",
            description="Tolerance around each library model span used as the soft recognizer gate.",
        ),
    )

    # --- Group conflict --------------------------------------------------
    axis_conflict_radius_mm: float = field(
        default=1.5,
        metadata=_ui_meta(
            "Axis conflict radius",
            minimum=0.0,
            maximum=10.0,
            decimals=2,
            suffix=" mm",
            description="Perpendicular radius within which two trajectories are considered to conflict.",
        ),
    )


@dataclass(frozen=True)
class DeepCoreBoltConfig:
    """RANSAC-based bolt detection parameters (upstream of Phase B)."""

    enabled: bool = field(
        default=True,
        metadata=_ui_meta(
            "Bolt detection enabled",
            minimum=0,
            maximum=1,
            control="bool",
            description="Run the RANSAC bolt detection stage before Phase B.",
        ),
    )
    span_min_mm: float = field(
        default=10.0,
        metadata=_ui_meta(
            "Bolt span min", minimum=1.0, maximum=30.0, decimals=1,
            suffix=" mm", description="Minimum axial span for a bolt candidate.",
        ),
    )
    span_max_mm: float = field(
        default=40.0,
        metadata=_ui_meta(
            "Bolt span max", minimum=5.0, maximum=80.0, decimals=1,
            suffix=" mm", description="Maximum axial span for a bolt candidate.",
        ),
    )
    inlier_tol_mm: float = field(
        default=1.5,
        metadata=_ui_meta(
            "Bolt inlier tolerance", minimum=0.5, maximum=5.0, decimals=2,
            suffix=" mm", description="Perpendicular tolerance for RANSAC inliers.",
        ),
    )
    min_inliers: int = field(
        default=15,
        metadata=_ui_meta(
            "Bolt min inliers", minimum=5, maximum=200, control="int",
            description="Minimum voxel count for an accepted line.",
        ),
    )
    fill_frac_min: float = field(
        default=0.80,
        metadata=_ui_meta(
            "Bolt fill fraction", minimum=0.3, maximum=1.0, decimals=2,
            description=("Fraction of sampled points along the span that must"
                         " have a supporting metal voxel."),
        ),
    )
    max_gap_mm: float = field(
        default=3.0,
        metadata=_ui_meta(
            "Bolt max gap", minimum=0.5, maximum=10.0, decimals=1,
            suffix=" mm", description="Longest allowed continuous gap along the span.",
        ),
    )
    shell_min_mm: float = field(
        default=-5.0,
        metadata=_ui_meta(
            "Bolt shell min", minimum=-20.0, maximum=10.0, decimals=1,
            suffix=" mm", description="Lower bound on head_distance at the bolt center.",
        ),
    )
    shell_max_mm: float = field(
        default=35.0,
        metadata=_ui_meta(
            "Bolt shell max", minimum=5.0, maximum=60.0, decimals=1,
            suffix=" mm", description="Upper bound on head_distance at the bolt center.",
        ),
    )
    axis_depth_delta_mm: float = field(
        default=8.0,
        metadata=_ui_meta(
            "Bolt axis-depth delta", minimum=0.0, maximum=30.0, decimals=1,
            suffix=" mm",
            description=("Minimum increase in head_distance when probing inward"
                         " along the bolt axis. Rejects lines that run along the skin."),
        ),
    )
    support_overlap_frac: float = field(
        default=0.70,
        metadata=_ui_meta(
            "Bolt support overlap", minimum=0.3, maximum=1.0, decimals=2,
            description=("Drop a candidate whose support voxels are this fraction"
                         " contained in another accepted candidate."),
        ),
    )
    collinear_angle_deg: float = field(
        default=10.0,
        metadata=_ui_meta(
            "Bolt collinear angle", minimum=0.0, maximum=30.0, decimals=1,
            suffix=" deg",
            description=("Maximum axis angle (in degrees) for a collinear-dedup"
                         " match when the two candidates share an electrode."),
        ),
    )
    collinear_perp_mm: float = field(
        default=5.0,
        metadata=_ui_meta(
            "Bolt collinear perp", minimum=0.0, maximum=15.0, decimals=1,
            suffix=" mm",
            description="Maximum perpendicular distance for a collinear-dedup match.",
        ),
    )
    max_lines: int = field(
        default=40,
        metadata=_ui_meta(
            "Bolt max lines", minimum=5, maximum=200, control="int",
            description="Cap on the number of RANSAC lines to extract.",
        ),
    )
    n_samples: int = field(
        default=4000,
        metadata=_ui_meta(
            "Bolt RANSAC samples", minimum=200, maximum=20000, control="int",
            description="RANSAC hypothesis count per iteration.",
        ),
    )


@dataclass(frozen=True)
class DeepCoreConfig:
    """Top-level Deep Core configuration."""

    mask: DeepCoreMaskConfig = field(default_factory=DeepCoreMaskConfig)
    support: DeepCoreSupportConfig = field(default_factory=DeepCoreSupportConfig)
    proposal: DeepCoreProposalConfig = field(default_factory=DeepCoreProposalConfig)
    annulus: DeepCoreAnnulusConfig = field(default_factory=DeepCoreAnnulusConfig)
    internal: DeepCoreInternalConfig = field(default_factory=DeepCoreInternalConfig)
    model_fit: DeepCoreModelFitConfig = field(default_factory=DeepCoreModelFitConfig)
    bolt: DeepCoreBoltConfig = field(default_factory=DeepCoreBoltConfig)

    def with_updates(self, updates: dict[str, Any]) -> "DeepCoreConfig":
        """Return a copy with dotted-path updates applied."""

        section_updates: dict[str, dict[str, Any]] = {}
        for path, value in dict(updates or {}).items():
            section_name, _, field_name = str(path).partition(".")
            if not section_name or not field_name:
                raise ValueError(f"Invalid Deep Core config path: {path}")
            section_updates.setdefault(section_name, {})[field_name] = value
        out = self
        for section_name, values in section_updates.items():
            section_obj = getattr(out, section_name)
            out = replace(out, **{section_name: replace(section_obj, **values)})
        return out

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested config sections for logging/debugging."""

        out: dict[str, Any] = {}
        for section_name in ("mask", "support", "proposal", "annulus", "internal", "model_fit", "bolt"):
            section_obj = getattr(self, section_name)
            for f in fields(section_obj):
                out[f"{section_name}.{f.name}"] = getattr(section_obj, f.name)
        return out


@dataclass(frozen=True)
class DeepCoreUiFieldSpec:
    """Resolved UI description for one Deep Core config field."""

    path: str
    label: str
    minimum: float | int
    maximum: float | int
    decimals: int
    suffix: str
    advanced: bool
    description: str
    control: str
    default: Any


_UI_FIELD_ORDER = (
    "mask.hull_threshold_hu",
    "mask.hull_clip_hu",
    "mask.hull_sigma_mm",
    "mask.hull_open_vox",
    "mask.hull_close_vox",
    "mask.deep_core_shrink_mm",
    "mask.metal_threshold_hu",
    "mask.bolt_metal_threshold_hu",
    "mask.metal_grow_vox",
    "support.support_spacing_mm",
    "support.component_min_elongation",
    "support.line_atom_diameter_max_mm",
    "support.line_atom_min_span_mm",
    "support.line_atom_min_pca_dominance",
    "support.contact_component_diameter_max_mm",
    "support.support_cube_size_mm",
    "annulus.pre_extension_annulus_reject_percentile",
    "annulus.cross_section_annulus_inner_mm",
    "annulus.cross_section_annulus_outer_mm",
    "annulus.profile_remaining_brain_fraction_req",
    "annulus.profile_simple_middle_nonbrain_max_run_mm",
    "annulus.profile_multi_middle_nonbrain_max_run_mm",
    "annulus.profile_endpoint_margin_mm",
    "proposal.outward_support_check_span_mm",
    "proposal.outward_support_radius_mm",
    "proposal.outward_support_search_mm",
    "proposal.outward_support_min_extension_mm",
    "proposal.outward_support_min_depth_gain_mm",
    "internal.complex_seed_preferred_annulus_percentile",
    "internal.complex_seed_min_head_depth_mm",
    "model_fit.use_bolt_detection",
    "model_fit.bolt_bridge_radial_tol_mm",
    "model_fit.bolt_endpoint_offset_mm",
    "bolt.enabled",
    "bolt.span_min_mm",
    "bolt.span_max_mm",
    "bolt.inlier_tol_mm",
    "bolt.min_inliers",
    "bolt.fill_frac_min",
    "bolt.max_gap_mm",
    "bolt.shell_min_mm",
    "bolt.shell_max_mm",
    "bolt.axis_depth_delta_mm",
    "bolt.support_overlap_frac",
    "bolt.collinear_angle_deg",
    "bolt.collinear_perp_mm",
    "bolt.max_lines",
    "bolt.n_samples",
)


def _resolve_field(path: str) -> tuple[str, Any]:
    config = DeepCoreConfig()
    section_name, _, field_name = str(path).partition(".")
    if not section_name or not field_name:
        raise KeyError(path)
    section_obj = getattr(config, section_name)
    for f in fields(section_obj):
        if f.name == field_name:
            return section_name, f
    raise KeyError(path)


def deep_core_ui_specs(*, advanced: bool | None = None) -> list[DeepCoreUiFieldSpec]:
    """Return UI field specs in a stable order."""

    config = DeepCoreConfig()
    specs: list[DeepCoreUiFieldSpec] = []
    for path in _UI_FIELD_ORDER:
        section_name, f = _resolve_field(path)
        meta = dict(f.metadata or {})
        if not bool(meta.get("ui")):
            continue
        is_advanced = bool(meta.get("advanced", False))
        if advanced is not None and bool(advanced) != is_advanced:
            continue
        section_obj = getattr(config, section_name)
        specs.append(
            DeepCoreUiFieldSpec(
                path=str(path),
                label=str(meta.get("label") or f.name),
                minimum=meta.get("minimum"),
                maximum=meta.get("maximum"),
                decimals=int(meta.get("decimals", 0)),
                suffix=str(meta.get("suffix", "")),
                advanced=is_advanced,
                description=str(meta.get("description", "")),
                control=str(meta.get("control", "double")),
                default=getattr(section_obj, f.name),
            )
        )
    return specs


def deep_core_default_config() -> DeepCoreConfig:
    """Return a fresh Deep Core config with module defaults."""

    return DeepCoreConfig()
