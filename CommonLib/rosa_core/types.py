"""Shared type aliases and TypedDict schemas for rosa_core."""

from __future__ import annotations

from typing import TypedDict

Point3D = list[float]
Matrix4x4 = list[list[float]]


class TokenBlock(TypedDict):
    """Parsed token payload block from ROS text."""

    token: str
    content: str


class DisplayRecord(TypedDict, total=False):
    """Volume display transform record parsed from ROS."""

    volume: str
    volume_path: str | None
    matrix: Matrix4x4
    index: int
    imagery_name: str
    serie_uid: str
    imagery_3dref: int


class TrajectoryRecord(TypedDict):
    """Single planned trajectory in ROSA/LPS coordinates."""

    name: str
    start: Point3D
    end: Point3D


class RosParseResult(TypedDict, total=False):
    """Top-level parse output from `.ros` content or file."""

    displays: list[DisplayRecord]
    trajectories: list[TrajectoryRecord]
    ros_path: str


class AssignmentRow(TypedDict, total=False):
    """User-editable per-trajectory electrode assignment."""

    trajectory: str
    model_id: str
    tip_at: str
    tip_shift_mm: float
    xyz_offset_mm: Point3D


class AssignmentTemplate(TypedDict):
    """Assignment file schema used by contacts generation."""

    schema_version: str
    assignments: list[AssignmentRow]


class ContactRecord(TypedDict):
    """Generated contact center in ROSA/LPS coordinates."""

    trajectory: str
    model_id: str
    index: int
    label: str
    position_lps: Point3D
    tip_at: str


class FCSVRow(TypedDict):
    """Generic FCSV row payload used by exporters."""

    label: str
    xyz: Point3D


class QCMetricsRow(TypedDict):
    """Per-trajectory QC metrics row."""

    trajectory: str
    entry_radial_mm: float
    target_radial_mm: float
    mean_contact_radial_mm: float
    max_contact_radial_mm: float
    rms_contact_radial_mm: float
    angle_deg: float
    matched_contacts: int


class ContactFitResult(TypedDict, total=False):
    """Auto-fit result payload returned by contact_fit."""

    success: bool
    reason: str
    entry_lps: Point3D
    target_lps: Point3D
    tip_shift_mm: float
    lateral_shift_mm: float
    angle_deg: float
    residual_mm: float
    one_d_residual_mm: float
    points_in_roi: int
    slab_centroids: int
    slab_inliers: int
    ransac_rms_mm: float
    axis_lps: Point3D
    deep_axis_lps: Point3D
    center_lps: Point3D
    deep_t_raw_mm: float
    deep_t_clamped_mm: float
    planned_target_t_mm: float
    planned_tip_lps: Point3D
    planned_tip_axis_lps: Point3D
    slab_t_min_mm: float
    slab_t_max_mm: float
    slab_t_used_count: int


class ElectrodeModel(TypedDict, total=False):
    """Single model entry in electrode_models.json."""

    id: str
    type: str
    contact_count: int
    contact_length_mm: float
    diameter_mm: float
    total_exploration_length_mm: float
    contact_center_offsets_from_tip_mm: list[float]


class ElectrodeLibrary(TypedDict):
    """Electrode library JSON root object."""

    models: list[ElectrodeModel]
