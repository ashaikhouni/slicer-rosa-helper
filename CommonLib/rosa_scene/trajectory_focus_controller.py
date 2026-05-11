"""Shared trajectory-focus controller for slice navigation and highlighting."""

from rosa_core import lps_to_ras_point

from .electrode_scene import ElectrodeSceneService
from .layout_service import LayoutService
from .trajectory_scene import TrajectorySceneService


class TrajectoryFocusController:
    """Coordinate trajectory highlight and slice focusing across modules."""

    def __init__(self, trajectory_scene=None, electrode_scene=None, layout_service=None):
        """Wire reusable scene/layout coordinators used by focus interactions."""
        self.trajectory_scene = trajectory_scene or TrajectorySceneService()
        self.electrode_scene = electrode_scene or ElectrodeSceneService()
        self.layout_service = layout_service or LayoutService()

    def focus_selected(
        self,
        trajectory,
        scope_node_ids=None,
        jump_cardinal=True,
        align_focus_views=True,
        focus="entry",
        placed_contacts_ras=None,
    ):
        """Highlight one trajectory and optionally move slice views to it.

        Args:
            placed_contacts_ras: optional list of (x, y, z) RAS contact
                positions. When supplied with ≥2 contacts, the slice
                long-axis is refit (PCA) through them so the focus slice
                plane passes through the actual implant rather than the
                seed line. Mitigates the "centerline drift" issue where
                slight placer-side drift or mild electrode bend causes
                contacts to sit 1-2 mm off the seed line. For curved
                electrodes the PCA fit is the best straight-line
                approximation; some contacts may still sit off the slice
                plane and require scrolling.
        """
        selected_id = ""
        if trajectory is not None:
            selected_id = str(trajectory.get("node_id", "") or "")
        self.trajectory_scene.highlight_selected_trajectory(
            selected_node_id=selected_id,
            scope_node_ids=scope_node_ids or [],
        )

        if trajectory is None:
            return False

        start_ras = lps_to_ras_point(trajectory["start"])
        end_ras = lps_to_ras_point(trajectory["end"])

        # When the user has already placed contacts, refit the slice axis
        # through them so the focus views show the implant. Falls back to
        # the seed line silently when no contacts are supplied or the fit
        # degenerates (single point / co-located).
        if placed_contacts_ras and len(placed_contacts_ras) >= 2:
            refit = _refit_axis_through_contacts(
                placed_contacts_ras,
                fallback_start=start_ras, fallback_end=end_ras,
            )
            if refit is not None:
                start_ras, end_ras = refit

        if bool(jump_cardinal):
            self.electrode_scene.jump_slice_views_to_point(start_ras, slice_views=("Red", "Yellow", "Green"))

        if bool(align_focus_views) and self.layout_service.has_focus_slice_views():
            self.electrode_scene.align_slice_to_trajectory(
                start_ras=start_ras,
                end_ras=end_ras,
                slice_view=self.layout_service.TRAJECTORY_LONG_VIEW,
                mode="long",
                focus=focus,
            )
            self.electrode_scene.align_slice_to_trajectory(
                start_ras=start_ras,
                end_ras=end_ras,
                slice_view=self.layout_service.TRAJECTORY_DOWN_VIEW,
                mode="down",
                focus=focus,
            )
        return True


def _refit_axis_through_contacts(contacts_ras, *, fallback_start, fallback_end):
    """PCA-fit a straight line through ``contacts_ras`` and return
    ``(start_ras, end_ras)`` spanning the contact range along the fit.

    The principal direction is sign-flipped to match the seed direction
    (``fallback_end - fallback_start``) so the bolt-side end stays the
    "start" — keeps slice ``focus="entry"`` semantics meaningful.

    Returns ``None`` when the fit degenerates (collinear input, all
    coincident points, etc.) — caller falls back to the seed line.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    pts = np.asarray(contacts_ras, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return None
    cm = pts.mean(axis=0)
    centered = pts - cm
    if not np.any(np.linalg.norm(centered, axis=1) > 1e-6):
        return None
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vh[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return None
    axis = axis / axis_norm
    seed_dir = np.asarray(fallback_end, dtype=float) - np.asarray(fallback_start, dtype=float)
    if float(np.linalg.norm(seed_dir)) > 1e-9 and float(np.dot(axis, seed_dir)) < 0.0:
        axis = -axis
    proj = centered @ axis
    return (
        (cm + axis * float(proj.min())).tolist(),
        (cm + axis * float(proj.max())).tolist(),
    )
