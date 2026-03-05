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
    ):
        """Highlight one trajectory and optionally move slice views to it."""
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
