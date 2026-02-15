"""Slicer scene helpers for trajectory markup nodes."""

from __main__ import slicer, vtk

from rosa_core import lps_to_ras_point


class TrajectorySceneService:
    """Read/write helpers for trajectory line markups in the active Slicer scene."""

    def find_line_markup_node(self, name):
        """Return trajectory line markup node by exact name, or None."""
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetName() == name:
                return node
        return None

    def trajectory_from_line_node(self, name, node):
        """Extract one line node as a trajectory dictionary in ROSA/LPS coordinates."""
        if node is None or node.GetNumberOfControlPoints() < 2:
            return None
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        return {
            "name": name,
            "start": lps_to_ras_point(p0),
            "end": lps_to_ras_point(p1),
        }

    def collect_planned_trajectory_map(self):
        """Return `Plan_*` line markups as `name -> trajectory` in ROSA/LPS."""
        out = {}
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            node_name = node.GetName() or ""
            if not node_name.startswith("Plan_"):
                continue
            traj_name = node_name[len("Plan_") :]
            traj = self.trajectory_from_line_node(traj_name, node)
            if traj is not None:
                out[traj_name] = traj
        return out

    def build_trajectory_map_with_scene_overrides(self, base_trajectories):
        """Overlay in-scene line edits on top of trajectory dictionaries."""
        out = {traj["name"]: traj for traj in base_trajectories}
        for name in list(out.keys()):
            node = self.find_line_markup_node(name)
            scene_traj = self.trajectory_from_line_node(name, node)
            if scene_traj is not None:
                out[name] = scene_traj
        return out

    def set_preview_line(self, trajectory_name, start_lps, end_lps, node_prefix="AutoFit_"):
        """Create/update one preview trajectory line in scene."""
        node_name = f"{node_prefix}{trajectory_name}"
        node = self.find_line_markup_node(node_name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", node_name)
            node.CreateDefaultDisplayNodes()
        start_ras = lps_to_ras_point(start_lps)
        end_ras = lps_to_ras_point(end_lps)
        node.RemoveAllControlPoints()
        node.AddControlPoint(vtk.vtkVector3d(*start_ras))
        node.AddControlPoint(vtk.vtkVector3d(*end_ras))
        display = node.GetDisplayNode()
        if display:
            display.SetSelectedColor(0.2, 0.8, 1.0)
            display.SetColor(0.2, 0.8, 1.0)
            display.SetLineThickness(0.5)

    def remove_preview_lines(self, trajectory_names=None, node_prefix="AutoFit_"):
        """Remove preview line markups from scene."""
        nodes = list(slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"))
        if trajectory_names is None:
            for node in nodes:
                node_name = (node.GetName() or "").lower()
                if node_name.startswith(node_prefix.lower()):
                    slicer.mrmlScene.RemoveNode(node)
            return

        expected = set()
        for name in trajectory_names:
            expected.add(f"{node_prefix}{name}".lower())

        for node in nodes:
            node_name = (node.GetName() or "").lower()
            if node_name in expected:
                slicer.mrmlScene.RemoveNode(node)
