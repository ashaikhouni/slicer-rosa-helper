"""Slicer scene helpers for grouped trajectory line markups."""

from __main__ import slicer, vtk

from rosa_core import lps_to_ras_point


TRAJECTORY_GROUP_CONFIG = {
    "planned_rosa": {
        "role": "PlannedTrajectoryLines",
        "folder": "Planned",
        "prefix": "Plan_",
        "display_color": (0.65, 0.65, 0.65),
        "selected_color": (0.65, 0.65, 0.65),
        "line_thickness": 0.35,
        "locked": True,
        "point_labels": False,
    },
    "imported_rosa": {
        "role": "ImportedTrajectoryLines",
        "folder": "ImportedROSA",
        "prefix": "",
        "display_color": (0.15, 0.65, 1.0),
        "selected_color": (0.35, 0.8, 1.0),
        "line_thickness": 0.5,
        "locked": False,
        "point_labels": True,
    },
    "manual": {
        "role": "ManualTrajectoryLines",
        "folder": "Manual",
        "prefix": "",
        "display_color": (1.0, 0.95, 0.1),
        "selected_color": (1.0, 0.9, 0.3),
        "line_thickness": 0.5,
        "locked": False,
        "point_labels": True,
    },
    "guided_fit": {
        "role": "GuidedFitTrajectoryLines",
        "folder": "GuidedFit",
        "prefix": "Guided_",
        "display_color": (0.95, 0.35, 0.8),
        "selected_color": (1.0, 0.55, 0.9),
        "line_thickness": 0.6,
        "locked": False,
        "point_labels": True,
    },
    "de_novo": {
        "role": "DeNovoTrajectoryLines",
        "folder": "DeNovo",
        "prefix": "DeNovo_",
        "display_color": (0.2, 0.95, 0.45),
        "selected_color": (0.45, 1.0, 0.6),
        "line_thickness": 0.55,
        "locked": False,
        "point_labels": True,
    },
    "autofit_preview": {
        "role": "",
        "folder": "Preview",
        "prefix": "AutoFit_",
        "display_color": (0.2, 0.8, 1.0),
        "selected_color": (0.2, 0.8, 1.0),
        "line_thickness": 0.5,
        "locked": False,
        "point_labels": False,
    },
}

DEFAULT_GROUP = "manual"


class TrajectorySceneService:
    """Read/write helpers for grouped trajectory line markups in the active scene."""

    def _normalize_group(self, group):
        key = str(group or DEFAULT_GROUP).strip().lower()
        return key if key in TRAJECTORY_GROUP_CONFIG else DEFAULT_GROUP

    def group_role(self, group):
        cfg = TRAJECTORY_GROUP_CONFIG.get(self._normalize_group(group), {})
        return cfg.get("role", "")

    def build_node_name(self, trajectory_name, group):
        """Return display node name for a trajectory logical name + group."""
        name = str(trajectory_name or "").strip()
        cfg = TRAJECTORY_GROUP_CONFIG.get(self._normalize_group(group), {})
        return f"{cfg.get('prefix', '')}{name}"

    def infer_group_from_node(self, node):
        """Infer trajectory group from explicit node attributes or known prefixes."""
        if node is None:
            return DEFAULT_GROUP
        attr = (node.GetAttribute("Rosa.TrajectoryGroup") or "").strip().lower()
        if attr in TRAJECTORY_GROUP_CONFIG:
            return attr
        name = (node.GetName() or "").strip()
        if name.startswith("Plan_"):
            return "planned_rosa"
        if name.startswith("Guided_"):
            return "guided_fit"
        if name.startswith("DeNovo_"):
            return "de_novo"
        if name.startswith("AutoFit_"):
            return "autofit_preview"
        return DEFAULT_GROUP

    def logical_name_from_node(self, node):
        """Return logical trajectory name (without group prefixes) for one node."""
        if node is None:
            return ""
        attr_name = (node.GetAttribute("Rosa.TrajectoryName") or "").strip()
        if attr_name:
            return attr_name
        name = (node.GetName() or "").strip()
        group = self.infer_group_from_node(node)
        prefix = TRAJECTORY_GROUP_CONFIG.get(group, {}).get("prefix", "")
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
        return name

    def find_line_markup_node(self, name):
        """Return trajectory line markup node by exact display name, or None."""
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetName() == name:
                return node
        return None

    def find_line_by_group_and_name(self, trajectory_name, group):
        """Return grouped trajectory node by logical name, or None."""
        target_group = self._normalize_group(group)
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if self.infer_group_from_node(node) != target_group:
                continue
            if self.logical_name_from_node(node) == trajectory_name:
                return node
        return None

    def _apply_group_display(self, node, group):
        cfg = TRAJECTORY_GROUP_CONFIG.get(self._normalize_group(group), TRAJECTORY_GROUP_CONFIG[DEFAULT_GROUP])
        node.SetLocked(bool(cfg.get("locked", False)))
        display = node.GetDisplayNode()
        if display is None:
            return
        color = cfg.get("display_color", (1.0, 1.0, 0.0))
        sel_color = cfg.get("selected_color", color)
        display.SetColor(float(color[0]), float(color[1]), float(color[2]))
        display.SetSelectedColor(float(sel_color[0]), float(sel_color[1]), float(sel_color[2]))
        display.SetLineThickness(float(cfg.get("line_thickness", 0.5)))
        display.SetVisibility(True)
        if hasattr(display, "SetPointLabelsVisibility"):
            display.SetPointLabelsVisibility(bool(cfg.get("point_labels", True)))
        if hasattr(display, "SetPropertiesLabelVisibility"):
            display.SetPropertiesLabelVisibility(False)

    def set_group_visibility(self, group, visible):
        """Set visibility for all trajectories in one group."""
        grp = self._normalize_group(group)
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if self.infer_group_from_node(node) != grp:
                continue
            display = node.GetDisplayNode()
            if display:
                display.SetVisibility(bool(visible))

    def show_only_groups(self, groups):
        """Hide all trajectory groups except the provided one(s)."""
        keep = {self._normalize_group(g) for g in (groups or [])}
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            group = self.infer_group_from_node(node)
            if group == "autofit_preview":
                continue
            display = node.GetDisplayNode()
            if display:
                display.SetVisibility(group in keep)

    def set_trajectory_metadata(self, node, trajectory_name, group, origin=""):
        """Stamp standard trajectory metadata on one line node."""
        if node is None:
            return
        grp = self._normalize_group(group)
        node.SetAttribute("Rosa.Managed", "1")
        node.SetAttribute("Rosa.TrajectoryName", str(trajectory_name or ""))
        node.SetAttribute("Rosa.TrajectoryGroup", grp)
        node.SetAttribute("Rosa.TrajectoryOrigin", str(origin or grp))

    def trajectory_from_line_node(self, name, node):
        """Extract one line node as a trajectory dictionary in ROSA/LPS coordinates."""
        if node is None or node.GetNumberOfControlPoints() < 2:
            return None
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        logical_name = (str(name or "").strip() or self.logical_name_from_node(node)).strip()
        group = self.infer_group_from_node(node)
        return {
            "name": logical_name,
            "node_name": node.GetName() or logical_name,
            "node_id": node.GetID(),
            "group": group,
            "start": lps_to_ras_point(p0),
            "end": lps_to_ras_point(p1),
        }

    def collect_planned_trajectory_map(self):
        """Return planned trajectory map as `name -> trajectory` in ROSA/LPS."""
        out = {}
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if self.infer_group_from_node(node) != "planned_rosa":
                continue
            name = self.logical_name_from_node(node)
            traj = self.trajectory_from_line_node(name, node)
            if traj is not None:
                out[name] = traj
        return out

    def build_trajectory_map_with_scene_overrides(self, base_trajectories):
        """Overlay in-scene line edits on top of trajectory dictionaries."""
        out = {}
        for traj in base_trajectories or []:
            name = traj.get("name", "")
            node = None
            node_id = traj.get("node_id", "")
            if node_id:
                node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                node_name = traj.get("node_name", "")
                if node_name:
                    node = self.find_line_markup_node(node_name)
            if node is None and name:
                node = self.find_line_markup_node(name)
            scene_traj = self.trajectory_from_line_node(name, node)
            out[name] = scene_traj if scene_traj is not None else traj
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
        self.set_trajectory_metadata(node, trajectory_name=trajectory_name, group="autofit_preview", origin="preview")
        self._apply_group_display(node, "autofit_preview")

    def remove_preview_lines(self, trajectory_names=None, node_prefix="AutoFit_"):
        """Remove preview line markups from scene."""
        nodes = list(slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"))
        if trajectory_names is None:
            for node in nodes:
                if self.infer_group_from_node(node) == "autofit_preview":
                    slicer.mrmlScene.RemoveNode(node)
                    continue
                node_name = (node.GetName() or "").lower()
                if node_name.startswith(node_prefix.lower()):
                    slicer.mrmlScene.RemoveNode(node)
            return

        expected = {f"{node_prefix}{name}".lower() for name in trajectory_names}
        for node in nodes:
            node_name = (node.GetName() or "").lower()
            if node_name in expected:
                slicer.mrmlScene.RemoveNode(node)

    def _ensure_subject_hierarchy_folder(self, parent_item_id, folder_name):
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return 0
        if hasattr(sh_node, "GetItemChildWithName"):
            existing = sh_node.GetItemChildWithName(parent_item_id, folder_name)
            if existing:
                return existing
        return sh_node.CreateFolderItem(parent_item_id, folder_name)

    def place_trajectory_nodes_in_hierarchy(self, context_id, nodes):
        """Place grouped trajectory nodes under `RosaWorkflow_<id>/Trajectories/<Group>/`."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return
        scene_item = sh_node.GetSceneItemID()
        root = self._ensure_subject_hierarchy_folder(scene_item, f"RosaWorkflow_{context_id}")
        traj_root = self._ensure_subject_hierarchy_folder(root, "Trajectories")
        group_folders = {}
        for node in nodes or []:
            if node is None:
                continue
            group = self.infer_group_from_node(node)
            folder_name = TRAJECTORY_GROUP_CONFIG.get(group, {}).get("folder", group.title())
            folder_id = group_folders.get(folder_name)
            if not folder_id:
                folder_id = self._ensure_subject_hierarchy_folder(traj_root, folder_name)
                group_folders[folder_name] = folder_id
            item = sh_node.GetItemByDataNode(node)
            if not item:
                # Some nodes may not have a hierarchy item yet at first publish.
                try:
                    item = sh_node.CreateItem(folder_id, node)
                except Exception:
                    item = sh_node.GetItemByDataNode(node)
            if item:
                sh_node.SetItemParent(item, folder_id)

    def create_or_update_trajectory_line(
        self,
        name,
        start_ras,
        end_ras,
        node_id=None,
        group=DEFAULT_GROUP,
        origin="",
        node_name=None,
    ):
        """Create/update one grouped trajectory line node."""
        grp = self._normalize_group(group)
        node = None
        if node_id:
            node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None and node_name:
            node = self.find_line_markup_node(node_name)
        if node is None:
            if not node_name:
                node_name = self.build_node_name(name, grp)
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", node_name)
            node.CreateDefaultDisplayNodes()
        elif node_name:
            node.SetName(node_name)

        node.RemoveAllControlPoints()
        node.AddControlPoint(vtk.vtkVector3d(*start_ras))
        node.AddControlPoint(vtk.vtkVector3d(*end_ras))
        node.SetNthControlPointLabel(0, f"{name}_start")
        node.SetNthControlPointLabel(1, f"{name}_end")
        self.set_trajectory_metadata(node, trajectory_name=name, group=grp, origin=origin or grp)
        self._apply_group_display(node, grp)
        return node

    def collect_working_trajectory_rows(self, groups=None):
        """Return sorted trajectory rows from scene, optionally filtered by group list."""
        group_filter = None
        if groups:
            group_filter = {self._normalize_group(g) for g in groups}
        rows = []
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetNumberOfControlPoints() < 2:
                continue
            group = self.infer_group_from_node(node)
            if group == "autofit_preview":
                continue
            if group == "planned_rosa" and group_filter is None:
                continue
            if group_filter is not None and group not in group_filter:
                continue
            name = self.logical_name_from_node(node)
            if not name:
                continue
            p0 = [0.0, 0.0, 0.0]
            p1 = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(0, p0)
            node.GetNthControlPointPositionWorld(1, p1)
            rows.append(
                {
                    "name": name,
                    "node_name": node.GetName() or name,
                    "node_id": node.GetID(),
                    "group": group,
                    "start_ras": [float(p0[0]), float(p0[1]), float(p0[2])],
                    "end_ras": [float(p1[0]), float(p1[1]), float(p1[2])],
                }
            )
        rows.sort(key=lambda item: (item.get("name", ""), item.get("group", "")))
        return rows
