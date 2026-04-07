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
        "line_thickness": 0.4,
        "locked": False,
        "point_labels": True,
    },
    "manual": {
        "role": "ManualTrajectoryLines",
        "folder": "Manual",
        "prefix": "",
        "display_color": (0.78, 0.42, 0.18),
        "selected_color": (0.88, 0.56, 0.30),
        "line_thickness": 0.4,
        "locked": False,
        "point_labels": True,
    },
    "guided_fit": {
        "role": "GuidedFitTrajectoryLines",
        "folder": "GuidedFit",
        "prefix": "Guided_",
        "display_color": (0.95, 0.35, 0.8),
        "selected_color": (1.0, 0.55, 0.9),
        "line_thickness": 0.45,
        "locked": False,
        "point_labels": True,
    },
    "de_novo": {
        "role": "DeNovoTrajectoryLines",
        "folder": "DeNovo",
        "prefix": "DeNovo_",
        "display_color": (0.2, 0.95, 0.45),
        "selected_color": (0.45, 1.0, 0.6),
        "line_thickness": 0.42,
        "locked": False,
        "point_labels": True,
    },
    "deep_core": {
        "role": "DeepCoreTrajectoryLines",
        "folder": "DeepCore",
        "prefix": "DeepCore_",
        "display_color": (0.10, 0.85, 0.95),
        "selected_color": (0.30, 0.95, 1.0),
        "line_thickness": 0.42,
        "locked": False,
        "point_labels": True,
    },
    "imported_external": {
        "role": "ImportedExternalTrajectoryLines",
        "folder": "ImportedExternal",
        "prefix": "Ext_",
        "display_color": (1.0, 0.65, 0.1),
        "selected_color": (1.0, 0.8, 0.35),
        "line_thickness": 0.42,
        "locked": False,
        "point_labels": True,
    },
    "autofit_preview": {
        "role": "",
        "folder": "Preview",
        "prefix": "AutoFit_",
        "display_color": (0.2, 0.8, 1.0),
        "selected_color": (0.2, 0.8, 1.0),
        "line_thickness": 0.4,
        "locked": False,
        "point_labels": False,
    },
}

DEFAULT_GROUP = "manual"
# Reserve yellow for active trajectory highlight only.
ACTIVE_HIGHLIGHT_COLOR = (1.0, 0.95, 0.10)


class TrajectorySceneService:
    """Read/write helpers for grouped trajectory line markups in the active scene."""

    def _normalize_group(self, group):
        """Normalize caller group key and fallback to default when unknown."""
        key = str(group or DEFAULT_GROUP).strip().lower()
        return key if key in TRAJECTORY_GROUP_CONFIG else DEFAULT_GROUP

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
        if name.startswith("DeepCore_"):
            return "deep_core"
        if name.startswith("Ext_"):
            return "imported_external"
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
        """Apply configured color/visibility/lock style for one trajectory group."""
        cfg = TRAJECTORY_GROUP_CONFIG.get(self._normalize_group(group), TRAJECTORY_GROUP_CONFIG[DEFAULT_GROUP])
        node.SetLocked(bool(cfg.get("locked", False)))
        # Compact labels: show trajectory name once on entry point plus E/T markers.
        traj_name = (node.GetAttribute("Rosa.TrajectoryName") or "").strip() or (node.GetName() or "").strip()
        if node.GetNumberOfControlPoints() >= 1:
            node.SetNthControlPointLabel(0, f"{traj_name} E" if traj_name else "E")
        if node.GetNumberOfControlPoints() >= 2:
            node.SetNthControlPointLabel(1, "T")
        display = node.GetDisplayNode()
        if display is None:
            return
        color = cfg.get("display_color", (1.0, 1.0, 0.0))
        sel_color = cfg.get("selected_color", color)
        display.SetColor(float(color[0]), float(color[1]), float(color[2]))
        display.SetSelectedColor(float(sel_color[0]), float(sel_color[1]), float(sel_color[2]))
        display.SetLineThickness(float(cfg.get("line_thickness", 0.5)))
        display.SetOpacity(1.0)
        display.SetVisibility(True)
        if hasattr(display, "SetPointLabelsVisibility"):
            display.SetPointLabelsVisibility(bool(cfg.get("point_labels", True)))
        if hasattr(display, "SetPropertiesLabelVisibility"):
            # Hide properties labels to avoid Slicer's auto-added length text.
            display.SetPropertiesLabelVisibility(False)
        if hasattr(display, "SetTextScale"):
            display.SetTextScale(1.5)
        if hasattr(display, "SetGlyphScale"):
            display.SetGlyphScale(1.5)
        # Keep default slice behavior: show markups only on slices that intersect
        # the markup geometry. Do not project lines onto all slices.
        if hasattr(display, "SetSliceProjection"):
            display.SetSliceProjection(False)
        if hasattr(display, "SetSliceProjectionUseFiducialColor"):
            display.SetSliceProjectionUseFiducialColor(True)
        if hasattr(display, "SetSliceProjectionOpacity"):
            display.SetSliceProjectionOpacity(0.9)

    def highlight_selected_trajectory(self, selected_node_id="", scope_node_ids=None):
        """Highlight one trajectory line and de-emphasize other lines in the same scope."""
        selected_id = str(selected_node_id or "").strip()
        scope = {str(node_id) for node_id in (scope_node_ids or []) if str(node_id)}
        if not scope and not selected_id:
            return

        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            node_id = str(node.GetID() or "")
            if scope and node_id not in scope:
                continue
            group = self.infer_group_from_node(node)
            self._apply_group_display(node, group)
            display = node.GetDisplayNode()
            if display is None:
                continue
            cfg = TRAJECTORY_GROUP_CONFIG.get(self._normalize_group(group), TRAJECTORY_GROUP_CONFIG[DEFAULT_GROUP])
            if selected_id and node_id == selected_id:
                base = float(cfg.get("line_thickness", 0.5))
                display.SetColor(
                    float(ACTIVE_HIGHLIGHT_COLOR[0]),
                    float(ACTIVE_HIGHLIGHT_COLOR[1]),
                    float(ACTIVE_HIGHLIGHT_COLOR[2]),
                )
                # Markups may render using selected-color state; keep both in sync
                # so highlight remains yellow regardless of MRML selected-state.
                display.SetSelectedColor(
                    float(ACTIVE_HIGHLIGHT_COLOR[0]),
                    float(ACTIVE_HIGHLIGHT_COLOR[1]),
                    float(ACTIVE_HIGHLIGHT_COLOR[2]),
                )
                display.SetLineThickness(base)
                display.SetOpacity(0.50)
                if hasattr(display, "SetSliceProjectionOpacity"):
                    display.SetSliceProjectionOpacity(0.50)
            elif selected_id:
                display.SetOpacity(0.45)

    def show_only_groups(self, groups):
        """Hide all trajectory groups except the provided one(s)."""
        keep = {self._normalize_group(g) for g in (groups or [])}
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            group = self.infer_group_from_node(node)
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
        if node.GetNumberOfControlPoints() >= 1:
            traj_name = str(trajectory_name or "").strip()
            node.SetNthControlPointLabel(0, f"{traj_name} E" if traj_name else "E")
        if node.GetNumberOfControlPoints() >= 2:
            node.SetNthControlPointLabel(1, "T")

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
            "best_model_id": str(node.GetAttribute("Rosa.BestModelId") or ""),
            "best_model_score": (
                None
                if not (node.GetAttribute("Rosa.BestModelScore") or "").strip()
                else float(node.GetAttribute("Rosa.BestModelScore"))
            ),
            "proposal_family": str(node.GetAttribute("Rosa.DeepCoreProposalFamily") or ""),
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
        """Create or reuse a subject-hierarchy folder under the given parent item."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return 0
        if hasattr(sh_node, "GetItemChildWithName"):
            existing = sh_node.GetItemChildWithName(parent_item_id, folder_name)
            if existing:
                return existing
        return sh_node.CreateFolderItem(parent_item_id, folder_name)

    def place_trajectory_nodes_in_hierarchy(self, context_id, nodes):
        """Place grouped trajectory nodes under `RosaWorkflow/Trajectories/<Group>/`."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return
        scene_item = sh_node.GetSceneItemID()
        root = self._ensure_subject_hierarchy_folder(scene_item, "RosaWorkflow")
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
        node.SetNthControlPointLabel(0, f"{name} E" if name else "E")
        node.SetNthControlPointLabel(1, "T")
        self.set_trajectory_metadata(node, trajectory_name=name, group=grp, origin=origin or grp)
        self._apply_group_display(node, grp)
        return node

    def rename_trajectory_node(self, node, new_name):
        """Rename one trajectory node while preserving its group/origin metadata."""
        if node is None:
            return False
        clean_name = str(new_name or "").strip()
        if not clean_name:
            return False
        group = self.infer_group_from_node(node)
        origin = node.GetAttribute("Rosa.TrajectoryOrigin") or group
        node.SetName(self.build_node_name(clean_name, group))
        self.set_trajectory_metadata(node, trajectory_name=clean_name, group=group, origin=origin)
        self._apply_group_display(node, group)
        return True

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
