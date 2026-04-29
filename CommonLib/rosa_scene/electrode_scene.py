"""Shared electrode/contact scene operations for ROSA modules."""

import math

from __main__ import slicer, vtk

from rosa_core import lps_to_ras_point
from rosa_workflow import WorkflowPublisher, WorkflowState

from .scene_utils import find_node_by_name as _find_node_by_name


class ElectrodeSceneService:
    """Scene-level operations for contacts, electrode models, and trajectory views."""

    def __init__(self, workflow_state=None, workflow_publish=None):
        """Initialize workflow-aware scene helpers used by multiple UI modules."""
        # Shared workflow handles let scene updates publish role-based outputs consistently.
        self.workflow_state = workflow_state or WorkflowState()
        self.workflow_publish = workflow_publish or WorkflowPublisher(self.workflow_state)

    def find_node_by_name(self, node_name, class_name):
        """Return first node with exact name and class, or None."""
        return _find_node_by_name(node_name=node_name, class_name=class_name)

    def create_contacts_fiducials_node(self, contacts, node_name="ROSA_Contacts"):
        """Create one fiducial node from contact list in ROSA/LPS space.

        The node is left unlocked so the user can drag individual
        control points in slice / 3D views (Slicer markup default).
        Interaction handles are NOT enabled — with 16-18 contacts
        per shank they would render an overlapping arrow widget at
        every contact, drowning the actual glyphs in clutter.
        """
        node = self.find_node_by_name(node_name, "vtkMRMLMarkupsFiducialNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
        else:
            node.RemoveAllControlPoints()
        for contact in contacts:
            ras = lps_to_ras_point(contact["position_lps"])
            point_index = node.AddControlPoint(vtk.vtkVector3d(*ras))
            contact_index = contact.get("index", point_index + 1)
            node.SetNthControlPointLabel(point_index, str(contact_index))
        # Explicit unlock so no code path elsewhere can stamp a lock
        # that breaks the GT-annotation drag flow.
        node.SetLocked(False)

        display_node = node.GetDisplayNode()
        if display_node:
            # Glyph picking in slice views requires the cursor on the
            # glyph itself; the prior 2.0 scale was too small to grab
            # reliably on tightly-spaced DBS / SEEG contacts.
            display_node.SetGlyphScale(4.00)
            display_node.SetTextScale(1.50)
        return node

    def create_contacts_fiducials_nodes_by_trajectory(self, contacts, node_prefix="ROSA_Contacts"):
        """Create one fiducial node per trajectory so visibility can be toggled independently."""
        by_traj = {}
        for contact in contacts:
            traj = contact.get("trajectory", "unknown")
            by_traj.setdefault(traj, []).append(contact)

        nodes = {}
        for traj_name in sorted(by_traj.keys()):
            node_name = f"{node_prefix}_{traj_name}"
            node = self.create_contacts_fiducials_node(
                by_traj[traj_name],
                node_name=node_name,
            )
            node.SetAttribute("Rosa.TrajectoryName", str(traj_name))
            nodes[traj_name] = node
        return nodes

    def create_or_update_table_node(self, node_name, columns, rows):
        """Create/update MRML table node from row dictionaries."""
        table_node = self.find_node_by_name(node_name, "vtkMRMLTableNode")
        if table_node is None:
            table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", node_name)
        table = table_node.GetTable()
        while table.GetNumberOfColumns() > 0:
            table.RemoveColumn(0)
        for col_name in columns:
            arr = vtk.vtkStringArray()
            arr.SetName(str(col_name))
            table.AddColumn(arr)
        for _ in rows:
            table.InsertNextBlankRow()
        for r, row in enumerate(rows):
            for c, col_name in enumerate(columns):
                table.SetValue(r, c, str(row.get(col_name, "")))
        table.Modified()
        table_node.Modified()
        return table_node

    def _ensure_subject_hierarchy_folder(self, parent_item_id, folder_name):
        """Create/reuse one subject-hierarchy folder under parent."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return 0
        if hasattr(sh_node, "GetItemChildWithName"):
            existing = sh_node.GetItemChildWithName(parent_item_id, folder_name)
            if existing:
                return existing
        return sh_node.CreateFolderItem(parent_item_id, folder_name)

    def place_electrode_nodes_in_hierarchy(
        self,
        context_id,
        contact_nodes_by_traj,
        model_nodes_by_traj,
    ):
        """Place electrode nodes under `RosaWorkflow/Electrodes/<Trajectory>/`."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return
        scene_item = sh_node.GetSceneItemID()
        root = self._ensure_subject_hierarchy_folder(scene_item, "RosaWorkflow")
        electrodes_root = self._ensure_subject_hierarchy_folder(root, "Electrodes")
        names = set((contact_nodes_by_traj or {}).keys()) | set((model_nodes_by_traj or {}).keys())
        for traj_name in sorted(names):
            traj_folder = self._ensure_subject_hierarchy_folder(electrodes_root, str(traj_name))
            cnode = (contact_nodes_by_traj or {}).get(traj_name)
            if cnode is not None:
                item = sh_node.GetItemByDataNode(cnode)
                if item:
                    sh_node.SetItemParent(item, traj_folder)
            pair = (model_nodes_by_traj or {}).get(traj_name, {})
            for key in ("shaft", "contacts"):
                node = pair.get(key)
                if node is None:
                    continue
                item = sh_node.GetItemByDataNode(node)
                if item:
                    sh_node.SetItemParent(item, traj_folder)

    def publish_contacts_outputs(
        self,
        contact_nodes_by_traj,
        model_nodes_by_traj,
        assignment_rows,
        qc_rows,
        workflow_node=None,
    ):
        """Publish generated contacts/models/assignment/QC artifacts to workflow roles."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        self.place_electrode_nodes_in_hierarchy(
            context_id=self.workflow_state.context_id(workflow_node=wf),
            contact_nodes_by_traj=contact_nodes_by_traj,
            model_nodes_by_traj=model_nodes_by_traj,
        )
        contact_nodes = list((contact_nodes_by_traj or {}).values())
        self.workflow_publish.publish_nodes(
            role="ContactFiducials",
            nodes=contact_nodes,
            source="contacts",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )

        shaft_nodes = []
        contact_model_nodes = []
        for pair in (model_nodes_by_traj or {}).values():
            shaft = pair.get("shaft")
            contact_model = pair.get("contacts")
            if shaft is not None:
                shaft_nodes.append(shaft)
            if contact_model is not None:
                contact_model_nodes.append(contact_model)
        self.workflow_publish.publish_nodes(
            role="ElectrodeShaftModelNodes",
            nodes=shaft_nodes,
            source="contacts",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )
        self.workflow_publish.publish_nodes(
            role="ElectrodeContactModelNodes",
            nodes=contact_model_nodes,
            source="contacts",
            space_name="ROSA_BASE",
            workflow_node=wf,
        )

        by_traj = {}
        for row in assignment_rows or []:
            by_traj[row.get("trajectory", "")] = dict(row)
        for traj_name, node in (contact_nodes_by_traj or {}).items():
            row = by_traj.setdefault(traj_name, {"trajectory": traj_name})
            row["contact_fiducial_node_id"] = node.GetID()
            pair = (model_nodes_by_traj or {}).get(traj_name, {})
            shaft = pair.get("shaft")
            contacts = pair.get("contacts")
            row["shaft_model_node_id"] = "" if shaft is None else shaft.GetID()
            row["contact_model_node_id"] = "" if contacts is None else contacts.GetID()
            row["source"] = row.get("source", "contacts")

        assignment_columns = [
            "trajectory",
            "model_id",
            "tip_at",
            "tip_shift_mm",
            "trajectory_length_mm",
            "electrode_length_mm",
            "shaft_model_node_id",
            "contact_model_node_id",
            "contact_fiducial_node_id",
            "source",
        ]
        assignment_table = self.create_or_update_table_node(
            node_name="Rosa_ElectrodeAssignments",
            columns=assignment_columns,
            rows=[by_traj[k] for k in sorted(by_traj.keys())],
        )
        self.workflow_state.set_single_role("ElectrodeAssignmentTable", assignment_table, workflow_node=wf)
        self.workflow_state.tag_node(
            assignment_table,
            role="ElectrodeAssignmentTable",
            source="contacts",
            space="ROSA_BASE",
            signature=assignment_table.GetID(),
            workflow_node=wf,
        )

        qc_columns = [
            "trajectory",
            "entry_radial_mm",
            "target_radial_mm",
            "mean_contact_radial_mm",
            "max_contact_radial_mm",
            "rms_contact_radial_mm",
            "angle_deg",
            "matched_contacts",
        ]
        qc_table = self.create_or_update_table_node(
            node_name="Rosa_QCMetrics",
            columns=qc_columns,
            rows=qc_rows or [],
        )
        self.workflow_state.set_single_role("QCMetricsTable", qc_table, workflow_node=wf)
        self.workflow_state.tag_node(
            qc_table,
            role="QCMetricsTable",
            source="contacts",
            space="ROSA_BASE",
            signature=qc_table.GetID(),
            workflow_node=wf,
        )

    def _vsub(self, a, b):
        """Return 3D vector subtraction ``a-b``."""
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def _vadd(self, a, b):
        """Return 3D vector addition ``a+b``."""
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    def _vmul(self, a, s):
        """Return vector ``a`` scaled by scalar ``s``."""
        return [a[0] * s, a[1] * s, a[2] * s]

    def _vdot(self, a, b):
        """Return 3D dot product."""
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _vcross(self, a, b):
        """Return 3D cross product."""
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def _vnorm(self, a):
        """Return Euclidean norm of a 3D vector."""
        return math.sqrt(self._vdot(a, a))

    def _vunit(self, a):
        """Return unit-length direction vector for ``a``."""
        n = self._vnorm(a)
        if n <= 1e-9:
            raise ValueError("Zero-length trajectory vector")
        return [a[0] / n, a[1] / n, a[2] / n]

    def _slice_widget_aspect(self, slice_widget):
        """Return width/height of the slice viewport, or 1.0 as fallback."""
        try:
            view = slice_widget.sliceView()
            w = float(view.width)
            h = float(view.height)
            if h > 0:
                return w / h
        except Exception:
            pass
        return 1.0

    def _tube_polydata(self, p0, p1, radius_mm, sides=24):
        """Build capped tube polydata between two 3D points."""
        line = vtk.vtkLineSource()
        line.SetPoint1(float(p0[0]), float(p0[1]), float(p0[2]))
        line.SetPoint2(float(p1[0]), float(p1[1]), float(p1[2]))

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(float(radius_mm))
        tube.SetNumberOfSides(int(sides))
        tube.CappingOn()
        tube.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(tube.GetOutput())
        return out

    def create_electrode_models_by_trajectory(
        self,
        contacts,
        trajectories_by_name,
        models_by_id,
        node_prefix="ROSA_Contacts",
    ):
        """Create per-trajectory model nodes for electrode shaft and contact segments."""
        by_traj = {}
        for contact in contacts:
            traj = contact.get("trajectory", "")
            by_traj.setdefault(traj, []).append(contact)

        created = {}
        for traj_name in sorted(by_traj.keys()):
            group = sorted(by_traj[traj_name], key=lambda c: int(c.get("index", 0)))
            if not group:
                continue

            model_id = group[0].get("model_id", "")
            if model_id not in models_by_id:
                continue
            model = models_by_id[model_id]
            offsets = list(model.get("contact_center_offsets_from_tip_mm", []))
            if not offsets:
                continue

            p_first = list(group[0]["position_lps"])
            trajectory = trajectories_by_name.get(traj_name)
            tip_at = (group[0].get("tip_at") or "target").lower()
            if len(group) >= 2:
                p_last = list(group[-1]["position_lps"])
                axis = self._vunit(self._vsub(p_last, p_first))
            else:
                if trajectory is None:
                    continue
                if tip_at == "entry":
                    axis = self._vunit(self._vsub(trajectory["end"], trajectory["start"]))
                else:
                    axis = self._vunit(self._vsub(trajectory["start"], trajectory["end"]))

            tip = self._vsub(p_first, self._vmul(axis, float(offsets[0])))
            nominal_len = float(model.get("total_exploration_length_mm", 0.0))
            # Extend the shaft cylinder all the way to the trajectory's
            # shallow endpoint so the insulated wire past the most-
            # proximal contact is visualized — important for DBS leads
            # where the contacts cover only a few cm but the lead
            # extends out to the burr-hole bolt. Project the endpoint
            # onto the axis so a small lateral offset between the
            # trajectory line and the contact-center line doesn't bend
            # the shaft. Falls back to the model's nominal length when
            # the trajectory dict is unavailable, or when the projected
            # distance is shorter than nominal (don't make the shaft
            # truncate to less than the contact-bearing region).
            shaft_len = nominal_len
            if trajectory is not None:
                shallow_lps = list(trajectory.get(
                    "end" if tip_at == "entry" else "start"
                ) or [])
                if len(shallow_lps) == 3:
                    along = self._vdot(self._vsub(shallow_lps, tip), axis)
                    if along > nominal_len:
                        shaft_len = float(along)
            shaft_end = self._vadd(tip, self._vmul(axis, shaft_len))
            radius = float(model.get("diameter_mm", 0.8)) / 2.0
            contact_len = float(model.get("contact_length_mm", 2.0))

            shaft_poly = self._tube_polydata(
                lps_to_ras_point(tip),
                lps_to_ras_point(shaft_end),
                radius_mm=radius,
                sides=24,
            )

            append = vtk.vtkAppendPolyData()
            for contact in group:
                center = contact["position_lps"]
                p0 = self._vsub(center, self._vmul(axis, contact_len / 2.0))
                p1 = self._vadd(center, self._vmul(axis, contact_len / 2.0))
                segment = self._tube_polydata(
                    lps_to_ras_point(p0),
                    lps_to_ras_point(p1),
                    radius_mm=radius,
                    sides=24,
                )
                append.AddInputData(segment)
            append.Update()
            contact_poly = vtk.vtkPolyData()
            contact_poly.DeepCopy(append.GetOutput())

            shaft_name = f"{node_prefix}_{traj_name}_shaft"
            contacts_name = f"{node_prefix}_{traj_name}_contacts"

            shaft_node = self.find_node_by_name(shaft_name, "vtkMRMLModelNode")
            if shaft_node is None:
                shaft_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", shaft_name)
            shaft_node.SetAndObservePolyData(shaft_poly)
            shaft_node.CreateDefaultDisplayNodes()
            shaft_display = shaft_node.GetDisplayNode()
            if shaft_display:
                shaft_display.SetColor(0.80, 0.80, 0.80)
                shaft_display.SetOpacity(0.40)
                if hasattr(shaft_display, "SetLineWidth"):
                    shaft_display.SetLineWidth(7)
                if hasattr(shaft_display, "SetSliceIntersectionThickness"):
                    shaft_display.SetSliceIntersectionThickness(7)
                if hasattr(shaft_display, "SetVisibility2D"):
                    shaft_display.SetVisibility2D(True)
                elif hasattr(shaft_display, "SetSliceIntersectionVisibility"):
                    shaft_display.SetSliceIntersectionVisibility(True)

            contacts_node = self.find_node_by_name(contacts_name, "vtkMRMLModelNode")
            if contacts_node is None:
                contacts_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", contacts_name)
            contacts_node.SetAndObservePolyData(contact_poly)
            contacts_node.CreateDefaultDisplayNodes()
            contacts_display = contacts_node.GetDisplayNode()
            if contacts_display:
                contacts_display.SetColor(1.00, 0.95, 0.20)
                contacts_display.SetOpacity(1.00)
                if hasattr(contacts_display, "SetLineWidth"):
                    contacts_display.SetLineWidth(7)
                if hasattr(contacts_display, "SetSliceIntersectionThickness"):
                    contacts_display.SetSliceIntersectionThickness(7)
                if hasattr(contacts_display, "SetVisibility2D"):
                    contacts_display.SetVisibility2D(True)
                elif hasattr(contacts_display, "SetSliceIntersectionVisibility"):
                    contacts_display.SetSliceIntersectionVisibility(True)

            created[traj_name] = {"shaft": shaft_node, "contacts": contacts_node}

        return created

    def align_slice_to_trajectory(self, start_ras, end_ras, slice_view="Red", mode="long", focus="entry"):
        """Align a slice node to a trajectory using two RAS points.

        Parameters
        ----------
        focus : str
            One of: ``entry`` (start point), ``target`` (end point), ``midpoint``.
            Ignored when ``mode='long'``: the long-axis view always
            centers on the trajectory midpoint and sizes its field of
            view to cover the full entry→target span (1.2× margin) so
            the whole shank is visible instead of clipped to one end.
        """
        direction = self._vunit(self._vsub(end_ras, start_ras))
        mode = (mode or "long").lower()
        if mode == "long":
            center = self._vmul(self._vadd(start_ras, end_ras), 0.5)
        else:
            focus_key = str(focus or "entry").strip().lower()
            if focus_key == "target":
                center = list(end_ras)
            elif focus_key == "midpoint":
                center = self._vmul(self._vadd(start_ras, end_ras), 0.5)
            else:
                center = list(start_ras)

        up = [0.0, 0.0, 1.0]
        if abs(self._vdot(direction, up)) > 0.9:
            up = [0.0, 1.0, 0.0]

        x_axis = self._vunit(self._vcross(up, direction))
        y_axis = self._vunit(self._vcross(direction, x_axis))

        if mode == "down":
            normal = direction
            transverse = y_axis
        else:
            normal = x_axis
            transverse = direction

        lm = slicer.app.layoutManager()
        if lm is None:
            raise RuntimeError("Slicer layout manager is not available")
        slice_widget = lm.sliceWidget(slice_view)
        if slice_widget is None:
            raise ValueError(f"Unknown slice view '{slice_view}'")
        slice_node = slice_widget.mrmlSliceNode()
        if slice_node is None:
            raise RuntimeError(f"Slice node not found for view '{slice_view}'")

        slice_node.SetSliceToRASByNTP(
            normal[0],
            normal[1],
            normal[2],
            transverse[0],
            transverse[1],
            transverse[2],
            center[0],
            center[1],
            center[2],
            0,
        )
        if hasattr(slice_node, "JumpSliceByOffsetting"):
            slice_node.JumpSliceByOffsetting(center[0], center[1], center[2])
        if hasattr(slice_node, "JumpSliceByCentering"):
            slice_node.JumpSliceByCentering(center[0], center[1], center[2])

        if mode == "long":
            length_mm = float(self._vnorm(self._vsub(end_ras, start_ras)))
            if length_mm > 1e-3:
                fov_h = length_mm * 1.2
                aspect = self._slice_widget_aspect(slice_widget)
                fov_v = fov_h / aspect if aspect > 1e-3 else fov_h * 0.5
                try:
                    slice_node.SetFieldOfView(float(fov_h), float(fov_v), 1.0)
                except Exception:
                    pass

    def jump_slice_views_to_point(self, point_ras, slice_views=("Red", "Yellow", "Green")):
        """Jump one or more slice views to the given RAS point without changing orientation."""
        center = [float(point_ras[0]), float(point_ras[1]), float(point_ras[2])]
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        for view_name in (slice_views or []):
            slice_widget = lm.sliceWidget(str(view_name))
            if slice_widget is None:
                continue
            slice_node = slice_widget.mrmlSliceNode()
            if slice_node is None:
                continue
            if hasattr(slice_node, "JumpSliceByOffsetting"):
                slice_node.JumpSliceByOffsetting(center[0], center[1], center[2])
            if hasattr(slice_node, "JumpSliceByCentering"):
                slice_node.JumpSliceByCentering(center[0], center[1], center[2])

    def set_slice_view_layers(self, slice_view, background_node=None, foreground_node=None, foreground_opacity=0.5):
        """Set background/foreground volume layers for one slice view."""
        lm = slicer.app.layoutManager()
        if lm is None:
            return False
        slice_widget = lm.sliceWidget(str(slice_view))
        if slice_widget is None:
            return False
        composite = slice_widget.mrmlSliceCompositeNode()
        if composite is None:
            return False
        bg_id = background_node.GetID() if background_node is not None else ""
        fg_id = foreground_node.GetID() if foreground_node is not None else ""
        composite.SetBackgroundVolumeID(bg_id)
        composite.SetForegroundVolumeID(fg_id)
        composite.SetForegroundOpacity(float(foreground_opacity) if fg_id else 0.0)
        return True

    def set_planned_trajectory_visibility(self, visible):
        """Show or hide all planned backup trajectory lines."""
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            name = node.GetName() or ""
            group = (node.GetAttribute("Rosa.TrajectoryGroup") or "").strip().lower()
            if group != "planned_rosa" and not name.startswith("Plan_"):
                continue
            display = node.GetDisplayNode()
            if display:
                display.SetVisibility(bool(visible))
