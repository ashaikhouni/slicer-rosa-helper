"""Import external contacts/trajectories into shared RosaWorkflow roles."""

import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

from __main__ import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),
    os.path.join(MODULE_DIR, "CommonLib"),
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from rosa_core import lps_to_ras_point
from rosa_scene import ElectrodeSceneService, TrajectorySceneService
from rosa_workflow import WorkflowPublisher, WorkflowState


class ContactImport(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Contact Import"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = "Import contacts or trajectories from external files (CSV/TSV/XLSX/POM)."


class ContactImportWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = ContactImportLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowPublisher = self.logic.workflow_publish
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        top_form = qt.QFormLayout()
        self.layout.addLayout(top_form)

        self.refreshButton = qt.QPushButton("Refresh Workflow Inputs")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        top_form.addRow(self.refreshButton)

        self.referenceSelector = slicer.qMRMLNodeComboBox()
        self.referenceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.referenceSelector.noneEnabled = True
        self.referenceSelector.addEnabled = False
        self.referenceSelector.removeEnabled = False
        self.referenceSelector.setMRMLScene(slicer.mrmlScene)
        top_form.addRow("Reference volume", self.referenceSelector)

        ref_load_row = qt.QHBoxLayout()
        self.referenceLoadPath = ctk.ctkPathLineEdit()
        self.referenceLoadPath.filters = ctk.ctkPathLineEdit.Files
        ref_load_row.addWidget(self.referenceLoadPath)
        self.referenceLoadButton = qt.QPushButton("Load Volume")
        self.referenceLoadButton.clicked.connect(self.onLoadReferenceVolumeClicked)
        ref_load_row.addWidget(self.referenceLoadButton)
        top_form.addRow("Load new reference", ref_load_row)

        self.coordSystemCombo = qt.QComboBox()
        self.coordSystemCombo.addItems(["RAS", "LPS"])
        self.coordSystemCombo.setCurrentText("RAS")
        top_form.addRow("Coordinate system", self.coordSystemCombo)

        self.coordTypeCombo = qt.QComboBox()
        self.coordTypeCombo.addItems(["world", "voxel"])
        self.coordTypeCombo.setCurrentText("world")
        top_form.addRow("Coordinate type", self.coordTypeCombo)

        self.unitsCombo = qt.QComboBox()
        self.unitsCombo.addItems(["mm", "m"])
        self.unitsCombo.setCurrentText("mm")
        top_form.addRow("Units", self.unitsCombo)

        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs)
        self._build_contacts_tab()
        self._build_trajectories_tab()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(3000)
        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)
        self.onRefreshClicked()

    def log(self, message):
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _build_contacts_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "Contacts")
        form = qt.QFormLayout(tab)

        self.contactsPath = ctk.ctkPathLineEdit()
        self.contactsPath.filters = ctk.ctkPathLineEdit.Files
        form.addRow("Contacts file", self.contactsPath)

        self.importContactsButton = qt.QPushButton("Import Contacts")
        self.importContactsButton.clicked.connect(self.onImportContactsClicked)
        form.addRow(self.importContactsButton)

        columns = qt.QLabel(
            "Required columns (CSV/TSV/XLSX): trajectory_name,index,x,y,z\n"
            "Optional: label\n"
            "POM: parsed from REMARK_LIST + LOCATION_LIST"
        )
        columns.wordWrap = True
        form.addRow(columns)

    def _build_trajectories_tab(self):
        tab = qt.QWidget()
        self.tabs.addTab(tab, "Trajectories")
        form = qt.QFormLayout(tab)

        self.trajectoriesPath = ctk.ctkPathLineEdit()
        self.trajectoriesPath.filters = ctk.ctkPathLineEdit.Files
        form.addRow("Trajectories file", self.trajectoriesPath)

        self.importTrajectoriesButton = qt.QPushButton("Import Trajectories")
        self.importTrajectoriesButton.clicked.connect(self.onImportTrajectoriesClicked)
        form.addRow(self.importTrajectoriesButton)

        columns = qt.QLabel("Required columns: name,ex,ey,ez,tx,ty,tz")
        columns.wordWrap = True
        form.addRow(columns)

    def _widget_text(self, widget):
        text_attr = getattr(widget, "currentText", "")
        return text_attr() if callable(text_attr) else text_attr

    def _require_reference_volume(self):
        node = self.referenceSelector.currentNode()
        if node is None:
            raise ValueError("Reference volume is required. Select one in scene or load a new volume.")
        return node

    def _units_scale(self):
        units = (self._widget_text(self.unitsCombo) or "mm").strip().lower()
        return 1000.0 if units == "m" else 1.0

    def _coord_system(self):
        return (self._widget_text(self.coordSystemCombo) or "RAS").strip().upper()

    def _coord_type(self):
        return (self._widget_text(self.coordTypeCombo) or "world").strip().lower()

    def _ijk_to_world_ras(self, volume_node, ijk):
        ijk_to_ras = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras)
        point4 = [float(ijk[0]), float(ijk[1]), float(ijk[2]), 1.0]
        ras_local = [0.0, 0.0, 0.0, 0.0]
        ijk_to_ras.MultiplyPoint(point4, ras_local)
        ras = [float(ras_local[0]), float(ras_local[1]), float(ras_local[2])]

        parent = volume_node.GetParentTransformNode()
        if parent is not None:
            tfm = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parent, None, tfm)
            ras = list(tfm.TransformPoint(ras))
        return ras

    def _world_ras_to_ijk(self, volume_node, ras_world):
        ras_local = list(ras_world)
        parent = volume_node.GetParentTransformNode()
        if parent is not None:
            tfm = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, parent, tfm)
            ras_local = list(tfm.TransformPoint(ras_world))

        ras_to_ijk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk)
        point4 = [float(ras_local[0]), float(ras_local[1]), float(ras_local[2]), 1.0]
        ijk = [0.0, 0.0, 0.0, 0.0]
        ras_to_ijk.MultiplyPoint(point4, ijk)
        return [float(ijk[0]), float(ijk[1]), float(ijk[2])]

    def _convert_xyz_to_world_ras(self, x, y, z, reference_volume):
        coord_type = self._coord_type()
        scale = self._units_scale()
        if coord_type == "voxel":
            return self._ijk_to_world_ras(reference_volume, [x, y, z])

        point = [float(x) * scale, float(y) * scale, float(z) * scale]
        if self._coord_system() == "LPS":
            point = lps_to_ras_point(point)
        return point

    def _validate_world_points_against_reference(self, points_ras, reference_volume):
        image = reference_volume.GetImageData()
        if image is None:
            return
        dims = image.GetDimensions()
        if not dims or len(dims) < 3:
            return
        inside = 0
        for point in points_ras:
            ijk = self._world_ras_to_ijk(reference_volume, point)
            if (
                -1.0 <= ijk[0] <= float(dims[0]) and
                -1.0 <= ijk[1] <= float(dims[1]) and
                -1.0 <= ijk[2] <= float(dims[2])
            ):
                inside += 1
        total = len(points_ras)
        if total:
            self.log(f"[import] reference-bounds check: {inside}/{total} points inside reference volume")

    def _normalize_headers(self, row):
        return {str(k).strip().lower(): v for k, v in (row or {}).items()}

    def _read_table_rows(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            try:
                import openpyxl
                workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
                sheet = workbook.worksheets[0]
                rows = list(sheet.iter_rows(values_only=True))
                if not rows:
                    return []
                headers = [str(h).strip() if h is not None else "" for h in rows[0]]
                output = []
                for values in rows[1:]:
                    if values is None:
                        continue
                    row = {}
                    empty = True
                    for idx, header in enumerate(headers):
                        if not header:
                            continue
                        value = values[idx] if idx < len(values) else None
                        row[header] = "" if value is None else str(value)
                        if row[header] != "":
                            empty = False
                    if not empty:
                        output.append(row)
                return output
            except Exception:
                return self._read_xlsx_rows_fallback(path)

        delimiter = ","
        if ext == ".tsv":
            delimiter = "\t"
        elif ext not in (".csv", ".txt"):
            raise ValueError(f"Unsupported file extension: {ext}")

        with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as stream:
            if ext == ".txt":
                sample = stream.read(4096)
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ","
            reader = csv.DictReader(stream, delimiter=delimiter)
            return [row for row in reader if row]

    def _read_xlsx_rows_fallback(self, path):
        """Read first worksheet from XLSX without openpyxl (fallback parser)."""
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        def _col_index(cell_ref):
            letters = "".join(ch for ch in cell_ref if ch.isalpha())
            idx = 0
            for ch in letters:
                idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
            return max(idx - 1, 0)

        shared_strings = []
        with zipfile.ZipFile(path, "r") as archive:
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for si in root.findall("x:si", ns):
                    text_parts = []
                    for t in si.findall(".//x:t", ns):
                        text_parts.append(t.text or "")
                    shared_strings.append("".join(text_parts))

            if "xl/worksheets/sheet1.xml" not in archive.namelist():
                raise RuntimeError("XLSX fallback parser: sheet1.xml not found.")
            sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

        rows = []
        for row_node in sheet_root.findall(".//x:sheetData/x:row", ns):
            row_values = {}
            for cell in row_node.findall("x:c", ns):
                ref = cell.attrib.get("r", "")
                col = _col_index(ref)
                cell_type = cell.attrib.get("t", "")
                value_node = cell.find("x:v", ns)
                inline_node = cell.find("x:is/x:t", ns)
                text = ""
                if inline_node is not None:
                    text = inline_node.text or ""
                elif value_node is not None:
                    raw = value_node.text or ""
                    if cell_type == "s":
                        try:
                            text = shared_strings[int(raw)]
                        except Exception:
                            text = raw
                    else:
                        text = raw
                row_values[col] = text
            if row_values:
                max_col = max(row_values.keys())
                rows.append([row_values.get(i, "") for i in range(max_col + 1)])

        if not rows:
            return []
        headers = [str(h).strip() if h is not None else "" for h in rows[0]]
        output = []
        for values in rows[1:]:
            row = {}
            empty = True
            for idx, header in enumerate(headers):
                if not header:
                    continue
                value = values[idx] if idx < len(values) else ""
                text = "" if value is None else str(value)
                row[header] = text
                if text != "":
                    empty = False
            if not empty:
                output.append(row)
        return output

    def _parse_pom_contacts(self, path):
        text = open(path, "r", encoding="utf-8-sig", errors="replace").read()
        lines = text.splitlines()

        def collect_block(name):
            inside = False
            output = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(f"{name} START_LIST"):
                    inside = True
                    continue
                if inside and stripped.startswith(f"{name} END_LIST"):
                    break
                if inside:
                    if not stripped or stripped.startswith("#"):
                        continue
                    output.append(stripped)
            return output

        location_lines = collect_block("LOCATION_LIST")
        remark_lines = collect_block("REMARK_LIST")
        if not location_lines or not remark_lines:
            raise ValueError("POM parse failed: LOCATION_LIST or REMARK_LIST is missing.")
        if len(location_lines) != len(remark_lines):
            raise ValueError(
                f"POM parse failed: LOCATION_LIST count ({len(location_lines)}) "
                f"!= REMARK_LIST count ({len(remark_lines)})."
            )

        rows = []
        for index, (remark, coords) in enumerate(zip(remark_lines, location_lines), start=1):
            parts = re.split(r"[\t ,]+", coords.strip())
            if len(parts) < 3:
                raise ValueError(f"POM parse failed at row {index}: invalid coordinates '{coords}'.")
            label = remark.strip()
            match = re.match(r"^(.*?)(\d+)$", label)
            if not match:
                raise ValueError(
                    f"POM label '{label}' does not match required pattern '<trajectory_name><index>'."
                )
            trajectory_name = match.group(1).strip()
            contact_index = int(match.group(2))
            rows.append(
                {
                    "trajectory_name": trajectory_name,
                    "index": contact_index,
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "z": float(parts[2]),
                    "label": label,
                }
            )
        return rows

    def _parse_contacts_rows(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pom":
            return self._parse_pom_contacts(path)

        raw_rows = self._read_table_rows(path)
        if not raw_rows:
            raise ValueError("Input file has no data rows.")

        required = {"trajectory_name", "index", "x", "y", "z"}
        parsed = []
        seen = set()
        for row_idx, row in enumerate(raw_rows, start=2):
            norm = self._normalize_headers(row)
            missing = [name for name in sorted(required) if name not in norm or str(norm.get(name, "")).strip() == ""]
            if missing:
                raise ValueError(f"Row {row_idx}: missing required column value(s): {', '.join(missing)}")
            trajectory_name = str(norm.get("trajectory_name", "")).strip()
            contact_index = int(float(norm.get("index")))
            x = float(norm.get("x"))
            y = float(norm.get("y"))
            z = float(norm.get("z"))
            key = (trajectory_name, contact_index)
            if key in seen:
                raise ValueError(f"Duplicate contact key found: trajectory_name={trajectory_name}, index={contact_index}")
            seen.add(key)
            label = str(norm.get("label", "")).strip() or f"{trajectory_name}{contact_index}"
            parsed.append(
                {
                    "trajectory_name": trajectory_name,
                    "index": contact_index,
                    "x": x,
                    "y": y,
                    "z": z,
                    "label": label,
                }
            )
        parsed.sort(key=lambda row: (row["trajectory_name"], int(row["index"])))
        return parsed

    def _validate_contacts_have_trajectory_lines(self, parsed_rows):
        """Require at least two contacts per trajectory so a line can be created."""
        counts = {}
        for row in parsed_rows or []:
            traj = str(row.get("trajectory_name", "")).strip()
            if not traj:
                continue
            counts[traj] = counts.get(traj, 0) + 1
        invalid = sorted([name for name, count in counts.items() if count < 2])
        if invalid:
            detail = ", ".join(invalid)
            raise ValueError(
                "Each trajectory must have at least 2 contacts. "
                f"Found trajectory(ies) with only one contact: {detail}"
            )

    def _parse_trajectory_rows(self, path):
        raw_rows = self._read_table_rows(path)
        if not raw_rows:
            raise ValueError("Input file has no data rows.")

        required = {"name", "ex", "ey", "ez", "tx", "ty", "tz"}
        parsed = []
        seen = set()
        for row_idx, row in enumerate(raw_rows, start=2):
            norm = self._normalize_headers(row)
            missing = [name for name in sorted(required) if name not in norm or str(norm.get(name, "")).strip() == ""]
            if missing:
                raise ValueError(f"Row {row_idx}: missing required column value(s): {', '.join(missing)}")
            name = str(norm.get("name", "")).strip()
            if name in seen:
                raise ValueError(f"Duplicate trajectory name: {name}")
            seen.add(name)
            parsed.append(
                {
                    "name": name,
                    "ex": float(norm.get("ex")),
                    "ey": float(norm.get("ey")),
                    "ez": float(norm.get("ez")),
                    "tx": float(norm.get("tx")),
                    "ty": float(norm.get("ty")),
                    "tz": float(norm.get("tz")),
                }
            )
        parsed.sort(key=lambda row: row["name"])
        return parsed

    def onLoadReferenceVolumeClicked(self):
        path = (self.referenceLoadPath.currentPath or "").strip()
        if not path:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contact Import", "Select a volume file to load.")
            return
        if not os.path.isfile(path):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contact Import", f"File not found:\n{path}")
            return
        try:
            loaded = slicer.util.loadVolume(path)
        except Exception as exc:
            self.log(f"[import] failed to load reference volume: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contact Import", str(exc))
            return
        node = loaded
        if isinstance(loaded, tuple):
            ok, node = loaded
            if not ok:
                qt.QMessageBox.critical(slicer.util.mainWindow(), "Contact Import", "Failed to load volume.")
                return
        self.referenceSelector.setCurrentNode(node)
        self.workflowPublisher.register_volume(
            volume_node=node,
            source_type="import",
            source_path=path,
            space_name="ROSA_BASE",
            role="AdditionalMRIVolumes",
            workflow_node=self.workflowNode,
        )
        self.log(f"[import] loaded reference volume: {node.GetName()}")

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        base = self.workflowNode.GetNodeReference("BaseVolume")
        if base is not None:
            self.referenceSelector.setCurrentNode(base)
        self.log("[refresh] ready")

    def _create_contact_nodes(self, contacts_rows):
        by_traj = {}
        for row in contacts_rows:
            by_traj.setdefault(row["trajectory_name"], []).append(row)

        nodes = {}
        for traj_name in sorted(by_traj.keys()):
            node_name = f"ROSA_Contacts_{traj_name}"
            node = self.logic.electrode_scene.find_node_by_name(node_name, "vtkMRMLMarkupsFiducialNode")
            if node is None:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
                node.CreateDefaultDisplayNodes()
            node.RemoveAllControlPoints()
            for row in sorted(by_traj[traj_name], key=lambda item: int(item["index"])):
                point = row["world_ras"]
                idx = node.AddControlPoint(vtk.vtkVector3d(*point))
                node.SetNthControlPointLabel(idx, str(row.get("label", f"{traj_name}{row['index']}")))
            node.SetAttribute("Rosa.Managed", "1")
            node.SetAttribute("Rosa.TrajectoryName", str(traj_name))
            node.SetAttribute("Rosa.Source", "contact_import")
            display = node.GetDisplayNode()
            if display is not None:
                display.SetGlyphScale(2.0)
                display.SetTextScale(1.5)
            nodes[traj_name] = node
        return nodes

    def _create_trajectory_nodes_from_contacts(self, contacts_rows):
        """Create/update one trajectory line per contact group using first/last contact index."""
        by_traj = {}
        for row in contacts_rows:
            by_traj.setdefault(row["trajectory_name"], []).append(row)

        nodes = []
        for traj_name in sorted(by_traj.keys()):
            ordered = sorted(by_traj[traj_name], key=lambda item: int(item["index"]))
            if len(ordered) < 2:
                self.log(
                    f"[contacts] skipped trajectory '{traj_name}' line creation: requires >=2 contacts."
                )
                continue
            start_ras = ordered[0]["world_ras"]
            end_ras = ordered[-1]["world_ras"]
            node = self.logic.trajectory_scene.create_or_update_trajectory_line(
                name=traj_name,
                start_ras=start_ras,
                end_ras=end_ras,
                group="imported_external",
                origin="contact_import",
            )
            nodes.append(node)
        return nodes

    def onImportContactsClicked(self):
        path = (self.contactsPath.currentPath or "").strip()
        if not path:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contact Import", "Select contacts file.")
            return
        if not os.path.isfile(path):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contact Import", f"File not found:\n{path}")
            return
        try:
            reference = self._require_reference_volume()
            rows = self._parse_contacts_rows(path)
            self._validate_contacts_have_trajectory_lines(rows)
            if self._coord_type() == "voxel" and abs(self._units_scale() - 1.0) > 1e-9:
                self.log("[contacts] units are ignored for voxel coordinates (indices are unitless)")
            world_points = []
            for row in rows:
                ras = self._convert_xyz_to_world_ras(row["x"], row["y"], row["z"], reference)
                row["world_ras"] = [float(ras[0]), float(ras[1]), float(ras[2])]
                world_points.append(row["world_ras"])
            self._validate_world_points_against_reference(world_points, reference)
            nodes_by_traj = self._create_contact_nodes(rows)
            trajectory_nodes = self._create_trajectory_nodes_from_contacts(rows)

            self.workflowPublisher.publish_nodes(
                role="ContactFiducials",
                nodes=list(nodes_by_traj.values()),
                source="contact_import",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.workflowPublisher.publish_nodes(
                role="ImportedExternalTrajectoryLines",
                nodes=trajectory_nodes,
                source="contact_import",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.workflowPublisher.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=trajectory_nodes,
                source="contact_import",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.logic.electrode_scene.place_electrode_nodes_in_hierarchy(
                context_id=self.workflowState.context_id(workflow_node=self.workflowNode),
                contact_nodes_by_traj=nodes_by_traj,
                model_nodes_by_traj={},
            )
            self.logic.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflowState.context_id(workflow_node=self.workflowNode),
                nodes=trajectory_nodes,
            )
            self.log(
                f"[contacts] imported {len(rows)} contacts across {len(nodes_by_traj)} trajectories; "
                f"created/updated {len(trajectory_nodes)} trajectory lines"
            )
        except Exception as exc:
            self.log(f"[contacts] import failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contact Import", str(exc))

    def onImportTrajectoriesClicked(self):
        path = (self.trajectoriesPath.currentPath or "").strip()
        if not path:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contact Import", "Select trajectories file.")
            return
        if not os.path.isfile(path):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Contact Import", f"File not found:\n{path}")
            return
        try:
            reference = self._require_reference_volume()
            rows = self._parse_trajectory_rows(path)
            if self._coord_type() == "voxel" and abs(self._units_scale() - 1.0) > 1e-9:
                self.log("[traj] units are ignored for voxel coordinates (indices are unitless)")
            nodes = []
            for row in rows:
                entry_ras = self._convert_xyz_to_world_ras(row["ex"], row["ey"], row["ez"], reference)
                target_ras = self._convert_xyz_to_world_ras(row["tx"], row["ty"], row["tz"], reference)
                node = self.logic.trajectory_scene.create_or_update_trajectory_line(
                    name=row["name"],
                    start_ras=entry_ras,
                    end_ras=target_ras,
                    group="imported_external",
                    origin="contact_import",
                )
                nodes.append(node)

            self.workflowPublisher.publish_nodes(
                role="ImportedExternalTrajectoryLines",
                nodes=nodes,
                source="contact_import",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.workflowPublisher.publish_nodes(
                role="WorkingTrajectoryLines",
                nodes=nodes,
                source="contact_import",
                space_name="ROSA_BASE",
                workflow_node=self.workflowNode,
            )
            self.logic.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflowState.context_id(workflow_node=self.workflowNode),
                nodes=nodes,
            )
            self.logic.trajectory_scene.show_only_groups(["imported_external"])
            self.log(f"[traj] imported {len(nodes)} trajectories (group=imported_external)")
        except Exception as exc:
            self.log(f"[traj] import failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contact Import", str(exc))


class ContactImportLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.trajectory_scene = TrajectorySceneService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
        )
