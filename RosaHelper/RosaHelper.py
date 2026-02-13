import os
import sys

from __main__ import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(MODULE_DIR, "Lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from rosa_core import (
    build_effective_matrices,
    choose_reference_volume,
    find_ros_file,
    invert_4x4,
    lps_to_ras_matrix,
    lps_to_ras_point,
    parse_ros_file,
    resolve_analyze_volume,
    resolve_reference_index,
)


class RosaHelper(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "ROSA Helper"
        self.parent.categories = ["ROSA"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar", "Codex"]
        self.parent.helpText = "Load a ROSA case folder into Slicer and apply ROSA transforms."


class RosaHelperWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.logic = RosaHelperLogic()

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.caseDirSelector = ctk.ctkPathLineEdit()
        self.caseDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.caseDirSelector.setToolTip("Case folder containing .ros and DICOM/")
        form.addRow("Case folder", self.caseDirSelector)

        self.referenceEdit = qt.QLineEdit()
        self.referenceEdit.setPlaceholderText("Optional (auto-detect if blank)")
        form.addRow("Reference volume", self.referenceEdit)

        self.invertCheck = qt.QCheckBox("Invert TRdicomRdisplay")
        self.invertCheck.setChecked(False)
        form.addRow("Transform option", self.invertCheck)

        self.hardenCheck = qt.QCheckBox("Harden transforms")
        self.hardenCheck.setChecked(True)
        form.addRow("Scene option", self.hardenCheck)

        self.markupsCheck = qt.QCheckBox("Load trajectories")
        self.markupsCheck.setChecked(True)
        form.addRow("Trajectory option", self.markupsCheck)

        self.loadButton = qt.QPushButton("Load ROSA case")
        self.loadButton.clicked.connect(self.onLoadClicked)
        self.layout.addWidget(self.loadButton)

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1000)
        self.layout.addWidget(self.statusText)

        self.layout.addStretch(1)

    def log(self, msg):
        self.statusText.appendPlainText(msg)
        print(msg)

    def onLoadClicked(self):
        case_dir = self.caseDirSelector.currentPath
        if not case_dir:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Please select a case folder")
            return

        reference = self.referenceEdit.text.strip() or None

        try:
            summary = self.logic.load_case(
                case_dir=case_dir,
                reference=reference,
                invert=self.invertCheck.checked,
                harden=self.hardenCheck.checked,
                load_trajectories=self.markupsCheck.checked,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[error] {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.log(
            f"[done] loaded {summary['loaded_volumes']} volumes, "
            f"created {summary['trajectory_count']} trajectories"
        )


class RosaHelperLogic(ScriptedLoadableModuleLogic):
    def load_case(
        self,
        case_dir,
        reference=None,
        invert=False,
        harden=True,
        load_trajectories=True,
        logger=None,
    ):
        def log(msg):
            if logger:
                logger(msg)
            else:
                print(msg)

        case_dir = os.path.abspath(case_dir)
        ros_path = find_ros_file(case_dir)
        analyze_root = os.path.join(case_dir, "DICOM")

        if not os.path.isdir(analyze_root):
            raise ValueError(f"Analyze root not found: {analyze_root}")

        parsed = parse_ros_file(ros_path)
        displays = parsed["displays"]
        trajectories = parsed["trajectories"]

        if not displays:
            raise ValueError("No TRdicomRdisplay/VOLUME entries found in ROS file")

        reference_volume = choose_reference_volume(displays, preferred=reference)
        root_index = resolve_reference_index(displays, reference_volume)
        effective_lps = build_effective_matrices(displays, root_index=root_index)
        log(f"[ros] {ros_path}")
        log(f"[ref] {reference_volume}")

        loaded_count = 0

        for i, disp in enumerate(displays):
            vol_name = disp["volume"]
            img_path = resolve_analyze_volume(analyze_root, disp)
            if not img_path:
                log(f"[skip] missing Analyze .img for {vol_name}")
                continue

            vol_node = self._load_volume(img_path)
            if vol_node is None:
                log(f"[skip] failed to load {img_path}")
                continue

            loaded_count += 1
            vol_node.SetName(vol_name)
            self._center_volume(vol_node)
            log(f"[load] {vol_name}")
            log(f"[center] {vol_name}")

            if vol_name != reference_volume:
                matrix_ras = lps_to_ras_matrix(effective_lps[i])
                if invert:
                    matrix_ras = invert_4x4(matrix_ras)
                tnode = self._apply_transform(vol_node, matrix_ras)
                ref_idx = disp.get("imagery_3dref", root_index)
                log(
                    f"[xform] {vol_name} {'inv ' if invert else ''}TRdicomRdisplay "
                    f"(ref idx {ref_idx} -> root idx {root_index})"
                )
                if harden:
                    slicer.vtkSlicerTransformLogic().hardenTransform(vol_node)
                    slicer.mrmlScene.RemoveNode(tnode)
                    log(f"[harden] {vol_name}")
            else:
                log(f"[xform] {vol_name} reference (none)")

        if load_trajectories and trajectories:
            self._add_trajectories(trajectories, logger=log)

        return {
            "loaded_volumes": loaded_count,
            "trajectory_count": len(trajectories) if load_trajectories else 0,
        }

    def _load_volume(self, path):
        try:
            result = slicer.util.loadVolume(path, returnNode=True)
            if isinstance(result, tuple):
                ok, node = result
                return node if ok else None
            return result
        except TypeError:
            return slicer.util.loadVolume(path)

    def _center_volume(self, volume_node):
        logic = slicer.modules.volumes.logic()
        if logic and hasattr(logic, "CenterVolume"):
            logic.CenterVolume(volume_node)
            return

        ijk_to_ras = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras)
        dims = volume_node.GetImageData().GetDimensions()
        center_ijk = [(dims[0] - 1) / 2.0, (dims[1] - 1) / 2.0, (dims[2] - 1) / 2.0, 1.0]
        center_ras = [0.0, 0.0, 0.0, 0.0]
        ijk_to_ras.MultiplyPoint(center_ijk, center_ras)
        for i in range(3):
            ijk_to_ras.SetElement(i, 3, ijk_to_ras.GetElement(i, 3) - center_ras[i])
        volume_node.SetIJKToRASMatrix(ijk_to_ras)

    def _apply_transform(self, volume_node, matrix4x4):
        tnode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        vtk_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_mat.SetElement(r, c, matrix4x4[r][c])
        tnode.SetMatrixTransformToParent(vtk_mat)
        volume_node.SetAndObserveTransformNodeID(tnode.GetID())
        return tnode

    def _add_trajectories(self, trajectories, logger=None):
        for traj in trajectories:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            node.SetName(traj["name"])
            start_ras = lps_to_ras_point(traj["start"])
            end_ras = lps_to_ras_point(traj["end"])
            node.AddControlPoint(vtk.vtkVector3d(*start_ras))
            node.AddControlPoint(vtk.vtkVector3d(*end_ras))
            node.SetNthControlPointLabel(0, f"{traj['name']}_start")
            node.SetNthControlPointLabel(1, f"{traj['name']}_end")

        if logger:
            logger(f"[markups] created {len(trajectories)} line trajectories")


def run(case_dir, reference=None, invert=False, harden=True, load_trajectories=True):
    return RosaHelperLogic().load_case(
        case_dir=case_dir,
        reference=reference,
        invert=invert,
        harden=harden,
        load_trajectories=load_trajectories,
    )
