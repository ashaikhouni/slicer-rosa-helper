"""Shared atlas/burn services used by AtlasSources, AtlasLabeling, NavigationBurn."""

import hashlib
import math
import os

try:
    import numpy as np
except ImportError:
    np = None

from __main__ import slicer, vtk

from rosa_core import lps_to_ras_point
from rosa_workflow import WorkflowState

from .electrode_scene import ElectrodeSceneService
from .freesurfer_service import FreeSurferService


class AtlasCoreService:
    """Common service layer for atlas loading, labeling, and navigation burn workflows."""

    def __init__(self, module_dir=None, workflow_state=None):
        self.workflow_state = workflow_state or WorkflowState()
        self.electrode_scene = ElectrodeSceneService(workflow_state=self.workflow_state)
        self.fs_service = FreeSurferService(module_dir=module_dir)

    def _find_node_by_name(self, node_name, class_name):
        for node in slicer.util.getNodesByClass(class_name):
            if node.GetName() == node_name:
                return node
        return None

    def _vtk_matrix_to_numpy_4x4(self, vtk_matrix):
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix.GetElement(r, c))
        return out

    def _volume_ijk_to_world_matrix(self, volume_node):
        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_world = self._vtk_matrix_to_numpy_4x4(ijk_to_ras_vtk)

        parent = volume_node.GetParentTransformNode()
        if parent is None:
            return ijk_to_world
        parent_to_world_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformToWorld(parent_to_world_vtk):
            return ijk_to_world
        parent_to_world = self._vtk_matrix_to_numpy_4x4(parent_to_world_vtk)
        return parent_to_world @ ijk_to_world

    def _world_to_node_ras_matrix(self, node):
        if node is None:
            return np.eye(4, dtype=float)
        parent = node.GetParentTransformNode()
        if parent is None:
            return np.eye(4, dtype=float)
        world_to_local_vtk = vtk.vtkMatrix4x4()
        if not parent.GetMatrixTransformFromWorld(world_to_local_vtk):
            return np.eye(4, dtype=float)
        return self._vtk_matrix_to_numpy_4x4(world_to_local_vtk)

    def _world_to_node_ras_point(self, node, point_world_ras):
        mat = self._world_to_node_ras_matrix(node)
        vec = np.array([float(point_world_ras[0]), float(point_world_ras[1]), float(point_world_ras[2]), 1.0])
        out = mat @ vec
        return [float(out[0]), float(out[1]), float(out[2])]

    def show_volume_in_all_slice_views(self, volume_node):
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        return self.fs_service.run_brainsfit_rigid_registration(
            fixed_volume_node=fixed_volume_node,
            moving_volume_node=moving_volume_node,
            output_transform_node=output_transform_node,
            initialize_mode=initialize_mode,
            logger=logger,
        )

    def list_freesurfer_parcellation_candidates(self, subject_dir):
        return self.fs_service.freesurfer_parcellation_candidates(subject_dir)

    def load_freesurfer_parcellation_volumes(
        self,
        subject_dir,
        selected_names=None,
        color_lut_path=None,
        apply_color_table=True,
        create_3d_geometry=False,
        logger=None,
    ):
        return self.fs_service.load_freesurfer_parcellation_volumes(
            subject_dir=subject_dir,
            selected_names=selected_names,
            color_lut_path=color_lut_path,
            apply_color_table=apply_color_table,
            create_3d_geometry=create_3d_geometry,
            logger=logger,
        )

    def apply_transform_to_nodes(self, nodes, transform_node, harden=False):
        if transform_node is None:
            raise ValueError("Transform node is required.")
        transform_id = transform_node.GetID()
        for node in nodes or []:
            if node is None:
                continue
            if hasattr(node, "SetAndObserveTransformNodeID"):
                node.SetAndObserveTransformNodeID(transform_id)
                if bool(harden):
                    slicer.vtkSlicerTransformLogic().hardenTransform(node)
        return nodes

    def _load_label_volume_node(self, path):
        try:
            result = slicer.util.loadLabelVolume(path, returnNode=True)
            if isinstance(result, tuple):
                ok, node = result
                return node if ok else None
            return result
        except Exception:
            try:
                result = slicer.util.loadVolume(path, properties={"labelmap": True}, returnNode=True)
                if isinstance(result, tuple):
                    ok, node = result
                    return node if ok else None
                return result
            except Exception:
                return None

    def _infer_thomas_side(self, path):
        text = (path or "").lower()
        if "/left/" in text or text.endswith("/left") or "lh" in os.path.basename(text):
            return "left"
        if "/right/" in text or text.endswith("/right") or "rh" in os.path.basename(text):
            return "right"
        return "unknown"

    def _thomas_segment_name_from_path(self, path):
        base = os.path.basename(path)
        name = base.replace(".nii.gz", "").replace(".nii", "")
        side = self._infer_thomas_side(path)
        if name.lower().endswith("_l") or name.lower().endswith("_r"):
            name = name[:-2]
        token = name
        if "-" in token:
            token = token.split("-", 1)[1]
        elif "_" in token:
            parts = token.split("_")
            if parts and parts[0].isdigit():
                token = "_".join(parts[1:]) or token
        token = token.strip("_- ") or name
        return f"{side.upper()}_{token}"

    def _thomas_color_for_label(self, segment_name, side):
        key = (segment_name or "UNKNOWN").upper()
        base_map = {
            "THALAMUS": (0.90, 0.80, 0.20),
            "CM": (0.95, 0.35, 0.35),
            "VA": (0.95, 0.55, 0.20),
            "VLA": (0.90, 0.55, 0.45),
            "VL": (0.85, 0.50, 0.55),
            "VLP": (0.85, 0.45, 0.70),
            "VLPD": (0.75, 0.45, 0.80),
            "VLPV": (0.75, 0.55, 0.85),
            "VPL": (0.45, 0.70, 0.95),
            "PUL": (0.40, 0.75, 0.70),
            "LGN": (0.35, 0.75, 0.55),
            "MGN": (0.35, 0.65, 0.50),
            "MD-PF": (0.60, 0.70, 0.45),
            "HB": (0.45, 0.85, 0.45),
            "MTT": (0.60, 0.85, 0.45),
            "AV": (0.75, 0.75, 0.35),
        }
        token = key.split("_", 1)[-1] if "_" in key else key
        color = base_map.get(token)
        if color is None:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            val = int(digest[:8], 16)
            r = 0.35 + ((val >> 16) & 0xFF) / 255.0 * 0.55
            g = 0.35 + ((val >> 8) & 0xFF) / 255.0 * 0.55
            b = 0.35 + (val & 0xFF) / 255.0 * 0.55
            color = (r, g, b)

        if side == "left":
            return (max(0.0, color[0] * 0.90), min(1.0, color[1] * 1.03), min(1.0, color[2] * 1.08))
        if side == "right":
            return (min(1.0, color[0] * 1.08), min(1.0, color[1] * 1.02), max(0.0, color[2] * 0.90))
        return color

    def _style_thomas_segmentation(self, seg_node, side):
        if seg_node is None:
            return
        seg_node.CreateDefaultDisplayNodes()
        display = seg_node.GetDisplayNode()
        if display:
            display.SetVisibility(True)
            if hasattr(display, "SetVisibility2D"):
                display.SetVisibility2D(True)
            if hasattr(display, "SetVisibility3D"):
                display.SetVisibility3D(True)
            if hasattr(display, "SetOpacity2DFill"):
                display.SetOpacity2DFill(0.25)
            if hasattr(display, "SetOpacity2DOutline"):
                display.SetOpacity2DOutline(1.0)
            if hasattr(display, "SetOpacity3D"):
                display.SetOpacity3D(0.55)
            if hasattr(display, "SetPreferredDisplayRepresentationName2D"):
                display.SetPreferredDisplayRepresentationName2D("Binary labelmap")
            if hasattr(display, "SetPreferredDisplayRepresentationName3D"):
                display.SetPreferredDisplayRepresentationName3D("Closed surface")

    def _find_thomas_mask_paths(self, thomas_dir):
        root = os.path.abspath(thomas_dir)
        if not os.path.isdir(root):
            raise ValueError(f"THOMAS output directory not found: {thomas_dir}")
        by_side = {"left": [], "right": []}
        skipped = []
        excluded_tokens = (
            "crop",
            "resampled",
            "ocrop",
            "thomasfull",
            "regn_",
            "sthomas",
            "mask_inp",
        )
        for side in ("left", "right"):
            side_dir = os.path.join(root, side)
            if not os.path.isdir(side_dir):
                continue
            files = sorted(os.listdir(side_dir))
            for fname in files:
                full = os.path.join(side_dir, fname)
                if not os.path.isfile(full):
                    continue
                lower = fname.lower()
                if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                    continue
                if any(tok in lower for tok in excluded_tokens):
                    skipped.append(full)
                    continue
                by_side[side].append(full)
        return by_side, skipped

    def load_thomas_thalamus_masks(self, thomas_dir, logger=None, replace_existing=True, node_name_prefix="THOMAS_"):
        if not hasattr(slicer.modules, "segmentations"):
            raise RuntimeError("Segmentations module is not available in this Slicer install.")
        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            raise RuntimeError("Segmentations logic is unavailable.")

        by_side, skipped = self._find_thomas_mask_paths(thomas_dir)
        total_candidates = sum(len(v) for v in by_side.values())
        if total_candidates == 0:
            raise ValueError(
                "No THOMAS structure masks found under left/right directories "
                "(top-level only; EXTRAS/cropped/resampled files are ignored)."
            )

        loaded_nodes = []
        loaded_paths = []
        failed_paths = []
        for side in ("left", "right"):
            paths = by_side.get(side, [])
            if not paths:
                continue
            seg_name = f"{node_name_prefix}{side.capitalize()}_Structures"
            if replace_existing:
                existing = self._find_node_by_name(seg_name, "vtkMRMLSegmentationNode")
                if existing is not None:
                    slicer.mrmlScene.RemoveNode(existing)
            seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", seg_name)
            seg_node.CreateDefaultDisplayNodes()
            self._style_thomas_segmentation(seg_node, side)
            segmentation = seg_node.GetSegmentation()
            for path in paths:
                label_node = self._load_label_volume_node(path)
                if label_node is None:
                    failed_paths.append(path)
                    continue
                pre_count = segmentation.GetNumberOfSegments()
                seg_logic.ImportLabelmapToSegmentationNode(label_node, seg_node)
                post_count = segmentation.GetNumberOfSegments()
                if post_count <= pre_count:
                    failed_paths.append(path)
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue

                new_segment_id = segmentation.GetNthSegmentID(post_count - 1)
                new_segment = segmentation.GetSegment(new_segment_id)
                if new_segment is not None:
                    seg_label = self._thomas_segment_name_from_path(path)
                    new_segment.SetName(seg_label)
                    color = self._thomas_color_for_label(seg_label, side)
                    new_segment.SetColor(float(color[0]), float(color[1]), float(color[2]))

                loaded_paths.append(path)
                slicer.mrmlScene.RemoveNode(label_node)
                if logger:
                    logger(f"[thomas] loaded structure: {path}")

            if segmentation.GetNumberOfSegments() > 0:
                loaded_nodes.append(seg_node)
            else:
                slicer.mrmlScene.RemoveNode(seg_node)

        return {
            "loaded_nodes": loaded_nodes,
            "loaded_mask_paths": loaded_paths,
            "failed_mask_paths": failed_paths,
            "missing_mask_paths": [],
            "skipped_mask_paths": skipped,
        }

    def load_dicom_scalar_volume_from_directory(self, dicom_dir, logger=None):
        root = os.path.abspath(dicom_dir)
        if not os.path.isdir(root):
            raise ValueError(f"DICOM directory not found: {dicom_dir}")

        files = []
        for walk_root, _dirs, names in os.walk(root):
            for name in names:
                if name.startswith("."):
                    continue
                path = os.path.join(walk_root, name)
                if os.path.isfile(path):
                    files.append(path)
        files.sort()
        if not files:
            raise ValueError(f"No files found under DICOM directory: {dicom_dir}")

        node = None
        try:
            from DICOMScalarVolumePlugin import DICOMScalarVolumePluginClass

            plugin = DICOMScalarVolumePluginClass()
            loadables = plugin.examine([files]) or []
            if loadables:
                loadables.sort(
                    key=lambda l: (float(getattr(l, "confidence", 0.0)), len(getattr(l, "files", []) or [])),
                    reverse=True,
                )
                chosen = loadables[0]
                chosen.selected = True
                if logger:
                    logger(
                        f"[thomas] DICOM candidate: '{getattr(chosen, 'name', 'series')}' "
                        f"confidence={float(getattr(chosen, 'confidence', 0.0)):.2f} "
                        f"files={len(getattr(chosen, 'files', []) or [])}"
                    )
                before_ids = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")}
                loaded = plugin.load(chosen)
                if hasattr(loaded, "GetID"):
                    node = loaded
                else:
                    after_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                    for after_node in after_nodes:
                        if after_node.GetID() not in before_ids:
                            node = after_node
                            break
        except Exception as exc:
            if logger:
                logger(f"[thomas] DICOM plugin import/examine failed, fallback to direct load: {exc}")

        if node is None:
            for path in files:
                try:
                    result = slicer.util.loadVolume(path, returnNode=True)
                    if isinstance(result, tuple):
                        ok, candidate = result
                        if ok and candidate is not None:
                            node = candidate
                            break
                    elif result is not None:
                        node = result
                        break
                except Exception:
                    continue

        if node is None:
            raise RuntimeError(
                "Failed to load DICOM scalar volume from directory. "
                "Select a single-series folder and try again."
            )

        if logger:
            logger(f"[thomas] loaded DICOM MRI: {node.GetName()}")
        return node

    def place_node_under_same_study(self, node, reference_node, logger=None):
        if node is None or reference_node is None:
            return False
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return False
        node_item = sh_node.GetItemByDataNode(node)
        ref_item = sh_node.GetItemByDataNode(reference_node)
        if node_item == 0 or ref_item == 0:
            return False
        ref_study = sh_node.GetItemParent(ref_item)
        if not ref_study:
            return False
        sh_node.SetItemParent(node_item, ref_study)
        if logger:
            logger(f"[thomas] moved {node.GetName()} under study of {reference_node.GetName()}")
        return True

    def export_scalar_volume_to_dicom_series(
        self,
        volume_node,
        reference_volume_node,
        export_dir,
        series_description,
        modality="MR",
        logger=None,
    ):
        if volume_node is None:
            raise ValueError("Volume node is required for DICOM export.")
        if reference_volume_node is None:
            raise ValueError("Reference volume node is required for DICOM export.")

        out_root = os.path.abspath(export_dir)
        os.makedirs(out_root, exist_ok=True)

        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            raise RuntimeError("Subject hierarchy is unavailable.")

        self.place_node_under_same_study(volume_node, reference_volume_node, logger=logger)

        volume_item = sh_node.GetItemByDataNode(volume_node)
        reference_item = sh_node.GetItemByDataNode(reference_volume_node)
        if volume_item == 0:
            raise RuntimeError("Output volume is not in subject hierarchy.")
        if reference_item == 0:
            raise RuntimeError("Reference volume is not in subject hierarchy.")

        reference_study_item = sh_node.GetItemParent(reference_item)
        reference_patient_item = sh_node.GetItemParent(reference_study_item) if reference_study_item else 0
        if reference_study_item:
            sh_node.SetItemParent(volume_item, reference_study_item)

        from DICOMScalarVolumePlugin import DICOMScalarVolumePluginClass

        plugin = DICOMScalarVolumePluginClass()
        exportables = plugin.examineForExport(volume_item) or []
        if not exportables:
            raise RuntimeError("DICOM scalar volume export is unavailable for selected output volume.")
        exportable = exportables[0]
        exportable.directory = out_root
        exportable.setTag("SeriesDescription", series_description or "THOMAS_BURNED")
        if modality:
            exportable.setTag("Modality", modality)

        constants = slicer.vtkMRMLSubjectHierarchyConstants
        patient_tags = [
            constants.GetDICOMPatientNameTagName,
            constants.GetDICOMPatientIDTagName,
            constants.GetDICOMPatientBirthDateTagName,
            constants.GetDICOMPatientSexTagName,
            constants.GetDICOMPatientCommentsTagName,
        ]
        study_tags = [
            constants.GetDICOMStudyIDTagName,
            constants.GetDICOMStudyDateTagName,
            constants.GetDICOMStudyTimeTagName,
            constants.GetDICOMStudyDescriptionTagName,
        ]
        for getter in patient_tags:
            tag_name = getter()
            value = sh_node.GetItemAttribute(reference_patient_item, tag_name) if reference_patient_item else ""
            if value:
                exportable.setTag(tag_name, value)
        for getter in study_tags:
            tag_name = getter()
            value = sh_node.GetItemAttribute(reference_study_item, tag_name) if reference_study_item else ""
            if value:
                exportable.setTag(tag_name, value)

        if logger:
            logger(
                f"[thomas] DICOM export start: volume={volume_node.GetName()} "
                f"series='{exportable.tag('SeriesDescription')}' out={out_root}"
            )
        err = plugin.export([exportable])
        if err:
            raise RuntimeError(err)

        series_dir = os.path.join(out_root, f"ScalarVolume_{int(volume_item)}")
        if logger:
            logger(f"[thomas] DICOM export wrote series directory: {series_dir}")
        return series_dir

    def _thomas_nucleus_from_segment_name(self, name):
        text = (name or "").strip().upper()
        if text.startswith("LEFT_"):
            return text[5:]
        if text.startswith("RIGHT_"):
            return text[6:]
        return text

    def _thomas_side_from_segment_name(self, name, node_name=""):
        segment_text = (name or "").upper()
        node_text = (node_name or "").upper()
        if segment_text.startswith("LEFT_") or node_text.startswith("THOMAS_LEFT"):
            return "left"
        if segment_text.startswith("RIGHT_") or node_text.startswith("THOMAS_RIGHT"):
            return "right"
        return "unknown"

    def collect_thomas_nuclei(self, segmentation_nodes):
        nuclei = set()
        for seg_node in segmentation_nodes or []:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                nucleus = self._thomas_nucleus_from_segment_name(segment.GetName())
                if nucleus:
                    nuclei.add(nucleus)
        return sorted(nuclei)

    def burn_thomas_nucleus_to_volume(
        self,
        segmentation_nodes,
        input_volume_node,
        nucleus,
        side="Both",
        fill_value=1200.0,
        output_name="THOMAS_Burned_MRI",
        logger=None,
    ):
        if np is None:
            raise RuntimeError("NumPy is required for burn workflow.")
        if input_volume_node is None:
            raise ValueError("Input volume node is required.")
        nucleus_token = (nucleus or "").strip().upper()
        if not nucleus_token:
            raise ValueError("Nucleus name is required.")

        side_text = (side or "Both").strip().lower()
        if side_text not in ("left", "right", "both"):
            raise ValueError("Side must be Left, Right, or Both.")
        target_sides = {"left", "right"} if side_text == "both" else {side_text}

        selected_segments = []
        for seg_node in segmentation_nodes or []:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            node_name = seg_node.GetName() or ""
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                seg_name = segment.GetName() or ""
                seg_side = self._thomas_side_from_segment_name(seg_name, node_name=node_name)
                seg_nucleus = self._thomas_nucleus_from_segment_name(seg_name)
                if seg_side in target_sides and seg_nucleus == nucleus_token:
                    selected_segments.append((seg_node, seg_id, seg_name))

        if not selected_segments:
            available = ", ".join(self.collect_thomas_nuclei(segmentation_nodes))
            raise ValueError(
                f"No THOMAS segments matched nucleus '{nucleus_token}' and side '{side}'. "
                f"Available nuclei: {available or 'none'}"
            )

        volumes_logic = slicer.modules.volumes.logic()
        if volumes_logic is None:
            raise RuntimeError("Volumes logic is unavailable.")
        out_volume = volumes_logic.CloneVolume(slicer.mrmlScene, input_volume_node, output_name)
        if out_volume is None:
            raise RuntimeError("Failed to create output burn volume.")

        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            raise RuntimeError("Segmentations logic is unavailable.")

        out_arr = slicer.util.arrayFromVolume(out_volume)
        fill_cast = np.asarray([fill_value], dtype=out_arr.dtype)[0]
        total_voxels = 0

        for seg_node, seg_id, seg_name in selected_segments:
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"__tmp_{seg_node.GetName()}_{seg_id}")
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(seg_id)
            ok = seg_logic.ExportSegmentsToLabelmapNode(seg_node, ids, labelmap_node, input_volume_node)
            if not ok:
                slicer.mrmlScene.RemoveNode(labelmap_node)
                raise RuntimeError(f"Failed exporting segment '{seg_name}' to labelmap.")
            label_arr = slicer.util.arrayFromVolume(labelmap_node)
            mask = label_arr > 0
            voxels = int(mask.sum())
            if voxels > 0:
                out_arr[mask] = fill_cast
                total_voxels += voxels
            slicer.mrmlScene.RemoveNode(labelmap_node)

        slicer.util.arrayFromVolumeModified(out_volume)
        if logger:
            logger(
                f"[thomas] burned nucleus {nucleus_token} ({side}) using {len(selected_segments)} segment(s), "
                f"voxels={total_voxels}, fill={float(fill_cast)} -> {out_volume.GetName()}"
            )
        return out_volume

    def _volume_label_lookup_name(self, volume_node, label_value):
        if volume_node is None:
            return ""
        display = volume_node.GetDisplayNode()
        if display is None:
            return ""
        color_node = display.GetColorNode()
        if color_node is None:
            return ""
        try:
            name = color_node.GetColorName(int(label_value))
        except Exception:
            name = ""
        return str(name) if name else ""

    def _build_volume_label_index(self, volume_node):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if volume_node is None:
            return None
        arr = slicer.util.arrayFromVolume(volume_node)
        idx = np.argwhere(arr > 0)
        if idx.size == 0:
            return None

        labels = arr[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)
        ijk_h = np.ones((idx.shape[0], 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2].astype(float)
        ijk_h[:, 1] = idx[:, 1].astype(float)
        ijk_h[:, 2] = idx[:, 0].astype(float)

        ijk_to_world = self._volume_ijk_to_world_matrix(volume_node)
        ras = (ijk_h @ ijk_to_world.T)[:, :3]

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(int(ras.shape[0]))
        for i in range(int(ras.shape[0])):
            points.SetPoint(i, float(ras[i, 0]), float(ras[i, 1]), float(ras[i, 2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        centroids = {}
        label_values = np.unique(labels)
        for label_value in label_values:
            xyz = ras[labels == int(label_value)]
            if xyz.size:
                centroids[int(label_value)] = xyz.mean(axis=0)

        label_names = {}
        for value in label_values:
            name = self._volume_label_lookup_name(volume_node, int(value))
            label_names[int(value)] = name or f"Label_{int(value)}"

        return {
            "locator": locator,
            "points": points,
            "labels": labels,
            "centroids": centroids,
            "label_names": label_names,
        }

    def _build_segmentation_label_index(self, segmentation_nodes, reference_volume_node, skip_generic_thalamus=True):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")
        if not segmentation_nodes:
            return None
        if reference_volume_node is None:
            raise ValueError("Reference volume node is required for segmentation atlas indexing.")
        if not hasattr(slicer.modules, "segmentations"):
            return None
        seg_logic = slicer.modules.segmentations.logic()
        if seg_logic is None:
            return None

        all_points = []
        all_labels = []
        label_names = {}
        label_value = 1
        for seg_node in segmentation_nodes:
            if seg_node is None:
                continue
            segmentation = seg_node.GetSegmentation()
            if segmentation is None:
                continue
            for i in range(segmentation.GetNumberOfSegments()):
                seg_id = segmentation.GetNthSegmentID(i)
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue
                seg_name = segment.GetName() or ""
                if skip_generic_thalamus and self._thomas_nucleus_from_segment_name(seg_name) == "THALAMUS":
                    continue
                label_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tmp_atlas_seg")
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(seg_id)
                ok = seg_logic.ExportSegmentsToLabelmapNode(seg_node, ids, label_node, reference_volume_node)
                if not ok:
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue
                arr = slicer.util.arrayFromVolume(label_node)
                idx = np.argwhere(arr > 0)
                if idx.size == 0:
                    slicer.mrmlScene.RemoveNode(label_node)
                    continue
                ijk_h = np.ones((idx.shape[0], 4), dtype=float)
                ijk_h[:, 0] = idx[:, 2].astype(float)
                ijk_h[:, 1] = idx[:, 1].astype(float)
                ijk_h[:, 2] = idx[:, 0].astype(float)
                ijk_to_ras_vtk = vtk.vtkMatrix4x4()
                label_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
                ijk_to_ras = self._vtk_matrix_to_numpy_4x4(ijk_to_ras_vtk)
                ras = (ijk_h @ ijk_to_ras.T)[:, :3]
                all_points.append(ras)
                all_labels.append(np.full((ras.shape[0],), int(label_value), dtype=np.int32))
                label_names[int(label_value)] = seg_name or f"Segment_{label_value}"
                label_value += 1
                slicer.mrmlScene.RemoveNode(label_node)

        if not all_points:
            if skip_generic_thalamus:
                return self._build_segmentation_label_index(
                    segmentation_nodes=segmentation_nodes,
                    reference_volume_node=reference_volume_node,
                    skip_generic_thalamus=False,
                )
            return None

        ras_all = np.vstack(all_points)
        labels_all = np.concatenate(all_labels)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(int(ras_all.shape[0]))
        for i in range(int(ras_all.shape[0])):
            points.SetPoint(i, float(ras_all[i, 0]), float(ras_all[i, 1]), float(ras_all[i, 2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        centroids = {}
        for value in np.unique(labels_all):
            xyz = ras_all[labels_all == int(value)]
            if xyz.size:
                centroids[int(value)] = xyz.mean(axis=0)

        return {
            "locator": locator,
            "points": points,
            "labels": labels_all,
            "centroids": centroids,
            "label_names": label_names,
        }

    def _query_label_index(self, index_cache, point_ras):
        if not index_cache:
            return {
                "label": "",
                "label_value": 0,
                "distance_to_voxel_mm": 0.0,
                "distance_to_centroid_mm": 0.0,
            }
        locator = index_cache["locator"]
        point_id = int(locator.FindClosestPoint(float(point_ras[0]), float(point_ras[1]), float(point_ras[2])))
        nearest = [0.0, 0.0, 0.0]
        index_cache["points"].GetPoint(point_id, nearest)
        dx = float(point_ras[0]) - float(nearest[0])
        dy = float(point_ras[1]) - float(nearest[1])
        dz = float(point_ras[2]) - float(nearest[2])
        dv = math.sqrt(dx * dx + dy * dy + dz * dz)

        label_value = int(index_cache["labels"][point_id])
        label_name = index_cache["label_names"].get(label_value, f"Label_{label_value}")
        centroid = index_cache["centroids"].get(label_value)
        dc = 0.0
        if centroid is not None:
            cx = float(point_ras[0]) - float(centroid[0])
            cy = float(point_ras[1]) - float(centroid[1])
            cz = float(point_ras[2]) - float(centroid[2])
            dc = math.sqrt(cx * cx + cy * cy + cz * cz)

        return {
            "label": label_name,
            "label_value": label_value,
            "distance_to_voxel_mm": float(dv),
            "distance_to_centroid_mm": float(dc),
        }

    def assign_contacts_to_atlases(
        self,
        contacts,
        freesurfer_volume_node=None,
        thomas_segmentation_nodes=None,
        wm_volume_node=None,
        reference_volume_node=None,
        prefer_thomas=True,
    ):
        if np is None:
            raise RuntimeError("NumPy is required for atlas assignment.")

        fs_idx = self._build_volume_label_index(freesurfer_volume_node) if freesurfer_volume_node else None
        wm_idx = self._build_volume_label_index(wm_volume_node) if wm_volume_node else None
        th_idx = None
        if thomas_segmentation_nodes:
            th_idx = self._build_segmentation_label_index(thomas_segmentation_nodes, reference_volume_node)
        th_native_node = None
        if thomas_segmentation_nodes:
            for node in thomas_segmentation_nodes:
                if node is not None:
                    th_native_node = node
                    break

        rows = []
        for contact in contacts or []:
            point_ras = lps_to_ras_point(contact.get("position_lps", [0.0, 0.0, 0.0]))
            th_native_ras = self._world_to_node_ras_point(th_native_node, point_ras) if th_native_node is not None else [0.0, 0.0, 0.0]
            fs_native_ras = self._world_to_node_ras_point(freesurfer_volume_node, point_ras) if freesurfer_volume_node is not None else [0.0, 0.0, 0.0]
            wm_native_ras = self._world_to_node_ras_point(wm_volume_node, point_ras) if wm_volume_node is not None else [0.0, 0.0, 0.0]
            th = self._query_label_index(th_idx, point_ras) if th_idx else None
            fs = self._query_label_index(fs_idx, point_ras) if fs_idx else None
            wm = self._query_label_index(wm_idx, point_ras) if wm_idx else None

            choices = []
            if fs is not None and fs.get("label"):
                choices.append(("freesurfer", fs))
            if th is not None and th.get("label"):
                choices.append(("thomas", th))
            if wm is not None and wm.get("label"):
                choices.append(("wm", wm))

            closest_source = ""
            closest = {"label": "", "label_value": 0, "distance_to_voxel_mm": 0.0, "distance_to_centroid_mm": 0.0}
            if choices:
                closest_source, closest = min(choices, key=lambda item: float(item[1].get("distance_to_voxel_mm", float("inf"))))

            primary_source = ""
            primary = {"label": "", "label_value": 0, "distance_to_voxel_mm": 0.0, "distance_to_centroid_mm": 0.0}
            if prefer_thomas and th is not None and th.get("label"):
                primary_source, primary = "thomas", th
            else:
                primary_source, primary = closest_source, closest

            rows.append(
                {
                    "trajectory": contact.get("trajectory", ""),
                    "contact_label": contact.get("label", ""),
                    "contact_index": int(contact.get("index", 0)),
                    "contact_ras": [float(point_ras[0]), float(point_ras[1]), float(point_ras[2])],
                    "closest_source": closest_source,
                    "closest_label": closest.get("label", ""),
                    "closest_label_value": int(closest.get("label_value", 0)),
                    "closest_distance_to_voxel_mm": float(closest.get("distance_to_voxel_mm", 0.0)),
                    "closest_distance_to_centroid_mm": float(closest.get("distance_to_centroid_mm", 0.0)),
                    "primary_source": primary_source,
                    "primary_label": primary.get("label", ""),
                    "primary_label_value": int(primary.get("label_value", 0)),
                    "primary_distance_to_voxel_mm": float(primary.get("distance_to_voxel_mm", 0.0)),
                    "primary_distance_to_centroid_mm": float(primary.get("distance_to_centroid_mm", 0.0)),
                    "thomas_label": "" if th is None else th.get("label", ""),
                    "thomas_label_value": 0 if th is None else int(th.get("label_value", 0)),
                    "thomas_distance_to_voxel_mm": 0.0 if th is None else float(th.get("distance_to_voxel_mm", 0.0)),
                    "thomas_distance_to_centroid_mm": 0.0 if th is None else float(th.get("distance_to_centroid_mm", 0.0)),
                    "freesurfer_label": "" if fs is None else fs.get("label", ""),
                    "freesurfer_label_value": 0 if fs is None else int(fs.get("label_value", 0)),
                    "freesurfer_distance_to_voxel_mm": 0.0 if fs is None else float(fs.get("distance_to_voxel_mm", 0.0)),
                    "freesurfer_distance_to_centroid_mm": 0.0 if fs is None else float(fs.get("distance_to_centroid_mm", 0.0)),
                    "wm_label": "" if wm is None else wm.get("label", ""),
                    "wm_label_value": 0 if wm is None else int(wm.get("label_value", 0)),
                    "wm_distance_to_voxel_mm": 0.0 if wm is None else float(wm.get("distance_to_voxel_mm", 0.0)),
                    "wm_distance_to_centroid_mm": 0.0 if wm is None else float(wm.get("distance_to_centroid_mm", 0.0)),
                    "thomas_native_ras": th_native_ras,
                    "freesurfer_native_ras": fs_native_ras,
                    "wm_native_ras": wm_native_ras,
                }
            )

        rows.sort(key=lambda r: (str(r.get("trajectory", "")), int(r.get("contact_index", 0))))
        return rows

    def publish_atlas_assignment_rows(self, atlas_rows, workflow_node=None):
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        columns = [
            "trajectory", "contact_label", "contact_index", "x_ras", "y_ras", "z_ras",
            "closest_source", "closest_label", "closest_label_value", "closest_distance_to_voxel_mm", "closest_distance_to_centroid_mm",
            "primary_source", "primary_label", "primary_label_value", "primary_distance_to_voxel_mm", "primary_distance_to_centroid_mm",
            "thomas_label", "thomas_label_value", "thomas_distance_to_voxel_mm", "thomas_distance_to_centroid_mm",
            "freesurfer_label", "freesurfer_label_value", "freesurfer_distance_to_voxel_mm", "freesurfer_distance_to_centroid_mm",
            "wm_label", "wm_label_value", "wm_distance_to_voxel_mm", "wm_distance_to_centroid_mm",
            "thomas_native_x_ras", "thomas_native_y_ras", "thomas_native_z_ras",
            "freesurfer_native_x_ras", "freesurfer_native_y_ras", "freesurfer_native_z_ras",
            "wm_native_x_ras", "wm_native_y_ras", "wm_native_z_ras",
        ]
        rows = []
        for row in atlas_rows or []:
            p_ras = row.get("contact_ras", [0.0, 0.0, 0.0])
            th_native = row.get("thomas_native_ras", [0.0, 0.0, 0.0])
            fs_native = row.get("freesurfer_native_ras", [0.0, 0.0, 0.0])
            wm_native = row.get("wm_native_ras", [0.0, 0.0, 0.0])
            rows.append(
                {
                    "trajectory": row.get("trajectory", ""),
                    "contact_label": row.get("contact_label", ""),
                    "contact_index": row.get("contact_index", 0),
                    "x_ras": float(p_ras[0]),
                    "y_ras": float(p_ras[1]),
                    "z_ras": float(p_ras[2]),
                    "closest_source": row.get("closest_source", ""),
                    "closest_label": row.get("closest_label", ""),
                    "closest_label_value": row.get("closest_label_value", 0),
                    "closest_distance_to_voxel_mm": row.get("closest_distance_to_voxel_mm", 0.0),
                    "closest_distance_to_centroid_mm": row.get("closest_distance_to_centroid_mm", 0.0),
                    "primary_source": row.get("primary_source", ""),
                    "primary_label": row.get("primary_label", ""),
                    "primary_label_value": row.get("primary_label_value", 0),
                    "primary_distance_to_voxel_mm": row.get("primary_distance_to_voxel_mm", 0.0),
                    "primary_distance_to_centroid_mm": row.get("primary_distance_to_centroid_mm", 0.0),
                    "thomas_label": row.get("thomas_label", ""),
                    "thomas_label_value": row.get("thomas_label_value", 0),
                    "thomas_distance_to_voxel_mm": row.get("thomas_distance_to_voxel_mm", 0.0),
                    "thomas_distance_to_centroid_mm": row.get("thomas_distance_to_centroid_mm", 0.0),
                    "freesurfer_label": row.get("freesurfer_label", ""),
                    "freesurfer_label_value": row.get("freesurfer_label_value", 0),
                    "freesurfer_distance_to_voxel_mm": row.get("freesurfer_distance_to_voxel_mm", 0.0),
                    "freesurfer_distance_to_centroid_mm": row.get("freesurfer_distance_to_centroid_mm", 0.0),
                    "wm_label": row.get("wm_label", ""),
                    "wm_label_value": row.get("wm_label_value", 0),
                    "wm_distance_to_voxel_mm": row.get("wm_distance_to_voxel_mm", 0.0),
                    "wm_distance_to_centroid_mm": row.get("wm_distance_to_centroid_mm", 0.0),
                    "thomas_native_x_ras": float(th_native[0]),
                    "thomas_native_y_ras": float(th_native[1]),
                    "thomas_native_z_ras": float(th_native[2]),
                    "freesurfer_native_x_ras": float(fs_native[0]),
                    "freesurfer_native_y_ras": float(fs_native[1]),
                    "freesurfer_native_z_ras": float(fs_native[2]),
                    "wm_native_x_ras": float(wm_native[0]),
                    "wm_native_y_ras": float(wm_native[1]),
                    "wm_native_z_ras": float(wm_native[2]),
                }
            )

        table = self.electrode_scene.create_or_update_table_node(
            node_name="Rosa_AtlasAssignments",
            columns=columns,
            rows=rows,
        )
        self.workflow_state.set_single_role("AtlasAssignmentTable", table, workflow_node=wf)
        self.workflow_state.tag_node(
            table,
            role="AtlasAssignmentTable",
            source="atlas",
            space="ROSA_BASE",
            signature=table.GetID(),
            workflow_node=wf,
        )
