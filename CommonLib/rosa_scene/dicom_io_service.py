"""DICOM import/export helpers for atlas/navigation workflows."""

from __future__ import annotations

import os

from __main__ import slicer


class DicomIOService:
    """Import/export scalar volumes from/to DICOM directories."""

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
