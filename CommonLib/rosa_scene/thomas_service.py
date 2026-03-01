"""THOMAS segmentation load/style/burn helpers."""

from __future__ import annotations

import hashlib
import os

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from __main__ import slicer, vtk


class ThomasService:
    """Operations related to THOMAS segmentation assets."""

    def __init__(self, utils):
        self.utils = utils

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
                existing = self.utils.find_node_by_name(seg_name, "vtkMRMLSegmentationNode")
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
