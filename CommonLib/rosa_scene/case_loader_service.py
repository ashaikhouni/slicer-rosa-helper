"""Slicer scene helpers for ROSA case loading and basic volume utilities."""

from __future__ import annotations

import os

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from __main__ import slicer, vtk

from rosa_core import (
    build_effective_matrices,
    choose_reference_volume,
    find_ros_file,
    invert_4x4,
    lps_to_ras_matrix,
    parse_ros_file,
    resolve_analyze_volume,
    resolve_reference_index,
)


class CaseLoaderService:
    """Scene-loading and volume-display utilities used across ROSA modules."""

    def load_case(
        self,
        case_dir,
        reference=None,
        invert=False,
        harden=True,
        load_trajectories=True,
        show_planned=False,
        add_trajectories_fn=None,
        logger=None,
    ):
        """Load a ROSA case directory into the active Slicer scene."""

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
        effective_used_lps = [invert_4x4(m) for m in effective_lps] if invert else effective_lps

        log(f"[ros] {ros_path}")
        log(f"[ref] {reference_volume}")

        loaded_count = 0
        loaded_volume_node_ids = {}
        loaded_volume_source_paths = {}
        ros_transform_records = []

        for i, disp in enumerate(displays):
            vol_name = disp["volume"]
            img_path = resolve_analyze_volume(analyze_root, disp)
            if not img_path:
                log(f"[skip] missing Analyze .img for {vol_name}")
                continue

            vol_node = self.load_volume(img_path)
            if vol_node is None:
                log(f"[skip] failed to load {img_path}")
                continue

            loaded_count += 1
            loaded_volume_node_ids[vol_name] = vol_node.GetID()
            loaded_volume_source_paths[vol_name] = img_path
            vol_node.SetName(vol_name)
            self.center_volume(vol_node)
            log(f"[load] {vol_name}")
            log(f"[center] {vol_name}")

            if vol_name != reference_volume:
                matrix_ras = lps_to_ras_matrix(effective_used_lps[i])
                tnode = self.apply_transform(vol_node, matrix_ras)
                tnode.SetName(f"{vol_name}_to_{reference_volume}_ROSA")
                ref_idx = disp.get("imagery_3dref", root_index)
                log(
                    f"[xform] {vol_name} {'inv ' if invert else ''}TRdicomRdisplay "
                    f"(ref idx {ref_idx} -> root idx {root_index})"
                )
                if harden:
                    slicer.vtkSlicerTransformLogic().hardenTransform(vol_node)
                    vol_node.SetAndObserveTransformNodeID(None)
                    log(f"[harden] {vol_name}")
                tnode.SetAttribute("Rosa.Managed", "1")
                tnode.SetAttribute("Rosa.Source", "rosa")
                tnode.SetAttribute("Rosa.Space", "ROSA_BASE")
                tnode.SetAttribute("Rosa.NativeToBaseForNodeID", vol_node.GetID())
                tnode.SetAttribute("Rosa.NativeToBaseForNodeName", vol_name)
                ros_transform_records.append(
                    {
                        "volume_name": vol_name,
                        "transform_node_id": tnode.GetID(),
                        "from_space": f"{vol_name}_NATIVE",
                        "to_space": "ROSA_BASE",
                    }
                )
            else:
                log(f"[xform] {vol_name} reference (none)")

        if load_trajectories and trajectories:
            if not callable(add_trajectories_fn):
                raise ValueError("add_trajectories_fn is required when load_trajectories=True")
            add_trajectories_fn(trajectories, logger=log, show_planned=show_planned)

        return {
            "loaded_volumes": loaded_count,
            "loaded_volume_node_ids": loaded_volume_node_ids,
            "loaded_volume_source_paths": loaded_volume_source_paths,
            "reference_volume": reference_volume,
            "trajectory_count": len(trajectories) if load_trajectories else 0,
            "trajectories": trajectories,
            "ros_path": ros_path,
            "analyze_root": analyze_root,
            "ros_transform_records": ros_transform_records,
        }

    def load_volume(self, path):
        """Load scalar volume by path and return MRML node."""
        try:
            result = slicer.util.loadVolume(path, returnNode=True)
            if isinstance(result, tuple):
                ok, node = result
                return node if ok else None
            return result
        except TypeError:
            return slicer.util.loadVolume(path)

    def center_volume(self, volume_node):
        """Center volume origin (Volumes module equivalent)."""
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

    def apply_transform(self, volume_node, matrix4x4):
        """Assign a new linear transform node built from a 4x4 matrix."""
        tnode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        vtk_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_mat.SetElement(r, c, matrix4x4[r][c])
        tnode.SetMatrixTransformToParent(vtk_mat)
        volume_node.SetAndObserveTransformNodeID(tnode.GetID())
        return tnode

    def vtk_matrix_to_numpy(self, vtk_matrix4x4):
        """Convert vtkMatrix4x4 to NumPy 4x4."""
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix4x4.GetElement(r, c))
        return out

    def extract_threshold_candidates_lps(self, volume_node, threshold, max_points=300000):
        """Extract thresholded candidate points in LPS coordinates."""
        if np is None:
            raise RuntimeError("NumPy is required for candidate extraction.")
        arr = slicer.util.arrayFromVolume(volume_node)  # K, J, I
        idx = np.argwhere(arr >= float(threshold))
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)

        if idx.shape[0] > int(max_points):
            rng = np.random.default_rng(0)
            keep = rng.choice(idx.shape[0], size=int(max_points), replace=False)
            idx = idx[keep]

        n = idx.shape[0]
        ijk_h = np.ones((n, 4), dtype=float)
        ijk_h[:, 0] = idx[:, 2].astype(float)
        ijk_h[:, 1] = idx[:, 1].astype(float)
        ijk_h[:, 2] = idx[:, 0].astype(float)

        ijk_to_ras_vtk = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_vtk)
        ijk_to_ras = self.vtk_matrix_to_numpy(ijk_to_ras_vtk)
        ras_h = ijk_h @ ijk_to_ras.T
        ras = ras_h[:, :3]
        lps = ras.copy()
        lps[:, 0] *= -1.0
        lps[:, 1] *= -1.0
        return lps

    def show_volume_in_all_slice_views(self, volume_node):
        """Set volume as slice background for all slice composite nodes."""
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)

    def apply_ct_window_from_threshold(self, volume_node, threshold):
        """Apply a CT display window derived from detection threshold."""
        if volume_node is None:
            return
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        lower = float(threshold) - 250.0
        upper = float(threshold) + 2200.0
        display.AutoWindowLevelOff()
        display.SetWindow(max(upper - lower, 1.0))
        display.SetLevel((upper + lower) * 0.5)

    def reset_ct_window(self, volume_node):
        """Reset CT display window/level to auto mode."""
        if volume_node is None:
            return
        display = volume_node.GetDisplayNode()
        if display is None:
            return
        display.AutoWindowLevelOn()
