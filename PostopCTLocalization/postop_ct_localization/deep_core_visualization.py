import json
import os

try:
    import numpy as np
except ImportError:
    np = None

from __main__ import qt, slicer, vtk

from shank_core.masking import build_preview_masks

class DeepCoreVisualizationLogicMixin:
    @staticmethod
    def _remove_node_if_exists(node_name):
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    @staticmethod
    def _remove_nodes_with_prefix(node_prefix):
        prefix = str(node_prefix or "")
        if not prefix:
            return
        for class_name in ("vtkMRMLMarkupsFiducialNode", "vtkMRMLModelNode"):
            try:
                nodes = list(slicer.util.getNodesByClass(class_name) or [])
            except Exception:
                nodes = []
            for node in nodes:
                try:
                    name = str(node.GetName() or "")
                except Exception:
                    name = ""
                if name.startswith(prefix):
                    slicer.mrmlScene.RemoveNode(node)

    def _get_or_create_labelmap_node(self, node_name):
        node = None
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", node_name)
            node.CreateDefaultDisplayNodes()
        return node

    @staticmethod

    @staticmethod

    @staticmethod
    def _copy_parent_transform(reference_node, target_node):
        if reference_node is None or target_node is None:
            return
        try:
            target_node.SetAndObserveTransformNodeID(reference_node.GetTransformNodeID())
        except Exception:
            pass

    def _update_labelmap_from_mask(self, reference_volume_node, node_name, mask_kji):
        if mask_kji is None:
            return None
        node = self._get_or_create_labelmap_node(node_name)
        arr = np.asarray(mask_kji)
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.integer):
            # Keep integer label depth for multi-component debug maps.
            if arr.max() > np.iinfo(np.uint8).max:
                arr = arr.astype(np.uint16)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        slicer.util.updateVolumeFromArray(node, arr)
        m = vtk.vtkMatrix4x4()
        reference_volume_node.GetIJKToRASMatrix(m)
        node.SetIJKToRASMatrix(m)
        self._copy_parent_transform(reference_volume_node, node)
        display = node.GetDisplayNode()
        if display:
            try:
                color = slicer.util.getNode("GenericAnatomyColors")
                display.SetAndObserveColorNodeID(color.GetID())
            except Exception:
                pass
            display.SetVisibility(True)
        return node

    def _get_or_create_scalar_volume_node(self, node_name):
        node = None
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", node_name)
            node.CreateDefaultDisplayNodes()
        return node

    def _update_scalar_volume_from_array(self, reference_volume_node, node_name, array_kji,
                                          ijk_to_ras_mat=None):
        """Push a numpy array into a Slicer scalar volume.

        ``ijk_to_ras_mat``: optional 4x4 matrix (numpy or VTK) to use
        for the new volume's geometry. When omitted, copies the
        reference volume's IJK->RAS — which is wrong if the array was
        computed on a different grid (e.g. canonical-1mm-resampled
        feature volumes on raw sub-mm CT input).
        """
        if array_kji is None:
            return None
        node = self._get_or_create_scalar_volume_node(node_name)
        arr = np.asarray(array_kji)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        slicer.util.updateVolumeFromArray(node, arr)
        m = vtk.vtkMatrix4x4()
        if ijk_to_ras_mat is not None:
            np_mat = np.asarray(ijk_to_ras_mat, dtype=float)
            for r in range(4):
                for c in range(4):
                    m.SetElement(r, c, float(np_mat[r, c]))
        else:
            reference_volume_node.GetIJKToRASMatrix(m)
        node.SetIJKToRASMatrix(m)
        self._copy_parent_transform(reference_volume_node, node)
        display = node.GetDisplayNode()
        if display is not None:
            try:
                display.AutoWindowLevelOn()
            except Exception:
                pass
            display.SetVisibility(True)
        return node

    def _get_or_create_model_node(self, node_name, color_rgb=(0.1, 0.9, 0.9)):
        node = None
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", node_name)
            node.CreateDefaultDisplayNodes()
            node.SetAttribute("Rosa.Managed", "1")
        display = node.GetDisplayNode()
        if display is not None:
            display.SetColor(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
            display.SetSelectedColor(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
            display.SetOpacity(1.0)
            display.SetVisibility(True)
            if hasattr(display, "SetLineWidth"):
                display.SetLineWidth(3.0)
            if hasattr(display, "SetRepresentation"):
                display.SetRepresentation(1)
        return node

    @staticmethod
    def _build_polydata_from_line_segments(segments_ras):
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for start_ras, end_ras in list(segments_ras or []):
            start = np.asarray(start_ras, dtype=float).reshape(3)
            end = np.asarray(end_ras, dtype=float).reshape(3)
            if float(np.linalg.norm(end - start)) <= 1e-6:
                continue
            start_id = points.InsertNextPoint(float(start[0]), float(start[1]), float(start[2]))
            end_id = points.InsertNextPoint(float(end[0]), float(end[1]), float(end[2]))
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(start_id))
            line.GetPointIds().SetId(1, int(end_id))
            lines.InsertNextCell(line)
        poly.SetPoints(points)
        poly.SetLines(lines)
        return poly

    def show_metal_and_head_masks(
        self,
        volume_node,
        metal_mask_kji=None,
        head_mask_kji=None,
        head_distance_map_kji=None,
        distance_surface_mask_kji=None,
        not_air_mask_kji=None,
        not_air_eroded_mask_kji=None,
        head_core_mask_kji=None,
        metal_gate_mask_kji=None,
        metal_in_gate_mask_kji=None,
        depth_window_mask_kji=None,
        metal_depth_pass_mask_kji=None,
        blob_kept_mask_kji=None,
        blob_rejected_mask_kji=None,
    ):
        """Display troubleshooting masks for de novo shank detection."""
        if metal_mask_kji is None:
            raise ValueError("metal_mask_kji is required for mask visualization")

        metal_bool = np.asarray(metal_mask_kji, dtype=bool)
        self._update_labelmap_from_mask(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_MetalMask",
            mask_kji=metal_mask_kji,
        )
        if head_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HeadMask",
                mask_kji=head_mask_kji,
            )
        distance_map_inside_mask = None
        if head_distance_map_kji is not None:
            distance_map_inside_mask = np.asarray(head_distance_map_kji, dtype=float) > 0.0
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DistanceMapMask",
                mask_kji=distance_map_inside_mask,
            )
        if distance_surface_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DistanceSurfaceMask",
                mask_kji=distance_surface_mask_kji,
            )
        if not_air_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_NotAirMask",
                mask_kji=not_air_mask_kji,
            )
        if not_air_eroded_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_NotAirErodedMask",
                mask_kji=not_air_eroded_mask_kji,
            )
        if head_core_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_HeadCoreMask",
                mask_kji=head_core_mask_kji,
            )
        if metal_gate_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_MetalGateMask",
                mask_kji=metal_gate_mask_kji,
            )
        if metal_in_gate_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_MetalInGateMask",
                mask_kji=metal_in_gate_mask_kji,
            )
        if depth_window_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DepthWindowMask",
                mask_kji=depth_window_mask_kji,
            )
        if metal_depth_pass_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_DepthPassMetalMask",
                mask_kji=metal_depth_pass_mask_kji,
            )
        if blob_kept_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_BlobKeptMetalMask",
                mask_kji=blob_kept_mask_kji,
            )
        if blob_rejected_mask_kji is not None:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_BlobRejectedMetalMask",
                mask_kji=blob_rejected_mask_kji,
            )

        # Overlay label IDs (for troubleshooting):
        # 1=head/gating, 2=metal, 3=depth window, 4=row/column distance surface,
        # 5=metal kept after depth gate, 6=distance-map inside mask, 7=metal-in-gate,
        # 8=blob-kept metal voxels, 9=blob-rejected metal voxels.
        combo = np.zeros_like(np.asarray(metal_mask_kji, dtype=np.uint8), dtype=np.uint8)
        if head_mask_kji is not None:
            combo[np.asarray(head_mask_kji, dtype=bool)] = 1
        if distance_map_inside_mask is not None:
            combo[np.asarray(distance_map_inside_mask, dtype=bool)] = 6
        if distance_surface_mask_kji is not None:
            combo[np.asarray(distance_surface_mask_kji, dtype=bool)] = 4
        if depth_window_mask_kji is not None:
            combo[np.asarray(depth_window_mask_kji, dtype=bool)] = 3
        combo[metal_bool] = 2
        if metal_in_gate_mask_kji is not None:
            combo[np.asarray(metal_in_gate_mask_kji, dtype=bool)] = 7
        if metal_depth_pass_mask_kji is not None:
            combo[np.asarray(metal_depth_pass_mask_kji, dtype=bool)] = 5
        if blob_kept_mask_kji is not None:
            combo[np.asarray(blob_kept_mask_kji, dtype=bool)] = 8
        if blob_rejected_mask_kji is not None:
            combo[np.asarray(blob_rejected_mask_kji, dtype=bool)] = 9
        combo_node = self._update_labelmap_from_mask(
            reference_volume_node=volume_node,
            node_name=f"{volume_node.GetName()}_MaskOverlay",
            mask_kji=combo,
        )
        slicer.util.setSliceViewerLayers(
            background=volume_node,
            foreground=None,
            foregroundOpacity=0.0,
            label=combo_node,
            labelOpacity=0.55,
        )

    @staticmethod
    def _deep_core_proposal_family_style(family):
        family_key = str(family or "").strip().lower()
        style_map = {
            "graph": {"color": (0.15, 0.85, 0.20), "tag": "G"},
            "blob_connectivity": {"color": (0.10, 0.85, 0.95), "tag": "B"},
            "blob_axis": {"color": (0.95, 0.60, 0.15), "tag": "A"},
        }
        return dict(style_map.get(family_key, {"color": (0.75, 0.75, 0.75), "tag": "X"}))

    def show_deep_core_proposals(self, volume_node, proposals):
        self.trajectory_scene.remove_preview_lines()
        star_node_name = f"{volume_node.GetName()}_DeepCoreSkullJumpStars" if volume_node is not None else "DeepCoreSkullJumpStars"
        shallow_node_name = f"{volume_node.GetName()}_DeepCoreProposalShallow" if volume_node is not None else "DeepCoreProposalShallow"
        deep_node_name = f"{volume_node.GetName()}_DeepCoreProposalDeep" if volume_node is not None else "DeepCoreProposalDeep"
        self._remove_node_if_exists(star_node_name)
        self._remove_node_if_exists(shallow_node_name)
        self._remove_node_if_exists(deep_node_name)
        nodes = []
        skull_jump_records = []
        shallow_records = []
        deep_records = []
        for idx, proposal in enumerate(list(proposals or []), start=1):
            family = str(proposal.get("proposal_family") or "unknown")
            style = self._deep_core_proposal_family_style(family)
            line_tag = str(style.get("tag") or "X")
            start_ras = list(proposal.get("start_ras") or [0.0, 0.0, 0.0])
            end_ras = list(proposal.get("end_ras") or [0.0, 0.0, 0.0])
            node = self.trajectory_scene.create_or_update_trajectory_line(
                name=f"{line_tag}{idx:02d}",
                start_ras=start_ras,
                end_ras=end_ras,
                node_id=None,
                group="auto_fit",
                origin="contact_pitch_v1",
                node_name=f"AutoFit_{family}_{idx:02d}",
            )
            # Immutable baseline RAS so the user can revert post-edit. The
            # control points themselves are user-draggable; these attrs
            # are written once at publish time and never re-stamped.
            node.SetAttribute(
                "Rosa.AutoFitStartRas",
                ",".join(f"{float(v):.6f}" for v in start_ras),
            )
            node.SetAttribute(
                "Rosa.AutoFitEndRas",
                ",".join(f"{float(v):.6f}" for v in end_ras),
            )
            node.SetAttribute("Rosa.DeepCoreProposalFamily", family)
            node.SetAttribute("Rosa.DeepCoreProposalScore", f"{float(proposal.get('score', 0.0)):.3f}")
            node.SetAttribute("Rosa.ProposalLengthMm", f"{float(proposal.get('span_mm', 0.0)):.3f}")
            # Per-trajectory confidence (contact_pitch_v1 score framework).
            # Stored as MRML attributes so downstream UI (Guided Fit
            # trajectory table, Auto Fit list) can show / filter by
            # confidence without re-running detection.
            confidence = proposal.get("confidence")
            if confidence is not None:
                try:
                    node.SetAttribute("Rosa.Confidence", f"{float(confidence):.3f}")
                except Exception:
                    node.SetAttribute("Rosa.Confidence", "")
            confidence_label = proposal.get("confidence_label")
            if confidence_label is not None:
                node.SetAttribute(
                    "Rosa.ConfidenceLabel", str(confidence_label),
                )
            bolt_source = proposal.get("bolt_source")
            if bolt_source is not None:
                node.SetAttribute("Rosa.BoltSource", str(bolt_source))
            annulus_mean = proposal.get("annulus_mean_ct_hu")
            annulus_samples = int(proposal.get("annulus_sample_count", 0) or 0)
            if annulus_mean is None:
                node.SetAttribute("Rosa.DeepCoreAnnulusMeanHu", "")
            else:
                node.SetAttribute("Rosa.DeepCoreAnnulusMeanHu", f"{float(annulus_mean):.3f}")
            node.SetAttribute("Rosa.DeepCoreAnnulusSampleCount", str(int(annulus_samples)))
            # contact_pitch_v1 emits `skull_entry_ras` = deepest bolt-tube
            # voxel along the shank axis (bone→brain transition). Trajectories
            # whose anchoring failed (bolt_source = "none") have no
            # skull_entry_ras and skip the skull-jump record entirely. The
            # annulus-gradient fallback that used to live here was removed
            # with the v1/v2 pipeline cleanup (commit 4916dae); call site
            # was overlooked then.
            entry_override = proposal.get("skull_entry_ras")
            node.SetAttribute("Rosa.DeepCoreSkullJumpHuPerMm", "")
            node.SetAttribute("Rosa.DeepCoreSkullJumpAxialMm", "")
            if entry_override is not None and len(list(entry_override)) >= 3:
                entry_pt = [float(v) for v in list(entry_override)[:3]]
                skull_jump_records.append(
                    {
                        "point_ras": entry_pt,
                        "label": f"{line_tag}{idx:02d}*",
                        "description": "bolt base (bone->brain)",
                    }
                )
            shallow_name = str(proposal.get("shallow_endpoint_name") or "").strip().lower()
            deep_name = str(proposal.get("deep_endpoint_name") or "").strip().lower()
            shallow_ras = end_ras if shallow_name == "end" else start_ras
            if deep_name == "start":
                deep_ras = start_ras
            elif deep_name == "end":
                deep_ras = end_ras
            else:
                deep_ras = end_ras if shallow_name == "start" else start_ras
            shallow_records.append(
                {
                    "point_ras": [float(v) for v in np.asarray(shallow_ras, dtype=float).reshape(3)],
                    "label": f"{line_tag}{idx:02d}Sh",
                    "description": "trajectory shallow end",
                }
            )
            deep_records.append(
                {
                    "point_ras": [float(v) for v in np.asarray(deep_ras, dtype=float).reshape(3)],
                    "label": f"{line_tag}{idx:02d}Dp",
                    "description": "trajectory deep end",
                }
            )
            atom_ids = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            blob_ids = [int(v) for v in list(proposal.get("blob_id_list") or []) if int(v) > 0]
            node.SetAttribute("Rosa.DeepCoreAtomIds", ",".join(str(v) for v in atom_ids))
            node.SetAttribute("Rosa.DeepCoreBlobIds", ",".join(str(v) for v in blob_ids))
            display = node.GetDisplayNode()
            if display is not None:
                color = tuple(style.get("color") or (0.75, 0.75, 0.75))
                display.SetColor(float(color[0]), float(color[1]), float(color[2]))
                display.SetSelectedColor(float(color[0]), float(color[1]), float(color[2]))
            self._copy_parent_transform(volume_node, node)
            nodes.append(node)
        self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
            context_id=self.workflow_state.context_id(),
            nodes=nodes,
        )
        if skull_jump_records:
            star_node = self._get_or_create_fiducial_node(
                star_node_name,
                color_rgb=(1.0, 0.1, 0.1),
                glyph_scale=1.6,
                text_scale=0.7,
                show_labels=True,
            )
            self._copy_parent_transform(volume_node, star_node)
            self._set_fiducials_from_point_records(star_node, skull_jump_records, show_labels=True, text_scale=0.7)
            star_display = star_node.GetDisplayNode()
            if star_display is not None and hasattr(star_display, "SetGlyphTypeFromString"):
                try:
                    star_display.SetGlyphTypeFromString("StarBurst2D")
                except Exception:
                    pass
        if shallow_records:
            shallow_node = self._get_or_create_fiducial_node(
                shallow_node_name,
                color_rgb=(0.10, 0.90, 1.00),
                glyph_scale=1.3,
                text_scale=1.2,
                show_labels=True,
            )
            self._copy_parent_transform(volume_node, shallow_node)
            self._set_fiducials_from_point_records(shallow_node, shallow_records, show_labels=True, text_scale=1.2)
        if deep_records:
            deep_node = self._get_or_create_fiducial_node(
                deep_node_name,
                color_rgb=(1.00, 0.90, 0.10),
                glyph_scale=1.3,
                text_scale=1.2,
                show_labels=True,
            )
            self._copy_parent_transform(volume_node, deep_node)
            self._set_fiducials_from_point_records(deep_node, deep_records, show_labels=True, text_scale=1.2)
        return nodes

    def _get_or_create_fiducial_node(self, node_name, color_rgb, glyph_scale=1.0, text_scale=0.8, show_labels=False):
        node = None
        try:
            node = slicer.util.getNode(node_name)
        except Exception:
            node = None
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", node_name)
            node.CreateDefaultDisplayNodes()
            node.SetAttribute("Rosa.Managed", "1")
        display = node.GetDisplayNode()
        if display is not None:
            display.SetColor(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
            display.SetSelectedColor(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
            display.SetGlyphScale(float(glyph_scale))
            display.SetTextScale(float(text_scale))
            display.SetVisibility(True)
            if hasattr(display, "SetPointLabelsVisibility"):
                display.SetPointLabelsVisibility(bool(show_labels))
            # Keep default slice behavior: points are only shown when the
            # current slice intersects them.
            if hasattr(display, "SetSliceProjection"):
                display.SetSliceProjection(False)
        return node

    def _set_fiducials_from_ras_points(self, node, points_ras):
        node.RemoveAllControlPoints()
        pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
        for p in pts:
            node.AddControlPoint(vtk.vtkVector3d(float(p[0]), float(p[1]), float(p[2])))
        node.SetLocked(True)
        return int(pts.shape[0])

    def _set_fiducials_from_point_records(self, node, point_records, show_labels=True, text_scale=0.65):
        node.RemoveAllControlPoints()
        records = list(point_records or [])
        for record in records:
            rec = dict(record or {})
            point = np.asarray(rec.get("point_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            label = str(rec.get("label") or "")
            description = str(rec.get("description") or "")
            idx = int(node.AddControlPoint(vtk.vtkVector3d(float(point[0]), float(point[1]), float(point[2]))))
            if label:
                node.SetNthControlPointLabel(int(idx), label)
            if description:
                node.SetNthControlPointDescription(int(idx), description)
        node.SetLocked(True)
        display = node.GetDisplayNode()
        if display is not None:
            display.SetTextScale(float(text_scale))
            if hasattr(display, "SetPointLabelsVisibility"):
                display.SetPointLabelsVisibility(bool(show_labels))
        return int(len(records))

    def show_blob_diagnostics(
        self,
        volume_node,
        blob_labelmap_kji=None,
        blob_centroids_all_ras=None,
        blob_centroids_kept_ras=None,
        blob_centroids_rejected_ras=None,
        line_blob_points_ras=None,
        contact_blob_points_ras=None,
        complex_blob_points_ras=None,
        complex_blob_chain_rows=None,
        contact_chain_rows=None,
    ):
        """Create/update blob debug overlays and centroid markups."""
        if blob_labelmap_kji is not None and np.asarray(blob_labelmap_kji).size > 0:
            self._update_labelmap_from_mask(
                reference_volume_node=volume_node,
                node_name=f"{volume_node.GetName()}_BlobLabelMap",
                mask_kji=blob_labelmap_kji,
            )
        self._remove_node_if_exists(f"{volume_node.GetName()}_BlobCentroids_All")
        self._remove_node_if_exists(f"{volume_node.GetName()}_BlobCentroids_Kept")
        self._remove_node_if_exists(f"{volume_node.GetName()}_BlobCentroids_Rejected")
        self._remove_nodes_with_prefix(f"{volume_node.GetName()}_ComplexBlobChain_")
        self._remove_nodes_with_prefix(f"{volume_node.GetName()}_ContactChain_")
        self._remove_node_if_exists(f"{volume_node.GetName()}_ComplexBlobChain_CandidateSeeds")
        all_node = self._get_or_create_fiducial_node(
            f"{volume_node.GetName()}_RawBlobCentroids",
            color_rgb=(1.0, 1.0, 0.0),
        )
        self._copy_parent_transform(volume_node, all_node)
        self._set_fiducials_from_ras_points(all_node, blob_centroids_all_ras)

        chain_rows = [dict(row or {}) for row in list(complex_blob_chain_rows or [])]
        if chain_rows:
            aggregated = {}
            for row in chain_rows:
                key = (int(row.get("blob_id", -1)), int(row.get("node_id", -1)))
                if key not in aggregated:
                    aggregated[key] = {
                        "blob_id": int(row.get("blob_id", -1)),
                        "node_id": int(row.get("node_id", -1)),
                        "node_role": str(row.get("node_role") or ""),
                        "bin_id": int(row.get("bin_id", -1)),
                        "degree": int(row.get("degree", -1)),
                        "point_ras": [
                            float(row.get("support_x_ras", 0.0)),
                            float(row.get("support_y_ras", 0.0)),
                            float(row.get("support_z_ras", 0.0)),
                        ],
                        "memberships": [],
                        "seed_line_ids": set(),
                        "candidate_seed_only": False,
                    }
                line_id = int(row.get("line_id", 0))
                line_order = int(row.get("line_order", 0))
                if line_id > 0:
                    aggregated[key]["memberships"].append((int(line_id), int(line_order)))
                    if bool(row.get("is_seed")):
                        aggregated[key]["seed_line_ids"].add(int(line_id))
                elif bool(row.get("is_candidate_seed")):
                    aggregated[key]["candidate_seed_only"] = True
            role_code_map = {"edge": "E", "core": "C", "crossing": "X"}
            line_palette = [
                (0.15, 0.75, 0.95),
                (0.95, 0.45, 0.15),
                (0.20, 0.85, 0.25),
                (0.95, 0.15, 0.75),
                (0.95, 0.85, 0.15),
                (0.65, 0.55, 0.95),
            ]
            role_color_map = {
                "edge": (1.0, 0.95, 0.15),
                "core": (0.10, 0.90, 0.95),
                "crossing": (0.95, 0.20, 0.85),
            }
            role_scale_map = {
                "edge": 1.8,
                "core": 1.1,
                "crossing": 2.1,
            }
            seed_records = []
            candidate_seed_records = []
            line_groups = {}
            unassigned_groups = {}
            has_memberships = False
            for item in aggregated.values():
                memberships = sorted(set((int(line_id), int(line_order)) for line_id, line_order in list(item.get("memberships") or [])))
                role = str(item.get("node_role") or "core")
                if role not in role_code_map:
                    role = "core"
                role_code = role_code_map.get(role, "C")
                membership_text = "-" if not memberships else ", ".join(f"L{line_id}@{line_order}" for line_id, line_order in memberships)
                base_description = (
                    f"blob_id={int(item.get('blob_id', -1))}\n"
                    f"node_id={int(item.get('node_id', -1))}\n"
                    f"node_role={role}\n"
                    f"line_membership={membership_text}\n"
                    f"bin_id={int(item.get('bin_id', -1))}\n"
                    f"degree={int(item.get('degree', -1))}"
                )
                if memberships:
                    has_memberships = True
                    for line_id, line_order in memberships:
                        group_key = (int(item.get("blob_id", -1)), int(line_id))
                        line_groups.setdefault(group_key, []).append(
                            {
                                "line_order": int(line_order),
                                "point_ras": list(item.get("point_ras") or [0.0, 0.0, 0.0]),
                                "node_role": role,
                                "node_id": int(item.get("node_id", -1)),
                                "is_seed": bool(int(line_id) in set(item.get("seed_line_ids") or set())),
                                "description": base_description,
                            }
                        )
                else:
                    blob_key = int(item.get("blob_id", -1))
                    unassigned_groups.setdefault(blob_key, {}).setdefault(role, []).append(
                        {
                            "point_ras": list(item.get("point_ras") or [0.0, 0.0, 0.0]),
                            "label": f"n{int(item.get('node_id', -1))}{role_code}",
                            "description": base_description,
                        }
                    )
                if bool(item.get("candidate_seed_only")):
                    candidate_seed_records.append(
                        {
                            "point_ras": list(item.get("point_ras") or [0.0, 0.0, 0.0]),
                            "label": "S?",
                            "description": "{base}\ncandidate_seed=1\naccepted_seed=0".format(
                                base=base_description,
                            ),
                        }
                    )
            for (blob_id, line_id), records in sorted(line_groups.items()):
                path_color = line_palette[(int(line_id) - 1) % len(line_palette)]
                path_name = f"{volume_node.GetName()}_ComplexBlobChain_b{int(blob_id)}_L{int(line_id)}_Path"
                sorted_records = sorted(records, key=lambda item: int(item.get("line_order", 0)))
                for record in sorted_records:
                    if bool(record.get("is_seed")):
                        seed_records.append(
                            {
                                "point_ras": list(record.get("point_ras") or [0.0, 0.0, 0.0]),
                                "label": f"L{int(line_id)}S",
                                "description": "{base}\nblob_line=b{blob}_L{line}\nseed_node=1\nline_order={order}\nnode_role={role}\nnode_id={node_id}".format(
                                    base=str(record.get("description") or ""),
                                    blob=int(blob_id),
                                    line=int(line_id),
                                    order=int(record.get("line_order", 0)),
                                    role=str(record.get("node_role") or ""),
                                    node_id=int(record.get("node_id", -1)),
                                ),
                            }
                        )
                line_segments = []
                for idx in range(len(sorted_records) - 1):
                    p0 = list(sorted_records[idx].get("point_ras") or [0.0, 0.0, 0.0])
                    p1 = list(sorted_records[idx + 1].get("point_ras") or [0.0, 0.0, 0.0])
                    line_segments.append((p0, p1))
                model_node = self._get_or_create_model_node(path_name, color_rgb=path_color)
                model_node.SetAndObservePolyData(self._build_polydata_from_line_segments(line_segments))
                self._copy_parent_transform(volume_node, model_node)
                for role in ("edge", "core", "crossing"):
                    role_records = [
                        {
                            "point_ras": list(record.get("point_ras") or [0.0, 0.0, 0.0]),
                            "label": "",
                            "description": "{base}\nblob_line=b{blob}_L{line}\nline_order={order}\nnode_role={role}\nnode_id={node_id}".format(
                                base=str(record.get("description") or ""),
                                blob=int(blob_id),
                                line=int(line_id),
                                order=int(record.get("line_order", 0)),
                                role=role,
                                node_id=int(record.get("node_id", -1)),
                            ),
                        }
                        for record in sorted_records
                        if str(record.get("node_role") or "") == role
                    ]
                    if not role_records:
                        continue
                    role_node_name = f"{volume_node.GetName()}_ComplexBlobChain_b{int(blob_id)}_L{int(line_id)}_{role.title()}Nodes"
                    role_node = self._get_or_create_fiducial_node(
                        role_node_name,
                        color_rgb=role_color_map.get(role, (0.8, 0.8, 0.8)),
                        glyph_scale=role_scale_map.get(role, 1.2),
                        text_scale=0.0,
                        show_labels=False,
                    )
                    self._copy_parent_transform(volume_node, role_node)
                    self._set_fiducials_from_point_records(role_node, role_records, show_labels=False, text_scale=0.0)
            seed_node_name = f"{volume_node.GetName()}_ComplexBlobChain_SeedNodes"
            if seed_records:
                seed_node = self._get_or_create_fiducial_node(
                    seed_node_name,
                    color_rgb=(1.0, 0.15, 0.15),
                    glyph_scale=2.6,
                    text_scale=1.1,
                    show_labels=True,
                )
                self._copy_parent_transform(volume_node, seed_node)
                self._set_fiducials_from_point_records(seed_node, seed_records, show_labels=True, text_scale=1.1)
            else:
                self._remove_node_if_exists(seed_node_name)
            candidate_seed_node_name = f"{volume_node.GetName()}_ComplexBlobChain_CandidateSeeds"
            if candidate_seed_records:
                candidate_seed_node = self._get_or_create_fiducial_node(
                    candidate_seed_node_name,
                    color_rgb=(1.0, 0.6, 0.1),
                    glyph_scale=2.0,
                    text_scale=0.95,
                    show_labels=True,
                )
                self._copy_parent_transform(volume_node, candidate_seed_node)
                self._set_fiducials_from_point_records(candidate_seed_node, candidate_seed_records, show_labels=True, text_scale=0.95)
            else:
                self._remove_node_if_exists(candidate_seed_node_name)
            if has_memberships:
                for blob_id, role_groups in sorted(unassigned_groups.items()):
                    for role, records in sorted(role_groups.items()):
                        node_name = f"{volume_node.GetName()}_ComplexBlobChain_b{int(blob_id)}_Unassigned{str(role).title()}Nodes"
                        unassigned_node = self._get_or_create_fiducial_node(
                            node_name,
                            color_rgb=role_color_map.get(str(role), (0.70, 0.70, 0.70)),
                            glyph_scale=max(0.95, 0.85 * float(role_scale_map.get(str(role), 1.2))),
                            text_scale=0.0,
                            show_labels=False,
                        )
                        self._copy_parent_transform(volume_node, unassigned_node)
                        self._set_fiducials_from_point_records(unassigned_node, records, show_labels=False, text_scale=0.0)
            else:
                aggregated_unassigned = {}
                for _blob_id, role_groups in sorted(unassigned_groups.items()):
                    for role, records in sorted(role_groups.items()):
                        aggregated_unassigned.setdefault(str(role), []).extend(list(records or []))
                for role, records in sorted(aggregated_unassigned.items()):
                    node_name = f"{volume_node.GetName()}_ComplexBlob{str(role).title()}Nodes"
                    unassigned_node = self._get_or_create_fiducial_node(
                        node_name,
                        color_rgb=role_color_map.get(str(role), (0.70, 0.70, 0.70)),
                        glyph_scale=max(0.95, 0.90 * float(role_scale_map.get(str(role), 1.2))),
                        text_scale=0.0,
                        show_labels=False,
                    )
                    self._copy_parent_transform(volume_node, unassigned_node)
                    self._set_fiducials_from_point_records(unassigned_node, records, show_labels=False, text_scale=0.0)
        else:
            self._remove_nodes_with_prefix(f"{volume_node.GetName()}_ComplexBlobChain_")
            self._remove_node_if_exists(f"{volume_node.GetName()}_ComplexBlobChain_SeedNodes")
            self._remove_node_if_exists(f"{volume_node.GetName()}_ComplexBlobChain_CandidateSeeds")

        contact_rows = [dict(row or {}) for row in list(contact_chain_rows or [])]
        if contact_rows:
            chain_groups = {}
            for row in contact_rows:
                chain_id = int(row.get("chain_id", 0))
                if chain_id <= 0:
                    continue
                chain_groups.setdefault(int(chain_id), []).append(
                    {
                        "chain_order": int(row.get("chain_order", 0)),
                        "atom_id": int(row.get("atom_id", -1)),
                        "parent_blob_id": int(row.get("parent_blob_id", -1)),
                        "node_role": str(row.get("node_role") or "core"),
                        "point_ras": [
                            float(row.get("support_x_ras", 0.0)),
                            float(row.get("support_y_ras", 0.0)),
                            float(row.get("support_z_ras", 0.0)),
                        ],
                    }
                )
            contact_path_color = (1.0, 0.45, 0.10)
            contact_role_colors = {
                "edge": (1.0, 0.95, 0.15),
                "core": (0.15, 0.90, 0.20),
            }
            contact_role_scales = {"edge": 1.9, "core": 1.3}
            for chain_id, records in sorted(chain_groups.items()):
                sorted_records = sorted(records, key=lambda item: int(item.get("chain_order", 0)))
                line_segments = []
                for idx in range(len(sorted_records) - 1):
                    p0 = list(sorted_records[idx].get("point_ras") or [0.0, 0.0, 0.0])
                    p1 = list(sorted_records[idx + 1].get("point_ras") or [0.0, 0.0, 0.0])
                    line_segments.append((p0, p1))
                path_name = f"{volume_node.GetName()}_ContactChain_L{int(chain_id)}_Path"
                model_node = self._get_or_create_model_node(path_name, color_rgb=contact_path_color)
                model_node.SetAndObservePolyData(self._build_polydata_from_line_segments(line_segments))
                self._copy_parent_transform(volume_node, model_node)
                for role in ("edge", "core"):
                    role_records = [
                        {
                            "point_ras": list(record.get("point_ras") or [0.0, 0.0, 0.0]),
                            "label": "",
                            "description": (
                                f"contact_chain={int(chain_id)}\n"
                                f"chain_order={int(record.get('chain_order', 0))}\n"
                                f"atom_id={int(record.get('atom_id', -1))}\n"
                                f"parent_blob_id={int(record.get('parent_blob_id', -1))}\n"
                                f"node_role={role}"
                            ),
                        }
                        for record in sorted_records
                        if str(record.get("node_role") or "") == role
                    ]
                    if not role_records:
                        continue
                    role_node_name = f"{volume_node.GetName()}_ContactChain_L{int(chain_id)}_{role.title()}Nodes"
                    role_node = self._get_or_create_fiducial_node(
                        role_node_name,
                        color_rgb=contact_role_colors.get(role, (0.85, 0.85, 0.85)),
                        glyph_scale=contact_role_scales.get(role, 1.2),
                        text_scale=0.0,
                        show_labels=False,
                    )
                    self._copy_parent_transform(volume_node, role_node)
                    self._set_fiducials_from_point_records(role_node, role_records, show_labels=False, text_scale=0.0)
        else:
            self._remove_nodes_with_prefix(f"{volume_node.GetName()}_ContactChain_")

        use_class_tokens = any(
            np.asarray(points if points is not None else [], dtype=float).reshape(-1, 3).shape[0] > 0
            for points in (line_blob_points_ras, contact_blob_points_ras, complex_blob_points_ras)
        )
        if use_class_tokens:
            self._remove_node_if_exists(f"{volume_node.GetName()}_SupportSamplePoints")
            self._remove_node_if_exists(f"{volume_node.GetName()}_RejectedSupportSamples")
            line_node = self._get_or_create_fiducial_node(
                f"{volume_node.GetName()}_LineBlobTokens",
                color_rgb=(0.05, 0.15, 0.75),
            )
            contact_node = self._get_or_create_fiducial_node(
                f"{volume_node.GetName()}_ContactBlobTokens",
                color_rgb=(0.10, 0.90, 0.10),
            )
            complex_node = self._get_or_create_fiducial_node(
                f"{volume_node.GetName()}_ComplexBlobTokens",
                color_rgb=(0.95, 0.55, 0.10),
            )
            self._copy_parent_transform(volume_node, line_node)
            self._copy_parent_transform(volume_node, contact_node)
            self._copy_parent_transform(volume_node, complex_node)
            self._set_fiducials_from_ras_points(line_node, line_blob_points_ras)
            self._set_fiducials_from_ras_points(contact_node, contact_blob_points_ras)
            self._set_fiducials_from_ras_points(complex_node, complex_blob_points_ras)
            return

        self._remove_node_if_exists(f"{volume_node.GetName()}_LineBlobTokens")
        self._remove_node_if_exists(f"{volume_node.GetName()}_ContactBlobTokens")
        self._remove_node_if_exists(f"{volume_node.GetName()}_ComplexBlobTokens")
        kept_node = self._get_or_create_fiducial_node(
            f"{volume_node.GetName()}_SupportSamplePoints",
            color_rgb=(0.1, 0.9, 0.1),
        )
        rej_node = self._get_or_create_fiducial_node(
            f"{volume_node.GetName()}_RejectedSupportSamples",
            color_rgb=(0.95, 0.25, 0.25),
        )
        self._copy_parent_transform(volume_node, kept_node)
        self._copy_parent_transform(volume_node, rej_node)
        self._set_fiducials_from_ras_points(kept_node, blob_centroids_kept_ras)
        self._set_fiducials_from_ras_points(rej_node, blob_centroids_rejected_ras)

    def show_deep_core_support_atoms(self, volume_node, support_atoms):
        self._remove_node_if_exists(f"{volume_node.GetName()}_DeepCoreSupportAtoms")
        self._remove_node_if_exists(f"{volume_node.GetName()}_DeepCoreSupportAtomCenters")
        line_node_name = f"{volume_node.GetName()}_SupportAtomLines"
        center_node_name = f"{volume_node.GetName()}_SupportAtomCenters"
        atoms = [dict(atom or {}) for atom in list(support_atoms or [])]
        if not atoms:
            self._remove_node_if_exists(line_node_name)
            self._remove_node_if_exists(center_node_name)
            return

        line_segments = []
        center_points = []
        for atom in atoms:
            center = np.asarray(atom.get("center_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            start = np.asarray(atom.get("start_ras") or center, dtype=float).reshape(3)
            end = np.asarray(atom.get("end_ras") or center, dtype=float).reshape(3)
            center_points.append(center.tolist())
            if float(np.linalg.norm(end - start)) >= 0.75:
                line_segments.append((start.tolist(), end.tolist()))

        line_node = self._get_or_create_model_node(
            line_node_name,
            color_rgb=(0.10, 0.85, 0.95),
        )
        line_node.SetAndObservePolyData(self._build_polydata_from_line_segments(line_segments))
        self._copy_parent_transform(volume_node, line_node)

        center_node = self._get_or_create_fiducial_node(
            center_node_name,
            color_rgb=(1.0, 0.75, 0.15),
        )
        self._copy_parent_transform(volume_node, center_node)
        self._set_fiducials_from_ras_points(center_node, np.asarray(center_points, dtype=float).reshape(-1, 3))

    @staticmethod

    @staticmethod

    @staticmethod
    def _build_detection_diagnostics(engine_result, params):
        """Build stable diagnostics payload for logs/tests/debug export."""
        diag = dict((engine_result.get("diagnostics") or {}))
        counts = dict(diag.get("counts") or {})
        timing = dict(diag.get("timing") or {})
        extras = dict(diag.get("extras") or {})
        lines = PostopCTLocalizationLogic._engine_trajectories_to_lines(engine_result)
        summary = {
            "candidate_mode": str((params or {}).get("candidate_mode", "voxel")),
            "candidate_points_total": int(counts.get("candidate_points_total", 0)),
            "candidate_points_after_mask": int(counts.get("candidate_points_after_mask", 0)),
            "candidate_points_after_depth": int(counts.get("candidate_points_after_depth", 0)),
            "effective_min_inliers": int(counts.get("effective_min_inliers", 0)),
            "effective_inlier_radius_mm": float(counts.get("effective_inlier_radius_mm", 0.0)) / 1000.0
            if float(counts.get("effective_inlier_radius_mm", 0.0)) > 100.0
            else float(counts.get("effective_inlier_radius_mm", 0.0)),
            "blob_count_total": int(counts.get("blob_count_total", 0)),
            "blob_count_kept": int(counts.get("blob_count_kept", 0)),
            "blob_reject_small": int(counts.get("blob_reject_small", 0)),
            "blob_reject_large": int(counts.get("blob_reject_large", 0)),
            "blob_reject_intensity": int(counts.get("blob_reject_intensity", 0)),
            "blob_reject_shape": int(counts.get("blob_reject_shape", 0)),
            "fit1_lines_proposed": int(counts.get("fit1_lines_proposed", 0)),
            "fit2_lines_kept": int(counts.get("fit2_lines_kept", 0)),
            "rescue_lines_kept": int(counts.get("rescue_lines_kept", 0)),
            "final_lines_kept": int(counts.get("final_lines_kept", len(lines))),
            "assigned_points_after_refine": int(counts.get("assigned_points_after_refine", 0)),
            "unassigned_points_after_refine": int(counts.get("unassigned_points_after_refine", 0)),
            "rescued_points": int(counts.get("rescued_points", 0)),
            "final_unassigned_points": int(counts.get("final_unassigned_points", 0)),
            "gap_reject_count": int(counts.get("gap_reject_count", 0)),
            "duplicate_reject_count": int(counts.get("duplicate_reject_count", 0)),
            "start_zone_reject_count": int(counts.get("start_zone_reject_count", 0)),
            "length_reject_count": int(counts.get("length_reject_count", 0)),
            "inlier_reject_count": int(counts.get("inlier_reject_count", 0)),
            "profile_ms": {
                "setup": float(timing.get("stage.gating.ms", 0.0)),
                "subsample": float(timing.get("stage.blob_extraction.ms", 0.0)),
                "depth_map": float(timing.get("stage.gating.ms", 0.0)),
                "ras_convert": float(timing.get("stage.blob_scoring.ms", 0.0)),
                "blob_stage": float(timing.get("stage.blob_extraction.ms", 0.0)),
                "exclude": float(timing.get("stage.blob_scoring.ms", 0.0)),
                "fit1_stage": float(timing.get("stage.seed_initialization.ms", 0.0)),
                "fit2_stage": float(timing.get("stage.em_refinement.ms", 0.0)),
                "rescue_stage": float(timing.get("stage.model_selection.ms", 0.0)),
                "total": float(timing.get("total_ms", 0.0)),
            },
            "gating_mask_type": str(extras.get("gating_mask_type", "engine")),
            "inside_method": str(extras.get("inside_method", "engine")),
            "params": dict(params or {}),
        }
        summary["lines"] = [
            {
                "index": int(i + 1),
                "start_ras": list(line.get("start_ras", [0.0, 0.0, 0.0])),
                "end_ras": list(line.get("end_ras", [0.0, 0.0, 0.0])),
                "length_mm": float(line.get("length_mm", 0.0)),
                "inlier_count": int(line.get("inlier_count", 0)),
                "support_weight": float(line.get("support_weight", 0.0)),
                "rms_mm": float(line.get("rms_mm", 0.0)),
                "inside_fraction": float(line.get("inside_fraction", 0.0)),
                "depth_span_mm": float(line.get("depth_span_mm", 0.0)),
                "best_model_id": str(line.get("best_model_id", "")),
                "best_model_score": (
                    None if line.get("best_model_score", None) is None else float(line.get("best_model_score"))
                ),
            }
            for i, line in enumerate(lines)
        ]
        return summary

    def write_detection_diagnostics_json(self, volume_node, diagnostics):
        """Persist debug diagnostics JSON under /tmp for troubleshooting."""
        out_dir = os.path.join("/tmp", "rosa_postopct_debug")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = str(volume_node.GetName() or "volume").replace(os.sep, "_")
        out_path = os.path.join(out_dir, f"{safe_name}_denovo_diagnostics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        return out_path

    def _vtk_matrix_to_numpy(self, vtk_matrix4x4):
        out = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                out[r, c] = float(vtk_matrix4x4.GetElement(r, c))
        return out

    def show_volume_in_all_slice_views(self, volume_node):
        if volume_node is None:
            return
        volume_id = volume_node.GetID()
        app_logic = slicer.app.applicationLogic() if hasattr(slicer.app, "applicationLogic") else None
        if app_logic is not None:
            sel = app_logic.GetSelectionNode()
            if sel is not None:
                sel.SetReferenceActiveVolumeID(volume_id)
                app_logic.PropagateVolumeSelection(0)
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(volume_id)
        bounds = [0.0] * 6
        volume_node.GetRASBounds(bounds)
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        for view_name in ("Red", "Yellow", "Green"):
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            logic = widget.sliceLogic()
            if logic is not None:
                logic.FitSliceToAll()
            slice_node = widget.mrmlSliceNode()
            if slice_node is not None:
                try:
                    slice_node.JumpSliceByCentering(cx, cy, cz)
                except Exception:
                    pass

    def reset_standard_slice_views(self):
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        layout_node = lm.layoutLogic().GetLayoutNode()
        if layout_node is not None:
            layout_node.SetViewArrangement(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        for view_name, orientation in (("Red", "Axial"), ("Yellow", "Sagittal"), ("Green", "Coronal")):
            widget = lm.sliceWidget(view_name)
            if widget is None:
                continue
            slice_node = widget.mrmlSliceNode()
            if slice_node:
                slice_node.SetOrientation(orientation)
            logic = widget.sliceLogic()
            if logic:
                logic.FitSliceToAll()
