"""Shared export workflow service used by multiple Slicer modules."""

import csv
import json
import os

import numpy as np
from __main__ import slicer, vtk

from rosa_core import lps_to_ras_point

from .export_profiles import get_export_profile
from .workflow_registry import table_to_dict_rows
from .workflow_state import WorkflowState


def _safe_filename(text):
    """Return filesystem-safe filename stem."""
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    stem = "".join(safe).strip("._")
    return stem or "volume"


def _to_float(value, default=0.0):
    if isinstance(value, str):
        value = value.strip()
        # Handle serialized string scalars such as '"0.0"' or "'0.0'".
        while len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1].strip()
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value, default=0):
    if isinstance(value, str):
        value = value.strip()
        while len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1].strip()
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _ras_to_lps_point(point_ras):
    return [-float(point_ras[0]), -float(point_ras[1]), float(point_ras[2])]


def _vtk_matrix_to_numpy_4x4(vtk_matrix):
    """Return 4x4 NumPy matrix copied from vtkMatrix4x4."""
    out = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            out[r, c] = float(vtk_matrix.GetElement(r, c))
    return out


def _world_to_node_ras_matrix(node):
    """Return 4x4 matrix mapping world RAS into node-local RAS coordinates."""
    if node is None:
        return np.eye(4, dtype=float)
    parent = node.GetParentTransformNode()
    if parent is None:
        return np.eye(4, dtype=float)
    world_to_local_vtk = vtk.vtkMatrix4x4()
    if not parent.GetMatrixTransformFromWorld(world_to_local_vtk):
        return np.eye(4, dtype=float)
    return _vtk_matrix_to_numpy_4x4(world_to_local_vtk)


def collect_volume_node_ids_from_registry(workflow_node):
    """Build `{volume_name: node_id}` map from image registry table."""
    out = {}
    table = workflow_node.GetNodeReference("ImageRegistryTable")
    rows = table_to_dict_rows(table) if table is not None else []
    for row in rows:
        node_id = str(row.get("node_id", ""))
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None:
            continue
        if not (node.IsA("vtkMRMLScalarVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode")):
            continue
        label = str(row.get("label", "")).strip() or node.GetName() or node_id
        name = label
        suffix = 2
        while name in out and out[name] != node_id:
            name = f"{label}_{suffix}"
            suffix += 1
        out[name] = node_id
    if out:
        return out

    # Backward-compatible fallback for scenes loaded before registry publication.
    state = WorkflowState()
    fallback_roles = [
        "BaseVolume",
        "PostopCT",
        "ROSAVolumes",
        "AdditionalMRIVolumes",
        "AdditionalCTVolumes",
        "DerivedVolumes",
        "FSParcellationVolumes",
        "WMParcellationVolumes",
    ]
    seen_ids = set()
    for role in fallback_roles:
        nodes = state.role_nodes(role, workflow_node=workflow_node)
        for node in nodes:
            if node is None:
                continue
            if not (node.IsA("vtkMRMLScalarVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode")):
                continue
            node_id = node.GetID()
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            label = (node.GetName() or "").strip() or node_id
            name = label
            suffix = 2
            while name in out and out[name] != node_id:
                name = f"{label}_{suffix}"
                suffix += 1
            out[name] = node_id
    return out


def _build_assignment_map(workflow_node):
    """Build lookup maps from workflow assignment table."""
    by_contact_node_id = {}
    by_traj = {}
    table = workflow_node.GetNodeReference("ElectrodeAssignmentTable")
    rows = table_to_dict_rows(table) if table is not None else []
    for row in rows:
        traj = str(row.get("trajectory", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        if traj:
            by_traj[traj] = row
        node_id = str(row.get("contact_fiducial_node_id", "")).strip()
        if node_id:
            by_contact_node_id[node_id] = {"trajectory": traj, "model_id": model_id}
    return by_contact_node_id, by_traj


def _trajectory_name_from_node_name(node_name):
    """Best-effort trajectory name parse from contact node name."""
    name = str(node_name or "").strip()
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[-1]


def collect_contacts_from_workflow(workflow_node):
    """Collect contact dictionaries from workflow ContactFiducials role nodes."""
    state = WorkflowState()
    contact_nodes = state.role_nodes("ContactFiducials", workflow_node=workflow_node)
    by_node_id, by_traj = _build_assignment_map(workflow_node)
    contacts = []
    for node in contact_nodes:
        node_id = node.GetID()
        mapped = by_node_id.get(node_id, {})
        traj = mapped.get("trajectory", "") or _trajectory_name_from_node_name(node.GetName())
        model_id = mapped.get("model_id", "")
        if not model_id and traj in by_traj:
            model_id = str(by_traj[traj].get("model_id", ""))
        n = int(node.GetNumberOfControlPoints())
        for i in range(n):
            point_ras = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(i, point_ras)
            label = node.GetNthControlPointLabel(i) or str(i + 1)
            index = _to_int(label, i + 1)
            contacts.append(
                {
                    "trajectory": traj,
                    "label": f"{traj}{index}",
                    "index": index,
                    "position_lps": _ras_to_lps_point(point_ras),
                    "model_id": model_id,
                }
            )
    contacts.sort(key=lambda c: (str(c.get("trajectory", "")), int(c.get("index", 0))))
    return contacts


def _collect_trajectory_map_from_nodes(nodes, strip_plan_prefix=False):
    """Collect line-node trajectory map in LPS coordinates."""
    out = {}
    for node in nodes or []:
        if node is None or not node.IsA("vtkMRMLMarkupsLineNode"):
            continue
        if int(node.GetNumberOfControlPoints()) < 2:
            continue
        name = str(node.GetAttribute("Rosa.TrajectoryName") or "").strip()
        if not name:
            name = str(node.GetName() or "").strip()
        if strip_plan_prefix and name.startswith("Plan_"):
            name = name[5:]
        if not name:
            continue
        p0 = [0.0, 0.0, 0.0]
        p1 = [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        out[name] = {
            "name": name,
            "start": _ras_to_lps_point(p0),
            "end": _ras_to_lps_point(p1),
        }
    return out


def collect_qc_rows_from_workflow(workflow_node):
    """Collect QC row dictionaries from workflow QC table."""
    table = workflow_node.GetNodeReference("QCMetricsTable")
    raw_rows = table_to_dict_rows(table) if table is not None else []
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "trajectory": row.get("trajectory", ""),
                "entry_radial_mm": _to_float(row.get("entry_radial_mm", 0.0)),
                "target_radial_mm": _to_float(row.get("target_radial_mm", 0.0)),
                "mean_contact_radial_mm": _to_float(row.get("mean_contact_radial_mm", 0.0)),
                "max_contact_radial_mm": _to_float(row.get("max_contact_radial_mm", 0.0)),
                "rms_contact_radial_mm": _to_float(row.get("rms_contact_radial_mm", 0.0)),
                "angle_deg": _to_float(row.get("angle_deg", 0.0)),
                "matched_contacts": _to_int(row.get("matched_contacts", 0)),
            }
        )
    return rows


def collect_atlas_rows_from_workflow(workflow_node):
    """Collect atlas assignment rows in export-ready structure."""
    table = workflow_node.GetNodeReference("AtlasAssignmentTable")
    raw_rows = table_to_dict_rows(table) if table is not None else []
    rows = []
    for row in raw_rows:
        world_x = row.get("x_world_ras", row.get("x_ras", 0.0))
        world_y = row.get("y_world_ras", row.get("y_ras", 0.0))
        world_z = row.get("z_world_ras", row.get("z_ras", 0.0))
        rows.append(
            {
                "trajectory": row.get("trajectory", ""),
                "contact_label": row.get("contact_label", ""),
                "contact_index": _to_int(row.get("contact_index", 0)),
                "contact_ras": [_to_float(world_x), _to_float(world_y), _to_float(world_z)],
                "closest_source": row.get("closest_source", ""),
                "closest_label": row.get("closest_label", ""),
                "closest_label_value": _to_int(row.get("closest_label_value", 0)),
                "closest_distance_to_voxel_mm": _to_float(row.get("closest_distance_to_voxel_mm", 0.0)),
                "closest_distance_to_centroid_mm": _to_float(row.get("closest_distance_to_centroid_mm", 0.0)),
                "primary_source": row.get("primary_source", ""),
                "primary_label": row.get("primary_label", ""),
                "primary_label_value": _to_int(row.get("primary_label_value", 0)),
                "primary_distance_to_voxel_mm": _to_float(row.get("primary_distance_to_voxel_mm", 0.0)),
                "primary_distance_to_centroid_mm": _to_float(row.get("primary_distance_to_centroid_mm", 0.0)),
                "thomas_label": row.get("thomas_label", ""),
                "thomas_label_value": _to_int(row.get("thomas_label_value", 0)),
                "thomas_distance_to_voxel_mm": _to_float(row.get("thomas_distance_to_voxel_mm", 0.0)),
                "thomas_distance_to_centroid_mm": _to_float(row.get("thomas_distance_to_centroid_mm", 0.0)),
                "freesurfer_label": row.get("freesurfer_label", ""),
                "freesurfer_label_value": _to_int(row.get("freesurfer_label_value", 0)),
                "freesurfer_distance_to_voxel_mm": _to_float(row.get("freesurfer_distance_to_voxel_mm", 0.0)),
                "freesurfer_distance_to_centroid_mm": _to_float(row.get("freesurfer_distance_to_centroid_mm", 0.0)),
                "wm_label": row.get("wm_label", ""),
                "wm_label_value": _to_int(row.get("wm_label_value", 0)),
                "wm_distance_to_voxel_mm": _to_float(row.get("wm_distance_to_voxel_mm", 0.0)),
                "wm_distance_to_centroid_mm": _to_float(row.get("wm_distance_to_centroid_mm", 0.0)),
                "thomas_native_ras": [
                    _to_float(row.get("thomas_native_x_ras", 0.0)),
                    _to_float(row.get("thomas_native_y_ras", 0.0)),
                    _to_float(row.get("thomas_native_z_ras", 0.0)),
                ],
                "freesurfer_native_ras": [
                    _to_float(row.get("freesurfer_native_x_ras", 0.0)),
                    _to_float(row.get("freesurfer_native_y_ras", 0.0)),
                    _to_float(row.get("freesurfer_native_z_ras", 0.0)),
                ],
                "wm_native_ras": [
                    _to_float(row.get("wm_native_x_ras", 0.0)),
                    _to_float(row.get("wm_native_y_ras", 0.0)),
                    _to_float(row.get("wm_native_z_ras", 0.0)),
                ],
            }
        )
    return rows


def collect_export_inputs_from_workflow(workflow_node, output_frame_node=None):
    """Collect volume/contact/trajectory/table inputs from workflow roles."""
    state = WorkflowState()
    volume_node_ids = collect_volume_node_ids_from_registry(workflow_node)
    contacts = collect_contacts_from_workflow(workflow_node)
    planned = _collect_trajectory_map_from_nodes(
        state.role_nodes("PlannedTrajectoryLines", workflow_node=workflow_node),
        strip_plan_prefix=True,
    )
    final = _collect_trajectory_map_from_nodes(
        state.role_nodes("WorkingTrajectoryLines", workflow_node=workflow_node),
        strip_plan_prefix=False,
    )
    qc_rows = collect_qc_rows_from_workflow(workflow_node)
    atlas_rows = collect_atlas_rows_from_workflow(workflow_node)
    frame_node = output_frame_node or workflow_node.GetNodeReference("BaseVolume")
    return {
        "volume_node_ids": volume_node_ids,
        "contacts": contacts,
        "planned_trajectories": planned,
        "final_trajectories": final,
        "qc_rows": qc_rows,
        "atlas_rows": atlas_rows,
        "output_frame_node": frame_node,
    }


def export_aligned_bundle(
    volume_node_ids,
    contacts,
    out_dir,
    node_prefix="ROSA_Contacts",
    planned_trajectories=None,
    final_trajectories=None,
    qc_rows=None,
    atlas_rows=None,
    output_frame_node=None,
    export_profile="full_bundle",
):
    """Export aligned scene outputs with explicit coordinate-frame declaration."""
    os.makedirs(out_dir, exist_ok=True)
    profile, resolved_profile_name = get_export_profile(export_profile)
    include_volumes = bool(profile.get("include_volumes", True))
    include_contacts = bool(profile.get("include_contacts", True))
    include_planned = bool(profile.get("include_planned", True))
    include_final = bool(profile.get("include_final", True))
    include_qc = bool(profile.get("include_qc", True))
    include_atlas = bool(profile.get("include_atlas", True))

    saved_paths = []
    if include_volumes:
        for volume_name in sorted(volume_node_ids.keys()):
            node_id = volume_node_ids[volume_name]
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                continue
            filename = f"{_safe_filename(volume_name)}.nii.gz"
            out_path = os.path.join(out_dir, filename)
            ok = slicer.util.saveNode(node, out_path)
            if not ok:
                raise RuntimeError(f"Failed to save volume '{volume_name}' to {out_path}")
            saved_paths.append(out_path)

    frame_node = output_frame_node
    if frame_node is None and volume_node_ids:
        first_name = sorted(volume_node_ids.keys())[0]
        frame_node = slicer.mrmlScene.GetNodeByID(volume_node_ids[first_name])
    frame_name = frame_node.GetName() if frame_node is not None else "WORLD_RAS"
    world_to_frame = _world_to_node_ras_matrix(frame_node) if frame_node is not None else np.eye(4)

    def _to_frame_ras(point_world_ras):
        p = np.array([float(point_world_ras[0]), float(point_world_ras[1]), float(point_world_ras[2]), 1.0])
        out = world_to_frame @ p
        return [float(out[0]), float(out[1]), float(out[2])]

    coord_path = ""
    if include_contacts:
        coord_path = os.path.join(out_dir, f"{node_prefix}_aligned_world_coords.txt")
        lines = []
        lines.append("# ROSA Helper aligned export")
        lines.append(f"# coordinate_frame_node: {frame_name}")
        lines.append("# coordinate_system: FRAME_RAS (x_frame_ras,y_frame_ras,z_frame_ras)")
        lines.append("# alternate_columns: WORLD_RAS + LPS")
        lines.append(
            "# columns: trajectory,label,index,x_frame_ras,y_frame_ras,z_frame_ras,"
            "x_world_ras,y_world_ras,z_world_ras,x_lps,y_lps,z_lps,model_id"
        )
        for contact in sorted(contacts, key=lambda c: (str(c.get("trajectory", "")), int(c.get("index", 0)))):
            p_lps = contact["position_lps"]
            p_world_ras = lps_to_ras_point(p_lps)
            p_frame_ras = _to_frame_ras(p_world_ras)
            lines.append(
                "{traj},{label},{idx},{x_frame:.6f},{y_frame:.6f},{z_frame:.6f},"
                "{x_world:.6f},{y_world:.6f},{z_world:.6f},"
                "{x_lps:.6f},{y_lps:.6f},{z_lps:.6f},{model}".format(
                    traj=contact.get("trajectory", ""),
                    label=contact.get("label", ""),
                    idx=int(contact.get("index", 0)),
                    x_frame=float(p_frame_ras[0]),
                    y_frame=float(p_frame_ras[1]),
                    z_frame=float(p_frame_ras[2]),
                    x_world=float(p_world_ras[0]),
                    y_world=float(p_world_ras[1]),
                    z_world=float(p_world_ras[2]),
                    x_lps=float(p_lps[0]),
                    y_lps=float(p_lps[1]),
                    z_lps=float(p_lps[2]),
                    model=contact.get("model_id", ""),
                )
            )
        with open(coord_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    planned_path = ""
    planned_map = planned_trajectories or {}
    if include_planned:
        planned_path = os.path.join(out_dir, f"{node_prefix}_planned_trajectory_points.csv")
        rows = []
        for traj_name in sorted(planned_map.keys()):
            traj = planned_map[traj_name]
            for point_type, p_lps in (("entry", traj["start"]), ("target", traj["end"])):
                p_world_ras = lps_to_ras_point(p_lps)
                p_frame_ras = _to_frame_ras(p_world_ras)
                rows.append(
                    "{traj},{ptype},{x_frame:.6f},{y_frame:.6f},{z_frame:.6f},"
                    "{x_world:.6f},{y_world:.6f},{z_world:.6f},"
                    "{x_lps:.6f},{y_lps:.6f},{z_lps:.6f}".format(
                        traj=str(traj_name),
                        ptype=point_type,
                        x_frame=float(p_frame_ras[0]),
                        y_frame=float(p_frame_ras[1]),
                        z_frame=float(p_frame_ras[2]),
                        x_world=float(p_world_ras[0]),
                        y_world=float(p_world_ras[1]),
                        z_world=float(p_world_ras[2]),
                        x_lps=float(p_lps[0]),
                        y_lps=float(p_lps[1]),
                        z_lps=float(p_lps[2]),
                    )
                )
        with open(planned_path, "w", encoding="utf-8") as f:
            f.write(
                "trajectory,point_type,x_frame_ras,y_frame_ras,z_frame_ras,"
                "x_world_ras,y_world_ras,z_world_ras,x_lps,y_lps,z_lps\n"
            )
            if rows:
                f.write("\n".join(rows) + "\n")

    final_path = ""
    final_map = final_trajectories or {}
    if include_final:
        final_path = os.path.join(out_dir, f"{node_prefix}_final_trajectory_points.csv")
        rows = []
        for traj_name in sorted(final_map.keys()):
            traj = final_map[traj_name]
            for point_type, p_lps in (("entry", traj["start"]), ("target", traj["end"])):
                p_world_ras = lps_to_ras_point(p_lps)
                p_frame_ras = _to_frame_ras(p_world_ras)
                rows.append(
                    "{traj},{ptype},{x_frame:.6f},{y_frame:.6f},{z_frame:.6f},"
                    "{x_world:.6f},{y_world:.6f},{z_world:.6f},{x_lps:.6f},{y_lps:.6f},{z_lps:.6f}".format(
                        traj=str(traj_name),
                        ptype=point_type,
                        x_frame=float(p_frame_ras[0]),
                        y_frame=float(p_frame_ras[1]),
                        z_frame=float(p_frame_ras[2]),
                        x_world=float(p_world_ras[0]),
                        y_world=float(p_world_ras[1]),
                        z_world=float(p_world_ras[2]),
                        x_lps=float(p_lps[0]),
                        y_lps=float(p_lps[1]),
                        z_lps=float(p_lps[2]),
                    )
                )
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(
                "trajectory,point_type,x_frame_ras,y_frame_ras,z_frame_ras,"
                "x_world_ras,y_world_ras,z_world_ras,x_lps,y_lps,z_lps\n"
            )
            if rows:
                f.write("\n".join(rows) + "\n")

    qc_path = ""
    if include_qc:
        qc_path = os.path.join(out_dir, f"{node_prefix}_qc_metrics.csv")
        with open(qc_path, "w", encoding="utf-8") as f:
            f.write(
                "trajectory,entry_radial_mm,target_radial_mm,mean_contact_radial_mm,"
                "max_contact_radial_mm,rms_contact_radial_mm,angle_deg,matched_contacts\n"
            )
            for row in qc_rows or []:
                f.write(
                    "{trajectory},{entry:.6f},{target:.6f},{mean:.6f},{maxv:.6f},{rms:.6f},{angle:.6f},{matched}\n".format(
                        trajectory=str(row.get("trajectory", "")),
                        entry=_to_float(row.get("entry_radial_mm", 0.0)),
                        target=_to_float(row.get("target_radial_mm", 0.0)),
                        mean=_to_float(row.get("mean_contact_radial_mm", 0.0)),
                        maxv=_to_float(row.get("max_contact_radial_mm", 0.0)),
                        rms=_to_float(row.get("rms_contact_radial_mm", 0.0)),
                        angle=_to_float(row.get("angle_deg", 0.0)),
                        matched=_to_int(row.get("matched_contacts", 0)),
                    )
                )

    atlas_path = ""
    if include_atlas:
        atlas_path = os.path.join(out_dir, f"{node_prefix}_atlas_assignment.csv")
        with open(atlas_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "trajectory",
                    "contact_label",
                    "contact_index",
                    "x_ras",
                    "y_ras",
                    "z_ras",
                    "x_world_ras",
                    "y_world_ras",
                    "z_world_ras",
                    "closest_source",
                    "closest_label",
                    "closest_label_value",
                    "closest_distance_to_voxel_mm",
                    "closest_distance_to_centroid_mm",
                    "primary_source",
                    "primary_label",
                    "primary_label_value",
                    "primary_distance_to_voxel_mm",
                    "primary_distance_to_centroid_mm",
                    "thomas_label",
                    "thomas_label_value",
                    "thomas_distance_to_voxel_mm",
                    "thomas_distance_to_centroid_mm",
                    "freesurfer_label",
                    "freesurfer_label_value",
                    "freesurfer_distance_to_voxel_mm",
                    "freesurfer_distance_to_centroid_mm",
                    "wm_label",
                    "wm_label_value",
                    "wm_distance_to_voxel_mm",
                    "wm_distance_to_centroid_mm",
                    "thomas_native_x_ras",
                    "thomas_native_y_ras",
                    "thomas_native_z_ras",
                    "freesurfer_native_x_ras",
                    "freesurfer_native_y_ras",
                    "freesurfer_native_z_ras",
                    "wm_native_x_ras",
                    "wm_native_y_ras",
                    "wm_native_z_ras",
                ]
            )
            for row in atlas_rows or []:
                p_world_ras = row.get("contact_ras", [0.0, 0.0, 0.0])
                p_frame_ras = _to_frame_ras(p_world_ras)
                th_native = row.get("thomas_native_ras", [0.0, 0.0, 0.0])
                fs_native = row.get("freesurfer_native_ras", [0.0, 0.0, 0.0])
                wm_native = row.get("wm_native_ras", [0.0, 0.0, 0.0])
                writer.writerow(
                    [
                        str(row.get("trajectory", "")),
                        str(row.get("contact_label", "")),
                        int(row.get("contact_index", 0)),
                        float(p_frame_ras[0]),
                        float(p_frame_ras[1]),
                        float(p_frame_ras[2]),
                        float(p_world_ras[0]),
                        float(p_world_ras[1]),
                        float(p_world_ras[2]),
                        str(row.get("closest_source", "")),
                        str(row.get("closest_label", "")),
                        int(row.get("closest_label_value", 0)),
                        float(row.get("closest_distance_to_voxel_mm", 0.0)),
                        float(row.get("closest_distance_to_centroid_mm", 0.0)),
                        str(row.get("primary_source", "")),
                        str(row.get("primary_label", "")),
                        int(row.get("primary_label_value", 0)),
                        float(row.get("primary_distance_to_voxel_mm", 0.0)),
                        float(row.get("primary_distance_to_centroid_mm", 0.0)),
                        str(row.get("thomas_label", "")),
                        int(row.get("thomas_label_value", 0)),
                        float(row.get("thomas_distance_to_voxel_mm", 0.0)),
                        float(row.get("thomas_distance_to_centroid_mm", 0.0)),
                        str(row.get("freesurfer_label", "")),
                        int(row.get("freesurfer_label_value", 0)),
                        float(row.get("freesurfer_distance_to_voxel_mm", 0.0)),
                        float(row.get("freesurfer_distance_to_centroid_mm", 0.0)),
                        str(row.get("wm_label", "")),
                        int(row.get("wm_label_value", 0)),
                        float(row.get("wm_distance_to_voxel_mm", 0.0)),
                        float(row.get("wm_distance_to_centroid_mm", 0.0)),
                        float(th_native[0]),
                        float(th_native[1]),
                        float(th_native[2]),
                        float(fs_native[0]),
                        float(fs_native[1]),
                        float(fs_native[2]),
                        float(wm_native[0]),
                        float(wm_native[1]),
                        float(wm_native[2]),
                    ]
                )

    manifest_path = os.path.join(out_dir, f"{node_prefix}_manifest.json")
    manifest = {
        "schema_version": "1.0",
        "export_profile": resolved_profile_name,
        "profile_options": profile,
        "output_frame_node": frame_name,
        "files": {
            "volumes": saved_paths,
            "contacts": coord_path,
            "planned_trajectories": planned_path,
            "final_trajectories": final_path,
            "qc_metrics": qc_path,
            "atlas_assignments": atlas_path,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return {
        "out_dir": out_dir,
        "volume_count": len(saved_paths),
        "volume_paths": saved_paths,
        "coordinates_path": coord_path,
        "planned_trajectories_path": planned_path,
        "final_trajectories_path": final_path,
        "qc_metrics_path": qc_path,
        "atlas_assignment_path": atlas_path,
        "manifest_path": manifest_path,
        "output_frame_node": frame_name,
    }
