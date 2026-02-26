"""QC metrics for planned vs final electrode trajectories and contacts."""

import math


def _vsub(a, b):
    """Return vector subtraction `a - b` for 3D vectors."""
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def _vdot(a, b):
    """Return dot product between 3D vectors."""
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _vmul(v, s):
    """Return scalar multiplication `v * s`."""
    return [float(v[0]) * float(s), float(v[1]) * float(s), float(v[2]) * float(s)]


def _vnorm(v):
    """Return Euclidean norm of a 3D vector."""
    return math.sqrt(_vdot(v, v))


def _vunit(v):
    """Return normalized 3D vector, raising for zero-length input."""
    n = _vnorm(v)
    if n <= 1e-9:
        raise ValueError("Zero-length vector")
    return [float(v[0]) / n, float(v[1]) / n, float(v[2]) / n]


def _radial_error_mm(delta_vec, axis_unit):
    """Return perpendicular component length of `delta_vec` relative to `axis_unit`."""
    axial_mm = _vdot(delta_vec, axis_unit)
    axial_vec = _vmul(axis_unit, axial_mm)
    radial_vec = _vsub(delta_vec, axial_vec)
    return _vnorm(radial_vec)


def _trajectory_axis_angle_deg(planned, final):
    """Return angle in degrees between planned and final trajectory axes."""
    planned_axis = _vunit(_vsub(planned["end"], planned["start"]))
    final_axis = _vunit(_vsub(final["end"], final["start"]))
    dot = max(-1.0, min(1.0, _vdot(planned_axis, final_axis)))
    return math.degrees(math.acos(dot))


def sorted_contacts_by_trajectory(contacts):
    """Return map `trajectory -> contacts sorted by contact index`."""
    by_traj = {}
    for contact in contacts:
        name = str(contact.get("trajectory", ""))
        if not name:
            continue
        by_traj.setdefault(name, []).append(contact)
    for name in list(by_traj.keys()):
        by_traj[name] = sorted(by_traj[name], key=lambda c: int(c.get("index", 0)))
    return by_traj


def compute_qc_metrics(
    planned_trajectories_by_name,
    final_trajectories_by_name,
    planned_contacts,
    final_contacts,
):
    """Compute per-trajectory radial and angular QC metrics.

    Inputs are expected in ROSA/LPS coordinates.
    Returns one dictionary row per trajectory with matching planned/final contacts.
    """
    planned_by_traj = sorted_contacts_by_trajectory(planned_contacts)
    final_by_traj = sorted_contacts_by_trajectory(final_contacts)

    rows = []
    for traj_name in sorted(final_by_traj.keys()):
        planned = planned_trajectories_by_name.get(traj_name)
        final = final_trajectories_by_name.get(traj_name)
        if planned is None or final is None:
            continue

        try:
            planned_axis = _vunit(_vsub(planned["end"], planned["start"]))
        except Exception:
            continue

        entry_delta = _vsub(final["start"], planned["start"])
        target_delta = _vsub(final["end"], planned["end"])
        entry_rad = _radial_error_mm(entry_delta, planned_axis)
        target_rad = _radial_error_mm(target_delta, planned_axis)
        angle_deg = _trajectory_axis_angle_deg(planned, final)

        planned_index_map = {
            int(c.get("index", 0)): c
            for c in planned_by_traj.get(traj_name, [])
            if int(c.get("index", 0)) > 0
        }
        final_index_map = {
            int(c.get("index", 0)): c
            for c in final_by_traj.get(traj_name, [])
            if int(c.get("index", 0)) > 0
        }
        matched_idx = sorted(set(planned_index_map.keys()) & set(final_index_map.keys()))
        if not matched_idx:
            continue

        radial_errors = []
        for idx in matched_idx:
            p_plan = planned_index_map[idx]["position_lps"]
            p_final = final_index_map[idx]["position_lps"]
            delta = _vsub(p_final, p_plan)
            radial_errors.append(_radial_error_mm(delta, planned_axis))

        mean_rad = sum(radial_errors) / float(len(radial_errors))
        max_rad = max(radial_errors)
        rms_rad = math.sqrt(sum((v * v for v in radial_errors)) / float(len(radial_errors)))

        rows.append(
            {
                "trajectory": traj_name,
                "entry_radial_mm": entry_rad,
                "target_radial_mm": target_rad,
                "mean_contact_radial_mm": mean_rad,
                "max_contact_radial_mm": max_rad,
                "rms_contact_radial_mm": rms_rad,
                "angle_deg": angle_deg,
                "matched_contacts": len(matched_idx),
            }
        )

    return rows
