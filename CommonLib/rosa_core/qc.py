"""QC metrics for planned vs final electrode trajectories and contacts."""

from __future__ import annotations

import math

from .types import ContactRecord, Point3D, QCMetricsRow, TrajectoryRecord


def _vsub(a: Point3D, b: Point3D) -> Point3D:
    """Return vector subtraction `a - b` for 3D vectors."""
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2])]


def _vdot(a: Point3D, b: Point3D) -> float:
    """Return dot product between 3D vectors."""
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _vmul(v: Point3D, s: float) -> Point3D:
    """Return scalar multiplication `v * s`."""
    return [float(v[0]) * float(s), float(v[1]) * float(s), float(v[2]) * float(s)]


def _vnorm(v: Point3D) -> float:
    """Return Euclidean norm of a 3D vector."""
    return math.sqrt(_vdot(v, v))


def _vunit(v: Point3D) -> Point3D:
    """Return normalized 3D vector, raising for zero-length input."""
    n = _vnorm(v)
    if n <= 1e-9:
        raise ValueError("Zero-length vector")
    return [float(v[0]) / n, float(v[1]) / n, float(v[2]) / n]


def _radial_error_mm(delta_vec: Point3D, axis_unit: Point3D) -> float:
    """Return perpendicular component length of `delta_vec` relative to `axis_unit`."""
    axial_mm = _vdot(delta_vec, axis_unit)
    axial_vec = _vmul(axis_unit, axial_mm)
    radial_vec = _vsub(delta_vec, axial_vec)
    return _vnorm(radial_vec)


def _trajectory_axis_angle_deg(planned: TrajectoryRecord, final: TrajectoryRecord) -> float:
    """Return angle in degrees between planned and final trajectory axes."""
    planned_axis = _vunit(_vsub(planned["end"], planned["start"]))
    final_axis = _vunit(_vsub(final["end"], final["start"]))
    dot = max(-1.0, min(1.0, _vdot(planned_axis, final_axis)))
    return math.degrees(math.acos(dot))


def sorted_contacts_by_trajectory(contacts: list[ContactRecord]) -> dict[str, list[ContactRecord]]:
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
    planned_trajectories_by_name: dict[str, TrajectoryRecord],
    final_trajectories_by_name: dict[str, TrajectoryRecord],
    planned_contacts: list[ContactRecord],
    final_contacts: list[ContactRecord],
    include_unmatched_planned: bool = False,
) -> list[QCMetricsRow]:
    """Compute per-trajectory radial and angular QC metrics.

    Inputs are expected in ROSA/LPS coordinates.
    Returns one dictionary row per trajectory with matching planned/final contacts.
    When ``include_unmatched_planned`` is True, planned trajectories that do not
    have a final counterpart are included with ``None`` metrics and
    ``matched_contacts=0``.
    """
    planned_by_traj = sorted_contacts_by_trajectory(planned_contacts)
    final_by_traj = sorted_contacts_by_trajectory(final_contacts)

    rows = []
    if include_unmatched_planned:
        traj_names = sorted(planned_trajectories_by_name.keys())
    else:
        traj_names = sorted(final_by_traj.keys())

    for traj_name in traj_names:
        planned = planned_trajectories_by_name.get(traj_name)
        final = final_trajectories_by_name.get(traj_name)
        if planned is None:
            continue
        if final is None:
            if include_unmatched_planned:
                rows.append(
                    {
                        "trajectory": traj_name,
                        "entry_radial_mm": None,
                        "target_radial_mm": None,
                        "mean_contact_radial_mm": None,
                        "max_contact_radial_mm": None,
                        "rms_contact_radial_mm": None,
                        "angle_deg": None,
                        "matched_contacts": 0,
                    }
                )
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
            if include_unmatched_planned:
                rows.append(
                    {
                        "trajectory": traj_name,
                        "entry_radial_mm": entry_rad,
                        "target_radial_mm": target_rad,
                        "mean_contact_radial_mm": None,
                        "max_contact_radial_mm": None,
                        "rms_contact_radial_mm": None,
                        "angle_deg": angle_deg,
                        "matched_contacts": 0,
                    }
                )
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


def compute_plan_vs_placement_qc(
    planned_by_name: dict[str, TrajectoryRecord],
    fitted_by_name: dict[str, dict],
) -> list[dict]:
    """Per-shank deviation of a fitted placement from its plan.

    This is the well-posed QC: it compares a *fitted result* against the *plan*
    it was generated from (e.g. the CLI ``fit-rosa`` seed→fit, or the Slicer
    Guided Fit seed→fit). Unlike the generic CTV table it replaced, it does NOT
    require synthesising nominal planned contacts — the perpendicular distance of
    a fitted contact from the planned LINE is the same regardless of which point
    on the line you reference, so we measure it directly from the planned entry.

    Args:
        planned_by_name: ``name -> {"start": entry, "end": target}``.
        fitted_by_name:  ``name -> {"start": fitted_entry, "end": fitted_tip,
            "contacts": [[x, y, z], ...]}``. MUST be in the SAME coordinate frame
            as ``planned`` (any consistent orthonormal frame — RAS or LPS; the
            metrics are radial/angular so they're frame-invariant).

    Returns one row per planned shank:
        ``trajectory``, ``entry_error_mm`` (3-D), ``target_error_mm`` (3-D),
        ``mean_contact_radial_mm`` / ``max_contact_radial_mm`` /
        ``rms_contact_radial_mm`` (perpendicular distance of each fitted contact
        from the planned line), ``angle_deg`` (axis deviation), ``n_contacts``.
        A planned shank with no fitted counterpart gets ``None`` metrics.
    """
    rows = []
    for name in sorted(planned_by_name.keys()):
        planned = planned_by_name[name]
        fitted = fitted_by_name.get(name)
        if fitted is None:
            rows.append({
                "trajectory": name, "entry_error_mm": None, "target_error_mm": None,
                "mean_contact_radial_mm": None, "max_contact_radial_mm": None,
                "rms_contact_radial_mm": None, "angle_deg": None, "n_contacts": 0,
            })
            continue
        try:
            planned_axis = _vunit(_vsub(planned["end"], planned["start"]))
        except Exception:
            continue
        entry_err = _vnorm(_vsub(fitted["start"], planned["start"]))
        target_err = _vnorm(_vsub(fitted["end"], planned["end"]))
        try:
            angle_deg = _trajectory_axis_angle_deg(planned, fitted)
        except Exception:
            angle_deg = None
        radials = [
            _radial_error_mm(_vsub(c, planned["start"]), planned_axis)
            for c in (fitted.get("contacts") or [])
        ]
        if radials:
            mean_rad = sum(radials) / float(len(radials))
            max_rad = max(radials)
            rms_rad = math.sqrt(sum(v * v for v in radials) / float(len(radials)))
        else:
            mean_rad = max_rad = rms_rad = None
        rows.append({
            "trajectory": name,
            "entry_error_mm": entry_err,
            "target_error_mm": target_err,
            "mean_contact_radial_mm": mean_rad,
            "max_contact_radial_mm": max_rad,
            "rms_contact_radial_mm": rms_rad,
            "angle_deg": angle_deg,
            "n_contacts": len(radials),
        })
    return rows


def summarize_plan_vs_placement_qc(rows: list[dict]) -> dict:
    """Cohort summary (median / p95 / max) of :func:`compute_plan_vs_placement_qc`
    rows, over the shanks with a non-None value for each metric."""
    def _stat(key):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        if not vals:
            return {"median": None, "p95": None, "max": None}
        n = len(vals)
        median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        p95 = vals[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
        return {"median": median, "p95": p95, "max": vals[-1]}

    return {
        k: _stat(k)
        for k in (
            "entry_error_mm", "target_error_mm",
            "mean_contact_radial_mm", "max_contact_radial_mm", "angle_deg",
        )
    }
