"""Contact generation from ROSA trajectories and electrode model library."""

import json

from .transforms import lps_to_ras_point


def build_assignment_template(trajectories, default_model_id="", default_tip_at="target"):
    """Build editable assignment template from trajectories."""
    rows = []
    for traj in trajectories:
        rows.append(
            {
                "trajectory": traj["name"],
                "model_id": default_model_id,
                "tip_at": default_tip_at,
                "tip_shift_mm": 0.0,
                "xyz_offset_mm": [0.0, 0.0, 0.0],
            }
        )
    return {"schema_version": "1.0", "assignments": rows}


def save_assignment_template(path, template):
    """Write assignment template JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)


def load_assignments(path):
    """Load assignment JSON.

    Supported formats:
    1) {"assignments":[{"trajectory":"RHH","model_id":"DIXI-15AM",...}]}
    2) {"RHH":"DIXI-15AM","LHH":"DIXI-15CM"}  # shorthand
    """
    data = json.loads(open(path, "r", encoding="utf-8").read())
    if isinstance(data, dict) and "assignments" in data:
        return data
    if isinstance(data, dict):
        rows = []
        for traj_name, model_id in data.items():
            rows.append(
                {
                    "trajectory": traj_name,
                    "model_id": model_id,
                    "tip_at": "target",
                    "tip_shift_mm": 0.0,
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }
            )
        return {"schema_version": "1.0", "assignments": rows}
    raise ValueError("Unsupported assignments format")


def _sub(a, b):
    """Return vector subtraction `a - b` for 3D points."""
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _add(a, b):
    """Return vector addition `a + b` for 3D points."""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mul(v, s):
    """Return scalar multiplication `v * s` for 3D vectors."""
    return [v[0] * s, v[1] * s, v[2] * s]


def _norm(v):
    """Return Euclidean norm of a 3D vector."""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def _unit(v):
    """Return unit-length version of `v` and validate non-zero length."""
    n = _norm(v)
    if n <= 1e-9:
        raise ValueError("Zero-length trajectory vector")
    return [v[0] / n, v[1] / n, v[2] / n]


def _tip_and_axis(trajectory, tip_at):
    """Resolve tip point and forward axis from entry/target and tip anchor choice."""
    entry = trajectory["start"]
    target = trajectory["end"]
    tip_at_norm = (tip_at or "target").lower()
    if tip_at_norm == "target":
        tip = list(target)
        axis = _unit(_sub(entry, target))
    elif tip_at_norm == "entry":
        tip = list(entry)
        axis = _unit(_sub(target, entry))
    else:
        raise ValueError(f"Invalid tip_at '{tip_at}'. Expected 'entry' or 'target'")
    return tip, axis


def generate_contacts_for_assignment(trajectory, model, assignment):
    """Generate contact centers for a single trajectory/model assignment.

    Returns contact points in ROSA/LPS coordinates.
    """
    tip_at = assignment.get("tip_at", "target")
    tip_shift = float(assignment.get("tip_shift_mm", 0.0))
    xyz_offset = assignment.get("xyz_offset_mm", [0.0, 0.0, 0.0])
    if len(xyz_offset) != 3:
        raise ValueError("xyz_offset_mm must have 3 values")
    xyz_offset = [float(x) for x in xyz_offset]

    tip, axis = _tip_and_axis(trajectory, tip_at)
    tip = _add(_add(tip, _mul(axis, tip_shift)), xyz_offset)

    contacts = []
    label_prefix = trajectory["name"]
    for idx, offset in enumerate(model["contact_center_offsets_from_tip_mm"], start=1):
        point = _add(tip, _mul(axis, float(offset)))
        contacts.append(
            {
                "trajectory": trajectory["name"],
                "model_id": model["id"],
                "index": idx,
                "label": f"{label_prefix}{idx}",
                "position_lps": point,
                "tip_at": tip_at,
            }
        )
    return contacts


def generate_contacts(trajectories, models_by_id, assignments):
    """Generate contacts for all assigned trajectories."""
    traj_map = {t["name"]: t for t in trajectories}
    out = []
    missing = []
    for a in assignments.get("assignments", []):
        traj_name = a.get("trajectory")
        model_id = a.get("model_id")
        if not model_id:
            continue
        if traj_name not in traj_map:
            missing.append(f"trajectory '{traj_name}'")
            continue
        if model_id not in models_by_id:
            missing.append(f"model '{model_id}'")
            continue
        contacts = generate_contacts_for_assignment(traj_map[traj_name], models_by_id[model_id], a)
        out.extend(contacts)
    if missing:
        raise ValueError("Missing references in assignments: " + ", ".join(missing))
    return out


def save_contacts_rosa_json(path, contacts, metadata=None):
    """Save contacts in ROSA/LPS coordinates."""
    payload = {"schema_version": "1.0", "coordinate_system": "ROSA_LPS", "contacts": contacts}
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def contacts_to_fcsv_rows(contacts, to_ras=True):
    """Convert contacts to FCSV row format used by exporters.save_fcsv."""
    rows = []
    for c in contacts:
        p = c["position_lps"]
        xyz = lps_to_ras_point(p) if to_ras else list(p)
        rows.append({"label": c["label"], "xyz": xyz})
    return rows


def build_contacts_markups(contacts, to_ras=True, node_name="contacts"):
    """Build a Slicer Markups JSON document with one Fiducial list."""
    cps = []
    for c in contacts:
        p = c["position_lps"]
        xyz = lps_to_ras_point(p) if to_ras else list(p)
        cps.append(
            {
                "id": c["label"],
                "label": c["label"],
                "description": c["model_id"],
                "position": xyz,
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
            }
        )

    coord = "RAS" if to_ras else "LPS"
    return {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json",
        "markups": [
            {
                "type": "Fiducial",
                "name": node_name,
                "coordinateSystem": coord,
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "controlPoints": cps,
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [1.0, 1.0, 0.2],
                    "selectedColor": [1.0, 0.8, 0.1],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.0,
                    "textScale": 1.0,
                },
            }
        ],
    }


def save_contacts_markups_json(path, contacts, to_ras=True, node_name="contacts"):
    """Write contact points as Slicer Markups JSON file."""
    doc = build_contacts_markups(contacts, to_ras=to_ras, node_name=node_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
