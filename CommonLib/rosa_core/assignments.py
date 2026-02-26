"""Helpers for trajectory/electrode assignment suggestions."""

import math


def trajectory_length_mm(trajectory):
    """Return Euclidean distance between `start` and `end` points in mm."""
    start = trajectory["start"]
    end = trajectory["end"]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    dz = float(end[2]) - float(start[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def electrode_length_mm(model):
    """Return exploration length (mm) from one electrode model dict."""
    return float(model.get("total_exploration_length_mm", 0.0))


def suggest_model_id_for_trajectory(
    trajectory,
    models_by_id,
    model_ids=None,
    tolerance_mm=5.0,
):
    """Select closest electrode model within tolerance of trajectory length.

    Tie-breaks:
    1) smallest absolute length delta
    2) higher contact count
    3) shorter model length
    4) lexical model ID
    """
    traj_len = trajectory_length_mm(trajectory)
    candidates = model_ids if model_ids is not None else sorted(models_by_id.keys())
    best_id = ""
    best_delta = None
    best_len = 0.0
    best_contacts = -1

    for model_id in candidates:
        model = models_by_id.get(model_id, {})
        model_len = electrode_length_mm(model)
        delta = abs(model_len - traj_len)
        if delta > float(tolerance_mm) + 1e-6:
            continue
        contact_count = int(model.get("contact_count", 0))
        if (
            best_delta is None
            or delta < best_delta - 1e-6
            or (abs(delta - best_delta) <= 1e-6 and contact_count > best_contacts)
            or (
                abs(delta - best_delta) <= 1e-6
                and contact_count == best_contacts
                and model_len < best_len - 1e-6
            )
            or (
                abs(delta - best_delta) <= 1e-6
                and contact_count == best_contacts
                and abs(model_len - best_len) <= 1e-6
                and model_id < best_id
            )
        ):
            best_id = model_id
            best_delta = delta
            best_len = model_len
            best_contacts = contact_count

    return best_id
