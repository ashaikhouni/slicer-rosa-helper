"""Pure policy helpers for atlas assignment orchestration."""

from __future__ import annotations

from typing import Mapping, Sequence


def collect_provider_samples(point_world_ras: Sequence[float], providers: Mapping[str, object]):
    """Collect one sample per ready provider."""
    samples = {}
    for source_id, provider in (providers or {}).items():
        if provider is None or not provider.is_ready():
            samples[source_id] = None
            continue
        samples[source_id] = provider.sample_contact(point_world_ras)
    return samples


def choose_closest_sample(samples):
    """Return (source_id, sample) for nearest valid sample by voxel distance."""
    choices = []
    for source_id, sample in (samples or {}).items():
        if sample is None or not sample.get("label"):
            continue
        choices.append((source_id, sample))
    if not choices:
        return "", {
            "label": "",
            "label_value": 0,
            "distance_to_voxel_mm": 0.0,
            "distance_to_centroid_mm": 0.0,
        }
    return min(choices, key=lambda item: float(item[1].get("distance_to_voxel_mm", float("inf"))))


def build_assignment_row(contact, point_ras, samples, closest_source, closest):
    """Build one assignment row using per-source samples and closest result."""
    th = samples.get("thomas")
    fs = samples.get("freesurfer")
    wm = samples.get("wm")

    th_native_ras = [0.0, 0.0, 0.0] if th is None else list(th.get("native_ras", [0.0, 0.0, 0.0]))
    fs_native_ras = [0.0, 0.0, 0.0] if fs is None else list(fs.get("native_ras", [0.0, 0.0, 0.0]))
    wm_native_ras = [0.0, 0.0, 0.0] if wm is None else list(wm.get("native_ras", [0.0, 0.0, 0.0]))

    return {
        "trajectory": contact.get("trajectory", ""),
        "contact_label": contact.get("label", ""),
        "contact_index": int(contact.get("index", 0)),
        "contact_ras": [float(point_ras[0]), float(point_ras[1]), float(point_ras[2])],
        "closest_source": closest_source,
        "closest_label": closest.get("label", ""),
        "closest_label_value": int(closest.get("label_value", 0)),
        "closest_distance_to_voxel_mm": float(closest.get("distance_to_voxel_mm", 0.0)),
        "closest_distance_to_centroid_mm": float(closest.get("distance_to_centroid_mm", 0.0)),
        "primary_source": closest_source,
        "primary_label": closest.get("label", ""),
        "primary_label_value": int(closest.get("label_value", 0)),
        "primary_distance_to_voxel_mm": float(closest.get("distance_to_voxel_mm", 0.0)),
        "primary_distance_to_centroid_mm": float(closest.get("distance_to_centroid_mm", 0.0)),
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

