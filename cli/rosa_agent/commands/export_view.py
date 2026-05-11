"""rosa-agent export-view — build a 3D viewer scene from ROSA + FreeSurfer.

What it does (mirrors what the Slicer extension does interactively):

1. Run the existing ``pipeline`` (load ROSA -> detect -> place contacts ->
   label) with the FreeSurfer parcellation wired in so each contact gets
   an anatomical label written to ``labels.tsv``.
2. Compute a rigid registration from the FreeSurfer T1 to the working
   CT (same algorithm the labelmap provider uses, run once and reused).
3. Read FreeSurfer surfaces (``surf/lh.pial`` + ``rh.pial``), convert
   them from FreeSurfer tkrRAS to scanner RAS, then through the rigid
   transform into the contact frame.
4. Optionally paint each surface vertex with its ``aparc.annot`` color
   so the brain looks like a FreeSurfer parcellation.
5. Pack everything (surfaces + trajectories as cylinders + contacts as
   spheres + per-node "extras" metadata) into a ``scene.glb`` and write
   a tiny ``index.html`` that loads it in any modern browser.

Inputs:

* ROSA case folder (positional; same target argument the pipeline uses).
* FreeSurfer recon-all subject directory (``--freesurfer-dir``).
* Optional external CT (``--ct``); same semantics as pipeline.
* Optional THOMAS directory (``--thomas``) for thalamic labeling.

Output directory layout:

    out_dir/
        trajectories.tsv
        contacts.tsv
        labels.tsv                # per-contact atlas labels
        ct.nii.gz                 # working CT (when ROSA-folder mode)
        manifest.json
        scene.glb                 # the 3D scene
        index.html                # static viewer
        scene_meta.json           # contacts + trajectories listings for the viewer JS
        view_manifest.json        # summary of inputs + counts

No Slicer / VTK / Qt imports — pure-Python on top of nibabel + SimpleITK +
numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..io.freesurfer import (
    FSSubject,
    load_fs_surfaces,
    parse_freesurfer_lut,
    resolve_fs_subject,
)
from ..io.glb_writer import GLBScene, write_glb
from ..io.trajectory_io import read_tsv_rows


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------
# FreeSurfer asset discovery
# ---------------------------------------------------------------------


_APARC_CANDIDATES = (
    "aparc+aseg.mgz",
    "aparc.DKTatlas+aseg.mgz",
    "aparc.a2009s+aseg.mgz",
)


def _discover_parcellation(fs: FSSubject, override: str | None) -> Path | None:
    """Resolve which FreeSurfer parcellation labelmap to use."""
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"FreeSurfer parcellation not found: {override}")
        return p
    for name in _APARC_CANDIDATES:
        candidate = fs.mri_dir / name
        if candidate.is_file():
            return candidate
    return None


def _discover_lut(fs: FSSubject, override: str | None) -> Path | None:
    """Locate a FreeSurferColorLUT.txt.

    Preference order: explicit override → ``$FREESURFER_HOME/FreeSurferColorLUT.txt``
    → bundled copy in ``CommonLib/resources/freesurfer/``.
    """
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"LUT not found: {override}")
        return p

    import os
    home = os.environ.get("FREESURFER_HOME")
    if home:
        candidate = Path(home) / "FreeSurferColorLUT.txt"
        if candidate.is_file():
            return candidate

    # Bundled fallback. Resolves relative to this file's repo location.
    here = Path(__file__).resolve()
    bundled = here.parents[3] / "CommonLib" / "resources" / "freesurfer" / "FreeSurferColorLUT20120827.txt"
    if bundled.is_file():
        return bundled
    return None


# ---------------------------------------------------------------------
# Registration: FreeSurfer T1 -> working CT
# ---------------------------------------------------------------------


def _load_image_as_sitk(path: Path):
    """Load any nibabel-supported format and return a SITK ``Image``.

    SITK's pip wheels don't always include the MGH ImageIO (the CI
    builders carry a slimmer SITK than typical local installs), so a
    bare ``sitk.ReadImage("...T1.mgz")`` fails with "Unable to determine
    ImageIO reader" on those systems. nibabel reads MGZ + NIfTI + many
    others everywhere, so we route through nibabel and stamp the
    resulting SITK image with the source affine.
    """
    import nibabel as nib
    import SimpleITK as sitk
    from rosa_detect.service import stamp_ijk_to_ras_on_sitk

    img = nib.load(str(path))
    arr_ijk = np.asarray(img.dataobj, dtype=np.float32)
    # nibabel arrays are (i, j, k); SITK's GetImageFromArray expects
    # (k, j, i) — transpose to match.
    arr_kji = np.transpose(arr_ijk, (2, 1, 0))
    sitk_img = sitk.GetImageFromArray(np.ascontiguousarray(arr_kji))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    sitk_img.SetSpacing(tuple(float(v) for v in zooms))
    stamp_ijk_to_ras_on_sitk(sitk_img, np.asarray(img.affine, dtype=float))
    return sitk_img


def _register_fs_to_ct(t1_path: Path, ct_path: Path):
    """Rigid Mattes-MI register FS T1 to CT.

    Returns a tuple ``(fs_to_ct_4x4, fixed_ct, moving_t1, sitk_transform)``
    so the caller can both push surface vertices through the 4×4 AND
    resample the T1 onto the CT grid via the same SITK transform —
    without paying for a second registration pass.
    """
    import SimpleITK as sitk
    from rosa_core.registration import register_rigid_mi

    # CT is a NIfTI written by the pipeline — SITK reads it natively
    # everywhere. T1 is FS-native (.mgz); route through nibabel so the
    # MGH IO requirement is satisfied via a portable backend.
    fixed = sitk.ReadImage(str(ct_path))
    moving = _load_image_as_sitk(t1_path)
    _stderr(
        f"[view] registering FS T1 ({t1_path.name}) -> CT ({ct_path.name}) "
        f"(rigid + Mattes MI)…"
    )
    result = register_rigid_mi(fixed=fixed, moving=moving, logger=_stderr)
    _stderr(
        f"[view] registration done: metric={result.final_metric:+.5f} "
        f"iters={result.n_iterations} ({result.converged_reason})"
    )
    # We want FS-RAS -> CT-RAS = moving -> fixed in the rigid convention.
    return result.moving_to_fixed_ras_4x4, fixed, moving, result.transform


def _write_t1_resampled_to_ct(
    fixed_ct, moving_t1, sitk_transform, out_path: Path,
) -> dict[str, Any]:
    """Resample the FS T1 onto the CT grid and write as NIfTI.

    The resampled volume lives in the CT RAS frame, which is also the
    frame the contacts + trajectories live in. The browser-side slice
    viewer parses this NIfTI and snaps the three orthogonal planes to
    a contact's RAS coordinate.

    Returns a dict describing what was written so the viewer's
    ``scene_meta.json`` can be populated.
    """
    import SimpleITK as sitk
    from rosa_core.registration import resample_volume

    resampled = resample_volume(
        moving_t1, sitk_transform,
        reference=fixed_ct, interp="linear",
    )
    # The FS T1 is intensity-normalized to [0, 255] by recon-all (uint8
    # on disk). _load_image_as_sitk widens to float32 so SITK Resample
    # behaves predictably, but for *storage* — and especially over HTTP
    # to the in-browser slice viewer — uint8 is the right end shape.
    # Clamp + cast: lossless for the FS T1 source, 4× smaller file.
    if resampled.GetPixelID() != sitk.sitkUInt8:
        resampled = sitk.Cast(
            sitk.Clamp(resampled, lowerBound=0.0, upperBound=255.0),
            sitk.sitkUInt8,
        )
    sitk.WriteImage(resampled, str(out_path))

    size = list(resampled.GetSize())
    spacing = list(resampled.GetSpacing())
    origin_lps = list(resampled.GetOrigin())
    direction_lps = list(resampled.GetDirection())
    # The viewer needs the vox->RAS affine. SITK stores origin/direction
    # in LPS; convert to RAS for the JS side.
    origin_ras = [-origin_lps[0], -origin_lps[1], origin_lps[2]]
    d = np.asarray(direction_lps, dtype=float).reshape(3, 3)
    # Flip the first two rows of the direction matrix to convert LPS to RAS.
    d_ras = d.copy()
    d_ras[0, :] *= -1.0
    d_ras[1, :] *= -1.0
    vox_to_ras = np.eye(4)
    for k in range(3):
        vox_to_ras[:3, k] = d_ras[:, k] * spacing[k]
    vox_to_ras[:3, 3] = origin_ras

    return {
        "path": out_path.name,
        "size": [int(s) for s in size],
        "spacing": [float(s) for s in spacing],
        "vox_to_ras": [[float(v) for v in row] for row in vox_to_ras.tolist()],
        "dtype": str(sitk.GetArrayViewFromImage(resampled).dtype),
    }


# ---------------------------------------------------------------------
# Reading pipeline output
# ---------------------------------------------------------------------


def _read_pipeline_trajectories(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv_rows(path)
    out = []
    for r in rows:
        try:
            out.append({
                "name": r.get("name", ""),
                "start": (float(r["start_x"]), float(r["start_y"]), float(r["start_z"])),
                "end": (float(r["end_x"]), float(r["end_y"]), float(r["end_z"])),
                "confidence": r.get("confidence", ""),
                "confidence_label": r.get("confidence_label", ""),
                "electrode_model": r.get("electrode_model", ""),
                "bolt_source": r.get("bolt_source", ""),
                "length_mm": r.get("length_mm", ""),
            })
        except (KeyError, ValueError):
            continue
    return out


def _read_pipeline_contacts(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv_rows(path)
    out = []
    for r in rows:
        try:
            out.append({
                "trajectory": r.get("trajectory", ""),
                "label": r.get("label", ""),
                "contact_index": int(r.get("contact_index", 0) or 0),
                "position": (float(r["x"]), float(r["y"]), float(r["z"])),
                "peak_detected": r.get("peak_detected", "1") == "1",
                "electrode_model": r.get("electrode_model", ""),
            })
        except (KeyError, ValueError):
            continue
    return out


def _read_labels_by_contact(path: Path | None) -> dict[str, dict[str, str]]:
    """Map ``contact_label`` -> labels-row dict. Empty when no labels TSV."""
    if path is None or not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    for r in read_tsv_rows(path):
        key = r.get("contact_label") or r.get("label") or ""
        if key:
            out[key] = r
    return out


# ---------------------------------------------------------------------
# GLB assembly
# ---------------------------------------------------------------------


_TRAJ_BAND_COLORS = {
    "high": (0.20, 0.90, 0.25, 1.0),
    "medium": (0.95, 0.75, 0.10, 1.0),
    "low": (0.95, 0.30, 0.20, 1.0),
    "": (0.70, 0.70, 0.70, 1.0),
}


def _color_for_label_value(
    label_value: int | str, lut_index: dict[int, dict[str, Any]],
) -> tuple[float, float, float, float]:
    """Return an RGBA tuple for an integer FS label. Falls back to gray."""
    try:
        value = int(label_value)
    except (TypeError, ValueError):
        return (0.7, 0.7, 0.7, 1.0)
    entry = lut_index.get(value)
    if not entry:
        return (0.7, 0.7, 0.7, 1.0)
    r, g, b, _a = entry["rgba"]
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def _build_scene(
    *,
    surfaces,
    trajectories,
    contacts,
    contact_labels,
    lut_index,
    shaft_radius_mm: float = 0.35,
    contact_band_radius_mm: float = 0.55,
    contact_band_length_mm: float = 2.0,
) -> tuple[GLBScene, list[dict[str, Any]]]:
    """Assemble the GLB scene and return (scene, contact-metadata-rows).

    Geometry layout per electrode (a "shank"):

      * ``shaft/<name>``  — thin dark cylinder running the full
        ``start`` → ``end`` length (radius ~0.35 mm, matching real SEEG
        electrode insulation).
      * ``contact/<label>`` — short cylinder centered on each contact,
        ~2 mm long along the trajectory axis, slightly thicker than the
        shaft so the bands look metallic and protrude (radius ~0.55 mm).
        Coloured by the contact's FreeSurfer region when available.

    Brain surfaces use ``alphaMode: BLEND`` so the electrodes show
    through the cortex by default.

    Every electrode node carries ``extras.shank`` (the trajectory name)
    so the in-browser viewer can isolate a single shank with one
    lookup. Every contact node also carries the contact label, index,
    electrode model, FS/THOMAS/WM region, and peak-detection flag.
    """
    scene = GLBScene()

    # Surfaces — one material per hemisphere. Default to OPAQUE +
    # white so vertex colors from ``aparc.annot`` come through at full
    # saturation in any static viewer (Preview, QuickLook,
    # gltf-viewer.donmccurdy.com, etc.). The HTML viewer flips opacity
    # at runtime via the Brain α slider (it sets material.transparent
    # = true, depthWrite = false when alpha < 1), so we don't sacrifice
    # the see-through electrodes view — we just stop showing a
    # washed-out ghost brain in tools that don't honor alphaMode=BLEND.
    lh_mat = scene.add_material(
        "fs_lh_pial", (1.0, 1.0, 1.0, 1.0),
        metallic=0.0, roughness=0.85, double_sided=True,
    )
    rh_mat = scene.add_material(
        "fs_rh_pial", (1.0, 1.0, 1.0, 1.0),
        metallic=0.0, roughness=0.85, double_sided=True,
    )
    for surf in surfaces:
        mat = lh_mat if surf.hemi == "lh" else rh_mat
        scene.add_surface(
            name=f"fs_{surf.name.replace('.', '_')}",
            positions=surf.vertices_ras,
            faces=surf.faces,
            material_index=mat,
            vertex_colors_rgba=surf.vertex_colors_rgba,
            normals=surf.vertex_normals,
            extras={
                "kind": "freesurfer_surface",
                "hemi": surf.hemi,
                "surface": surf.kind,
                "annotation": surf.annotation_name,
                "n_vertices": int(surf.vertices_ras.shape[0]),
            },
        )

    # Per-shank shaft colors. Static viewers (Preview, QuickLook,
    # online glTF viewers) don't run our JS, so a single-material
    # "all shafts dark gray" makes the scene visually homogeneous in
    # those tools. A deterministic hue per shank name produces
    # distinguishable electrodes everywhere.
    def _shaft_color_for(name: str) -> tuple[float, float, float, float]:
        import hashlib
        h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
        hue = (h % 360) / 360.0
        # HSV -> RGB with high value (0.85) and moderate saturation
        # (0.65): saturated enough to read against the white brain,
        # not so bright they overwhelm the parcellation colors.
        c = 0.85 * 0.65
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = 0.85 - c
        if hue < 1 / 6:    r, g, b = c, x, 0.0
        elif hue < 2 / 6:  r, g, b = x, c, 0.0
        elif hue < 3 / 6:  r, g, b = 0.0, c, x
        elif hue < 4 / 6:  r, g, b = 0.0, x, c
        elif hue < 5 / 6:  r, g, b = x, 0.0, c
        else:              r, g, b = c, 0.0, x
        return (r + m, g + m, b + m, 1.0)

    # Per-shank axis cache so each contact band can be oriented along
    # its trajectory.
    traj_axis: dict[str, np.ndarray] = {}
    shaft_mat_by_shank: dict[str, int] = {}
    for traj in trajectories:
        start = np.asarray(traj["start"], dtype=float)
        end = np.asarray(traj["end"], dtype=float)
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            # Degenerate shank — fall back to +Z so add_segment can still
            # emit a tiny placeholder cylinder.
            unit = np.array([0.0, 0.0, 1.0])
        else:
            unit = (end - start) / length
        traj_axis[traj["name"]] = unit

        color = _shaft_color_for(str(traj["name"]))
        shaft_mat = scene.add_material(
            f"shaft_{traj['name']}", color,
            metallic=0.3, roughness=0.55,
        )
        shaft_mat_by_shank[str(traj["name"])] = shaft_mat

        scene.add_segment(
            name=f"shaft_{traj['name']}",
            p0=traj["start"], p1=traj["end"],
            radius=shaft_radius_mm,
            material_index=shaft_mat,
            extras={
                "kind": "shaft",
                "shank": traj["name"],
                "confidence": traj.get("confidence", ""),
                "confidence_label": traj.get("confidence_label", ""),
                "electrode_model": traj.get("electrode_model", ""),
                "bolt_source": traj.get("bolt_source", ""),
                "length_mm": traj.get("length_mm", ""),
            },
        )

    # Contact bands — short cylinder per contact, aligned along the
    # trajectory axis, coloured by FS region.
    contact_meta: list[dict[str, Any]] = []
    color_to_material: dict[tuple[float, float, float, float], int] = {}
    fallback_mat = scene.add_material(
        "contact_unlabeled", (0.95, 0.95, 0.95, 1.0),
        metallic=0.7, roughness=0.3,
    )
    half_len = 0.5 * float(contact_band_length_mm)
    for contact in contacts:
        label_row = contact_labels.get(contact["label"], {})
        fs_label_name = label_row.get("freesurfer_label", "") or label_row.get("closest_label", "")
        fs_label_value = label_row.get("freesurfer_label_value") or label_row.get("closest_label_value")
        fs_distance = label_row.get("freesurfer_distance_to_voxel_mm") or label_row.get("closest_distance_to_voxel_mm")
        thomas_label = label_row.get("thomas_label", "")
        wm_label = label_row.get("wm_label", "")

        rgba = _color_for_label_value(fs_label_value, lut_index) if fs_label_value else (0.95, 0.95, 0.95, 1.0)
        mat = color_to_material.get(rgba)
        if mat is None:
            mat = scene.add_material(
                f"contact_lab{int(round(rgba[0]*255))}_{int(round(rgba[1]*255))}_{int(round(rgba[2]*255))}",
                rgba, metallic=0.7, roughness=0.3,
            )
            color_to_material[rgba] = mat
        if not fs_label_value:
            mat = fallback_mat

        center = np.asarray(contact["position"], dtype=float)
        axis = traj_axis.get(contact["trajectory"], np.array([0.0, 0.0, 1.0]))
        p0 = center - half_len * axis
        p1 = center + half_len * axis

        extras = {
            "kind": "contact",
            "shank": contact["trajectory"],
            "label": contact["label"],
            "contact_index": contact["contact_index"],
            "electrode_model": contact["electrode_model"],
            "peak_detected": bool(contact["peak_detected"]),
            "freesurfer_label": fs_label_name,
            "thomas_label": thomas_label,
            "wm_label": wm_label,
            "distance_mm": fs_distance,
            "position": [float(center[0]), float(center[1]), float(center[2])],
        }
        scene.add_segment(
            name=f"contact_{contact['label']}",
            p0=p0, p1=p1,
            radius=contact_band_radius_mm,
            material_index=mat,
            extras=extras,
        )
        contact_meta.append({
            "label": contact["label"],
            "trajectory": contact["trajectory"],
            "contact_index": contact["contact_index"],
            "position": [float(center[0]), float(center[1]), float(center[2])],
            "electrode_model": contact["electrode_model"],
            "peak_detected": bool(contact["peak_detected"]),
            "freesurfer_label": fs_label_name,
            "thomas_label": thomas_label,
            "wm_label": wm_label,
            "distance_mm": fs_distance,
        })

    return scene, contact_meta


# ---------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111; color: #eee; overflow: hidden; }}
  #app {{ display: grid; grid-template-columns: 1fr 320px 320px; height: 100%; }}
  #canvas-host {{ position: relative; }}
  #canvas-host canvas {{ display: block; }}
  #toolbar {{ position: absolute; top: 10px; left: 10px; right: 10px; z-index: 5; display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }}
  #toolbar button {{ background: rgba(40,40,40,0.85); color: #eee; border: 1px solid #444; padding: 6px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
  #toolbar button:hover {{ background: rgba(70,70,70,0.95); }}
  .plane-ctl {{ display: inline-flex; align-items: center; gap: 6px; background: rgba(40,40,40,0.85); border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #ddd; }}
  .plane-ctl input[type="range"] {{ width: 120px; }}
  .plane-ctl .axis {{ font-weight: 600; min-width: 52px; }}
  .plane-ctl .coord {{ color: #888; font-family: ui-monospace, monospace; min-width: 42px; text-align: right; }}
  #status-bar {{
    position: absolute; left: 10px; right: 10px; bottom: 8px; z-index: 4;
    background: rgba(15,15,15,0.92); color: #ccc; border: 1px solid #333;
    border-radius: 4px; padding: 6px 10px; font-family: ui-monospace, monospace;
    font-size: 11px; display: flex; gap: 18px; flex-wrap: wrap;
    pointer-events: none;
  }}
  #status-bar .lbl {{ color: #888; margin-right: 4px; }}
  #status-bar .ok {{ color: #6fcf6f; }}
  #status-bar .warn {{ color: #f5b13b; }}
  #status-bar .err {{ color: #ff6464; }}
  #slices {{ background: #0d0d0d; border-left: 1px solid #2a2a2a; padding: 8px 8px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }}
  .slice-panel {{ position: relative; background: #000; border: 1px solid #1d1d1d; }}
  .slice-panel canvas {{ display: block; width: 100%; height: auto; image-rendering: crisp-edges; }}
  .slice-label {{ position: absolute; top: 4px; left: 6px; font-size: 10px; color: #bbb; letter-spacing: 0.08em; text-transform: uppercase; pointer-events: none; }}
  .slice-coord {{ position: absolute; bottom: 4px; right: 6px; font-size: 10px; color: #888; pointer-events: none; font-family: ui-monospace, monospace; }}
  .slice-axes {{ position: absolute; bottom: 4px; left: 6px; font-size: 9px; color: #555; pointer-events: none; font-family: ui-monospace, monospace; }}
  #side {{ overflow-y: auto; padding: 12px 14px; border-left: 1px solid #2a2a2a; font-size: 13px; background: #161616; }}
  #side h2 {{ font-size: 12px; margin: 14px 0 6px; color: #999; letter-spacing: 0.08em; text-transform: uppercase; }}
  #subject {{ font-size: 14px; color: #ddd; font-weight: 500; }}
  .small {{ color: #888; font-size: 11px; }}
  .shank-card {{ border: 1px solid #2a2a2a; border-radius: 4px; margin-bottom: 6px; overflow: hidden; }}
  .shank-header {{ padding: 6px 8px; background: #1f1f1f; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
  .shank-header:hover {{ background: #262626; }}
  .shank-card.active .shank-header {{ background: #2a3a2a; }}
  .shank-name {{ font-weight: 500; }}
  .shank-meta {{ color: #888; font-size: 11px; }}
  .shank-contacts {{ display: none; }}
  .shank-card.expanded .shank-contacts {{ display: block; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; background: #333; color: #ddd; margin-left: 4px; }}
  .band-high {{ background: #225a25; }}
  .band-medium {{ background: #6a5418; }}
  .band-low {{ background: #6a2515; }}
  .contact-row {{ padding: 4px 8px 4px 18px; cursor: pointer; display: flex; justify-content: space-between; font-size: 12px; border-top: 1px solid #1c1c1c; }}
  .contact-row:hover {{ background: #232323; }}
  .contact-row.selected {{ background: #4a1f1f; color: #ffdcdc; }}
  .contact-row .label {{ font-family: ui-monospace, monospace; }}
  .contact-row .region {{ color: #aaa; }}
  .contact-row.selected .region {{ color: #ffdcdc; }}
</style>
<!-- es-module-shims polyfills <script type="importmap"> on browsers that
     don't support it natively (older Safari/Firefox). Modern Chrome/Safari/
     Firefox ignore it. -->
<script async src="https://unpkg.com/es-module-shims@1.10.0/dist/es-module-shims.js"></script>
<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="app">
  <div id="canvas-host">
    <div id="toolbar">
      <button id="btn-reset">Show all</button>
      <button id="btn-fit">Fit view</button>
      <label class="plane-ctl" data-control="brain-color">
        <span class="axis">Surface</span>
        <select>
          <option value="skin">Skin</option>
          <option value="parcellation" selected>Parcellation</option>
          <option value="white">White</option>
        </select>
      </label>
      <label class="plane-ctl" data-control="brain-alpha">
        <span class="axis">Brain α</span>
        <input type="range" min="0" max="1" step="0.05" value="0.45" />
        <span class="coord">0.45</span>
      </label>
      <label class="plane-ctl" data-control="plane-alpha">
        <span class="axis">Slice α</span>
        <input type="range" min="0" max="1" step="0.05" value="0.95" />
        <span class="coord">0.95</span>
      </label>
      <label class="plane-ctl" data-axis="axial">
        <input type="checkbox" /><span class="axis">Axial</span>
        <input type="range" min="0" max="1" step="0.5" disabled />
        <span class="coord">—</span>
      </label>
      <label class="plane-ctl" data-axis="coronal">
        <input type="checkbox" /><span class="axis">Coronal</span>
        <input type="range" min="0" max="1" step="0.5" disabled />
        <span class="coord">—</span>
      </label>
      <label class="plane-ctl" data-axis="sagittal">
        <input type="checkbox" /><span class="axis">Sagittal</span>
        <input type="range" min="0" max="1" step="0.5" disabled />
        <span class="coord">—</span>
      </label>
    </div>
    <div id="status-bar">
      <span><span class="lbl">MRI:</span><span id="dbg-mri">init</span></span>
      <span><span class="lbl">GLB:</span><span id="dbg-glb">init</span></span>
      <span><span class="lbl">Click:</span><span id="dbg-click">—</span></span>
      <span><span class="lbl">Snap:</span><span id="dbg-snap">—</span></span>
    </div>
  </div>
  <div id="slices">
    <div class="slice-panel" data-axis="axial">
      <canvas></canvas>
      <div class="slice-label">Axial</div>
      <div class="slice-axes">R&rarr;  A&uarr;</div>
      <div class="slice-coord"></div>
    </div>
    <div class="slice-panel" data-axis="coronal">
      <canvas></canvas>
      <div class="slice-label">Coronal</div>
      <div class="slice-axes">R&rarr;  S&uarr;</div>
      <div class="slice-coord"></div>
    </div>
    <div class="slice-panel" data-axis="sagittal">
      <canvas></canvas>
      <div class="slice-label">Sagittal</div>
      <div class="slice-axes">A&rarr;  S&uarr;</div>
      <div class="slice-coord"></div>
    </div>
  </div>
  <div id="side">
    <div id="subject"></div>
    <div class="small" id="summary"></div>
    <h2>Trajectories &amp; contacts</h2>
    <div id="shanks"></div>
  </div>
</div>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ GLTFLoader }} from "three/addons/loaders/GLTFLoader.js";

// Width reserved for everything to the right of the 3D canvas:
// slices column (320px) + sidebar column (320px).
const rightStripWidth = 320 + 320;

// Tiny debug-strip writer — every key event funnels through here so
// the user can see what's actually happening without opening devtools.
const _dbg = {{}};
function setDbg(slot, msg, cls) {{
  const el = document.getElementById("dbg-" + slot);
  if (!el) return;
  el.textContent = msg;
  el.className = cls || "";
  _dbg[slot] = msg;
}}

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
const host = document.getElementById("canvas-host");
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101010);
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(1, 1.2, 0.8);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.45);
fill.position.set(-1, -0.7, -0.6);
scene.add(fill);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
camera.position.set(220, 140, 220);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

function fitToObject(obj) {{
  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const diag = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  const dir = new THREE.Vector3(0.7, 0.45, 0.8).normalize();
  camera.position.copy(center).addScaledVector(dir, diag * 0.95);
  camera.near = Math.max(0.1, diag * 0.001);
  camera.far = diag * 10;
  camera.updateProjectionMatrix();
  controls.update();
}}

function resize() {{
  // Drive the canvas from the host's measured rect — that's the
  // grid-column width. setSize(w, h, true) syncs both the WebGL buffer
  // AND the canvas's CSS size, so the canvas fills its column instead
  // of collapsing to the default 300x150.
  const r = host.getBoundingClientRect();
  const w = Math.max(64, Math.floor(r.width));
  const h = Math.max(64, Math.floor(r.height));
  renderer.setSize(w, h, true);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
window.addEventListener("resize", resize);
// CSS grid resizes don't always fire window 'resize' — observe the
// host directly so the canvas tracks column reflows too.
new ResizeObserver(resize).observe(host);
resize();

(function loop() {{
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
}})();

// ---------- load GLB + index nodes ---------------------------------
//
// glTF puts our metadata (kind / shank / label / region) on the *node*
// extras, not the mesh. GLTFLoader mirrors that as two Object3Ds per
// element: an outer Object3D named "contact/RHH1" carrying the extras
// in userData, with a child Mesh named "contact/RHH1_mesh" carrying
// the geometry + material. We index the outer Object3D (so click /
// isolate logic uses the same names the metadata uses) and stash a
// reference to its mesh child for live material swapping.

const nodesByName = new Map();           // node.name -> outer Object3D
const shankNodes = new Map();            // shank id -> [Object3D, ...]
const surfaceNodes = [];                 // FS pial Object3D nodes
const originalMaterials = new Map();     // outer Object3D -> child Mesh's original material
let gltfRoot = null;

const RED = new THREE.MeshStandardMaterial({{ color: 0xff2030, metalness: 0.75, roughness: 0.25 }});

function _firstMeshChild(obj) {{
  let mesh = null;
  obj.traverse(c => {{ if (!mesh && c.isMesh) mesh = c; }});
  return mesh;
}}

const loader = new GLTFLoader();
setDbg("glb", "loading…");
loader.load("scene.glb", gltf => {{
  gltfRoot = gltf.scene;
  scene.add(gltfRoot);
  let nContacts = 0, nShafts = 0, nSurf = 0;
  gltfRoot.traverse(obj => {{
    const extras = obj.userData || {{}};
    const kind = extras.kind;
    if (!kind) return;  // only the outer nodes carry kind/shank
    if (kind === "contact") nContacts++;
    else if (kind === "shaft") nShafts++;
    else if (kind === "freesurfer_surface") nSurf++;
    // Key the lookup by extras (the canonical, unmangled identifier),
    // not by obj.name. Three.js's GLTFLoader runs every node name
    // through PropertyBinding.sanitizeNodeName, which strips any char
    // outside [A-Za-z0-9_-] — so a Python-side name like "contact/RHH5"
    // becomes "contactRHH5" on the JS side and the slash lookup misses.
    // Using extras.label / extras.shank sidesteps that entirely.
    let lookupKey = obj.name;
    if (kind === "contact" && extras.label) lookupKey = "contact:" + extras.label;
    else if (kind === "shaft" && extras.shank) lookupKey = "shaft:" + extras.shank;
    nodesByName.set(lookupKey, obj);
    const mesh = _firstMeshChild(obj);
    if (mesh) {{
      originalMaterials.set(obj, mesh.material);
      // Cache the mesh so material swaps (red highlight) don't have
      // to re-traverse.
      obj.userData._mesh = mesh;
    }}
    if (kind === "freesurfer_surface") {{
      // We deliberately do NOT toggle transparent here. The GLB ships
      // alphaMode=OPAQUE; the Brain α slider initializer below
      // sets transparent + opacity from a clean baseline. Forcing
      // transparent=true at load was the v2 default, but it left the
      // material's shader stuck in BLEND mode even after the slider
      // dialed back to 1.0 — the source of the "fractured cortex at
      // α=1" report.
      surfaceNodes.push(obj);
    }} else if (extras.shank) {{
      if (!shankNodes.has(extras.shank)) shankNodes.set(extras.shank, []);
      shankNodes.get(extras.shank).push(obj);
    }}
  }});
  setDbg("glb", `${{nSurf}} surface · ${{nShafts}} shafts · ${{nContacts}} contacts`, "ok");
  // Apply the default Surface color + Brain α slider value now that
  // surfaces are indexed. Without these, the GLB's authored material
  // state (opaque white) would persist until the user touched a
  // slider — even though both controls have non-default initial
  // positions in the toolbar. The result was a counter-intuitive
  // "slider says 0.45 but brain is solid" boot state.
  if (typeof _applyInitialSurfaceColor === "function") _applyInitialSurfaceColor();
  if (typeof _applyInitialBrainAlpha === "function") _applyInitialBrainAlpha();
  fitToObject(gltfRoot);
}}, undefined, err => {{
  console.error("GLB load failed", err);
  setDbg("glb", "load failed: " + (err.message || err), "err");
  document.getElementById("subject").textContent = "Failed to load scene.glb: " + err;
}});

// ---------- MRI slice viewer --------------------------------------
//
// Minimal NIfTI-1 parser + slice renderer. The exporter resamples the
// FreeSurfer T1 onto the CT grid so it lives in the same RAS frame as
// the contacts; we draw three orthogonal canvases and snap to a
// contact's voxel coordinates when the user clicks one.
//
// Convention: the input volume is recon-all's intensity-normalized T1
// (uint8, 256³). We don't do any windowing.

function _invert4x4(m) {{
  // Affine inverse; fine for rigid + spacing matrices used here.
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h = m[2][1], i = m[2][2];
  const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
  if (Math.abs(det) < 1e-12) return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const inv = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]];
  inv[0][0] = (e*i - f*h) / det;
  inv[0][1] = (c*h - b*i) / det;
  inv[0][2] = (b*f - c*e) / det;
  inv[1][0] = (f*g - d*i) / det;
  inv[1][1] = (a*i - c*g) / det;
  inv[1][2] = (c*d - a*f) / det;
  inv[2][0] = (d*h - e*g) / det;
  inv[2][1] = (b*g - a*h) / det;
  inv[2][2] = (a*e - b*d) / det;
  inv[0][3] = -(inv[0][0]*m[0][3] + inv[0][1]*m[1][3] + inv[0][2]*m[2][3]);
  inv[1][3] = -(inv[1][0]*m[0][3] + inv[1][1]*m[1][3] + inv[1][2]*m[2][3]);
  inv[2][3] = -(inv[2][0]*m[0][3] + inv[2][1]*m[1][3] + inv[2][2]*m[2][3]);
  return inv;
}}

function _apply4x4(m, v) {{
  return [
    m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2] + m[0][3],
    m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2] + m[1][3],
    m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2] + m[2][3],
  ];
}}

async function _maybeInflate(buffer) {{
  // .nii.gz support via the native DecompressionStream API
  // (Chrome 80+, Safari 16.4+, Firefox 113+). Returns the buffer
  // unchanged if it's not gzipped (magic != 0x1f 0x8b).
  const view = new Uint8Array(buffer);
  if (view.length < 2 || view[0] !== 0x1f || view[1] !== 0x8b) return buffer;
  if (typeof DecompressionStream === "undefined") {{
    throw new Error("Browser lacks DecompressionStream for .gz; can't load t1_in_ct.nii.gz");
  }}
  const ds = new DecompressionStream("gzip");
  const writer = ds.writable.getWriter();
  writer.write(view); writer.close();
  return await new Response(ds.readable).arrayBuffer();
}}

class NiftiVolume {{
  constructor(buffer) {{
    const v = new DataView(buffer);
    this.dim = [v.getInt16(42, true), v.getInt16(44, true), v.getInt16(46, true)];
    this.pixdim = [v.getFloat32(80, true), v.getFloat32(84, true), v.getFloat32(88, true)];
    this.datatype = v.getInt16(70, true);
    const voxOffset = v.getFloat32(108, true);
    // sform affine (preferred) — NIFTI stores vox->scanner RAS as 3 rows.
    this.affine = [
      [v.getFloat32(280, true), v.getFloat32(284, true), v.getFloat32(288, true), v.getFloat32(292, true)],
      [v.getFloat32(296, true), v.getFloat32(300, true), v.getFloat32(304, true), v.getFloat32(308, true)],
      [v.getFloat32(312, true), v.getFloat32(316, true), v.getFloat32(320, true), v.getFloat32(324, true)],
      [0, 0, 0, 1],
    ];
    this.affineInv = _invert4x4(this.affine);
    const off = Math.max(352, Math.floor(voxOffset));
    if (this.datatype === 2) {{
      this.data = new Uint8Array(buffer, off);
    }} else if (this.datatype === 4) {{
      this.data = new Int16Array(buffer, off);
    }} else if (this.datatype === 16) {{
      this.data = new Float32Array(buffer, off);
    }} else {{
      throw new Error("Unsupported NIfTI datatype " + this.datatype);
    }}
    // Auto-window from sampled percentile-ish range. For uint8 T1.mgz
    // (recon-all output) this collapses to [0, 255] which is exactly
    // the right rendering range; for float32 inputs we sample 1/256
    // of the voxels for a coarse min/max.
    let lo = Infinity, hi = -Infinity;
    const step = Math.max(1, Math.floor(this.data.length / 65536));
    for (let i = 0; i < this.data.length; i += step) {{
      const x = this.data[i];
      if (x < lo) lo = x; if (x > hi) hi = x;
    }}
    this.displayMin = lo;
    this.displayMax = Math.max(hi, lo + 1);
  }}
  rasToVox(ras) {{ return _apply4x4(this.affineInv, ras); }}
  voxel(i, j, k) {{
    if (i < 0 || j < 0 || k < 0) return 0;
    const [Nx, Ny, Nz] = this.dim;
    if (i >= Nx || j >= Ny || k >= Nz) return 0;
    return this.data[i + Nx * (j + Ny * k)];
  }}
}}

const slicePanels = {{}};   // axis -> {{ canvas, ctx, coordEl }}
let mriVolume = null;
let mriRasCursor = null;   // last selected RAS point [x,y,z]
let mriVoxCursor = [0, 0, 0]; // last selected vox indices

function _initSlicePanels() {{
  for (const panel of document.querySelectorAll(".slice-panel")) {{
    const axis = panel.dataset.axis;
    const canvas = panel.querySelector("canvas");
    // Fixed device-pixel size; CSS scales to width 100%.
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    slicePanels[axis] = {{ canvas, ctx, coordEl: panel.querySelector(".slice-coord") }};
  }}
}}

function renderSlice(axis) {{
  if (!mriVolume) return;
  const panel = slicePanels[axis];
  if (!panel) return;
  const [Nx, Ny, Nz] = mriVolume.dim;
  // Pick the in-plane axes (U/V) plus the through-plane axis index.
  // Naming convention for RAS-oriented data: the volume's i-axis ~ X (R),
  // j-axis ~ Y (A), k-axis ~ Z (S). For non-axis-aligned affines this
  // is approximate but still gives a sensible orthogonal slice.
  let W, H, throughIdx, throughLabel;
  if (axis === "axial") {{
    W = Nx; H = Ny; throughIdx = mriVoxCursor[2]; throughLabel = "k";
  }} else if (axis === "coronal") {{
    W = Nx; H = Nz; throughIdx = mriVoxCursor[1]; throughLabel = "j";
  }} else {{ // sagittal
    W = Ny; H = Nz; throughIdx = mriVoxCursor[0]; throughLabel = "i";
  }}
  const canvas = panel.canvas;
  if (canvas.width !== W || canvas.height !== H) {{
    canvas.width = W; canvas.height = H;
  }}
  const ctx = panel.ctx;
  const img = ctx.createImageData(W, H);
  const range = Math.max(1e-6, mriVolume.displayMax - mriVolume.displayMin);
  const lo = mriVolume.displayMin;
  for (let v = 0; v < H; v++) {{
    // Flip the vertical axis so anatomically "up" (anterior/superior)
    // appears at the top of the canvas.
    const vSrc = H - 1 - v;
    for (let u = 0; u < W; u++) {{
      let i, j, k;
      if (axis === "axial") {{ i = u; j = vSrc; k = throughIdx; }}
      else if (axis === "coronal") {{ i = u; j = throughIdx; k = vSrc; }}
      else {{ i = throughIdx; j = u; k = vSrc; }}
      const raw = mriVolume.voxel(i, j, k);
      const g = Math.max(0, Math.min(255, Math.round((raw - lo) / range * 255)));
      const px = (v * W + u) * 4;
      img.data[px] = g; img.data[px+1] = g; img.data[px+2] = g; img.data[px+3] = 255;
    }}
  }}
  ctx.putImageData(img, 0, 0);

  // Crosshair
  if (mriVoxCursor) {{
    let cu, cv;
    if (axis === "axial") {{ cu = mriVoxCursor[0]; cv = H - 1 - mriVoxCursor[1]; }}
    else if (axis === "coronal") {{ cu = mriVoxCursor[0]; cv = H - 1 - mriVoxCursor[2]; }}
    else {{ cu = mriVoxCursor[1]; cv = H - 1 - mriVoxCursor[2]; }}
    if (cu >= 0 && cu < W && cv >= 0 && cv < H) {{
      // Lines are drawn in canvas-buffer pixels but the canvas is
      // displayed at ~half size via CSS, so a 1px stroke ends up
      // sub-CSS-pixel and gets anti-aliased to nothing. Use 3px +
      // bright red so the crosshair is unmissable.
      ctx.strokeStyle = "#ff0040";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(0, cv + 0.5); ctx.lineTo(W, cv + 0.5);
      ctx.moveTo(cu + 0.5, 0); ctx.lineTo(cu + 0.5, H);
      ctx.stroke();
      ctx.fillStyle = "#ff0040";
      ctx.beginPath();
      ctx.arc(cu + 0.5, cv + 0.5, 6, 0, Math.PI * 2);
      ctx.fill();
      // White inner pip so the dot reads even on bright cortex.
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(cu + 0.5, cv + 0.5, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }}
    const ras = mriRasCursor || [0, 0, 0];
    panel.coordEl.textContent =
      `${{throughLabel}}=${{throughIdx|0}} | RAS ${{ras[0].toFixed(1)}}, ${{ras[1].toFixed(1)}}, ${{ras[2].toFixed(1)}}`;
  }}
}}

function snapSlicesToRas(ras) {{
  if (!mriVolume) {{
    setDbg("snap", `RAS ${{ras[0].toFixed(1)}},${{ras[1].toFixed(1)}},${{ras[2].toFixed(1)}} (MRI not loaded)`, "warn");
    return;
  }}
  const vox = mriVolume.rasToVox(ras);
  mriVoxCursor = [Math.round(vox[0]), Math.round(vox[1]), Math.round(vox[2])];
  mriRasCursor = ras;
  setDbg("snap",
    `RAS ${{ras[0].toFixed(1)}},${{ras[1].toFixed(1)}},${{ras[2].toFixed(1)}} ` +
    `→ vox ${{mriVoxCursor[0]}},${{mriVoxCursor[1]}},${{mriVoxCursor[2]}}`,
    "ok");
  renderSlice("axial");
  renderSlice("coronal");
  renderSlice("sagittal");
}}

async function loadMri(t1Meta) {{
  if (!t1Meta || !t1Meta.path) {{
    setDbg("mri", "not exported", "warn");
    // No T1 was exported (no FS surfaces); leave the slice panels blank.
    for (const axis of ["axial","coronal","sagittal"]) {{
      const p = slicePanels[axis]; if (!p) continue;
      const ctx = p.ctx;
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, p.canvas.width, p.canvas.height);
      ctx.fillStyle = "#666";
      ctx.font = "11px ui-sans-serif";
      ctx.fillText("No MRI available", 10, 20);
    }}
    return;
  }}
  try {{
    setDbg("mri", "fetching…");
    const resp = await fetch(t1Meta.path);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const raw = await resp.arrayBuffer();
    const inflated = await _maybeInflate(raw);
    mriVolume = new NiftiVolume(inflated);
    // Default cursor: volume center.
    mriVoxCursor = [
      Math.floor(mriVolume.dim[0] / 2),
      Math.floor(mriVolume.dim[1] / 2),
      Math.floor(mriVolume.dim[2] / 2),
    ];
    mriRasCursor = _apply4x4(mriVolume.affine, mriVoxCursor);
    renderSlice("axial"); renderSlice("coronal"); renderSlice("sagittal");
    setDbg("mri", `loaded ${{mriVolume.dim[0]}}×${{mriVolume.dim[1]}}×${{mriVolume.dim[2]}}`, "ok");
    // Kick the 3D cut planes — they reuse the same parsed volume.
    initCutPlanes(mriVolume);
    // If the user clicked a contact while the 16MB T1 was still
    // downloading, snapSlicesToRas silently no-oped (mriVolume was
    // null). Retroactively snap now that the volume is ready.
    if (selectedContact) {{
      const pos = selectedContact.getWorldPosition(new THREE.Vector3());
      snapSlicesToRas([pos.x, pos.y, pos.z]);
      snapCutPlanesToRas([pos.x, pos.y, pos.z]);
    }}
  }} catch (err) {{
    console.error("MRI load failed", err);
    setDbg("mri", "load failed: " + (err.message || err), "err");
    for (const axis of ["axial","coronal","sagittal"]) {{
      const p = slicePanels[axis]; if (!p) continue;
      const ctx = p.ctx;
      ctx.fillStyle = "#1a1a1a"; ctx.fillRect(0, 0, p.canvas.width, p.canvas.height);
      ctx.fillStyle = "#c66"; ctx.font = "11px ui-sans-serif";
      ctx.fillText("MRI load failed: " + err.message, 6, 18);
    }}
  }}
}}

// ---------- 3D cut planes ------------------------------------------
//
// Once the NIfTI is parsed we upload it as a 3D texture and add three
// textured quads (axial / coronal / sagittal) to the scene. A custom
// shader maps each fragment's world position back to voxel coords via
// the inverse affine and samples the volume.
//
// Defaults: planes are HIDDEN. Each plane has its own checkbox+slider
// in the toolbar; clicking a contact also moves all three sliders so
// the planes intersect at the contact.

const cutPlanes = {{}};     // axis -> THREE.Mesh
let cutVolumeBox = null;   // world-RAS bbox of the volume corners

function _volumeRasBbox(vol) {{
  const [Nx, Ny, Nz] = vol.dim;
  let xmin = Infinity, xmax = -Infinity;
  let ymin = Infinity, ymax = -Infinity;
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i <= 1; i++) for (let j = 0; j <= 1; j++) for (let k = 0; k <= 1; k++) {{
    const [x, y, z] = _apply4x4(vol.affine, [i * Nx, j * Ny, k * Nz]);
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    if (z < zmin) zmin = z; if (z > zmax) zmax = z;
  }}
  return {{ xmin, xmax, ymin, ymax, zmin, zmax }};
}}

function _buildVolumeTexture(vol) {{
  // Data3DTexture wants Uint8Array (R-channel) for our use case. The
  // FS T1 is already uint8; float32 inputs get windowed to uint8.
  let arr = vol.data;
  if (!(arr instanceof Uint8Array)) {{
    const out = new Uint8Array(arr.length);
    const range = Math.max(1e-6, vol.displayMax - vol.displayMin);
    const lo = vol.displayMin;
    for (let i = 0; i < arr.length; i++) {{
      out[i] = Math.max(0, Math.min(255, Math.round((arr[i] - lo) / range * 255)));
    }}
    arr = out;
  }}
  const tex = new THREE.Data3DTexture(arr, vol.dim[0], vol.dim[1], vol.dim[2]);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}}

function _makeCutPlane(axis, bbox, volTex, vol) {{
  // axis: "axial" (world Z), "coronal" (world Y), "sagittal" (world X).
  // PlaneGeometry default is XY plane with +Z normal; we rotate as
  // needed so the plane lies in the chosen RAS plane.
  let width, height;
  if (axis === "axial") {{ width = bbox.xmax - bbox.xmin; height = bbox.ymax - bbox.ymin; }}
  else if (axis === "coronal") {{ width = bbox.xmax - bbox.xmin; height = bbox.zmax - bbox.zmin; }}
  else {{ width = bbox.ymax - bbox.ymin; height = bbox.zmax - bbox.zmin; }}

  const geo = new THREE.PlaneGeometry(width, height);

  // Inverse affine: world RAS -> voxel index. mat4 stored row-major in
  // affineInv; THREE.Matrix4 wants column-major (set() takes row-major
  // and transposes for you).
  const inv = vol.affineInv;
  const rasToVox = new THREE.Matrix4().set(
    inv[0][0], inv[0][1], inv[0][2], inv[0][3],
    inv[1][0], inv[1][1], inv[1][2], inv[1][3],
    inv[2][0], inv[2][1], inv[2][2], inv[2][3],
    0, 0, 0, 1,
  );

  const mat = new THREE.ShaderMaterial({{
    uniforms: {{
      uVolume: {{ value: volTex }},
      uRasToVox: {{ value: rasToVox }},
      uVolumeSize: {{ value: new THREE.Vector3(vol.dim[0], vol.dim[1], vol.dim[2]) }},
      uOpacity: {{ value: 0.95 }},
    }},
    vertexShader: `
      varying vec3 vWorldPos;
      void main() {{
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }}
    `,
    fragmentShader: `
      precision highp float;
      precision highp sampler3D;
      uniform sampler3D uVolume;
      uniform mat4 uRasToVox;
      uniform vec3 uVolumeSize;
      uniform float uOpacity;
      varying vec3 vWorldPos;
      void main() {{
        vec3 vox = (uRasToVox * vec4(vWorldPos, 1.0)).xyz;
        vec3 uvw = (vox + 0.5) / uVolumeSize;
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) discard;
        float intensity = texture(uVolume, uvw).r;
        gl_FragColor = vec4(vec3(intensity), uOpacity);
      }}
    `,
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
  }});

  const plane = new THREE.Mesh(geo, mat);
  plane.userData.axis = axis;
  // Rotate so the plane lies in the right RAS plane.
  if (axis === "sagittal") plane.rotation.y = Math.PI / 2;          // normal = +X
  else if (axis === "coronal") plane.rotation.x = -Math.PI / 2;     // normal = +Y
  // axial keeps default normal = +Z
  // Initial center (mid of bbox):
  const cx = 0.5 * (bbox.xmin + bbox.xmax);
  const cy = 0.5 * (bbox.ymin + bbox.ymax);
  const cz = 0.5 * (bbox.zmin + bbox.zmax);
  if (axis === "axial") plane.position.set(cx, cy, cz);
  else if (axis === "coronal") plane.position.set(cx, cy, cz);
  else plane.position.set(cx, cy, cz);
  plane.visible = false;
  return plane;
}}

function _bboxRangeForAxis(bbox, axis) {{
  if (axis === "axial") return [bbox.zmin, bbox.zmax];
  if (axis === "coronal") return [bbox.ymin, bbox.ymax];
  return [bbox.xmin, bbox.xmax];
}}

function _setCutPlaneValue(axis, value) {{
  const plane = cutPlanes[axis];
  if (!plane) return;
  if (axis === "axial") plane.position.z = value;
  else if (axis === "coronal") plane.position.y = value;
  else plane.position.x = value;
  // Reflect into the slider + coord label.
  const ctl = document.querySelector(`.plane-ctl[data-axis="${{axis}}"]`);
  if (ctl) {{
    const slider = ctl.querySelector('input[type="range"]');
    if (slider && !slider.disabled) slider.value = String(value);
    const coord = ctl.querySelector(".coord");
    if (coord) coord.textContent = value.toFixed(1);
  }}
}}

function initCutPlanes(vol) {{
  // Wipe any previous planes (defensive — loadMri can fire twice).
  for (const axis of Object.keys(cutPlanes)) {{
    scene.remove(cutPlanes[axis]);
    delete cutPlanes[axis];
  }}
  cutVolumeBox = _volumeRasBbox(vol);
  const tex = _buildVolumeTexture(vol);
  for (const axis of ["axial", "coronal", "sagittal"]) {{
    const plane = _makeCutPlane(axis, cutVolumeBox, tex, vol);
    cutPlanes[axis] = plane;
    scene.add(plane);
    // Wire toolbar control
    const ctl = document.querySelector(`.plane-ctl[data-axis="${{axis}}"]`);
    if (!ctl) continue;
    const cb = ctl.querySelector('input[type="checkbox"]');
    const slider = ctl.querySelector('input[type="range"]');
    const coord = ctl.querySelector(".coord");
    const [lo, hi] = _bboxRangeForAxis(cutVolumeBox, axis);
    slider.min = String(lo.toFixed(1));
    slider.max = String(hi.toFixed(1));
    slider.step = "0.5";
    slider.value = String(((lo + hi) / 2).toFixed(1));
    slider.disabled = false;
    coord.textContent = slider.value;
    cb.checked = false;
    plane.visible = false;
    cb.addEventListener("change", () => {{ plane.visible = cb.checked; }});
    slider.addEventListener("input", () => {{
      const v = parseFloat(slider.value);
      _setCutPlaneValue(axis, v);
    }});
  }}
}}

function snapCutPlanesToRas(ras) {{
  // Move each plane onto the RAS coordinate; do not toggle visibility.
  if (!cutVolumeBox) return;
  _setCutPlaneValue("axial", ras[2]);
  _setCutPlaneValue("coronal", ras[1]);
  _setCutPlaneValue("sagittal", ras[0]);
}}

// ---------- sidebar -----------------------------------------------

let selectedContact = null;
let selectedShank = null;

function renderSidebar(meta) {{
  document.getElementById("subject").textContent = meta.subject_label || "(unnamed)";
  document.getElementById("summary").textContent =
    `${{meta.trajectories.length}} trajectories · ${{meta.contacts.length}} contacts`;

  const contactsByShank = new Map();
  for (const c of meta.contacts) {{
    if (!contactsByShank.has(c.trajectory)) contactsByShank.set(c.trajectory, []);
    contactsByShank.get(c.trajectory).push(c);
  }}
  for (const arr of contactsByShank.values()) {{
    arr.sort((a, b) => (a.contact_index|0) - (b.contact_index|0));
  }}

  const host = document.getElementById("shanks");
  host.innerHTML = "";
  for (const t of meta.trajectories) {{
    const card = document.createElement("div");
    card.className = "shank-card";
    card.dataset.shank = t.name;
    const band = (t.confidence_label || "").toLowerCase();
    const contacts = contactsByShank.get(t.name) || [];
    card.innerHTML = `
      <div class="shank-header">
        <div>
          <span class="shank-name">${{t.name}}</span>
          <span class="badge band-${{band}}">${{band || "?"}}</span>
        </div>
        <div class="shank-meta">${{t.electrode_model || ""}} · ${{contacts.length}}</div>
      </div>
      <div class="shank-contacts"></div>
    `;
    const list = card.querySelector(".shank-contacts");
    for (const c of contacts) {{
      const row = document.createElement("div");
      row.className = "contact-row";
      row.dataset.label = c.label;
      const region = c.freesurfer_label || c.thomas_label || c.wm_label || "—";
      const dist = c.distance_mm ? ` <span class="small">${{(+c.distance_mm).toFixed(1)}}mm</span>` : "";
      row.innerHTML = `<span class="label">${{c.label}}</span><span class="region">${{region}}${{dist}}</span>`;
      row.addEventListener("click", ev => {{ ev.stopPropagation(); selectContact(c.label, c.trajectory); }});
      list.appendChild(row);
    }}
    card.querySelector(".shank-header").addEventListener("click", () => {{
      card.classList.toggle("expanded");
      isolateShank(t.name, /*toggle=*/true);
    }});
    host.appendChild(card);
  }}
}}

function isolateShank(shank, toggle) {{
  // Hide every electrode node that doesn't belong to `shank`.
  // Brain surfaces stay visible (semi-transparent). Pass shank=null
  // (or toggle=true on an already-selected one) to restore all.
  if (toggle && selectedShank === shank) {{
    showAll();
    return;
  }}
  selectedShank = shank;
  for (const [s, nodes] of shankNodes) {{
    const visible = s === shank;
    for (const n of nodes) n.visible = visible;
  }}
  // Highlight active shank card
  document.querySelectorAll(".shank-card").forEach(c => c.classList.remove("active", "expanded"));
  const card = document.querySelector(`.shank-card[data-shank="${{shank}}"]`);
  if (card) card.classList.add("active", "expanded");
  // Frame the shank in the camera.
  const sNodes = shankNodes.get(shank) || [];
  if (sNodes.length) {{
    const box = new THREE.Box3();
    for (const n of sNodes) box.expandByObject(n);
    if (!box.isEmpty()) {{
      const center = box.getCenter(new THREE.Vector3());
      const diag = box.getSize(new THREE.Vector3()).length();
      controls.target.copy(center);
      const dir = camera.position.clone().sub(controls.target).normalize();
      camera.position.copy(center).addScaledVector(dir, Math.max(60, diag * 1.6));
      controls.update();
    }}
  }}
}}

function _setContactMaterial(node, material) {{
  // node is the outer Object3D; the visible geometry is on its child
  // mesh, which we cached during the GLB traversal.
  const mesh = node && node.userData && node.userData._mesh;
  if (mesh) mesh.material = material;
}}

function selectContact(label, shank) {{
  isolateShank(shank, /*toggle=*/false);

  // Restore previously selected contact's material
  if (selectedContact && originalMaterials.has(selectedContact)) {{
    _setContactMaterial(selectedContact, originalMaterials.get(selectedContact));
  }}
  const node = nodesByName.get("contact:" + label);
  setDbg("click",
    `${{label}} (shank=${{shank}}) — ${{node ? "node FOUND" : "node MISSING"}}`,
    node ? "ok" : "err");
  if (node) {{
    selectedContact = node;
    _setContactMaterial(node, RED);
    // Snap the controls target onto the contact center. Camera
    // distance is sized off the trajectory's own length (or fallback
    // to scene diag) so a typical shank fills the view comfortably,
    // not "inside the contact" the way a hard-coded 35mm would.
    const pos = node.getWorldPosition(new THREE.Vector3());
    controls.target.copy(pos);
    const dir = camera.position.clone().sub(pos).normalize();
    if (!isFinite(dir.x) || dir.lengthSq() < 1e-9) dir.set(0.7, 0.45, 0.8).normalize();
    let targetDist = 90;
    const sNodes = shankNodes.get(shank) || [];
    if (sNodes.length) {{
      const box = new THREE.Box3();
      for (const n of sNodes) box.expandByObject(n);
      if (!box.isEmpty()) targetDist = Math.max(70, box.getSize(new THREE.Vector3()).length() * 1.3);
    }}
    camera.position.copy(pos).addScaledVector(dir, targetDist);
    controls.update();
    // Snap the MRI slice viewer to the contact's RAS coordinate so the
    // axial/coronal/sagittal planes pass through it.
    snapSlicesToRas([pos.x, pos.y, pos.z]);
    snapCutPlanesToRas([pos.x, pos.y, pos.z]);
  }}
  // Toggle selection in sidebar UI
  document.querySelectorAll(".contact-row").forEach(r => r.classList.remove("selected"));
  const row = document.querySelector(`.contact-row[data-label="${{label}}"]`);
  if (row) {{
    row.classList.add("selected");
    row.scrollIntoView({{ block: "nearest" }});
  }}
}}

function showAll() {{
  selectedShank = null;
  for (const [, nodes] of shankNodes) for (const n of nodes) n.visible = true;
  if (selectedContact && originalMaterials.has(selectedContact)) {{
    _setContactMaterial(selectedContact, originalMaterials.get(selectedContact));
    selectedContact = null;
  }}
  document.querySelectorAll(".shank-card").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".contact-row").forEach(r => r.classList.remove("selected"));
}}

document.getElementById("btn-reset").addEventListener("click", showAll);
document.getElementById("btn-fit").addEventListener("click", () => {{ if (gltfRoot) fitToObject(gltfRoot); }});

// Surface color mode dropdown. The GLB already has per-vertex RGBA
// colors painted from the FS aparc.annot; the surface material's
// baseColor multiplies with them. Setting baseColor to white lets
// the parcellation come through at full saturation; the "skin" tan
// gives the soft cortex look the v2 viewer shipped with.
const SURFACE_COLORS = {{
  skin: [0.92, 0.88, 0.84],
  parcellation: [1.0, 1.0, 1.0],
  white: [0.95, 0.95, 0.95],
}};
function _applySurfaceColor(modeKey) {{
  const rgb = SURFACE_COLORS[modeKey] || SURFACE_COLORS.parcellation;
  for (const node of surfaceNodes) {{
    const mesh = node.userData && node.userData._mesh;
    if (!mesh || !mesh.material || !mesh.material.color) continue;
    mesh.material.color.setRGB(rgb[0], rgb[1], rgb[2]);
    // Vertex colors are already enabled on the material from glTF
    // (COLOR_0 attribute); nothing else to flip.
    mesh.material.needsUpdate = true;
  }}
}}
(function _wireSurfaceColor() {{
  const ctl = document.querySelector('.plane-ctl[data-control="brain-color"]');
  if (!ctl) return;
  const sel = ctl.querySelector("select");
  sel.addEventListener("change", () => _applySurfaceColor(sel.value));
  // Apply once on first GLB completion (surfaces may not be indexed
  // yet at module init time). The MutationObserver-free version: defer
  // a beat after gltfRoot loads — we trigger from inside the GLB
  // callback below.
  ctl.dataset.initialApplied = "0";
}})();
// Apply the initial selection right after surfaces are indexed.
function _applyInitialSurfaceColor() {{
  const ctl = document.querySelector('.plane-ctl[data-control="brain-color"]');
  if (!ctl) return;
  const sel = ctl.querySelector("select");
  _applySurfaceColor(sel.value);
}}

// Brain α slider — opacity for every FS surface mesh. Surfaces are
// indexed during GLB load, so this works regardless of how many
// hemispheres / surface kinds got loaded.
function _applyBrainAlpha(v) {{
  for (const node of surfaceNodes) {{
    const mesh = node.userData && node.userData._mesh;
    if (!mesh || !mesh.material) continue;
    mesh.material.opacity = v;
    mesh.material.transparent = v < 0.999;
    // Keep depthWrite TRUE even at α<1. Three.js otherwise draws the
    // brain's triangles in index order, so deep cortical folds bleed
    // through the front of the brain at intermediate alpha and the
    // surface looks fragmented. With depthWrite enabled, the closest
    // brain fragment at each pixel wins; the alpha-blend happens
    // against opaque electrodes that were already drawn underneath.
    // Trade-off: the back of the brain is occluded by the front (which
    // is what the user wants — they're looking from outside).
    mesh.material.depthWrite = true;
    mesh.material.needsUpdate = true;
  }}
}}
function _applyInitialBrainAlpha() {{
  const ctl = document.querySelector('.plane-ctl[data-control="brain-alpha"]');
  if (!ctl) return;
  const slider = ctl.querySelector('input[type="range"]');
  const coord = ctl.querySelector(".coord");
  if (coord) coord.textContent = parseFloat(slider.value).toFixed(2);
  _applyBrainAlpha(parseFloat(slider.value));
}}
(function _wireBrainAlpha() {{
  const ctl = document.querySelector('.plane-ctl[data-control="brain-alpha"]');
  if (!ctl) return;
  const slider = ctl.querySelector('input[type="range"]');
  const coord = ctl.querySelector(".coord");
  slider.addEventListener("input", () => {{
    const v = parseFloat(slider.value);
    coord.textContent = v.toFixed(2);
    _applyBrainAlpha(v);
  }});
}})();

// Slice plane α slider — sets the uOpacity uniform on the three cut
// planes. Sliders that fire before initCutPlanes runs are harmless.
(function _wirePlaneAlpha() {{
  const ctl = document.querySelector('.plane-ctl[data-control="plane-alpha"]');
  if (!ctl) return;
  const slider = ctl.querySelector('input[type="range"]');
  const coord = ctl.querySelector(".coord");
  slider.addEventListener("input", () => {{
    const v = parseFloat(slider.value);
    coord.textContent = v.toFixed(2);
    for (const axis of Object.keys(cutPlanes)) {{
      const plane = cutPlanes[axis];
      if (plane && plane.material && plane.material.uniforms && plane.material.uniforms.uOpacity) {{
        plane.material.uniforms.uOpacity.value = v;
      }}
    }}
  }});
}})();

_initSlicePanels();
fetch("scene_meta.json").then(r => r.json()).then(meta => {{
  renderSidebar(meta);
  // Kick off MRI load in the background; doesn't block sidebar / 3D.
  loadMri(meta.t1_volume);
}}).catch(err => {{
  console.error(err);
  document.getElementById("subject").textContent = "Failed to load metadata: " + err.message;
}});
</script>
</body>
</html>
"""


def _write_html(out_dir: Path, *, title: str) -> Path:
    html = _HTML_TEMPLATE.format(title=title)
    p = out_dir / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------------
# Command entry
# ---------------------------------------------------------------------


def run_export_view(
    target: str,
    *,
    freesurfer_dir: str,
    out_dir: str | Path,
    ct_override: str | None = None,
    ref_volume: str | None = None,
    parcellation_path: str | None = None,
    lut_path: str | None = None,
    thomas_dir: str | None = None,
    seeds_path: str | None = None,
    surface_kinds: tuple[str, ...] = ("pial",),
    annotation: str = "aparc",
    shaft_radius_mm: float = 0.35,
    contact_band_radius_mm: float = 0.55,
    contact_band_length_mm: float = 2.0,
    skip_registration: bool = False,
    output_frame: str = "ct",
    write_figures: bool = True,
    library: str | None = None,
) -> dict[str, Any]:
    """Run the pipeline + assemble the GLB viewer."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    fs = resolve_fs_subject(freesurfer_dir)
    _stderr(
        f"[view] resolved FreeSurfer subject {fs.subject_dir} "
        f"(surfaces={fs.available_surfaces}, annot={fs.available_annotations})"
    )

    parcellation = _discover_parcellation(fs, parcellation_path)
    if parcellation is None:
        _stderr("[view] no FreeSurfer parcellation found; labels.tsv will be unavailable")
    lut = _discover_lut(fs, lut_path)
    if lut is None:
        _stderr("[view] no FreeSurferColorLUT found; vertex/contact coloring will be flat gray")

    # 1. Run pipeline -> trajectories.tsv, contacts.tsv, optionally labels.tsv.
    from .pipeline import run_pipeline

    pipeline_summary = run_pipeline(
        target=target,
        out_dir=out,
        ct_override=ct_override,
        seeds_path=seeds_path,
        ref_volume=ref_volume,
        output_frame=output_frame,
        skip_registration=skip_registration,
        thomas_dir=thomas_dir,
        freesurfer_path=str(parcellation) if parcellation else None,
        freesurfer_lut=str(lut) if lut else None,
        atlas_base_path=str(fs.t1_path),
        write_figures=write_figures,
        library=library,
    )

    ct_path = Path(pipeline_summary["ct_path"])
    traj_tsv = out / "trajectories.tsv"
    contacts_tsv = out / "contacts.tsv"
    labels_tsv = out / "labels.tsv"

    trajectories = _read_pipeline_trajectories(traj_tsv)
    contacts = _read_pipeline_contacts(contacts_tsv)
    contact_labels = _read_labels_by_contact(labels_tsv if labels_tsv.exists() else None)
    lut_index = parse_freesurfer_lut(lut) if lut else {}

    # 2. Register FS T1 -> CT. Run only when we have surfaces to push.
    # The same registration is reused twice: to push surface vertices
    # into CT RAS, and to resample the T1 onto the CT grid so the
    # in-browser slice viewer can render axial/coronal/sagittal MRI
    # planes that line up with the contacts.
    surfaces = []
    t1_slice_meta: dict[str, Any] | None = None
    if fs.available_surfaces:
        fs_to_ct_4x4, fixed_ct_img, moving_t1_img, sitk_transform = _register_fs_to_ct(
            fs.t1_path, ct_path,
        )
        # 3. Load surfaces (in FS scanner-RAS) and push through FS->CT.
        surfaces = load_fs_surfaces(
            fs,
            surface_kinds=surface_kinds,
            annotation=annotation if annotation else None,
            transform_4x4=fs_to_ct_4x4,
        )
        for s in surfaces:
            _stderr(
                f"[view] surface {s.name}: {s.vertices_ras.shape[0]} verts / "
                f"{s.faces.shape[0]} faces"
                + (f" + {s.annotation_name}" if s.annotation_name else "")
            )
        # Resample T1 -> CT grid for the in-browser MRI slice viewer.
        t1_out = out / "t1_in_ct.nii.gz"
        t1_slice_meta = _write_t1_resampled_to_ct(
            fixed_ct_img, moving_t1_img, sitk_transform, t1_out,
        )
        _stderr(
            f"[view] wrote {t1_out} (T1 resampled onto CT grid; "
            f"{t1_slice_meta['size']} {t1_slice_meta['dtype']})"
        )
    else:
        _stderr("[view] no FreeSurfer surfaces under surf/; skipping brain mesh")

    # 4. Build the scene.
    scene, contact_meta = _build_scene(
        surfaces=surfaces,
        trajectories=trajectories,
        contacts=contacts,
        contact_labels=contact_labels,
        lut_index=lut_index,
        shaft_radius_mm=shaft_radius_mm,
        contact_band_radius_mm=contact_band_radius_mm,
        contact_band_length_mm=contact_band_length_mm,
    )
    glb_path = out / "scene.glb"
    glb_bytes = write_glb(glb_path, scene)
    _stderr(f"[view] wrote {glb_path} ({glb_bytes / (1024*1024):.1f} MB, {len(scene.nodes)} nodes)")

    # 5. Sidecar metadata for the HTML page.
    meta = {
        "subject_label": Path(target).name,
        "ct_path": str(ct_path),
        "freesurfer_subject": str(fs.subject_dir),
        "parcellation": str(parcellation) if parcellation else "",
        "lut": str(lut) if lut else "",
        "annotation": annotation,
        "trajectories": trajectories,
        "contacts": contact_meta,
        "t1_volume": t1_slice_meta,
    }
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    html_path = _write_html(out, title=f"rosa-agent export-view — {Path(target).name}")
    _stderr(f"[view] wrote {html_path}")

    view_manifest = {
        "pipeline": pipeline_summary,
        "freesurfer_subject": str(fs.subject_dir),
        "parcellation": str(parcellation) if parcellation else "",
        "lut": str(lut) if lut else "",
        "annotation": annotation,
        "surfaces_loaded": [s.name for s in surfaces],
        "scene_glb": str(glb_path),
        "viewer_html": str(html_path),
    }
    (out / "view_manifest.json").write_text(json.dumps(view_manifest, indent=2), encoding="utf-8")
    return view_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent export-view",
        description="Run the SEEG pipeline against a ROSA case and bundle the "
                    "trajectories + contacts + FreeSurfer brain into a "
                    "browser-loadable GLB scene.",
    )
    parser.add_argument(
        "target",
        help="ROSA case folder, OR dataset subject id (e.g. T22). Same input as 'pipeline'.",
    )
    parser.add_argument(
        "--freesurfer-dir", required=True,
        help="FreeSurfer recon-all subject directory (the one with surf/, mri/, label/).",
    )
    parser.add_argument("--out-dir", "-o", required=True)
    parser.add_argument("--ct", default="", help="External CT (overrides ROSA-folder reference + dataset lookup)")
    parser.add_argument("--ref-volume", default="", help="ROSA display name to use as the reference frame")
    parser.add_argument("--seeds", default="", help="Optional explicit seed TSV (overrides ROSA-derived seeds)")
    parser.add_argument("--thomas", default="", help="THOMAS segmentation directory (optional labeling source)")
    parser.add_argument("--parcellation", default="", help="Override the FS parcellation labelmap path")
    parser.add_argument("--lut", default="", help="Override the FreeSurferColorLUT path")
    parser.add_argument(
        "--surfaces", default="pial",
        help="Comma-separated FS surface kinds to load (default: pial). "
             "Accepted: pial, white, inflated, smoothwm.",
    )
    parser.add_argument(
        "--annotation", default="aparc",
        help="FreeSurfer annotation to paint onto surfaces (default: aparc). "
             "Pass empty string to disable vertex coloring.",
    )
    parser.add_argument(
        "--shaft-radius-mm", type=float, default=0.35,
        help="Electrode shaft (insulation) radius in mm — default 0.35.",
    )
    parser.add_argument(
        "--contact-band-radius-mm", type=float, default=0.55,
        help="Metallic contact band radius in mm — slightly larger than the "
             "shaft so the contacts look raised. Default 0.55.",
    )
    parser.add_argument(
        "--contact-band-length-mm", type=float, default=2.0,
        help="Metallic contact band length along the trajectory axis (mm). Default 2.0.",
    )
    parser.add_argument(
        "--skip-registration", action="store_true",
        help="When --ct is supplied alongside a ROSA folder, assume the external CT "
             "is already aligned to the ROSA reference (no rigid CT<->ROSA pass).",
    )
    parser.add_argument(
        "--output-frame", default="ct", choices=("ct", "rosa"),
        help="Frame for trajectory/contact coordinates in the output TSVs and GLB.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip the per-trajectory QC PNG render step (default: figures "
             "are written to <out-dir>/figures/ when matplotlib is available).",
    )
    parser.add_argument(
        "--library", default="",
        help="Electrode-library strategy key — restrict matching to a "
             "vendor subset (e.g. 'dixi', 'pmt_35', 'adtech'). "
             "Default: full bundled library.",
    )
    args = parser.parse_args(argv)

    surface_kinds = tuple(
        s.strip().lower() for s in args.surfaces.split(",") if s.strip()
    )
    if not surface_kinds:
        surface_kinds = ("pial",)

    summary = run_export_view(
        target=args.target,
        freesurfer_dir=args.freesurfer_dir,
        out_dir=args.out_dir,
        ct_override=args.ct or None,
        ref_volume=args.ref_volume or None,
        parcellation_path=args.parcellation or None,
        lut_path=args.lut or None,
        thomas_dir=args.thomas or None,
        seeds_path=args.seeds or None,
        surface_kinds=surface_kinds,
        annotation=args.annotation,
        shaft_radius_mm=float(args.shaft_radius_mm),
        contact_band_radius_mm=float(args.contact_band_radius_mm),
        contact_band_length_mm=float(args.contact_band_length_mm),
        skip_registration=bool(args.skip_registration),
        output_frame=args.output_frame,
        write_figures=not bool(args.no_figures),
        library=args.library or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
