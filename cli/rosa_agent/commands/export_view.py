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
    → the LUT packaged inside the ``rosa_core`` package
    (``rosa_core/resources/freesurfer/``, shipped in the wheel).
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

    # Bundled fallback — the LUT ships inside the rosa_core package, so this
    # resolves in a real wheel / site-packages install, not only an editable
    # checkout (the old Path(__file__).parents[3] guess broke in a wheel).
    from rosa_core.atlas_index import default_freesurfer_lut_path
    bundled = default_freesurfer_lut_path()
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


def _slice_volume_meta(img, out_path: Path, *, vid: str, label: str) -> dict[str, Any]:
    """Write a uint8 SITK image as NIfTI and return the slice/MIP-volume meta
    (id + label + path + vox->RAS affine) the in-browser viewer consumes.

    Shared by the FreeSurfer T1 (resampled onto the CT grid) and the windowed
    working CT, so both feed the same generic slice/MIP renderer.
    """
    import SimpleITK as sitk

    sitk.WriteImage(img, str(out_path))
    size = list(img.GetSize())
    spacing = list(img.GetSpacing())
    origin_lps = list(img.GetOrigin())
    direction_lps = list(img.GetDirection())
    # SITK stores origin/direction in LPS; the JS viewer needs vox->RAS.
    origin_ras = [-origin_lps[0], -origin_lps[1], origin_lps[2]]
    d = np.asarray(direction_lps, dtype=float).reshape(3, 3)
    d_ras = d.copy()
    d_ras[0, :] *= -1.0
    d_ras[1, :] *= -1.0
    vox_to_ras = np.eye(4)
    for k in range(3):
        vox_to_ras[:3, k] = d_ras[:, k] * spacing[k]
    vox_to_ras[:3, 3] = origin_ras
    return {
        "id": vid,
        "label": label,
        "path": out_path.name,
        "size": [int(s) for s in size],
        "spacing": [float(s) for s in spacing],
        "vox_to_ras": [[float(v) for v in row] for row in vox_to_ras.tolist()],
        "dtype": str(sitk.GetArrayViewFromImage(img).dtype),
    }


def _brain_mri_dvr_volume(native_mri_path, native_mask_path, sitk_transform, ct_path):
    """Brain-only MRI resampled into the CT frame, windowed to uint8.

    The 3D texture the in-browser volume renderer (DVR) raycasts. Masking to the
    brain + windowing on the brain's own intensity means a shader isosurface
    lands on the pial surface (skull/scalp gone) — a Curry/Medtronic-style
    volume-rendered brain, no mesh. Returns a SITK uint8 image on the CT grid.
    """
    import SimpleITK as sitk
    from rosa_core.registration import resample_volume

    mri = sitk.Cast(sitk.ReadImage(str(native_mri_path)), sitk.sitkFloat32)
    mask = sitk.Cast(sitk.ReadImage(str(native_mask_path)) > 0, sitk.sitkFloat32)
    masked = mri * mask                                   # same native grid
    inct = resample_volume(masked, sitk_transform, reference=sitk.ReadImage(str(ct_path)),
                           interp="linear")
    arr = sitk.GetArrayViewFromImage(inct)
    brain = arr[arr > 0]
    hi = float(np.percentile(brain, 99.0)) if brain.size else 1.0
    scaled = sitk.Clamp(inct * (255.0 / max(hi, 1e-3)), lowerBound=0.0, upperBound=255.0)
    return sitk.Cast(scaled, sitk.sitkUInt8)


def _write_ct_slice_volume(
    ct_path: Path, out_path: Path, window: tuple[float, float] = (-150.0, 1500.0),
) -> dict[str, Any]:
    """Window the working CT to uint8 and write it as a browser slice/MIP volume.

    The CT already lives in the contact/trajectory frame, so there's no
    resampling — just a HU window (default: brain reads mid-gray, the metal
    contacts saturate bright). This is what makes a FreeSurfer-free, CT-only
    view work, and it doubles as the MIP source (metal = brightest voxels).
    """
    import SimpleITK as sitk

    img = sitk.Cast(sitk.ReadImage(str(ct_path)), sitk.sitkFloat32)
    lo, hi = float(window[0]), float(window[1])
    if hi <= lo:
        hi = lo + 1.0
    scaled = sitk.Clamp(
        (img - lo) * (255.0 / (hi - lo)),
        lowerBound=0.0, upperBound=255.0,
    )
    return _slice_volume_meta(
        sitk.Cast(scaled, sitk.sitkUInt8), out_path, vid="ct", label="CT",
    )


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
    return _slice_volume_meta(resampled, out_path, vid="t1", label="FreeSurfer T1")


# ---------------------------------------------------------------------
# Reading pipeline output
# ---------------------------------------------------------------------


def _read_pipeline_trajectories(path: Path) -> list[dict[str, Any]]:
    """Read a trajectories TSV into ``{name, start, end, electrode_model, ...}``.

    Schema-tolerant: accepts the documented pipeline/place contract
    (``start_x/y/z`` + ``end_x/y/z`` + ``electrode_model``) AND fit-rosa's
    diagnostic variant (``entry_x/y/z`` + ``tip_x/y/z`` + ``predicted_model``),
    so any of the toolkit's trajectory outputs renders without per-tool hacks.
    """
    rows = read_tsv_rows(path)
    out = []
    for r in rows:
        sx, ex = r.get("start_x", r.get("entry_x")), r.get("end_x", r.get("tip_x"))
        if sx in (None, "") or ex in (None, ""):
            continue
        try:
            start = (float(sx),
                     float(r.get("start_y", r.get("entry_y"))),
                     float(r.get("start_z", r.get("entry_z"))))
            end = (float(ex),
                   float(r.get("end_y", r.get("tip_y"))),
                   float(r.get("end_z", r.get("tip_z"))))
        except (TypeError, ValueError):
            continue
        out.append({
            "name": r.get("name", ""),
            "start": start,
            "end": end,
            "confidence": r.get("confidence", ""),
            "confidence_label": r.get("confidence_label", ""),
            "electrode_model": r.get("electrode_model", r.get("predicted_model", "")),
            "bolt_source": r.get("bolt_source", ""),
            "length_mm": r.get("length_mm", ""),
        })
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
  #app {{ display: grid; grid-template-columns: minmax(0, 1fr) clamp(260px, 22vw, 440px) clamp(240px, 18vw, 360px); height: 100%; }}
  #canvas-host {{ position: relative; }}
  #canvas-host canvas {{ display: block; }}
  #toolbar {{ position: absolute; top: 10px; left: 10px; right: 10px; z-index: 5; display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }}
  /* Collapsed by default so the controls don't cover the 3D view — the toggle
     plus Show all / Fit view stay; the grouped controls hide until expanded. */
  #toolbar.collapsed .tb-group, #toolbar.collapsed .tb-sep {{ display: none; }}
  .tb-group {{ display: inline-flex; align-items: center; gap: 6px 10px; flex-wrap: wrap; }}
  .tb-sep {{ width: 1px; align-self: stretch; min-height: 20px; background: #3a3a3a; margin: 0 2px; }}
  #btn-tools {{ font-weight: 600; }}
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
  #slices-head {{ display: flex; align-items: center; gap: 8px; font-size: 10px; color: #888; padding: 0 2px 6px; border-bottom: 1px solid #1d1d1d; flex-wrap: wrap; }}
  #slices-head .ttl {{ font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #aaa; }}
  #slices-head .hint {{ color: #666; flex: 1; min-width: 90px; }}
  #slices-head .palpha {{ display: inline-flex; align-items: center; gap: 5px; color: #bbb; }}
  #slices-head .palpha input[type="range"] {{ width: 70px; }}
  /* Equal square panels: the canvas letterboxes (object-fit) inside a fixed
     1:1 box, so axial (512²) and coronal/sagittal (512×192) read the same size
     and the column resizes with the window instead of each panel picking its
     own aspect-driven height. */
  .slice-panel {{ position: relative; background: #000; border: 1px solid #1d1d1d; aspect-ratio: 1 / 1; overflow: hidden; }}
  .slice-panel canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; image-rendering: crisp-edges; }}
  .slice-3d {{ position: absolute; top: 4px; right: 6px; font-size: 10px; letter-spacing: 0.06em; color: #bfe9ff; display: inline-flex; align-items: center; gap: 3px; cursor: pointer; background: rgba(0,0,0,0.5); padding: 1px 5px 1px 3px; border-radius: 3px; user-select: none; z-index: 2; }}
  .slice-3d input {{ margin: 0; accent-color: #22d0ff; }}
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
  /* Picker-mode drop zone (GitHub Pages app; hidden in served mode). */
  #dropzone {{ display: none; position: absolute; inset: 0; z-index: 50; background: rgba(8,8,8,0.92); align-items: center; justify-content: center; }}
  #dropzone.hover .dz-inner {{ border-color: #4a90d9; background: #15202b; }}
  .dz-inner {{ border: 2px dashed #555; border-radius: 10px; padding: 36px 48px; max-width: 540px; text-align: center; color: #ccc; }}
  .dz-title {{ font-size: 18px; margin-bottom: 10px; color: #fff; }}
  .dz-sub {{ font-size: 13px; color: #9aa; line-height: 1.55; margin-bottom: 18px; }}
  .dz-inner code {{ color: #cdd; background: #222; padding: 1px 4px; border-radius: 3px; }}
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
  <div id="dropzone">
    <div class="dz-inner">
      <div class="dz-title">Drop a rosa-agent viewer export</div>
      <div class="dz-sub">Select or drop <code>scene.glb</code> + <code>scene_meta.json</code> + the
        <code>.nii.gz</code> volume(s) from an <code>export-view</code> output dir.
        Nothing is uploaded — everything renders locally in your browser.</div>
      <input type="file" id="file-input" multiple accept=".glb,.json,.nii,.gz" />
    </div>
  </div>
  <div id="canvas-host">
    <div id="toolbar" class="collapsed">
      <button id="btn-tools" title="Show/hide controls">Controls ▾</button>
      <button id="btn-reset" title="Un-hide every electrode">Show all</button>
      <button id="btn-fit" title="Frame the whole scene">Fit view</button>
      <span class="tb-group" title="Anatomy shown in the 3D surface + slice navigators">
        <label class="plane-ctl" data-control="brain-color">
          <span class="axis">Brain surface</span>
          <select>
            <option value="parcellation" selected>Parcellation</option>
            <option value="skin">Plain</option>
            <option value="white">White</option>
          </select>
        </label>
        <label class="plane-ctl" data-control="brain-alpha">
          <span class="axis">Brain opacity</span>
          <input type="range" min="0" max="1" step="0.05" value="0.45" />
          <span class="coord">0.45</span>
        </label>
      </span>
      <span class="tb-sep"></span>
      <span class="tb-group" title="Slice-navigator volume (the axial/coronal/sagittal panels + any planes shown in 3D)">
        <label class="plane-ctl" data-control="volume">
          <span class="axis">Slice volume</span>
          <select id="vol-select"></select>
        </label>
      </span>
      <span class="tb-sep"></span>
      <span class="tb-group">
        <label class="plane-ctl" data-control="mip" title="Metal projection — a rotatable maximum-intensity projection of the CT; metal contacts read as bright streaks. Always from the CT.">
          <input type="checkbox" /><span class="axis">Metal (CT)</span>
          <input type="range" data-mip="threshold" min="0" max="1" step="0.02" value="0.32" title="Threshold — raise to drop soft tissue and isolate bone + metal" />
          <input type="range" data-mip="opacity" min="0" max="1" step="0.05" value="0.6" title="Opacity" />
        </label>
        <label class="plane-ctl" data-control="dvr" title="Volume render — a gradient-shaded isosurface of the brain MRI (Curry/Medtronic look, no mesh). Always from the brain MRI.">
          <input type="checkbox" /><span class="axis">Volume render</span>
          <input type="range" data-dvr="threshold" min="0.02" max="0.9" step="0.01" value="0.28" title="Isosurface level — raise to peel outer tissue" />
        </label>
      </span>
    </div>
    <div id="status-bar">
      <span><span class="lbl">Surface:</span><span id="info-surface">—</span></span>
      <span><span class="lbl">Slices:</span><span id="info-slices">—</span></span>
      <span><span class="lbl">MRI:</span><span id="info-mri" class="warn">—</span></span>
    </div>
  </div>
  <div id="slices">
    <div id="slices-head">
      <span class="ttl">Slices</span>
      <span class="hint">scroll = scrub · click = locate · 3D = show plane</span>
      <label class="palpha" title="Opacity of the cut planes shown in 3D">&alpha;<input type="range" id="plane-alpha" min="0" max="1" step="0.05" value="0.9"></label>
    </div>
    <div class="slice-panel" data-axis="axial">
      <canvas></canvas>
      <div class="slice-label">Axial</div>
      <label class="slice-3d" title="Show this plane inside the 3D scene"><input type="checkbox" data-plane="axial">3D</label>
      <div class="slice-axes">R&rarr;  A&uarr;</div>
      <div class="slice-coord"></div>
    </div>
    <div class="slice-panel" data-axis="coronal">
      <canvas></canvas>
      <div class="slice-label">Coronal</div>
      <label class="slice-3d" title="Show this plane inside the 3D scene"><input type="checkbox" data-plane="coronal">3D</label>
      <div class="slice-axes">R&rarr;  S&uarr;</div>
      <div class="slice-coord"></div>
    </div>
    <div class="slice-panel" data-axis="sagittal">
      <canvas></canvas>
      <div class="slice-label">Sagittal</div>
      <label class="slice-3d" title="Show this plane inside the 3D scene"><input type="checkbox" data-plane="sagittal">3D</label>
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

// VIEWER_MODE: "served" (auto-fetch scene.glb/scene_meta.json/volumes from this
// dir — export-view + `rosa-agent view`) or "picker" (the GitHub Pages app:
// no auto-load, the user drag-and-drops their own files and everything renders
// client-side — nothing uploaded). The render engine below is identical for
// both; only how the initial bytes arrive differs.
const VIEWER_MODE = "{viewer_mode}";

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
// Ambient kept low so sulci actually cast shadow (high ambient washes the
// folds flat — the "mushy from some angles" look). The key + fill lights are
// repositioned every frame RELATIVE to the camera (see updateLights) so the
// raking that makes gyri read crisply follows the view instead of only working
// from one fixed side.
scene.add(new THREE.AmbientLight(0xffffff, 0.32));
const key = new THREE.DirectionalLight(0xffffff, 0.95);
const fill = new THREE.DirectionalLight(0xffffff, 0.35);
scene.add(key); scene.add(fill);
scene.add(key.target); scene.add(fill.target);
const _kOff = new THREE.Vector3(), _fOff = new THREE.Vector3();
const _camDir = new THREE.Vector3(), _right = new THREE.Vector3();
// DVR mesh + its camera-following light dir — declared here (before the render
// loop / updateLights) to avoid a temporal-dead-zone crash; assigned in
// buildDvr / used by _makeDvrMesh.
let dvrMesh = null;
const _dvrLightDir = new THREE.Vector3(0.4, 0.85, 0.55).normalize();
function updateLights() {{
  // Rake the key light across the surface facing the camera: offset it up +
  // left + toward the viewer from the orbit centre, so folds shadow from any
  // orientation. Fill comes from the opposite lower-right to lift the darks.
  camera.getWorldDirection(_camDir);                       // view direction
  _right.crossVectors(_camDir, camera.up).normalize();
  _kOff.copy(_camDir).multiplyScalar(-1)
       .addScaledVector(camera.up, 0.85).addScaledVector(_right, -0.55).normalize();
  key.position.copy(controls.target).addScaledVector(_kOff, 400);
  key.target.position.copy(controls.target); key.target.updateMatrixWorld();
  _fOff.copy(_camDir).multiplyScalar(-1)
       .addScaledVector(camera.up, -0.6).addScaledVector(_right, 0.6).normalize();
  fill.position.copy(controls.target).addScaledVector(_fOff, 400);
  fill.target.position.copy(controls.target); fill.target.updateMatrixWorld();
  // Rake the DVR isosurface the same way (direction from surface toward the key).
  if (dvrMesh && dvrMesh.visible) dvrMesh.material.uniforms.uLightDir.value.copy(_kOff);
}}

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
  updateLights();          // key/fill track the camera so folds rake from any view
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

// A bright marker sphere that jumps to the selected contact so it's obvious
// which one is highlighted — a lone recoloured band is easy to lose among many
// contacts or inside an opaque brain. depthTest off + high renderOrder draws it
// on top of everything (visible even when the contact is buried).
const selectionMarker = new THREE.Mesh(
  new THREE.SphereGeometry(2.2, 24, 18),
  new THREE.MeshBasicMaterial({{
    color: 0xff2233, transparent: true, opacity: 0.55,
    depthTest: false, depthWrite: false,
  }})
);
selectionMarker.renderOrder = 999;
selectionMarker.visible = false;
scene.add(selectionMarker);

// A second, cyan marker that tracks the SLICE navigator crosshair — i.e. a
// free "locate" point the user scrubbed to in the 2D panels (distinct from the
// red contact-selection beacon above). Lets you see, in 3D, where the slices
// are cutting even when no contact is selected.
const locateMarker = new THREE.Mesh(
  new THREE.SphereGeometry(1.6, 20, 14),
  new THREE.MeshBasicMaterial({{
    color: 0x22d0ff, transparent: true, opacity: 0.6,
    depthTest: false, depthWrite: false,
  }})
);
locateMarker.renderOrder = 999;
locateMarker.visible = false;
scene.add(locateMarker);

function _firstMeshChild(obj) {{
  let mesh = null;
  obj.traverse(c => {{ if (!mesh && c.isMesh) mesh = c; }});
  return mesh;
}}

const loader = new GLTFLoader();
// The GLB load handler is named so picker mode can re-invoke it on a dropped
// file (see the VIEWER_MODE bootstrap at the bottom). Served mode auto-loads.
const onGltf = gltf => {{
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
  // Re-apply any visibility a parent frame requested before the GLB finished
  // loading (embedded rosa-app: rejected contacts start hidden).
  if (_pendingVisibility) applyContactVisibility(_pendingVisibility.hideShanks, _pendingVisibility.hideContacts);
}};
const onGltfErr = err => {{
  console.error("GLB load failed", err);
  setDbg("glb", "load failed: " + (err.message || err), "err");
  document.getElementById("subject").textContent = "Failed to load scene.glb: " + err;
}};
function loadGlb(url) {{ setDbg("glb", "loading…"); loader.load(url, onGltf, undefined, onGltfErr); }}
if (VIEWER_MODE !== "picker") loadGlb("scene.glb");

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
// Contacts, in the shared CT/contact RAS frame, overlaid on every slice so the
// electrodes stay visible no matter which volume (CT or MRI) is displayed —
// the metal only shows on CT, but the markers show everywhere.
let allContacts = [];      // [{{ label, shank, ras:[x,y,z] }}]
let sceneMeta = null;      // the loaded scene_meta.json (surface source, atlas…)
let _sliceVolLabel = "—";  // label of the current slice-navigator volume
let _mriAvailable = false; // whether a brain/MRI volume was exported

// The clinician-facing info line (replaces the old dev status strip): what the
// 3D surface is derived from, which volume the slices show, and MRI status.
function updateInfoBar() {{
  const set = (id, txt) => {{ const el = document.getElementById(id); if (el) el.textContent = txt; }};
  const src = (sceneMeta && sceneMeta.surface_source) || "—";
  set("info-surface", src);
  set("info-slices", _sliceVolLabel);
  const mri = document.getElementById("info-mri");
  if (mri) {{ mri.textContent = _mriAvailable ? "loaded" : "not loaded";
              mri.className = _mriAvailable ? "ok" : "warn"; }}
}}

// axis -> which voxel index is the through-plane (the one scroll steps).
const _THROUGH_IDX = {{ axial: 2, coronal: 1, sagittal: 0 }};

// In-scene cut planes (the textured slices that float in the 3D view). They're
// kept — a slice you can see relative to the electrodes is the one thing the 2D
// navigators can't show — but driven ENTIRELY from the side: each navigator
// panel has a "3D" toggle, a shared α sets their opacity, and scrubbing/clicking
// a navigator moves the matching plane. No slice controls overlay the canvas.
const cutPlanes = {{}};     // axis -> THREE.Mesh
const planeWanted = {{ axial: false, coronal: false, sagittal: false }};
let planeAlpha = 0.9;
function setPlaneVisible(axis, on) {{
  planeWanted[axis] = on;
  if (cutPlanes[axis]) cutPlanes[axis].visible = on;
}}

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

    // Click / drag → move the crosshair to that in-plane point.
    let dragging = false;
    const pick = (ev) => _sliceClickToRas(axis, canvas, ev);
    canvas.addEventListener("pointerdown", (ev) => {{ dragging = true; canvas.setPointerCapture(ev.pointerId); pick(ev); }});
    canvas.addEventListener("pointermove", (ev) => {{ if (dragging) pick(ev); }});
    canvas.addEventListener("pointerup", (ev) => {{ dragging = false; try {{ canvas.releasePointerCapture(ev.pointerId); }} catch (_e) {{}} }});
    // Wheel → scrub the through-plane slice, like a real navigator.
    canvas.addEventListener("wheel", (ev) => {{ ev.preventDefault(); _sliceScroll(axis, ev.deltaY > 0 ? 1 : -1); }}, {{ passive: false }});
    canvas.style.cursor = "crosshair";
    // Per-panel "3D" toggle → show/hide THIS axis's cut plane in the scene.
    const plane3d = panel.querySelector('input[data-plane]');
    if (plane3d) plane3d.addEventListener("change", () => setPlaneVisible(axis, plane3d.checked));
  }}
  // Shared opacity for all three in-scene cut planes.
  const pa = document.getElementById("plane-alpha");
  if (pa) pa.addEventListener("input", () => {{
    planeAlpha = parseFloat(pa.value);
    for (const ax of Object.keys(cutPlanes)) {{
      const u = cutPlanes[ax].material && cutPlanes[ax].material.uniforms;
      if (u && u.uOpacity) u.uOpacity.value = planeAlpha;
    }}
  }});
}}

// Map a pointer event on a slice canvas to a RAS point and locate there.
function _sliceClickToRas(axis, canvas, ev) {{
  if (!mriVolume) return;
  const rect = canvas.getBoundingClientRect();
  const W = canvas.width, H = canvas.height;
  const u = Math.max(0, Math.min(W - 1, Math.round((ev.clientX - rect.left) / rect.width * W)));
  const v = Math.max(0, Math.min(H - 1, Math.round((ev.clientY - rect.top) / rect.height * H)));
  const vSrc = H - 1 - v;   // renderSlice flips vertical so "up" = anterior/superior
  const c = mriVoxCursor.slice();
  if (axis === "axial") {{ c[0] = u; c[1] = vSrc; }}
  else if (axis === "coronal") {{ c[0] = u; c[2] = vSrc; }}
  else {{ c[1] = u; c[2] = vSrc; }}   // sagittal
  locateAtRas(_apply4x4(mriVolume.affine, c));
}}

// Wheel-scrub the through-plane index of one panel.
function _sliceScroll(axis, step) {{
  if (!mriVolume) return;
  const ax = _THROUGH_IDX[axis];
  const c = mriVoxCursor.slice();
  c[ax] = Math.max(0, Math.min(mriVolume.dim[ax] - 1, c[ax] + step));
  locateAtRas(_apply4x4(mriVolume.affine, c));
}}

// Single entry point for "put the crosshair at this RAS point": scrub the 2D
// panels, move the in-scene cut planes onto it, drop the 3D locate beacon, and
// tell the parent frame.
function locateAtRas(ras) {{
  snapSlicesToRas(ras);
  snapCutPlanesToRas(ras);
  locateMarker.position.set(ras[0], ras[1], ras[2]);
  locateMarker.visible = true;
  emitLocated(ras);
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

  // Contacts near this slice — drawn under the crosshair so electrodes stay
  // visible even on the MRI (where metal doesn't show).
  _drawContactDots(ctx, axis, W, H, throughIdx);

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

// Project the contacts within a thin slab of this slice onto the panel. Keeps
// electrodes locatable on any volume (metal only shows on CT); the selected
// contact reads red, the rest cyan. Same (u,v) mapping as the crosshair.
function _drawContactDots(ctx, axis, W, H, throughIdx) {{
  if (!mriVolume || !allContacts.length) return;
  const ax = _THROUGH_IDX[axis];
  const slabVox = Math.max(1, Math.round(1.6 / (mriVolume.pixdim[ax] || 1)));
  const selLabel = (selectedContact && selectedContact.userData) ? selectedContact.userData.label : null;
  for (const c of allContacts) {{
    const vox = mriVolume.rasToVox(c.ras);
    const ci = Math.round(vox[0]), cj = Math.round(vox[1]), ck = Math.round(vox[2]);
    let through, u, vSrc;
    if (axis === "axial") {{ through = ck; u = ci; vSrc = cj; }}
    else if (axis === "coronal") {{ through = cj; u = ci; vSrc = ck; }}
    else {{ through = ci; u = cj; vSrc = ck; }}   // sagittal
    if (Math.abs(through - throughIdx) > slabVox) continue;
    const cu = u, cv = H - 1 - vSrc;
    if (cu < 0 || cu >= W || cv < 0 || cv >= H) continue;
    const sel = selLabel && c.label === selLabel;
    ctx.beginPath();
    ctx.arc(cu + 0.5, cv + 0.5, sel ? 5 : 3.2, 0, Math.PI * 2);
    ctx.fillStyle = sel ? "#ff3b30" : "rgba(90,200,255,0.92)";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = sel ? "#fff" : "rgba(10,30,45,0.9)";
    ctx.stroke();
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

// Fetch + parse a NIfTI once, cached by path — the CT (MIP), brain MRI (DVR)
// and the slice-navigator volume are now separate, so the same file can be
// requested by more than one role without re-downloading.
const _volCache = new Map();
async function _fetchVolume(volMeta) {{
  if (!volMeta || !volMeta.path) return null;
  if (_volCache.has(volMeta.path)) return _volCache.get(volMeta.path);
  const resp = await fetch(volMeta.path);
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  const vol = new NiftiVolume(await _maybeInflate(await resp.arrayBuffer()));
  _volCache.set(volMeta.path, vol);
  return vol;
}}

function _blankSlicePanels(msg, color) {{
  for (const axis of ["axial","coronal","sagittal"]) {{
    const p = slicePanels[axis]; if (!p) continue;
    const ctx = p.ctx;
    ctx.fillStyle = "#1a1a1a"; ctx.fillRect(0, 0, p.canvas.width, p.canvas.height);
    ctx.fillStyle = color || "#666"; ctx.font = "11px ui-sans-serif";
    ctx.fillText(msg, 8, 18);
  }}
}}

// Load the SLICE-navigator volume (also drives the cut planes). The MIP/DVR are
// pinned to their own volumes (buildMip/buildDvr) and are NOT touched here.
async function loadMri(volMeta) {{
  if (!volMeta || !volMeta.path) {{ _blankSlicePanels("No MRI available"); updateInfoBar(); return; }}
  try {{
    mriVolume = await _fetchVolume(volMeta);
    // Center the crosshair only the first time — keep it put across volume switches.
    if (!mriRasCursor) {{
      mriVoxCursor = [Math.floor(mriVolume.dim[0]/2), Math.floor(mriVolume.dim[1]/2), Math.floor(mriVolume.dim[2]/2)];
      mriRasCursor = _apply4x4(mriVolume.affine, mriVoxCursor);
    }} else {{
      const v = mriVolume.rasToVox(mriRasCursor);
      mriVoxCursor = [Math.round(v[0]), Math.round(v[1]), Math.round(v[2])];
    }}
    renderSlice("axial"); renderSlice("coronal"); renderSlice("sagittal");
    buildCutPlanes(mriVolume);              // cut planes track the slice volume
    _sliceVolLabel = volMeta.label || volMeta.id || "MRI";
    updateInfoBar();
    // If a contact was clicked while the volume was still downloading, snap now.
    if (selectedContact) {{
      const pos = selectedContact.getWorldPosition(new THREE.Vector3());
      snapSlicesToRas([pos.x, pos.y, pos.z]);
      snapCutPlanesToRas([pos.x, pos.y, pos.z]);
    }}
  }} catch (err) {{
    console.error("slice volume load failed", err);
    _blankSlicePanels("Slice volume load failed: " + err.message, "#c66");
  }}
}}

// ---------- 3D volume overlays (cut planes + MIP + DVR) ------------
//
// Once the NIfTI is parsed we upload it as a 3D texture, shared by: the three
// orthogonal cut planes (textured slices that float in the scene), a rotatable
// maximum-intensity projection (metal reads as bright streaks), and a
// gradient-shaded isosurface (DVR). All are OFF by default. The cut planes are
// driven from the side navigators (toggle + α + crosshair), NOT from a toolbar
// overlaying the canvas; MIP + DVR toggle from the toolbar.

let volumeBox = null;      // world-RAS bbox of the volume corners
let mipMesh = null;        // rotatable maximum-intensity-projection of the volume

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
      uOpacity: {{ value: planeAlpha }},
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
  const cx = 0.5 * (bbox.xmin + bbox.xmax);
  const cy = 0.5 * (bbox.ymin + bbox.ymax);
  const cz = 0.5 * (bbox.zmin + bbox.zmax);
  plane.position.set(cx, cy, cz);
  plane.visible = false;
  return plane;
}}

// Move one cut plane onto a RAS through-coordinate (no toolbar reflection — the
// side navigators are the control surface now).
function _setCutPlaneValue(axis, value) {{
  const plane = cutPlanes[axis];
  if (!plane) return;
  if (axis === "axial") plane.position.z = value;
  else if (axis === "coronal") plane.position.y = value;
  else plane.position.x = value;
}}

function snapCutPlanesToRas(ras) {{
  // All three planes pass through the crosshair point; visibility is unchanged
  // (each is toggled from its navigator's "3D" checkbox).
  _setCutPlaneValue("axial", ras[2]);
  _setCutPlaneValue("coronal", ras[1]);
  _setCutPlaneValue("sagittal", ras[0]);
}}

function _makeMipMesh(bbox, volTex, vol) {{
  // Rotatable maximum-intensity projection. A box spanning the volume's RAS
  // AABB; the fragment shader marches each view ray through the SAME 3D texture
  // the DVR uses (world->vox via uRasToVox) and keeps the brightest
  // sample — so the metal contacts (the brightest voxels on a windowed CT) read
  // as crisp streaks you can spin. depthTest off + renderOrder -1 makes it a
  // backdrop the electrode meshes draw over.
  const w = bbox.xmax - bbox.xmin, h = bbox.ymax - bbox.ymin, d = bbox.zmax - bbox.zmin;
  const geo = new THREE.BoxGeometry(w, h, d);
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
      uBoxMin: {{ value: new THREE.Vector3(bbox.xmin, bbox.ymin, bbox.zmin) }},
      uBoxMax: {{ value: new THREE.Vector3(bbox.xmax, bbox.ymax, bbox.zmax) }},
      uOpacity: {{ value: 0.6 }},
      uThreshold: {{ value: 0.32 }},
      uSteps: {{ value: 256.0 }},
      uColor: {{ value: new THREE.Color(1.0, 0.98, 0.92) }},
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
      uniform vec3 uBoxMin;
      uniform vec3 uBoxMax;
      uniform float uOpacity;
      uniform float uThreshold;
      uniform float uSteps;
      uniform vec3 uColor;
      varying vec3 vWorldPos;
      void main() {{
        vec3 ro = cameraPosition;
        vec3 rd = normalize(vWorldPos - ro);
        vec3 ta = (uBoxMin - ro) / rd;
        vec3 tb = (uBoxMax - ro) / rd;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        float tnear = max(max(tlo.x, tlo.y), tlo.z);
        float tfar = min(min(thi.x, thi.y), thi.z);
        tnear = max(tnear, 0.0);
        if (tnear > tfar) discard;
        int steps = int(uSteps);
        float dt = (tfar - tnear) / float(steps);
        float maxv = 0.0;
        for (int i = 0; i < 1024; i++) {{
          if (i >= steps) break;
          float t = tnear + (float(i) + 0.5) * dt;
          vec3 p = ro + rd * t;
          vec3 vox = (uRasToVox * vec4(p, 1.0)).xyz;
          vec3 uvw = (vox + 0.5) / uVolumeSize;
          if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {{
            maxv = max(maxv, texture(uVolume, uvw).r);
          }}
        }}
        if (maxv < uThreshold) discard;
        float v = (maxv - uThreshold) / max(1e-4, 1.0 - uThreshold);
        gl_FragColor = vec4(uColor * v, uOpacity * v);
      }}
    `,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
    depthTest: false,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(
    0.5 * (bbox.xmin + bbox.xmax),
    0.5 * (bbox.ymin + bbox.ymax),
    0.5 * (bbox.zmin + bbox.zmax),
  );
  mesh.renderOrder = -1;
  mesh.visible = false;
  return mesh;
}}

// Direct volume rendering (DVR): first-hit isosurface raycast with gradient
// (Blinn-ish Lambert) shading — the Curry/Medtronic "volume brain" look, no
// mesh. Best on the brain-only MRI volume (masked → the isosurface is the pial
// surface). Gradient is computed in voxel space and rotated to world via
// uVoxToRas for lighting; light follows the camera (uLightDir, see updateLights).
// (dvrMesh + _dvrLightDir are declared up near updateLights to avoid a TDZ crash.)
function _makeDvrMesh(bbox, volTex, vol) {{
  const w = bbox.xmax - bbox.xmin, h = bbox.ymax - bbox.ymin, d = bbox.zmax - bbox.zmin;
  const geo = new THREE.BoxGeometry(w, h, d);
  const inv = vol.affineInv, A = vol.affine;
  const rasToVox = new THREE.Matrix4().set(
    inv[0][0], inv[0][1], inv[0][2], inv[0][3],
    inv[1][0], inv[1][1], inv[1][2], inv[1][3],
    inv[2][0], inv[2][1], inv[2][2], inv[2][3], 0, 0, 0, 1);
  const voxToRas = new THREE.Matrix4().set(
    A[0][0], A[0][1], A[0][2], A[0][3],
    A[1][0], A[1][1], A[1][2], A[1][3],
    A[2][0], A[2][1], A[2][2], A[2][3], 0, 0, 0, 1);
  const mat = new THREE.ShaderMaterial({{
    uniforms: {{
      uVolume: {{ value: volTex }},
      uRasToVox: {{ value: rasToVox }},
      uVoxToRas: {{ value: voxToRas }},
      uVolumeSize: {{ value: new THREE.Vector3(vol.dim[0], vol.dim[1], vol.dim[2]) }},
      uBoxMin: {{ value: new THREE.Vector3(bbox.xmin, bbox.ymin, bbox.zmin) }},
      uBoxMax: {{ value: new THREE.Vector3(bbox.xmax, bbox.ymax, bbox.zmax) }},
      uThreshold: {{ value: 0.28 }},
      uSteps: {{ value: 384.0 }},
      uColor: {{ value: new THREE.Color(0.86, 0.86, 0.9) }},
      uLightDir: {{ value: _dvrLightDir.clone() }},
    }},
    vertexShader: `
      varying vec3 vWorldPos;
      void main() {{ vec4 wp = modelMatrix * vec4(position, 1.0); vWorldPos = wp.xyz;
                    gl_Position = projectionMatrix * viewMatrix * wp; }}
    `,
    fragmentShader: `
      precision highp float;
      precision highp sampler3D;
      uniform sampler3D uVolume;
      uniform mat4 uRasToVox; uniform mat4 uVoxToRas;
      uniform vec3 uVolumeSize, uBoxMin, uBoxMax, uColor, uLightDir;
      uniform float uThreshold, uSteps;
      varying vec3 vWorldPos;
      float sVox(vec3 vox) {{
        vec3 uvw = (vox + 0.5) / uVolumeSize;
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 0.0;
        return texture(uVolume, uvw).r;
      }}
      void main() {{
        vec3 ro = cameraPosition;
        vec3 rd = normalize(vWorldPos - ro);
        vec3 ta = (uBoxMin - ro) / rd, tb = (uBoxMax - ro) / rd;
        vec3 tlo = min(ta, tb), thi = max(ta, tb);
        float tnear = max(max(tlo.x, tlo.y), tlo.z);
        float tfar = min(min(thi.x, thi.y), thi.z);
        tnear = max(tnear, 0.0);
        if (tnear > tfar) discard;
        int steps = int(uSteps);
        float dt = (tfar - tnear) / float(steps);
        float prev = 0.0, tprev = tnear;
        for (int i = 0; i < 1024; i++) {{
          if (i >= steps) break;
          float t = tnear + (float(i) + 0.5) * dt;
          vec3 vox = (uRasToVox * vec4(ro + rd * t, 1.0)).xyz;
          float v = sVox(vox);
          if (v >= uThreshold && prev < uThreshold) {{
            float f = (uThreshold - prev) / max(1e-4, v - prev);   // sub-step refine
            vec3 hv = (uRasToVox * vec4(ro + rd * mix(tprev, t, f), 1.0)).xyz;
            vec3 g = vec3(sVox(hv + vec3(1.,0,0)) - sVox(hv - vec3(1.,0,0)),
                          sVox(hv + vec3(0,1.,0)) - sVox(hv - vec3(0,1.,0)),
                          sVox(hv + vec3(0,0,1.)) - sVox(hv - vec3(0,0,1.)));
            vec3 n = normalize((mat3(uVoxToRas) * (-g)) + 1e-6);   // outward (toward CSF)
            float diff = max(dot(n, normalize(uLightDir)), 0.0);
            gl_FragColor = vec4(uColor * (0.25 + 0.75 * diff), 1.0);
            return;
          }}
          prev = v; tprev = t;
        }}
        discard;
      }}
    `,
    side: THREE.BackSide,
    transparent: false,
    depthWrite: false,
    depthTest: false,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0.5 * (bbox.xmin + bbox.xmax), 0.5 * (bbox.ymin + bbox.ymax),
                    0.5 * (bbox.zmin + bbox.zmax));
  mesh.renderOrder = -2;   // behind the MIP + electrodes
  mesh.visible = false;
  return mesh;
}}

// The three overlays are now driven by INDEPENDENT volumes, so switching the
// slice-navigator volume never drags the MIP off the CT:
//   • cut planes  → follow the slice navigator (buildCutPlanes, on volume switch)
//   • MIP         → pinned to the CT   (metal projection is a CT concept)
//   • DVR         → pinned to the brain MRI (isosurface = pial)
function buildCutPlanes(vol) {{
  for (const axis of Object.keys(cutPlanes)) {{ scene.remove(cutPlanes[axis]); delete cutPlanes[axis]; }}
  volumeBox = _volumeRasBbox(vol);
  const tex = _buildVolumeTexture(vol);
  for (const axis of ["axial", "coronal", "sagittal"]) {{
    const plane = _makeCutPlane(axis, volumeBox, tex, vol);
    plane.visible = planeWanted[axis];
    plane.material.uniforms.uOpacity.value = planeAlpha;
    cutPlanes[axis] = plane;
    scene.add(plane);
  }}
  if (mriRasCursor) snapCutPlanesToRas(mriRasCursor);
}}

function buildMip(vol) {{
  if (mipMesh) {{ scene.remove(mipMesh); mipMesh = null; }}
  mipMesh = _makeMipMesh(_volumeRasBbox(vol), _buildVolumeTexture(vol), vol);
  const cb = document.querySelector('.plane-ctl[data-control="mip"] input[type="checkbox"]');
  mipMesh.visible = !!(cb && cb.checked);   // respect the current toggle
  scene.add(mipMesh);
}}

function buildDvr(vol) {{
  if (dvrMesh) {{ scene.remove(dvrMesh); dvrMesh = null; }}
  dvrMesh = _makeDvrMesh(_volumeRasBbox(vol), _buildVolumeTexture(vol), vol);
  const cb = document.querySelector('.plane-ctl[data-control="dvr"] input[type="checkbox"]');
  dvrMesh.visible = !!(cb && cb.checked);
  scene.add(dvrMesh);
}}

// Wire the MIP/DVR toolbar controls ONCE — the handlers read the current
// module-level mipMesh/dvrMesh, so rebuilding those meshes keeps them live.
let _overlayControlsWired = false;
function wireOverlayControls() {{
  if (_overlayControlsWired) return;
  _overlayControlsWired = true;
  const mipCtl = document.querySelector('.plane-ctl[data-control="mip"]');
  if (mipCtl) {{
    const mcb = mipCtl.querySelector('input[type="checkbox"]');
    const thr = mipCtl.querySelector('input[data-mip="threshold"]');
    const op = mipCtl.querySelector('input[data-mip="opacity"]');
    mcb.addEventListener("change", () => {{ if (mipMesh) mipMesh.visible = mcb.checked; }});
    if (thr) thr.addEventListener("input", () => {{ if (mipMesh) mipMesh.material.uniforms.uThreshold.value = parseFloat(thr.value); }});
    if (op) op.addEventListener("input", () => {{ if (mipMesh) mipMesh.material.uniforms.uOpacity.value = parseFloat(op.value); }});
  }}
  const dvrCtl = document.querySelector('.plane-ctl[data-control="dvr"]');
  if (dvrCtl) {{
    const cb = dvrCtl.querySelector('input[type="checkbox"]');
    const thr = dvrCtl.querySelector('input[data-dvr="threshold"]');
    cb.addEventListener("change", () => {{ if (dvrMesh) dvrMesh.visible = cb.checked; }});
    if (thr) thr.addEventListener("input", () => {{ if (dvrMesh) dvrMesh.material.uniforms.uThreshold.value = parseFloat(thr.value); }});
  }}
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
    selectionMarker.position.copy(pos);   // beacon on the highlighted contact
    selectionMarker.visible = true;
    locateMarker.visible = false;          // the contact beacon supersedes the free crosshair
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
    // Snap the navigator panels + the in-scene cut planes to the contact's
    // RAS coordinate, and tell any parent frame where we landed.
    snapSlicesToRas([pos.x, pos.y, pos.z]);
    snapCutPlanesToRas([pos.x, pos.y, pos.z]);
    emitSelected(label, shank, [pos.x, pos.y, pos.z]);
  }}
  // Toggle selection in sidebar UI
  document.querySelectorAll(".contact-row").forEach(r => r.classList.remove("selected"));
  const row = document.querySelector(`.contact-row[data-label="${{label}}"]`);
  if (row) {{
    row.classList.add("selected");
    row.scrollIntoView({{ block: "nearest" }});
  }}
}}

// Embedded mode: let a parent frame (e.g. the rosa-app UI) drive selection +
// visibility so its contact list and this 3D scene stay in sync. No-op standalone.
let _pendingVisibility = null;
function applyContactVisibility(hideShanks, hideContacts) {{
  _pendingVisibility = {{ hideShanks: hideShanks || [], hideContacts: hideContacts || [] }};
  if (!gltfRoot) return;  // GLB not loaded yet — re-applied from onGltf
  const hs = new Set(_pendingVisibility.hideShanks);
  const hc = new Set(_pendingVisibility.hideContacts.map(c => (c.shank || "") + "::" + c.label));
  gltfRoot.traverse(obj => {{
    const ex = obj.userData || {{}};
    if (ex.kind === "contact")
      obj.visible = !(hs.has(ex.shank) || hc.has((ex.shank || "") + "::" + ex.label));
    else if (ex.kind === "shaft")
      obj.visible = !hs.has(ex.shank);
  }});
}}
// ---------- scene message API -------------------------------------
// A documented seam so a parent frame (the rosa-app shell, or a future
// standalone 2D navigator / editor) can drive this 3D scene and stay in
// sync with it. RAS millimetres + contact identity (label + shank) are the
// only currency — never pixels or voxels — so the same messages work across
// the iframe boundary today and a module boundary tomorrow.
//
//   Inbound  (parent → scene)
//     rosa:select     {{ label, shank }}                 highlight + frame a contact
//     rosa:visibility {{ hideShanks, hideContacts }}     hide rejected electrodes
//     rosa:locate     {{ ras: [x,y,z] }}                 move the crosshair / slices
//   Outbound (scene → parent)
//     rosa:selected   {{ label, shank, ras }}            user picked a contact here
//     rosa:located    {{ ras }}                          user moved the crosshair here
function _emit(type, payload) {{
  try {{
    if (window.parent && window.parent !== window) window.parent.postMessage({{ type, ...payload }}, "*");
  }} catch (_e) {{}}
}}
function emitSelected(label, shank, ras) {{ _emit("rosa:selected", {{ label, shank, ras }}); }}
function emitLocated(ras) {{ _emit("rosa:located", {{ ras }}); }}

window.addEventListener("message", (e) => {{
  const m = e.data || {{}};
  if (!m) return;
  if (m.type === "rosa:select" && m.label) selectContact(m.label, m.shank || "");
  else if (m.type === "rosa:visibility") applyContactVisibility(m.hideShanks, m.hideContacts);
  else if (m.type === "rosa:locate" && Array.isArray(m.ras)) locateAtRas(m.ras);
}});

function showAll() {{
  selectedShank = null;
  for (const [, nodes] of shankNodes) for (const n of nodes) n.visible = true;
  if (selectedContact && originalMaterials.has(selectedContact)) {{
    _setContactMaterial(selectedContact, originalMaterials.get(selectedContact));
    selectedContact = null;
  }}
  selectionMarker.visible = false;   // clear the highlight beacon
  locateMarker.visible = false;      // and the free-locate crosshair
  document.querySelectorAll(".shank-card").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".contact-row").forEach(r => r.classList.remove("selected"));
}}

document.getElementById("btn-reset").addEventListener("click", showAll);
document.getElementById("btn-fit").addEventListener("click", () => {{ if (gltfRoot) fitToObject(gltfRoot); }});
(() => {{
  const bar = document.getElementById("toolbar");
  const btn = document.getElementById("btn-tools");
  const sync = () => {{ btn.textContent = bar.classList.contains("collapsed") ? "Controls ▾" : "Controls ▴"; }};
  btn.addEventListener("click", () => {{ bar.classList.toggle("collapsed"); sync(); }});
  sync();
}})();

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
  // Only "parcellation" uses the per-vertex COLOR_0; White/Skin are FLAT, so
  // vertexColors must be turned OFF (otherwise the material colour just
  // multiplies the parcellation and White/Skin never render).
  const useVertexColors = (modeKey === "parcellation");
  for (const node of surfaceNodes) {{
    const mesh = node.userData && node.userData._mesh;
    if (!mesh || !mesh.material || !mesh.material.color) continue;
    mesh.material.color.setRGB(rgb[0], rgb[1], rgb[2]);
    if (mesh.material.vertexColors !== useVertexColors) {{
      mesh.material.vertexColors = useVertexColors;
    }}
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

// Metadata handler — named so picker mode can call it with a dropped (and
// path-rewritten) scene_meta.json. Identical logic for served + picker.
function onMeta(meta) {{
  renderSidebar(meta);
  // Contacts for the slice overlay (already in the CT/contact RAS frame).
  sceneMeta = meta;
  allContacts = (meta.contacts || [])
    .map(c => ({{ label: c.label, shank: c.trajectory, ras: c.position || [c.x, c.y, c.z] }}))
    .filter(c => Array.isArray(c.ras) && c.ras.length === 3);
  // Volume roles (CT / MRI (brain) / FreeSurfer T1). Backend lists [MRI(brain),
  // T1, CT]; falls back to legacy `t1_volume`.
  const volSel = document.getElementById("vol-select");
  const vols = (meta.volumes && meta.volumes.length)
    ? meta.volumes
    : (meta.t1_volume ? [meta.t1_volume] : []);
  const isCt = v => (v.id || "").toLowerCase() === "ct" || /(^|[^a-z])ct($|[^a-z])/i.test(v.label || "");
  const isBrain = v => /mri|brain|t1/i.test((v.id || "") + " " + (v.label || ""));
  const ctMeta = vols.find(isCt) || vols[vols.length - 1] || null;
  const brainMeta = vols.find(isBrain) || null;
  _mriAvailable = !!brainMeta;
  wireOverlayControls();
  // Pinned volume overlays: MIP on the CT (metal projection), DVR on the brain
  // MRI (isosurface = pial). Independent of the slice-navigator volume below, so
  // changing the Volume dropdown never drags the MIP off the CT.
  (async () => {{
    try {{ const c = await _fetchVolume(ctMeta); if (c) buildMip(c); }} catch (e) {{ console.error("MIP volume", e); }}
    try {{ const b = await _fetchVolume(brainMeta || ctMeta); if (b) buildDvr(b); }} catch (e) {{ console.error("DVR volume", e); }}
  }})();
  if (volSel && vols.length) {{
    volSel.innerHTML = "";
    vols.forEach((v, i) => {{
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = v.label || v.id || ("Volume " + i);
      volSel.appendChild(opt);
    }});
    volSel.addEventListener("change", () => {{
      const v = vols[parseInt(volSel.value, 10)];
      if (v) loadMri(v);
    }});
    if (vols.length < 2) volSel.parentElement.style.display = "none";
    // Slice navigator defaults to the CT — metal contacts are visible there.
    let di = ctMeta ? vols.indexOf(ctMeta) : 0;
    if (di < 0) di = 0;
    volSel.value = String(di);
    loadMri(vols[di]);
  }} else {{
    loadMri(meta.t1_volume);
  }}
  updateInfoBar();
}}
function onMetaErr(err) {{
  console.error(err);
  document.getElementById("subject").textContent = "Failed to load metadata: " + err.message;
}}

// Picker mode (GitHub Pages app): no fetch — the user drops their own export
// (scene.glb + scene_meta.json + the .nii.gz volumes). We map each file to a
// blob: URL, rewrite the volume paths in the metadata to those blobs, then call
// the SAME onMeta / loadGlb the served path uses. Nothing is uploaded.
function __initPicker() {{
  const dz = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  if (dz) dz.style.display = "flex";

  function handleFiles(fileList) {{
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const blobByName = {{}};
    let metaFile = null;
    for (const f of files) {{
      blobByName[f.name] = URL.createObjectURL(f);
      if (f.name === "scene_meta.json") metaFile = f;
    }}
    if (!metaFile) {{ for (const f of files) if (f.name.endsWith(".json")) {{ metaFile = f; break; }} }}
    if (!metaFile) {{ setDbg("glb", "drop scene_meta.json too", "err"); return; }}
    metaFile.text().then(txt => {{
      let meta;
      try {{ meta = JSON.parse(txt); }} catch (e) {{ setDbg("glb", "bad scene_meta.json", "err"); return; }}
      const fix = v => (v && v.path && blobByName[v.path])
        ? Object.assign({{}}, v, {{ path: blobByName[v.path] }}) : v;
      if (Array.isArray(meta.volumes)) meta.volumes = meta.volumes.map(fix);
      if (meta.t1_volume) meta.t1_volume = fix(meta.t1_volume);
      if (dz) dz.style.display = "none";
      onMeta(meta);
      const glbUrl = blobByName["scene.glb"];
      if (glbUrl) loadGlb(glbUrl);
      else setDbg("glb", "no scene.glb in the drop", "err");
    }});
  }}

  if (fileInput) fileInput.addEventListener("change", e => handleFiles(e.target.files));
  if (dz) {{
    dz.addEventListener("dragover", e => {{ e.preventDefault(); dz.classList.add("hover"); }});
    dz.addEventListener("dragleave", () => dz.classList.remove("hover"));
    dz.addEventListener("drop", e => {{ e.preventDefault(); dz.classList.remove("hover"); handleFiles(e.dataTransfer.files); }});
  }}
}}

_initSlicePanels();
if (VIEWER_MODE === "picker") {{
  __initPicker();
}} else {{
  fetch("scene_meta.json").then(r => r.json()).then(onMeta).catch(onMetaErr);
}}
</script>
</body>
</html>
"""


def render_viewer_html(*, title: str, mode: str = "served") -> str:
    """Render the viewer index.html. ``mode`` is the JS VIEWER_MODE:

    * ``served``  — auto-loads scene.glb / scene_meta.json / volumes from the
      same directory via fetch (what ``export-view`` writes + ``view`` serves).
    * ``picker``  — no auto-load; shows a drag-and-drop zone and renders the
      user's own dropped files entirely client-side (the GitHub Pages app —
      nothing uploaded, no server). Reuses the exact same render engine.
    """
    return _HTML_TEMPLATE.format(title=title, viewer_mode=mode)


def _write_html(out_dir: Path, *, title: str, mode: str = "served") -> Path:
    p = out_dir / "index.html"
    p.write_text(render_viewer_html(title=title, mode=mode), encoding="utf-8")
    return p


# ---------------------------------------------------------------------
# Shared scene assembly (export-view AND view-results)
# ---------------------------------------------------------------------


def _assemble_viewer(
    out: Path,
    *,
    ct_path: Path,
    trajectories: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    contact_labels: dict[str, dict[str, str]],
    fs,
    brain_volume_path: Path | None = None,
    brain_mask_cache_path: Path | None = None,
    brain_gyri: bool = False,
    brain_native_volume_path: Path | None = None,
    brain_to_ct_transform_path: Path | None = None,
    brain_surface_cache_path: Path | None = None,
    fastsurfer_aseg_path: Path | None = None,
    fastsurfer: bool = False,
    drop_cerebellum: bool = True,
    parcellation,
    lut,
    annotation: str,
    surface_kinds: tuple[str, ...],
    shaft_radius_mm: float,
    contact_band_radius_mm: float,
    contact_band_length_mm: float,
    ct_window: tuple[float, float],
    subject_label: str,
    title: str,
    base_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the GLB scene + windowed CT (+ FS surfaces/T1 when available) +
    viewer HTML from ALREADY-COMPUTED trajectories/contacts.

    Shared by ``export-view`` (after the pipeline) and ``view-results`` (after
    reading existing TSVs) — the difference between the two is only how
    ``trajectories``/``contacts`` are produced.
    """
    lut_index = parse_freesurfer_lut(lut) if lut else {}

    # Register FS T1 -> CT (only when surfaces exist). The transform is reused
    # to push surface vertices into CT RAS AND to resample the T1 onto the CT
    # grid for the in-browser slice planes.
    surfaces = []
    t1_slice_meta: dict[str, Any] | None = None
    surface_source = ""   # human-readable provenance of the 3D surface (for the viewer info line)
    if fs is not None and fs.available_surfaces:
        surface_source = "MRI — FreeSurfer recon"
        fs_to_ct_4x4, fixed_ct_img, moving_t1_img, sitk_transform = _register_fs_to_ct(
            fs.t1_path, ct_path,
        )
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
        t1_out = out / "t1_in_ct.nii.gz"
        t1_slice_meta = _write_t1_resampled_to_ct(
            fixed_ct_img, moving_t1_img, sitk_transform, t1_out,
        )
        _stderr(
            f"[view] wrote {t1_out} (T1 resampled onto CT grid; "
            f"{t1_slice_meta['size']} {t1_slice_meta['dtype']})"
        )
    elif brain_native_volume_path is not None or (
            brain_surface_cache_path is not None and Path(brain_surface_cache_path).is_file()):
        # No FreeSurfer recon, but we have the NATIVE MRI (1mm, isotropic) +
        # the MRI→CT rigid transform the labeling step already computed. Mesh
        # the gyri in the native frame (clean, no interpolation ramps) and push
        # the surface into CT via that transform — the same trick FS surfaces
        # use. This is the accurate, bundleable "gyri without FreeSurfer" path;
        # meshing the MRI *resampled onto the anisotropic CT grid* looks bumpy.
        try:
            import tempfile
            import numpy as np
            from types import SimpleNamespace
            from rosa_core.brain_mesh import BrainSurface
            scache = Path(brain_surface_cache_path) if brain_surface_cache_path else None
            if scache is not None and scache.is_file():
                # The surface is atlas-independent (it depends only on the MRI +
                # CT + registration, all fixed per case), so it's built ONCE per
                # case and every later label reuses it — no re-mesh, no re-warp.
                d = np.load(scache)
                bs = BrainSurface(vertices_ras=d["v"], faces=d["f"], vertex_normals=d["n"])
                brain_colors = d["c"] if "c" in d.files else None
                backend = "cached-surface"
                _stderr(f"[view] reusing cached brain surface {scache.name} "
                        f"({bs.n_vertices} verts{', parcellated' if brain_colors is not None else ''})")
            else:
                from rosa_detect.services.mask_backend import select_brain_mask_to_path
                from rosa_core.brain_mesh import (
                    gyral_surface_from_mri, brain_tissue_from_fastsurfer_aseg, transform_surface)
                from rosa_core.registration import transform_to_4x4_ras, load_transform
                # Native brain mask (SynthStrip), cached once per case.
                cache = Path(brain_mask_cache_path) if brain_mask_cache_path else None
                if cache is not None and cache.is_file():
                    nmask, backend = cache, "cached"
                else:
                    nmask = cache or (Path(tempfile.mkdtemp()) / "brain_mask_native.nii.gz")
                    if cache is not None:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                    backend = select_brain_mask_to_path(
                        brain_native_volume_path, nmask, backend="auto", log=_stderr)
                    if backend is None:
                        raise RuntimeError("no brain-extract backend available")
                # Prefer FastSurfer's LEARNED tissue as the surface support (dura
                # never enters; cerebellum dropped cleanly). Fall back to the Otsu
                # grayscale-iso path when FastSurfer isn't available. Either way,
                # the same step_size=1 grayscale-iso mesher runs.
                aseg = Path(fastsurfer_aseg_path) if fastsurfer_aseg_path else None
                if aseg is None and fastsurfer:
                    from rosa_detect.services.fastsurfer_seg import run_fastsurfer_seg
                    aseg = run_fastsurfer_seg(
                        brain_native_volume_path,
                        (scache.parent if scache is not None else Path(tempfile.mkdtemp())) / "fastsurfer",
                        sid="subject", log=_stderr)
                if aseg is not None and aseg.is_file():
                    tissue = brain_tissue_from_fastsurfer_aseg(
                        aseg, brain_native_volume_path, drop_cerebellum=drop_cerebellum)
                    bs = gyral_surface_from_mri(
                        brain_native_volume_path, nmask, step_size=1, brain_tissue=tissue)
                    backend = f"{backend}+fastsurfer"
                else:
                    bs = gyral_surface_from_mri(brain_native_volume_path, nmask, step_size=1)
                # aparc surface coloring (when a FastSurfer/FS aseg is available):
                # sample the parcellation at each vertex → per-vertex RGBA for the
                # viewer's "Parcellation" mode. Sample in the NATIVE frame (aseg +
                # surface share RAS) before the CT transform; colors are per-vertex
                # so they survive the transform + cache unchanged.
                brain_colors = None
                if aseg is not None and aseg.is_file():
                    try:
                        import rosa_core
                        from rosa_core.brain_mesh import aparc_vertex_colors
                        lut = parse_freesurfer_lut(   # module-level import
                            Path(rosa_core.__file__).parent / "resources" / "freesurfer"
                            / "FreeSurferColorLUT20120827.txt")
                        brain_colors = aparc_vertex_colors(bs.vertices_ras, aseg, lut)
                        _stderr(f"[view] parcellated surface from aparc ({len(lut)} LUT entries)")
                    except Exception as exc:  # noqa: BLE001
                        _stderr(f"[view] aparc coloring skipped ({exc})")
                # Native-MRI RAS → CT RAS. Reuse the cached t1_to_ct.tfm (fixed=CT,
                # moving=T1) when present; otherwise register here as a fallback.
                tfp = Path(brain_to_ct_transform_path) if brain_to_ct_transform_path else None
                if tfp is not None and tfp.is_file():
                    sitk_tf = load_transform(str(tfp))
                    _stderr(f"[view] reusing cached MRI→CT transform {tfp.name}")
                else:
                    _, _, _, sitk_tf = _register_fs_to_ct(brain_native_volume_path, ct_path)
                ct_from_t1 = np.linalg.inv(transform_to_4x4_ras(sitk_tf))
                bs = transform_surface(bs, ct_from_t1)   # colors are per-vertex, order preserved
                if scache is not None:
                    scache.parent.mkdir(parents=True, exist_ok=True)
                    extra = {"c": brain_colors} if brain_colors is not None else {}
                    np.savez(scache, v=bs.vertices_ras, f=bs.faces, n=bs.vertex_normals, **extra)
                    _stderr(f"[view] cached brain surface → {scache.name}")
                    # Brain-extracted MRI in the CT frame → the DVR volume (cached
                    # with the surface, atlas-independent). An isosurface raycast
                    # of this in the browser is the mesh-free "volume brain".
                    try:
                        import SimpleITK as sitk
                        dvr = _brain_mri_dvr_volume(brain_native_volume_path, nmask, sitk_tf, ct_path)
                        sitk.WriteImage(dvr, str(scache.parent / "brain_mri_in_ct.nii.gz"))
                        _stderr("[view] cached DVR volume → brain_mri_in_ct.nii.gz")
                    except Exception as exc:  # noqa: BLE001
                        _stderr(f"[view] DVR volume failed ({exc})")
            surfaces = [SimpleNamespace(
                name="brain", hemi="lh", kind="brain",
                annotation_name=("aparc" if brain_colors is not None else None),
                vertices_ras=bs.vertices_ras, faces=bs.faces,
                vertex_normals=bs.vertex_normals, vertex_colors_rgba=brain_colors,
            )]
            # Provenance: the surface is always meshed from the native MRI; the
            # tissue/parcellation is FastSurfer (learned, colored) or Otsu.
            surface_source = ("MRI — FastSurfer recon" if brain_colors is not None
                              else "MRI — intensity surface")
            _stderr(f"[view] subject brain surface (native+gyri, {backend}): "
                    f"{bs.n_vertices} verts / {bs.n_faces} faces")
        except Exception as exc:  # noqa: BLE001 — brain mesh is optional context
            _stderr(f"[view] native gyral mesh failed ({exc}); no brain mesh")
    elif brain_volume_path is not None or brain_mask_cache_path is not None:
        # No FreeSurfer recon, but we have a volume already in the CT frame
        # (typically the MRI resampled to CT): brain-extract it and marching-cubes
        # the mask into ONE subject-own brain surface — the patient's actual
        # anatomy (accurate), unlike a template warp. Vertices come out in the
        # mask's RAS (== CT RAS), so no registration is needed. The mask is
        # cached (brain_mask_cache_path) so SynthStrip runs once per case, not
        # per atlas/label.
        try:
            import tempfile
            from types import SimpleNamespace
            from rosa_detect.services.mask_backend import select_brain_mask_to_path
            from rosa_core.brain_mesh import surface_from_mask
            cache = Path(brain_mask_cache_path) if brain_mask_cache_path else None
            if cache is not None and cache.is_file():
                _mask, backend = cache, "cached"
            elif brain_volume_path is not None:
                _mask = cache or (Path(tempfile.mkdtemp()) / "brain_mask.nii.gz")
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                backend = select_brain_mask_to_path(
                    brain_volume_path, _mask, backend="auto", log=_stderr)
                if backend is None:
                    raise RuntimeError("no brain-extract backend available")
            else:
                raise RuntimeError("brain mask cache absent and no brain volume to extract")
            if brain_gyri and brain_volume_path is not None:
                # Fold the surface into the sulci: Otsu-drop CSF from the T1 so
                # the mesh follows GM/WM (gyri) instead of the filled envelope.
                # Low smooth_sigma / few Taubin passes so the folds survive;
                # step_size=3 keeps the gyral detail at ~half the weight of
                # step 2 (~100k verts / ~5MB vs ~240k / ~11MB — step 2 is overkill).
                from rosa_core.brain_mesh import gyral_mask_from_mri
                gmwm = gyral_mask_from_mri(brain_volume_path, _mask)
                bs = surface_from_mask(
                    gmwm, smooth_sigma=0.5, taubin_iterations=6, step_size=3)
                backend = f"{backend}+gyri"
            else:
                # step_size=4 (~26k verts vs ~103k at the default 2) — a translucent
                # context envelope needs no gyral detail, and it loads ~4x smaller.
                bs = surface_from_mask(_mask, step_size=4)
            surfaces = [SimpleNamespace(
                name="brain", hemi="lh", kind="brain", annotation_name=None,
                vertices_ras=bs.vertices_ras, faces=bs.faces,
                vertex_normals=bs.vertex_normals, vertex_colors_rgba=None,
            )]
            surface_source = "MRI — intensity surface"
            _stderr(f"[view] subject brain surface ({backend}): "
                    f"{bs.n_vertices} verts / {bs.n_faces} faces")
        except Exception as exc:  # noqa: BLE001 — brain mesh is optional context
            _stderr(f"[view] brain-extract/mesh failed ({exc}); no brain mesh")
    else:
        _stderr("[view] no FreeSurfer surfaces / brain volume; skipping brain mesh")

    # Always export the working CT as a windowed uint8 slice/MIP volume so the
    # viewer has anatomy even without a FreeSurfer recon. The CT already lives
    # in the contact frame, so no resampling is needed.
    ct_slice_meta = _write_ct_slice_volume(
        ct_path, out / "ct_in_view.nii.gz", window=ct_window,
    )
    _stderr(
        f"[view] wrote ct_in_view.nii.gz (windowed CT slice/MIP volume; "
        f"{ct_slice_meta['size']})"
    )

    # Volume list for the in-browser selector. T1 first when present (keeps the
    # existing FS-mode default), CT always available.
    volumes: list[dict[str, Any]] = []
    if t1_slice_meta is not None:
        volumes.append(t1_slice_meta)
    # Brain-only MRI in the CT frame (cached with the surface) — the DVR source.
    # First when present so it's the default and the Volume 3D toggle renders a
    # brain immediately.
    _dvr_cache = (Path(brain_surface_cache_path).parent / "brain_mri_in_ct.nii.gz"
                  if brain_surface_cache_path else None)
    if _dvr_cache is not None and _dvr_cache.is_file():
        try:
            import SimpleITK as sitk
            dvr_meta = _slice_volume_meta(
                sitk.ReadImage(str(_dvr_cache)), out / "mri_in_view.nii.gz",
                vid="mri_brain", label="MRI (brain)")
            volumes.insert(0, dvr_meta)
            _stderr(f"[view] wrote mri_in_view.nii.gz (brain MRI for DVR; {dvr_meta['size']})")
        except Exception as exc:  # noqa: BLE001
            _stderr(f"[view] DVR volume export failed ({exc})")
    volumes.append(ct_slice_meta)

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

    meta = {
        "subject_label": subject_label,
        "ct_path": str(ct_path),
        "surface_source": surface_source,
        "freesurfer_subject": str(fs.subject_dir) if fs is not None else "",
        "parcellation": str(parcellation) if parcellation else "",
        "lut": str(lut) if lut else "",
        "annotation": annotation if fs is not None else "",
        "trajectories": trajectories,
        "contacts": contact_meta,
        "volumes": volumes,
        # Back-compat: the original viewer reads `t1_volume`. Point it at the
        # default volume (T1 in FS mode, CT otherwise) so old behavior holds.
        "t1_volume": volumes[0] if volumes else None,
    }
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    html_path = _write_html(out, title=title)
    _stderr(f"[view] wrote {html_path}")

    view_manifest = dict(base_manifest or {})
    view_manifest.update({
        "freesurfer_subject": str(fs.subject_dir) if fs is not None else "",
        "parcellation": str(parcellation) if parcellation else "",
        "lut": str(lut) if lut else "",
        "annotation": annotation if fs is not None else "",
        "surfaces_loaded": [s.name for s in surfaces],
        "volumes": [v["id"] for v in volumes],
        "scene_glb": str(glb_path),
        "viewer_html": str(html_path),
    })
    (out / "view_manifest.json").write_text(json.dumps(view_manifest, indent=2), encoding="utf-8")
    return view_manifest


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
    ct_window: tuple[float, float] = (-150.0, 1500.0),
) -> dict[str, Any]:
    """Run the pipeline + assemble the GLB viewer.

    ``freesurfer_dir`` is optional: without it the viewer is CT-only (CT slice
    volume + 3D electrodes, no brain mesh and no atlas labels).
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    fs = resolve_fs_subject(freesurfer_dir) if freesurfer_dir else None
    if fs is not None:
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
    else:
        _stderr("[view] no --freesurfer-dir: CT-only view (no brain mesh, no atlas labels)")
        parcellation = None
        lut = None

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
        atlas_base_path=str(fs.t1_path) if fs is not None else None,
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
    # 2-5. Register FS, window the CT, build the scene, write the viewer.
    # Shared with `view-results` (which feeds it pre-computed trajectories /
    # contacts instead of pipeline output).
    return _assemble_viewer(
        out,
        ct_path=ct_path,
        trajectories=trajectories,
        contacts=contacts,
        contact_labels=contact_labels,
        fs=fs,
        parcellation=parcellation,
        lut=lut,
        annotation=annotation,
        surface_kinds=surface_kinds,
        shaft_radius_mm=shaft_radius_mm,
        contact_band_radius_mm=contact_band_radius_mm,
        contact_band_length_mm=contact_band_length_mm,
        ct_window=ct_window,
        subject_label=Path(target).name,
        title=f"rosa-agent export-view — {Path(target).name}",
        base_manifest={"pipeline": pipeline_summary},
    )


def run_view_results(
    out_dir: str | Path,
    *,
    ct_path: str | Path,
    contacts_tsv: str | Path,
    trajectories_tsv: str | Path | None = None,
    labels_tsv: str | Path | None = None,
    freesurfer_dir: str = "",
    brain_volume: str | Path | None = None,
    brain_mask_cache: str | Path | None = None,
    brain_gyri: bool = False,
    brain_native_volume: str | Path | None = None,
    brain_to_ct_transform: str | Path | None = None,
    brain_surface_cache: str | Path | None = None,
    fastsurfer_aseg: str | Path | None = None,
    fastsurfer: bool = False,
    drop_cerebellum: bool = True,
    surface_kinds: tuple[str, ...] = ("pial",),
    annotation: str = "aparc",
    shaft_radius_mm: float = 0.35,
    contact_band_radius_mm: float = 0.55,
    contact_band_length_mm: float = 2.0,
    ct_window: tuple[float, float] = (-150.0, 1500.0),
    subject_label: str = "",
) -> dict[str, Any]:
    """Assemble the viewer from ALREADY-COMPUTED results — no pipeline re-run.

    Reads ``contacts_tsv`` (+ optional ``trajectories_tsv`` / ``labels_tsv``),
    which must be in ``ct_path``'s RAS frame, and renders them with the same
    engine ``export-view`` uses. FreeSurfer is optional (adds the brain mesh).
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ct = Path(ct_path)
    if not ct.exists():
        raise FileNotFoundError(f"CT not found: {ct}")

    contacts = _read_pipeline_contacts(Path(contacts_tsv))
    trajectories = (
        _read_pipeline_trajectories(Path(trajectories_tsv)) if trajectories_tsv else []
    )
    contact_labels = _read_labels_by_contact(
        Path(labels_tsv) if labels_tsv else None
    )
    _stderr(
        f"[view-results] {len(trajectories)} trajectories, {len(contacts)} contacts "
        f"from existing TSVs (no detection re-run)"
    )

    fs = resolve_fs_subject(freesurfer_dir) if freesurfer_dir else None
    parcellation = _discover_parcellation(fs, None) if fs is not None else None
    lut = _discover_lut(fs, None) if fs is not None else None

    label = subject_label or ct.stem
    return _assemble_viewer(
        out,
        ct_path=ct,
        trajectories=trajectories,
        contacts=contacts,
        contact_labels=contact_labels,
        fs=fs,
        brain_volume_path=(Path(brain_volume) if brain_volume else None),
        brain_mask_cache_path=(Path(brain_mask_cache) if brain_mask_cache else None),
        brain_gyri=brain_gyri,
        brain_native_volume_path=(Path(brain_native_volume) if brain_native_volume else None),
        brain_to_ct_transform_path=(Path(brain_to_ct_transform) if brain_to_ct_transform else None),
        brain_surface_cache_path=(Path(brain_surface_cache) if brain_surface_cache else None),
        fastsurfer_aseg_path=(Path(fastsurfer_aseg) if fastsurfer_aseg else None),
        fastsurfer=fastsurfer,
        drop_cerebellum=drop_cerebellum,
        parcellation=parcellation,
        lut=lut,
        annotation=annotation,
        surface_kinds=surface_kinds,
        shaft_radius_mm=shaft_radius_mm,
        contact_band_radius_mm=contact_band_radius_mm,
        contact_band_length_mm=contact_band_length_mm,
        ct_window=ct_window,
        subject_label=label,
        title=f"rosa-agent view-results — {label}",
        base_manifest={
            "source": "view-results",
            "inputs": {
                "ct": str(ct),
                "contacts": str(contacts_tsv),
                "trajectories": str(trajectories_tsv) if trajectories_tsv else "",
                "labels": str(labels_tsv) if labels_tsv else "",
            },
        },
    )


def _parse_window(spec: str) -> tuple[float, float]:
    """Parse a 'lo,hi' HU window string; falls back to the CT default."""
    default = (-150.0, 1500.0)
    if not spec:
        return default
    try:
        lo, hi = (float(x) for x in spec.split(","))
    except (ValueError, TypeError):
        return default
    return (lo, hi) if hi > lo else default


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
        "--freesurfer-dir", default="",
        help="FreeSurfer recon-all subject directory (surf/, mri/, label/). "
             "OPTIONAL — omit it for a CT-only view (CT slices + 3D electrodes, "
             "no brain mesh and no atlas labels).",
    )
    parser.add_argument(
        "--ct-window", default="",
        help="HU window 'lo,hi' for the CT slice/MIP volume "
             "(default '-150,1500': brain mid-gray, metal contacts saturate bright).",
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
        ct_window=_parse_window(args.ct_window),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
