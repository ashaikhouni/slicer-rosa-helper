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


def _register_fs_to_ct(t1_path: Path, ct_path: Path) -> np.ndarray:
    """Rigid Mattes-MI register FS T1 to CT; return the FS-RAS -> CT-RAS 4×4."""
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
    return result.moving_to_fixed_ras_4x4


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
    contact_radius_mm: float,
    trajectory_radius_mm: float,
) -> tuple[GLBScene, list[dict[str, Any]]]:
    """Assemble the GLB scene and return (scene, contact-metadata-rows).

    ``contact-metadata-rows`` is the listing the HTML viewer hydrates into
    a sidebar so a click on a node can show its anatomical label.
    """
    scene = GLBScene()

    # Surfaces — one material per hemisphere, semi-transparent so contacts
    # show through. Vertex colors from the .annot override the material's
    # base color when present.
    lh_mat = scene.add_material(
        "fs_lh_pial", (0.92, 0.88, 0.84, 0.35),
        metallic=0.0, roughness=0.85, double_sided=True,
    )
    rh_mat = scene.add_material(
        "fs_rh_pial", (0.84, 0.88, 0.92, 0.35),
        metallic=0.0, roughness=0.85, double_sided=True,
    )
    for surf in surfaces:
        mat = lh_mat if surf.hemi == "lh" else rh_mat
        scene.add_surface(
            name=f"fs/{surf.name}",
            positions=surf.vertices_ras,
            faces=surf.faces,
            material_index=mat,
            vertex_colors_rgba=surf.vertex_colors_rgba,
            extras={
                "kind": "freesurfer_surface",
                "hemi": surf.hemi,
                "surface": surf.kind,
                "annotation": surf.annotation_name,
                "n_vertices": int(surf.vertices_ras.shape[0]),
            },
        )

    # Trajectories — one cylinder per shank, colored by confidence band.
    band_materials = {
        band: scene.add_material(f"trajectory_{band or 'unknown'}", color, metallic=0.4, roughness=0.4)
        for band, color in _TRAJ_BAND_COLORS.items()
    }
    for traj in trajectories:
        band = (traj.get("confidence_label") or "").strip().lower()
        mat = band_materials.get(band, band_materials[""])
        scene.add_segment(
            name=f"traj/{traj['name']}",
            p0=traj["start"], p1=traj["end"],
            radius=trajectory_radius_mm,
            material_index=mat,
            extras={
                "kind": "trajectory",
                "name": traj["name"],
                "confidence": traj.get("confidence", ""),
                "confidence_label": traj.get("confidence_label", ""),
                "electrode_model": traj.get("electrode_model", ""),
                "bolt_source": traj.get("bolt_source", ""),
                "length_mm": traj.get("length_mm", ""),
            },
        )

    # Contacts — one sphere per contact, colored by FS label when available.
    contact_meta: list[dict[str, Any]] = []
    color_to_material: dict[tuple[float, float, float, float], int] = {}
    fallback_mat = scene.add_material("contact_unlabeled", (0.95, 0.95, 0.95, 1.0),
                                      metallic=0.7, roughness=0.3)
    for contact in contacts:
        label_row = contact_labels.get(contact["label"], {})
        # Prefer the FS-specific column; fall back to whichever source was closest.
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

        extras = {
            "kind": "contact",
            "trajectory": contact["trajectory"],
            "label": contact["label"],
            "contact_index": contact["contact_index"],
            "electrode_model": contact["electrode_model"],
            "peak_detected": bool(contact["peak_detected"]),
            "freesurfer_label": fs_label_name,
            "thomas_label": thomas_label,
            "wm_label": wm_label,
            "distance_mm": fs_distance,
        }
        scene.add_sphere(
            name=f"contact/{contact['label']}",
            center=contact["position"],
            radius=contact_radius_mm,
            material_index=mat,
            extras=extras,
        )
        contact_meta.append({
            "label": contact["label"],
            "trajectory": contact["trajectory"],
            "contact_index": contact["contact_index"],
            "position": list(contact["position"]),
            "electrode_model": contact["electrode_model"],
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
  html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #1a1a1a; color: #eee; }}
  #app {{ display: grid; grid-template-columns: 1fr 320px; height: 100%; }}
  model-viewer {{ width: 100%; height: 100%; background: #1a1a1a; --poster-color: transparent; }}
  #side {{ overflow-y: auto; padding: 12px 14px; border-left: 1px solid #333; font-size: 13px; }}
  #side h2 {{ font-size: 14px; margin: 16px 0 6px; color: #ccc; letter-spacing: 0.04em; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 3px 6px; border-bottom: 1px solid #2b2b2b; vertical-align: top; }}
  th {{ color: #999; font-weight: 500; font-size: 11px; }}
  tr.contact {{ cursor: pointer; }}
  tr.contact:hover {{ background: #2a2a2a; }}
  tr.highlight {{ background: #3a4a30; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; background: #333; color: #ddd; margin-right: 4px; }}
  .band-high {{ background: #225a25; }}
  .band-medium {{ background: #6a5418; }}
  .band-low {{ background: #6a2515; }}
  .small {{ color: #888; font-size: 11px; }}
</style>
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
</head>
<body>
<div id="app">
  <model-viewer id="viewer" src="scene.glb" camera-controls touch-action="pan-y" exposure="1" shadow-intensity="0" interaction-prompt="none"></model-viewer>
  <div id="side">
    <h2>Subject</h2>
    <div class="small" id="subject"></div>
    <h2>Trajectories</h2>
    <table id="traj-table"><thead><tr><th>Name</th><th>Band</th><th>Model</th></tr></thead><tbody></tbody></table>
    <h2>Contacts</h2>
    <table id="contact-table"><thead><tr><th>Label</th><th>Region</th></tr></thead><tbody></tbody></table>
  </div>
</div>
<script>
async function main() {{
  const meta = await fetch("scene_meta.json").then(r => r.json());
  document.getElementById("subject").textContent = meta.subject_label || "(unnamed)";
  const trajTbody = document.querySelector("#traj-table tbody");
  for (const t of meta.trajectories) {{
    const tr = document.createElement("tr");
    const band = (t.confidence_label || "").toLowerCase();
    tr.innerHTML = `<td>${{t.name}}</td><td><span class="badge band-${{band}}">${{band || "?"}}</span></td><td class="small">${{t.electrode_model || ""}}</td>`;
    trajTbody.appendChild(tr);
  }}
  const cTbody = document.querySelector("#contact-table tbody");
  for (const c of meta.contacts) {{
    const tr = document.createElement("tr");
    tr.className = "contact";
    tr.dataset.label = c.label;
    tr.innerHTML = `<td>${{c.label}}</td><td>${{c.freesurfer_label || c.thomas_label || c.wm_label || "<span class=small>—</span>"}}</td>`;
    tr.addEventListener("click", () => {{
      document.querySelectorAll("tr.contact").forEach(r => r.classList.remove("highlight"));
      tr.classList.add("highlight");
    }});
    cTbody.appendChild(tr);
  }}
}}
main().catch(err => {{ console.error(err); document.getElementById("subject").textContent = "Failed to load metadata: " + err.message; }});
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
    contact_radius_mm: float = 0.7,
    trajectory_radius_mm: float = 0.5,
    skip_registration: bool = False,
    output_frame: str = "ct",
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
    surfaces = []
    if fs.available_surfaces:
        fs_to_ct_4x4 = _register_fs_to_ct(fs.t1_path, ct_path)
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
    else:
        _stderr("[view] no FreeSurfer surfaces under surf/; skipping brain mesh")

    # 4. Build the scene.
    scene, contact_meta = _build_scene(
        surfaces=surfaces,
        trajectories=trajectories,
        contacts=contacts,
        contact_labels=contact_labels,
        lut_index=lut_index,
        contact_radius_mm=contact_radius_mm,
        trajectory_radius_mm=trajectory_radius_mm,
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
    parser.add_argument("--contact-radius-mm", type=float, default=0.7)
    parser.add_argument("--trajectory-radius-mm", type=float, default=0.5)
    parser.add_argument(
        "--skip-registration", action="store_true",
        help="When --ct is supplied alongside a ROSA folder, assume the external CT "
             "is already aligned to the ROSA reference (no rigid CT<->ROSA pass).",
    )
    parser.add_argument(
        "--output-frame", default="ct", choices=("ct", "rosa"),
        help="Frame for trajectory/contact coordinates in the output TSVs and GLB.",
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
        contact_radius_mm=float(args.contact_radius_mm),
        trajectory_radius_mm=float(args.trajectory_radius_mm),
        skip_registration=bool(args.skip_registration),
        output_frame=args.output_frame,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
