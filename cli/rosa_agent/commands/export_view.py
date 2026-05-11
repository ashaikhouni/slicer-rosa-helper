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

    # Surfaces — one material per hemisphere, alpha-blended so the
    # electrodes inside the cortex stay visible. Vertex colors from the
    # .annot override the material's base color when present.
    lh_mat = scene.add_material(
        "fs_lh_pial", (0.92, 0.88, 0.84, 0.35),
        metallic=0.0, roughness=0.85, double_sided=True, alpha_mode="BLEND",
    )
    rh_mat = scene.add_material(
        "fs_rh_pial", (0.84, 0.88, 0.92, 0.35),
        metallic=0.0, roughness=0.85, double_sided=True, alpha_mode="BLEND",
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

    # Electrode shaft material — dark insulation. One material reused
    # across all shanks so the JS viewer can recolor / hide one shank
    # without forking the material list.
    shaft_mat = scene.add_material(
        "electrode_shaft", (0.08, 0.08, 0.10, 1.0),
        metallic=0.0, roughness=0.95,
    )

    # Per-shank axis cache so each contact band can be oriented along
    # its trajectory.
    traj_axis: dict[str, np.ndarray] = {}
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

        scene.add_segment(
            name=f"shaft/{traj['name']}",
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
            name=f"contact/{contact['label']}",
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
  #app {{ display: grid; grid-template-columns: 1fr 360px; height: 100%; }}
  #canvas-host {{ position: relative; }}
  #canvas-host canvas {{ display: block; }}
  #toolbar {{ position: absolute; top: 10px; left: 10px; z-index: 5; display: flex; gap: 6px; }}
  #toolbar button {{ background: rgba(40,40,40,0.85); color: #eee; border: 1px solid #444; padding: 6px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
  #toolbar button:hover {{ background: rgba(70,70,70,0.95); }}
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

const sideWidth = 360;

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
  const w = window.innerWidth - sideWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
window.addEventListener("resize", resize);
resize();

(function loop() {{
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
}})();

// ---------- load GLB + index nodes ---------------------------------

const nodesByName = new Map();           // mesh.name -> THREE.Mesh
const shankNodes = new Map();            // shank id -> [mesh, ...]
const surfaceNodes = [];                 // FS pial meshes
const originalMaterials = new Map();     // mesh -> material
let gltfRoot = null;

const RED = new THREE.MeshStandardMaterial({{ color: 0xff2030, metalness: 0.75, roughness: 0.25 }});

const loader = new GLTFLoader();
loader.load("scene.glb", gltf => {{
  gltfRoot = gltf.scene;
  scene.add(gltfRoot);
  gltfRoot.traverse(obj => {{
    if (!obj.isMesh) return;
    nodesByName.set(obj.name, obj);
    originalMaterials.set(obj, obj.material);
    const extras = obj.userData || {{}};
    if (extras.kind === "freesurfer_surface") {{
      // Brain mesh: ensure it's transparent so electrodes show through.
      // glTF's alphaMode=BLEND already sets transparent=true; this is
      // belt-and-suspenders for renderers that don't honor it perfectly.
      if (obj.material) {{
        obj.material.transparent = true;
        obj.material.depthWrite = false;
      }}
      surfaceNodes.push(obj);
    }} else if (extras.shank) {{
      if (!shankNodes.has(extras.shank)) shankNodes.set(extras.shank, []);
      shankNodes.get(extras.shank).push(obj);
    }}
  }});
  fitToObject(gltfRoot);
}}, undefined, err => {{
  console.error("GLB load failed", err);
  document.getElementById("subject").textContent = "Failed to load scene.glb: " + err;
}});

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

function selectContact(label, shank) {{
  isolateShank(shank, /*toggle=*/false);

  // Restore previously selected contact's material
  if (selectedContact && originalMaterials.has(selectedContact)) {{
    selectedContact.material = originalMaterials.get(selectedContact);
  }}
  const node = nodesByName.get("contact/" + label);
  if (node) {{
    selectedContact = node;
    node.material = RED;
    // Snap the controls target onto the contact center and pull camera in close.
    const pos = node.getWorldPosition(new THREE.Vector3());
    controls.target.copy(pos);
    const dir = camera.position.clone().sub(pos).normalize();
    if (!isFinite(dir.x) || dir.lengthSq() < 1e-9) dir.set(0.7, 0.45, 0.8).normalize();
    camera.position.copy(pos).addScaledVector(dir, 35);
    controls.update();
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
    selectedContact.material = originalMaterials.get(selectedContact);
    selectedContact = null;
  }}
  document.querySelectorAll(".shank-card").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".contact-row").forEach(r => r.classList.remove("selected"));
}}

document.getElementById("btn-reset").addEventListener("click", showAll);
document.getElementById("btn-fit").addEventListener("click", () => {{ if (gltfRoot) fitToObject(gltfRoot); }});

fetch("scene_meta.json").then(r => r.json()).then(renderSidebar).catch(err => {{
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
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
