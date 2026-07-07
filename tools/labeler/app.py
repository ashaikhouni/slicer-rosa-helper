"""Flask app for SEEG GT labeling.

Routes:
    GET  /                                          → main UI
    GET  /api/subjects                              → subject list + GT status
    GET  /api/subject/<sid>                         → trajectories for one subject
    GET  /api/library                               → electrode model library
    GET  /api/trajectory/<sid>/<tid>                → algorithm-guessed landmarks + axis + extent
    GET  /api/slab.png?sid=...&tid=...              → slab PNG (regenerated on demand)
    POST /api/gt/<sid>/<tid>                        → upsert one trajectory's GT
    DELETE /api/gt/<sid>/<tid>                      → unmark a trajectory

GT persistence:
    <dataset>/gt/trajectories_gt.tsv  ← one row per trajectory (canonical)
    <dataset>/gt/contacts_gt.tsv      ← derived from the model + corrected tip + axis
                                        (regenerated atomically on every save)
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_file
import numpy as np
import SimpleITK as sitk

# Persistent state — lives in app config


GT_COLUMNS = (
    "subject", "trajectory", "electrode_model",
    "tip_arc_mm", "bone_inner_arc_mm",
    "bolt_start_arc_mm", "bolt_end_arc_mm",
    "notes",
    "axis_x", "axis_y", "axis_z",
    "entry_x", "entry_y", "entry_z",
    "tip_x", "tip_y", "tip_z",
    "reviewed_at",
)

CONTACT_COLUMNS = (
    "subject", "trajectory", "contact_index", "label",
    "x", "y", "z",
    "electrode_model", "source",
)


def create_app(dataset_dir: Path) -> Flask:
    """Build a Flask app rooted at ``dataset_dir``. The dataset is expected
    to have the snap_ct_dataset's layout (T{N}/masks + T{N}/snap subdirs)."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    # Disable static-file caching so JS/CSS edits show up on plain refresh
    # without needing Cmd+Shift+R every time.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.after_request
    def _no_cache(response):
        if response.headers.get("Content-Type", "").startswith(
            ("text/", "application/javascript", "application/json", "image/png")
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.config["DATASET_DIR"] = dataset_dir
    app.config["GT_DIR"] = dataset_dir / "gt"
    app.config["GT_DIR"].mkdir(exist_ok=True)
    app.config["LOCK"] = threading.Lock()
    app.config["SUBJECT_CACHE"] = {}  # lazy per-subject feature cache

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        from flask import render_template
        return render_template("index.html")

    @app.route("/api/subjects")
    def list_subjects():
        return jsonify(_collect_subjects(app))

    @app.route("/api/subject/<sid>")
    def get_subject(sid):
        return jsonify(_collect_subject_trajectories(app, sid))

    @app.route("/api/library")
    def get_library():
        return jsonify(_load_electrode_library())

    @app.route("/api/trajectory/<sid>/<tid>")
    def get_trajectory(sid, tid):
        data = _get_trajectory_data(app, sid, tid)
        if data is None:
            abort(404)
        return jsonify(data)

    @app.route("/api/slab.png")
    def get_slab():
        sid = request.args.get("sid")
        tid = request.args.get("tid")
        which = request.args.get("which", "axis_perp1")  # axis_perp1 | axis_perp2
        if not sid or not tid:
            abort(400, "sid + tid required")
        png_bytes = _render_slab_png(app, sid, tid, which)
        if png_bytes is None:
            abort(404)
        return send_file(io.BytesIO(png_bytes), mimetype="image/png")

    @app.route("/api/gt/<sid>/<tid>", methods=["POST"])
    def save_gt(sid, tid):
        payload = request.get_json(force=True) or {}
        row = _save_gt_row(app, sid, tid, payload)
        return jsonify({"ok": True, "row": row})

    @app.route("/api/gt/<sid>/<tid>", methods=["DELETE"])
    def delete_gt(sid, tid):
        _delete_gt_row(app, sid, tid)
        return jsonify({"ok": True})


# ---------------------------------------------------------------------
# Subject / trajectory discovery
# ---------------------------------------------------------------------


def _collect_subjects(app: Flask) -> list[dict]:
    dataset_dir = app.config["DATASET_DIR"]
    ct_dir = dataset_dir / "ct"
    if not ct_dir.exists():
        return []
    subjects = sorted(
        p.name.removesuffix("_ct.nii.gz") for p in ct_dir.glob("T*_ct.nii.gz")
    )
    subjects.sort(key=lambda s: int(s[1:]) if s[1:].isdigit() else 9999)

    gt = _load_gt_rows(app)
    out = []
    for sid in subjects:
        import_tsv = dataset_dir / "rosa_helper_import" / sid / "trajectories.tsv"
        if not import_tsv.is_file():
            continue
        with open(import_tsv) as f:
            n_traj = sum(1 for _ in f) - 1  # subtract header
        n_labeled = sum(1 for r in gt.values()
                        if r.get("subject") == sid and r.get("electrode_model"))
        out.append({
            "subject": sid,
            "n_trajectories": int(max(0, n_traj)),
            "n_labeled": int(n_labeled),
        })
    return out


def _collect_subject_trajectories(app: Flask, sid: str) -> dict:
    dataset_dir = app.config["DATASET_DIR"]
    import_tsv = dataset_dir / "rosa_helper_import" / sid / "trajectories.tsv"
    if not import_tsv.is_file():
        abort(404, f"no trajectories.tsv for {sid}")

    gt = _load_gt_rows(app)
    trajectories = []
    with open(import_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("name") or row.get("trajectory") or ""
            if not name:
                continue
            key = f"{sid}/{name}"
            gt_row = gt.get(key) or {}
            trajectories.append({
                "trajectory": name,
                "labeled": bool(gt_row.get("electrode_model")),
                "electrode_model": gt_row.get("electrode_model") or "",
                "tip_arc_mm": _f(gt_row.get("tip_arc_mm")),
                "bone_inner_arc_mm": _f(gt_row.get("bone_inner_arc_mm")),
                "bolt_start_arc_mm": _f(gt_row.get("bolt_start_arc_mm")),
                "bolt_end_arc_mm": _f(gt_row.get("bolt_end_arc_mm")),
                "notes": gt_row.get("notes") or "",
            })
    return {"subject": sid, "trajectories": trajectories}


def _f(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Electrode library
# ---------------------------------------------------------------------


def _load_electrode_library() -> dict:
    """Return the bundled electrode library + per-model offsets in a
    JSON-serialisable shape."""
    here = Path(__file__).resolve()
    lib_path = (
        here.parents[2]
        / "CommonLib/rosa_core/resources/electrodes/electrode_models.json"
    )
    if not lib_path.is_file():
        return {"models": []}
    payload = json.loads(lib_path.read_text())
    models = []
    for m in payload.get("models", []):
        offsets = m.get("contact_center_offsets_from_tip_mm") or []
        models.append({
            "id": m.get("id"),
            "vendor": m.get("vendor") or _guess_vendor(m.get("id") or ""),
            "contact_count": int(m.get("contact_count") or len(offsets)),
            "total_length_mm": float(m.get("total_exploration_length_mm") or 0.0),
            "offsets_from_tip_mm": offsets,
        })
    models.sort(key=lambda m: (m.get("vendor") or "", m.get("id") or ""))
    return {"models": models}


def _guess_vendor(model_id: str) -> str:
    if model_id.startswith("DIXI"):
        return "Dixi"
    if model_id.startswith("PMT"):
        return "PMT"
    if model_id.startswith("Medtronic"):
        return "Medtronic"
    if model_id.startswith("NeuroPace"):
        return "NeuroPace"
    return "Other"


# ---------------------------------------------------------------------
# Per-subject feature cache (canonical CT + mask + snap chains)
# ---------------------------------------------------------------------


def _get_subject_state(app: Flask, sid: str) -> dict | None:
    """Load + memoise the canonical CT, log_arr, metal_evidence, snap chains
    for one subject. First call may take ~3-5 s; subsequent calls are
    near-instant."""
    cache = app.config["SUBJECT_CACHE"]
    if sid in cache:
        return cache[sid]

    dataset_dir = app.config["DATASET_DIR"]
    masks_dir = dataset_dir / sid / "masks"
    canon_path = masks_dir / "ct_canon.nii.gz"
    mask_path = masks_dir / "brain_mask.nii.gz"
    if not (canon_path.is_file() and mask_path.is_file()):
        return None

    img_canon = sitk.ReadImage(str(canon_path))
    mask_img = sitk.ReadImage(str(mask_path))
    if mask_img.GetSize() != img_canon.GetSize():
        mask_img = sitk.Resample(mask_img, img_canon, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0.0)

    from shank_core.io import image_ijk_ras_matrices
    from rosa_detect.primitives.preprocessing import log_sigma, frangi_single

    i2r, r2i = image_ijk_ras_matrices(img_canon)
    ct_arr = sitk.GetArrayFromImage(img_canon).astype("float32", copy=False)
    log_arr = log_sigma(img_canon, sigma_mm=1.0)
    log_neg = np.clip(-log_arr, 0.0, None).astype("float32", copy=False)
    ct_clip = np.clip(ct_arr, -1024.0, 1500.0)
    metal_evidence = np.maximum(
        np.abs(log_arr) / 800.0,
        np.maximum(ct_clip, 0.0) / 2000.0,
    ).astype("float32", copy=False)
    intracranial = sitk.GetArrayFromImage(mask_img).astype(bool)

    # Run the snap pipeline for this subject's trajectories.
    from rosa_core.case_loader import find_ros_file  # noqa: F401  (unused but reused side-effects?)
    from rosa_core.seeded_fit import run_seeded_fit

    # Read planned trajectories from rosa_helper_import.
    import_tsv = dataset_dir / "rosa_helper_import" / sid / "trajectories.tsv"
    planned = _read_planned(import_tsv)
    chains = run_seeded_fit(
        planned,
        signal_vol=log_neg, threshold=300.0,
        ras_to_ijk=np.asarray(r2i, dtype=float),
        bolt_signal_vol=metal_evidence,
        log=lambda _m: None, label=sid,
    )
    chain_by_name = {p["name"]: c for p, c in zip(planned, chains)}

    # Native-grid CT for SLAB DISPLAY only — keeps fine-resolution detail
    # (typical 0.415 mm in-plane) which the 1 mm canonical grid loses. The
    # snap pipeline and feature compute use img_canon; only the slab
    # renderer pulls from the native CT.
    native_ct_path = app.config["DATASET_DIR"] / "ct" / f"{sid}_ct.nii.gz"
    img_native = None
    ct_arr_native = None
    ras_to_ijk_native = None
    if native_ct_path.is_file():
        try:
            img_native = sitk.ReadImage(str(native_ct_path))
            from shank_core.io import image_ijk_ras_matrices as _i2r2
            _i2r_native, _r2i_native = _i2r2(img_native)
            ct_arr_native = sitk.GetArrayFromImage(img_native).astype("float32", copy=False)
            ras_to_ijk_native = np.asarray(_r2i_native, dtype=float)
        except Exception:
            img_native = ct_arr_native = ras_to_ijk_native = None

    state = {
        "subject": sid,
        "img_canon": img_canon,
        "ct_arr": ct_arr,
        "log_arr": log_arr,
        "metal_evidence": metal_evidence,
        "intracranial_mask": intracranial,
        "ijk_to_ras": np.asarray(i2r, dtype=float),
        "ras_to_ijk": np.asarray(r2i, dtype=float),
        "ct_arr_native": ct_arr_native,
        "ras_to_ijk_native": ras_to_ijk_native,
        "planned": planned,
        "chains": chain_by_name,
    }
    cache[sid] = state
    return state


def _read_planned(tsv_path: Path) -> list[dict]:
    out = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("name") or row.get("trajectory") or ""
            try:
                start = (float(row["ex"]), float(row["ey"]), float(row["ez"]))
                end = (float(row["tx"]), float(row["ty"]), float(row["tz"]))
            except (KeyError, TypeError, ValueError):
                continue
            out.append({"name": name, "start": list(start), "end": list(end)})
    return out


def _get_trajectory_data(app: Flask, sid: str, tid: str) -> dict | None:
    state = _get_subject_state(app, sid)
    if state is None:
        return None
    chain = state["chains"].get(tid)

    # Even if snap failed, return enough info to label manually.
    planned = next((p for p in state["planned"] if p["name"] == tid), None)
    if planned is None:
        return None

    out = {
        "subject": sid,
        "trajectory": tid,
        "planned_start": planned["start"],
        "planned_end": planned["end"],
    }
    if chain is None:
        out["snap_failed"] = True
        return out

    from rosa_core.centerline import unit
    axis = unit(np.asarray(chain["axis"], dtype=float))
    entry = np.asarray(chain["entry_ras"], dtype=float)
    tip = np.asarray(chain["tip_ras"], dtype=float)
    kept_pts = np.asarray(chain["kept_pts"], dtype=float)

    # Project the snap peaks onto the axis to get arc coordinates.
    snap_arcs = ((kept_pts - entry) @ axis).tolist()
    tip_arc = float(np.linalg.norm(tip - entry))
    bolt_walked_mm = float(chain.get("bolt_walked_mm", 0.0))

    # Compute bone landmark from the bone-width profile (uses the same
    # algorithm as the picker).
    bone_arc = _compute_bone_arc(state, entry, tip, axis)

    out.update({
        "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
        "entry_ras": [float(entry[0]), float(entry[1]), float(entry[2])],
        "tip_ras": [float(tip[0]), float(tip[1]), float(tip[2])],
        "snap_arcs": [float(a) for a in snap_arcs],
        "snap_peaks_ras": [[float(p[0]), float(p[1]), float(p[2])] for p in kept_pts],
        "tip_arc": tip_arc,
        "bolt_walked_mm": bolt_walked_mm,
        "bone_arc_mm": None if bone_arc is None else float(bone_arc),
        "snap_failed": False,
    })
    # Slab extent so the frontend can map pixel→arc.
    slab_pad_entry = 6.0
    slab_pad_tip = 20.0
    slab_half_width = 6.0
    out["slab_extent"] = {
        "arc_lo": -slab_pad_entry,
        "arc_hi": tip_arc + slab_pad_tip,
        "perp_lo": -slab_half_width,
        "perp_hi": slab_half_width,
    }
    return out


def _compute_bone_arc(state: dict, entry, tip, axis) -> float | None:
    """Run the same bone-width landmark estimator the picker uses."""
    from rosa_core.centerline import (
        sample_trajectory_profile, collect_intracranial_hu_along_trajectory,
        otsu_threshold,
    )
    from rosa_core.seeded_fit import _bone_inner_edge_half_fall
    intra_hu = collect_intracranial_hu_along_trajectory(
        entry, tip, axis,
        ct_vol=state["ct_arr"], ras_to_ijk_mat=state["ras_to_ijk"],
        intracranial_mask_arr=state["intracranial_mask"],
    )
    metal_thr = otsu_threshold(intra_hu)
    arc_rel, prof = sample_trajectory_profile(
        entry, tip, axis,
        ct_vol=state["ct_arr"], ras_to_ijk_mat=state["ras_to_ijk"],
        metal_hu=metal_thr,
    )
    return _bone_inner_edge_half_fall(arc_rel, prof["bone_width_mm"])


# ---------------------------------------------------------------------
# Slab rendering (on-demand)
# ---------------------------------------------------------------------


def _render_slab_png(app: Flask, sid: str, tid: str, which: str) -> bytes | None:
    state = _get_subject_state(app, sid)
    if state is None:
        return None
    chain = state["chains"].get(tid)
    if chain is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rosa_core.centerline import unit, orthonormal_basis_for_axis
    from rosa_core.volume_sampling import sample_trilinear_batch

    SLAB_HALF_WIDTH_MM = 6.0
    SLAB_PAD_ENTRY_MM = 6.0
    SLAB_PAD_TIP_MM = 20.0
    # 0.25 mm step pairs well with the native 0.42 mm in-plane CT — avoids
    # aliasing artefacts that the 0.5 mm step had on the canonical grid.
    SLAB_STEP_MM = 0.25

    axis = unit(np.asarray(chain["axis"], dtype=float))
    entry = np.asarray(chain["entry_ras"], dtype=float)
    tip = np.asarray(chain["tip_ras"], dtype=float)
    p1, p2 = orthonormal_basis_for_axis(axis)
    perp = p1 if which == "axis_perp1" else p2
    L = float(np.linalg.norm(tip - entry))
    arc_lo = -SLAB_PAD_ENTRY_MM
    arc_hi = L + SLAB_PAD_TIP_MM
    n_along = int((arc_hi - arc_lo) / SLAB_STEP_MM) + 1
    n_perp = int((2 * SLAB_HALF_WIDTH_MM) / SLAB_STEP_MM) + 1
    arcs = np.linspace(arc_lo, arc_hi, n_along)
    perps = np.linspace(-SLAB_HALF_WIDTH_MM, SLAB_HALF_WIDTH_MM, n_perp)
    pts = (entry[None, None, :]
           + arcs[:, None, None] * axis[None, None, :]
           + perps[None, :, None] * perp[None, None, :])
    flat = pts.reshape(-1, 3)
    # Prefer native-grid CT (typically 0.42 mm in-plane) for slab display
    # — sharper contacts, fewer aliasing artefacts than the 1 mm canonical.
    # Falls back to canonical if native CT isn't available.
    ct_for_slab = state.get("ct_arr_native")
    r2i_for_slab = state.get("ras_to_ijk_native")
    if ct_for_slab is None or r2i_for_slab is None:
        ct_for_slab = state["ct_arr"]
        r2i_for_slab = state["ras_to_ijk"]
    vals = sample_trilinear_batch(
        ct_for_slab, r2i_for_slab, flat,
    ).reshape(n_along, n_perp)

    fig, ax = plt.subplots(figsize=(11, 2.4))
    extent = [arcs[0], arcs[-1], -SLAB_HALF_WIDTH_MM, SLAB_HALF_WIDTH_MM]
    ax.imshow(vals.T, cmap="gray", extent=extent, origin="lower",
              aspect="equal", vmin=-200, vmax=1500)
    ax.set_xlabel("along axis (mm)")
    ax.set_ylabel("perp (mm)")
    ax.set_title(f"{sid} / {tid}  ({which})")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------
# GT persistence — TSV-backed, append-or-update per trajectory.
# ---------------------------------------------------------------------


def _gt_tsv_path(app: Flask) -> Path:
    return app.config["GT_DIR"] / "trajectories_gt.tsv"


def _contacts_tsv_path(app: Flask) -> Path:
    return app.config["GT_DIR"] / "contacts_gt.tsv"


def _load_gt_rows(app: Flask) -> dict[str, dict[str, Any]]:
    p = _gt_tsv_path(app)
    if not p.is_file():
        return {}
    rows = {}
    with open(p) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            sid = r.get("subject") or ""
            tid = r.get("trajectory") or ""
            if sid and tid:
                rows[f"{sid}/{tid}"] = dict(r)
    return rows


def _write_gt_rows(app: Flask, rows: dict[str, dict[str, Any]]) -> None:
    p = _gt_tsv_path(app)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tsv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GT_COLUMNS), delimiter="\t")
        writer.writeheader()
        for key in sorted(rows.keys()):
            r = rows[key]
            writer.writerow({col: r.get(col, "") for col in GT_COLUMNS})
    tmp.replace(p)


def _save_gt_row(app: Flask, sid: str, tid: str, payload: dict) -> dict:
    """Upsert one trajectory's GT, then regenerate contacts_gt.tsv from
    the model + corrected tip + axis. Holds a single mutex so concurrent
    POSTs can't corrupt the TSV.
    """
    state = _get_subject_state(app, sid)
    if state is None:
        abort(404, f"subject {sid} not loadable")
    chain = state["chains"].get(tid)
    planned = next((p for p in state["planned"] if p["name"] == tid), None)
    if planned is None:
        abort(404, f"trajectory {sid}/{tid} not in plan")

    # Resolve axis + entry: use the snap chain's if available, else the
    # planned start/end (so labels can still be entered when snap failed).
    if chain is not None:
        from rosa_core.centerline import unit
        axis = unit(np.asarray(chain["axis"], dtype=float))
        entry = np.asarray(chain["entry_ras"], dtype=float)
        tip_alg = np.asarray(chain["tip_ras"], dtype=float)
    else:
        from rosa_core.centerline import unit
        start = np.asarray(planned["start"], dtype=float)
        end = np.asarray(planned["end"], dtype=float)
        axis = unit(end - start)
        entry = start
        tip_alg = end

    # Corrected tip arc → tip_ras.
    tip_arc = payload.get("tip_arc_mm")
    if tip_arc is None or tip_arc == "":
        # No correction = use the algorithm's tip.
        tip_arc_f = float(np.linalg.norm(tip_alg - entry))
    else:
        tip_arc_f = float(tip_arc)
    tip_ras = entry + tip_arc_f * axis

    row = {
        "subject": sid,
        "trajectory": tid,
        "electrode_model": str(payload.get("electrode_model") or ""),
        "tip_arc_mm": _fmt(tip_arc_f),
        "bone_inner_arc_mm": _fmt(payload.get("bone_inner_arc_mm")),
        "bolt_start_arc_mm": _fmt(payload.get("bolt_start_arc_mm")),
        "bolt_end_arc_mm": _fmt(payload.get("bolt_end_arc_mm")),
        "notes": str(payload.get("notes") or ""),
        "axis_x": _fmt(axis[0]),
        "axis_y": _fmt(axis[1]),
        "axis_z": _fmt(axis[2]),
        "entry_x": _fmt(entry[0]),
        "entry_y": _fmt(entry[1]),
        "entry_z": _fmt(entry[2]),
        "tip_x": _fmt(tip_ras[0]),
        "tip_y": _fmt(tip_ras[1]),
        "tip_z": _fmt(tip_ras[2]),
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with app.config["LOCK"]:
        rows = _load_gt_rows(app)
        rows[f"{sid}/{tid}"] = row
        _write_gt_rows(app, rows)
        _regenerate_contacts_tsv(app, rows)
    return row


def _delete_gt_row(app: Flask, sid: str, tid: str) -> None:
    with app.config["LOCK"]:
        rows = _load_gt_rows(app)
        rows.pop(f"{sid}/{tid}", None)
        _write_gt_rows(app, rows)
        _regenerate_contacts_tsv(app, rows)


def _polyline_walk(snap_peaks_ras: np.ndarray, tip_ras: np.ndarray,
                    offsets_from_deepest: np.ndarray) -> np.ndarray:
    """Walk along the snap chain's 3D polyline from the deepest peak (or the
    user-corrected tip) through interpolated positions at each library
    offset.

    Returns a (N, 3) array of contact RAS positions, where N = len(offsets).
    Falls back to straight-line walk along the deepest two peaks' direction
    if snap_peaks_ras has < 2 peaks.

    The snap chain captures the actual electrode geometry (small deviations
    from a single PCA axis). Walking along it places contacts where the
    real contacts sit even if the snap-fit PCA axis is slightly tilted
    relative to the true shank.
    """
    peaks = np.asarray(snap_peaks_ras, dtype=float)
    if len(peaks) < 2:
        # No usable polyline — fall back to placing all contacts at tip.
        return np.tile(np.asarray(tip_ras, dtype=float), (len(offsets_from_deepest), 1))

    # Cumulative polyline arc from deepest peak, walking shallower.
    deepest = peaks[-1]
    cum_arc = [0.0]
    cum_pts = [deepest]
    for i in range(len(peaks) - 2, -1, -1):
        d = float(np.linalg.norm(peaks[i] - peaks[i + 1]))
        cum_arc.append(cum_arc[-1] + d)
        cum_pts.append(peaks[i])
    cum_arc = np.asarray(cum_arc)
    cum_pts = np.asarray(cum_pts)

    # Shift the polyline so its deepest point lands at the user's tip.
    shift = np.asarray(tip_ras, dtype=float) - deepest

    # Direction at the shallowest end for extrapolation past the last peak.
    head_dir = cum_pts[-1] - cum_pts[max(0, len(cum_pts) - 2)]
    head_len = float(np.linalg.norm(head_dir))
    if head_len > 1e-6:
        head_dir = head_dir / head_len
    else:
        head_dir = np.zeros(3)

    out = np.empty((len(offsets_from_deepest), 3), dtype=float)
    for i, lib_off in enumerate(offsets_from_deepest):
        # Find the polyline segment containing lib_off.
        k = int(np.searchsorted(cum_arc, lib_off, side="right") - 1)
        k = max(0, min(k, len(cum_arc) - 1))
        if k >= len(cum_arc) - 1:
            past = lib_off - cum_arc[-1]
            pt = cum_pts[-1] + past * head_dir
        else:
            seg_lo, seg_hi = cum_arc[k], cum_arc[k + 1]
            seg_len = seg_hi - seg_lo
            t = (lib_off - seg_lo) / seg_len if seg_len > 0 else 0.0
            pt = cum_pts[k] + t * (cum_pts[k + 1] - cum_pts[k])
        out[i] = pt + shift
    return out


def _regenerate_contacts_tsv(app: Flask, rows: dict[str, dict]) -> None:
    """Walk along each saved trajectory's snap-chain polyline to place
    contacts at their actual 3D positions (not along the straight slab axis).
    Requires the subject's snap state to be cached; if not yet loaded, the
    snap chain is computed on demand via ``_get_subject_state``.

    Contacts whose ``source`` starts with ``gt_file`` (e.g. T22 recovered
    directly from a ROSA-Helper GT export because snap failed) are
    PRESERVED verbatim — never recomputed. The trajectory keys they cover
    are skipped in the snap-polyline derivation below.
    """
    library = {m["id"]: m for m in _load_electrode_library().get("models", [])}

    # Preserve any externally-sourced (gt_file) contacts across regen.
    preserved = []
    preserved_keys = set()
    cp = _contacts_tsv_path(app)
    if cp.is_file():
        with open(cp) as f:
            for cr in csv.DictReader(f, delimiter="\t"):
                if str(cr.get("source") or "").startswith("gt_file"):
                    preserved.append(dict(cr))
                    preserved_keys.add((cr.get("subject"), cr.get("trajectory")))

    out_rows = list(preserved)
    for key, r in rows.items():
        # Skip trajectories whose contacts are externally sourced.
        if (r.get("subject"), r.get("trajectory")) in preserved_keys:
            continue
        model_id = r.get("electrode_model")
        if not model_id or model_id not in library:
            continue
        model = library[model_id]
        offsets = model.get("offsets_from_tip_mm") or []
        if not offsets:
            continue
        try:
            # tip = PHYSICAL ELECTRODE TIP (deep edge of bright metal),
            # NOT the deepest contact center. Contact i sits offsets[i]
            # bolt-ward of this point.
            physical_tip = np.array(
                [float(r["tip_x"]), float(r["tip_y"]), float(r["tip_z"])],
                dtype=float,
            )
        except (KeyError, ValueError):
            continue

        offsets_arr = np.asarray(offsets, dtype=float)
        axis = np.array(
            [float(r["axis_x"]), float(r["axis_y"]), float(r["axis_z"])],
            dtype=float,
        )
        # Deepest contact placement. An alignment probe over 103 clean-
        # contact GT trajectories found the auto-derived contacts sat a
        # systematic ~offsets[0] (1 mm) too deep — the saved tip click
        # runs ~offsets[0] deeper than the model's physical-tip reference.
        # So the deepest contact is 2*offsets[0] bolt-ward (-axis) of the
        # saved tip; the correction zeroes the bias (median -1.0 -> -0.1
        # mm). Walk the polyline from there using offsets[i] - offsets[0].
        deepest_contact = physical_tip - 2.0 * float(offsets_arr[0]) * axis
        offsets_from_deepest = offsets_arr - float(offsets_arr[0])

        # Pull snap chain from the cached subject state for the polyline.
        sid = r.get("subject"); tid = r.get("trajectory")
        snap_peaks = None
        try:
            state = _get_subject_state(app, sid)
            if state is not None:
                chain = state["chains"].get(tid)
                if chain is not None:
                    snap_peaks = np.asarray(chain["kept_pts"], dtype=float)
        except Exception:
            snap_peaks = None
        if snap_peaks is None or len(snap_peaks) < 2:
            # No polyline available — fall back to straight-line walk along
            # the saved axis from the deepest contact.
            direction = -axis
            ras_arr = deepest_contact[None, :] + offsets_from_deepest[:, None] * direction[None, :]
        else:
            ras_arr = _polyline_walk(snap_peaks, deepest_contact, offsets_from_deepest)

        for i, ras in enumerate(ras_arr, start=1):
            out_rows.append({
                "subject": r["subject"],
                "trajectory": r["trajectory"],
                "contact_index": str(i),
                "label": f"{r['trajectory']}{i}",
                "x": _fmt(ras[0]),
                "y": _fmt(ras[1]),
                "z": _fmt(ras[2]),
                "electrode_model": model_id,
                "source": "gt_derived_polyline",
            })

    p = _contacts_tsv_path(app)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tsv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CONTACT_COLUMNS), delimiter="\t")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({col: row.get(col, "") for col in CONTACT_COLUMNS})
    tmp.replace(p)


def _fmt(v: Any, *, precision: int = 4) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{precision}f}"
    except (TypeError, ValueError):
        return ""
