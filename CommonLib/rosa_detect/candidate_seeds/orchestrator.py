"""Two-stage SEEG shank detection orchestrator.

Wires the candidate-seeds package together:

  preprocess (hull, intracranial, distance, LoG, Frangi)
    -> stage 1 walker (blob extraction + pitch chaining + arbitration
       + extension + dedup)
    -> bolt extraction + per-trajectory anchoring (with synth fallback)
    -> wire-class extension (PCA-fit unmatched bolt CCs)
    -> trajectory dedup
    -> deep-end refinement + crossing-tip retreat
    -> per-trajectory electrode-model suggestion (PaCER -> walker
       signature -> length-only dispatcher)
    -> continuous physical-evidence confidence score per emission.

Strategy-scoped walker bounds (span / length / gap) are computed once
at entry from the caller's ``pitch_strategy`` via
``bounds_for_strategy`` and threaded explicitly to every stage that
needs them as a :class:`WalkerBounds` record. No module-global
mutation, no decorator magic.
"""
from __future__ import annotations

from typing import Any

import numpy as np


from ..primitives.bolt_anchor import (
    BOLT_HULL_PROXIMITY_MM,
    METAL_BOLT_THRESHOLD,
    anchor_trajectory_to_bolt,
    extract_bolt_candidates,
)
from ..primitives.geometry import (
    kji_to_ras_fn_from_matrix,
    orient_shallow_to_deep,
)
from ..primitives.preprocessing import (
    FRANGI_STAGE1_SIGMA,
    LOG_SIGMA_MM,
    build_masks,
    frangi_single,
    log_sigma,
    prepare_volume,
)
from .blob_extraction import extract_blobs
from .confidence_score import compute_trajectory_score
from .constants import (
    DEEP_TIP_MIN_MM,
    DEEP_TIP_SHORT_MAX_AVG_PITCH_MM,
    FRANGI_LINE_MIN_MEDIAN,
    LOG_BLOB_MAX_VOXELS,
    LOG_BLOB_THRESHOLD,
    MIN_BLOBS_PER_LINE,
    PITCH_MM,
    WIRE_CLASS_MIN_DEPTH_MM,
    WIRE_CLASS_MIN_ELONGATION,
    WIRE_CLASS_MIN_SPAN_MM,
    WIRE_CLASS_MIN_VOXELS,
)
from .crossing_tips import retreat_crossing_tips
from .deep_end_refine import (
    clip_deep_end_to_inliers,
    refine_deep_end_via_axis_log,
)
from .dedup import dedup_trajectories
from .frangi_sampling import frangi_along_line_stats
from .metal_evidence import (
    compute_metal_evidence_volume,
    frac_strong_metal_along_line,
)
from .pitch_library import bounds_for_strategy
from .pitch_resolution import (
    detect_pitch_from_intracranial_blobs,
    resolve_pitches_for_strategy,
)
from .stage1_runner import run_stage1
from .synth_anchor import axis_to_skull_synth


def run_two_stage_detection(img, ijk_to_ras_mat, ras_to_ijk_mat,
                            return_features=False, progress_logger=None,
                            pitch_strategy=None,
                            pitches_mm=None):
    """Run the full SEEG shank detector on a SITK image.

    Args:
        img: SimpleITK image (raw CT).
        ijk_to_ras_mat: 4x4 numpy matrix.
        ras_to_ijk_mat: 4x4 numpy matrix.
        return_features: if True, return (trajectories, feature_arrays)
            where feature_arrays is a dict with the LoG, Frangi, hull
            head-distance, intracranial and hull arrays (KJI-order).
        progress_logger: optional callable(message: str) invoked at each
            major checkpoint. The Slicer widget passes a callback that
            updates the status panel and runs `app.processEvents()` so
            the UI doesn't appear hung during the ~10–20 s detection.

    Returns:
        list[dict] or (list[dict], dict): trajectories list (always) and
        optionally a feature_arrays dict for debugging / visualization.
    """
    # Strategy-scoped walker bounds — computed once here, threaded
    # explicitly to every stage that needs them.
    bounds = bounds_for_strategy(pitch_strategy)
    MIN_POST_ANCHOR_LEN_MM = bounds.min_post_anchor_len_mm
    MAX_POST_ANCHOR_LEN_MM = bounds.max_post_anchor_len_mm

    def _log(msg):
        if progress_logger is not None:
            try:
                progress_logger(msg)
            except Exception:
                pass

    import SimpleITK as sitk
    # Canonicalize spacing + anti-alias + HU clamp. Shared with
    # ``guided_fit_engine.compute_features`` so both paths see the
    # identical preprocessed volume — any drift here is a P0 parity
    # bug per ``feedback_cli_slicer_parity.md``.
    img, ijk_to_ras_mat, ras_to_ijk_mat = prepare_volume(
        img, ijk_to_ras_mat, ras_to_ijk_mat,
    )
    _log("preprocessing: hull, head-distance, intracranial mask…")
    ct_arr_kji = sitk.GetArrayFromImage(img).astype(np.float32)
    # Input fingerprint — lets us compare Slicer vs CLI runs byte-for-byte.
    # If Slicer returns a different trajectory count, the most common
    # causes are (a) HU rescaling (NIfTI scl_slope/scl_inter applied
    # differently) and (b) IJK→RAS matrix mismatch; this trace exposes both.
    try:
        _sp = img.GetSpacing()
        _dg = [ijk_to_ras_mat[i, i] for i in range(3)]
        _org = [ijk_to_ras_mat[i, 3] for i in range(3)]
        _log(
            f"input fingerprint: shape={ct_arr_kji.shape} "
            f"HU[min/mean/max]={ct_arr_kji.min():.1f}/"
            f"{ct_arr_kji.mean():.1f}/{ct_arr_kji.max():.1f} "
            f"spacing={tuple(f'{s:.4f}' for s in _sp)} "
            f"ijk2ras_diag={tuple(f'{d:+.4f}' for d in _dg)} "
            f"origin={tuple(f'{o:+.2f}' for o in _org)}"
        )
    except Exception:
        pass
    hull, intracranial, dist_arr = build_masks(img)
    _log("preprocessing: LoG σ=1…")
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    _log("preprocessing: Frangi σ=1…")
    frangi_s1 = frangi_single(img, sigma=FRANGI_STAGE1_SIGMA)
    kji_to_ras = kji_to_ras_fn_from_matrix(ijk_to_ras_mat)

    # Resolve walker pitches from the caller's strategy. Explicit
    # ``pitches_mm`` override takes precedence (used by unit tests and
    # power users). Otherwise fall back to strategy lookup — "auto"
    # auto-detects pitch from the intracranial blob cloud here so
    # stage-1 sees the right pitches from its first pass.
    if pitches_mm is not None and len(tuple(pitches_mm)) > 0:
        resolved_pitches = tuple(float(p) for p in pitches_mm)
    elif pitch_strategy is not None:
        strat_key = str(pitch_strategy).lower()
        if strat_key == "auto":
            _log("auto-detect pitch: extracting blobs…")
            _blobs_preview = extract_blobs(log1, threshold=LOG_BLOB_THRESHOLD)
            _pts_preview = (
                np.array([kji_to_ras(b["kji"]) for b in _blobs_preview])
                if _blobs_preview
                else np.empty((0, 3), dtype=float)
            )
            if _pts_preview.shape[0] > 0:
                _n_vox_preview = np.array(
                    [b["n_vox"] for b in _blobs_preview], dtype=int,
                )
                _pts_c_preview = _pts_preview[_n_vox_preview <= LOG_BLOB_MAX_VOXELS]
            else:
                _pts_c_preview = _pts_preview
            _raw_detected = detect_pitch_from_intracranial_blobs(
                _pts_c_preview, dist_arr, ras_to_ijk_mat,
            ) if _pts_c_preview.shape[0] > 0 else []
            resolved_pitches = resolve_pitches_for_strategy(
                "auto",
                pts_c=_pts_c_preview,
                dist_arr=dist_arr,
                ras_to_ijk_mat=ras_to_ijk_mat,
            )
            if _raw_detected:
                _log(
                    f"auto-detect pitch: raw={[f'{p:.2f}' for p in _raw_detected]} "
                    f"→ snapped={[f'{p:.2f}' for p in resolved_pitches]} mm"
                )
            else:
                _log(
                    f"auto-detect pitch: using {[f'{p:.2f}' for p in resolved_pitches]} mm (fallback)"
                )
        else:
            resolved_pitches = resolve_pitches_for_strategy(strat_key)
    else:
        resolved_pitches = (PITCH_MM,)

    _log(
        f"stage 1: blob-pitch walker — pitches={[f'{p:.2f}' for p in resolved_pitches]} mm"
    )
    stage1_lines, pts_blobs = run_stage1(
        log1, kji_to_ras, dist_arr, ras_to_ijk_mat,
        pitches_mm=resolved_pitches,
        frangi_arr=frangi_s1,
        bounds=bounds,
    )
    _log(f"stage 1: {len(stage1_lines)} candidate lines after walk + arbitrate + extend")
    # Attach inlier RAS coords AND LoG amplitudes to each stage-1 line
    # so post-anchor refinement can clip the deep end to the last
    # STRONG real contact (weak/noisy blobs added by extension don't
    # count as legit deep endpoints).
    # Re-derive LoG amplitudes at each contact-sized blob position —
    # pts_blobs is already the contact-filtered cloud, so indexing
    # matches line["inlier_idx"]. If this fails (shape mismatch, etc.)
    # the deep-end strong-contact clipping silently degrades to "no
    # amplitude data" — log it loudly so a regression can't hide here.
    try:
        K, J, I = log1.shape
        h_all = np.concatenate([pts_blobs, np.ones((pts_blobs.shape[0], 1))], axis=1)
        ijk_all = (ras_to_ijk_mat @ h_all.T).T[:, :3]
        ii = np.clip(np.round(ijk_all[:, 0]).astype(int), 0, I - 1)
        jj = np.clip(np.round(ijk_all[:, 1]).astype(int), 0, J - 1)
        kk = np.clip(np.round(ijk_all[:, 2]).astype(int), 0, K - 1)
        blob_amps = np.abs(log1[kk, jj, ii]).astype(np.float32)
    except Exception as exc:
        _log(
            f"warning: blob_amps re-derivation failed ({exc}); "
            f"deep-end strong-contact clipping disabled for all stage-1 lines"
        )
        blob_amps = None
    for l in stage1_lines:
        try:
            l["inlier_ras"] = np.asarray(pts_blobs[l["inlier_idx"]], dtype=float)
            if blob_amps is not None:
                l["inlier_amps"] = np.asarray(blob_amps[l["inlier_idx"]], dtype=float)
        except Exception as exc:
            _log(
                f"warning: line inlier_idx → blob lookup failed "
                f"({exc}); inlier_ras / inlier_amps cleared for one line"
            )
            l["inlier_ras"] = None
            l["inlier_amps"] = None

    _log("bolt extraction (unified metal-evidence)…")
    metal_evidence_vol = compute_metal_evidence_volume(log1, ct_arr_kji)
    bolts, bolt_mask = extract_bolt_candidates(
        log1, dist_arr, ijk_to_ras_mat, img.GetSpacing(),
        ct_arr=metal_evidence_vol, hu_threshold=METAL_BOLT_THRESHOLD,
        hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
    )
    _log(f"bolt extraction: {len(bolts)} bolt candidates "
         f"(metal_evidence ≥ {METAL_BOLT_THRESHOLD:.2f}, "
         f"hull_prox ≤ {BOLT_HULL_PROXIMITY_MM:.1f} mm)")

    def _assemble(l):
        rec = dict(
            start_ras=np.asarray(l["start_ras"], dtype=float),
            end_ras=np.asarray(l["end_ras"], dtype=float),
            shallow_endpoint_name="start",
            deep_endpoint_name="end",
            length_mm=float(l.get("span_mm", l.get("length_mm", 0.0))),
            n_inliers=int(l.get("n_blobs", l.get("n_inliers", 0))),
            dist_min_mm=float(l.get("dist_min_mm", float("nan"))),
            dist_max_mm=float(l.get("dist_max_mm", float("nan"))),
            dist_mean_mm=float(l.get("dist_mean_mm", float("nan"))),
            amp_sum=float(l.get("amp_sum", 0.0)),
        )
        # Carry the walker's blob-inlier set through anchoring so
        # ``dedup_trajectories`` can apply the inlier-subset rule
        # (drop only when no blob is orphaned).
        inlier_idx = l.get("inlier_idx")
        if inlier_idx is not None:
            rec["inlier_idx"] = list(int(b) for b in inlier_idx)
        # Preserve the pre-anchor inlier span — the actual distance
        # between the shallowest and deepest detected contacts. The
        # post-anchor ``length_mm`` overwrites this with the
        # bolt-tip → deep-tip length, so downstream classifiers
        # need this field to see the true contact span.
        rec["contact_span_mm"] = float(l.get("span_mm", 0.0))
        rec["original_span_mm"] = float(
            l.get("original_span_mm", l.get("span_mm", 0.0))
        )
        rec["original_median_pitch_mm"] = float(l.get(
            "original_median_pitch_mm",
            (float(l.get("original_span_mm", l.get("span_mm", 0.0)))
             / max(1, int(l.get("n_blobs", 2)) - 1)),
        ))
        if "frangi_mean_mm" in l:
            rec["frangi_mean_mm"] = float(l["frangi_mean_mm"])
        if "frangi_median_mm" in l:
            rec["frangi_median_mm"] = float(l["frangi_median_mm"])
        inlier_ras = l.get("inlier_ras")
        if inlier_ras is not None:
            rec["inlier_ras"] = np.asarray(inlier_ras, dtype=float)
        inlier_amps = l.get("inlier_amps")
        if inlier_amps is not None:
            rec["inlier_amps"] = np.asarray(inlier_amps, dtype=float)
        return rec

    def _is_genuine_seeg_line(rec):
        """Strong-SEEG-chain gate for the synth fallback.

        A real SEEG electrode's walker pre-extension line has Dixi/PMT
        pitch (3-5 mm). Uses the pre-extend MEDIAN pitch — robust to
        one walker-absorbed outlier that would skew a mean-based
        statistic past the 7 mm cap even when most inliers sit on a
        regular chain. Falls back to min(span_pre, span_post)/(n-1)
        when median isn't available.
        """
        n = int(rec.get("n_inliers", 0))
        if n < MIN_BLOBS_PER_LINE:
            return False
        dist_max = float(rec.get("dist_max_mm", 0.0))
        if dist_max < DEEP_TIP_MIN_MM:
            return False
        span_post = float(rec.get("contact_span_mm", rec.get("length_mm", 0.0)))
        span_pre = float(rec.get("original_span_mm", span_post))
        span_for_pitch = min(span_pre, span_post) if span_pre > 0 else span_post
        fallback_avg = span_for_pitch / (n - 1) if n > 1 else float("inf")
        median_pitch = float(rec.get("original_median_pitch_mm", fallback_avg))
        if median_pitch > DEEP_TIP_SHORT_MAX_AVG_PITCH_MM:
            return False
        return True

    # Anchor each candidate to the unified metal-evidence bolt CC pool
    # BEFORE dedup. Length and Frangi-tubularity filters catch the rare
    # walker false positives that survive the anchor step.
    def _anchor_or_reject(rec):
        # ``orient_shallow_to_deep`` upstream uses hull head-distance
        # to pick which endpoint is the shallow one, but that's
        # ambiguous for trajectories whose deep tip sits in a deep
        # sulcus as close to its local hull surface as the bolt side
        # (T22 LGR: orbital-floor tip is ~10 mm from hull, skull-top
        # bolt is ~15 mm from hull — orientation flipped). Let the
        # bolt CC decide by trying both orientations and keeping the
        # one whose bolt anchor has more tube voxels.
        fwd = anchor_trajectory_to_bolt(
            rec["start_ras"], rec["end_ras"], bolts,
        )
        bwd = anchor_trajectory_to_bolt(
            rec["end_ras"], rec["start_ras"], bolts,
        )
        fwd_n = int(fwd[2].get("tube_n_vox", 0)) if fwd[2] is not None else 0
        bwd_n = int(bwd[2].get("tube_n_vox", 0)) if bwd[2] is not None else 0

        if bwd_n > fwd_n:
            # Orientation was wrong; flip before writing results back.
            rec["start_ras"], rec["end_ras"] = (
                np.asarray(rec["end_ras"], dtype=float),
                np.asarray(rec["start_ras"], dtype=float),
            )
            new_start, skull_entry, bolt = bwd
        else:
            new_start, skull_entry, bolt = fwd
        # Synth fallback: when the unified bolt CC pass found no anchor
        # (T4-class shanks whose bolts sit outside the CT acquisition
        # window) AND the walker line looks unambiguously like a real
        # SEEG chain, synthesize a skull_entry + bolt_tip by walking the
        # walker axis outward until it crosses the hull surface.
        bolt_from_synth = None
        if new_start is None and _is_genuine_seeg_line(rec):
            s0, e0 = orient_shallow_to_deep(
                rec["start_ras"], rec["end_ras"],
                dist_arr, ras_to_ijk_mat,
            )
            synth_skull, synth_tip = axis_to_skull_synth(
                s0, e0, dist_arr, ras_to_ijk_mat,
            )
            if synth_skull is not None:
                rec["start_ras"] = np.asarray(s0, dtype=float)
                rec["end_ras"] = np.asarray(e0, dtype=float)
                new_start = synth_tip
                skull_entry = synth_skull
                bolt_from_synth = {"n_vox": 0, "dist_min_mm": float("nan"),
                                    "id": -1}
        if new_start is None:
            # Recall-first emission: no metal-evidence bolt CC AND
            # axis-to-skull synth never crossed the hull (e.g. T4 RPOG —
            # bolt sits outside the CT acquisition window AND the axis
            # never reaches the hull). Emit the walker's line as-is.
            # Confidence scoring will downweight these no-anchor
            # emissions.
            rec["bolt_source"] = "none"
            bolt = {"n_vox": 0, "dist_min_mm": float("nan"), "id": -1}
        else:
            if bolt_from_synth is not None:
                bolt = bolt_from_synth
                rec["bolt_source"] = "synthesized"
            else:
                rec["bolt_source"] = "metal"
            rec["start_ras"] = np.asarray(new_start, dtype=float)
            if skull_entry is not None:
                rec["skull_entry_ras"] = np.asarray(skull_entry, dtype=float)
        rec["length_mm"] = float(np.linalg.norm(rec["end_ras"] - rec["start_ras"]))
        # Length sanity: real SEEG total length (bolt + shank) is bounded.
        if (rec["length_mm"] < MIN_POST_ANCHOR_LEN_MM
                or rec["length_mm"] > MAX_POST_ANCHOR_LEN_MM):
            return None
        # Re-validate Frangi tubular evidence on the FULL extended axis
        # (bolt-tip → deep-tip), not just the contact span. Stage 1's
        # Frangi gate only saw the contact span; the bolt anchor extends
        # ``start_ras`` outward by 15-60 mm without checking what's in
        # between. Real shanks have continuous wire+bolt support and
        # keep frangi_median ≥ 30 even on the extended line. Synthesized
        # FPs extend through brain/air with no metal support and drop
        # out here. Overwrite the score's ``frangi_median_mm`` so it
        # sees the true post-anchor value.
        new_fmean, new_fmed = frangi_along_line_stats(
            rec["start_ras"], rec["end_ras"], frangi_s1, ras_to_ijk_mat,
        )
        rec["frangi_mean_mm"] = float(new_fmean)
        rec["frangi_median_mm"] = float(new_fmed)
        # Metal-continuity score feature: fraction of full-axis samples
        # whose unified metal evidence saturates (|LoG|≥LOG_BOLT_NORMALIZER
        # OR HU≥HU_BOLT_NORMALIZER). Real shanks have many discrete contact
        # peaks along the axis; cross-shank bone-assembled chains have a
        # few saturating spots clustered at one end with empty middle.
        # Computed BEFORE the Frangi gate so the gate can defer to it on
        # thin-wire shanks (long wire between bolt and contact array
        # contributes no Frangi response, dragging median below 30 even
        # though both ends have strong metal evidence).
        rec["frac_strong_metal"] = frac_strong_metal_along_line(
            rec["start_ras"], rec["end_ras"],
            log1, ct_arr_kji, ras_to_ijk_mat,
        )
        # Frangi gate fires ONLY when the bolt anchor failed
        # (synthesized / none). When ``bolt_source == "metal"`` the
        # anchor latched onto a real bolt CC and that's already proof
        # the line is real — even if the segment between bolt and
        # contacts is a wire too thin to register as tubular (e.g.
        # ct88 L_37: 50 mm wire gap, frangi_median=8.6, but bolt
        # anchored at 122 tube voxels). The score's Frangi component
        # penalizes weak Frangi proportionally; no need for a hard
        # cliff. Synthesized / none lines still get the gate because
        # the synth fallback can extend through brain/air with no
        # metal evidence on a cross-shank trajectory.
        if (rec["bolt_source"] != "metal"
                and new_fmed < FRANGI_LINE_MIN_MEDIAN):
            return None
        rec["bolt_n_vox"] = int(bolt["n_vox"])
        rec["bolt_dist_min_mm"] = float(bolt["dist_min_mm"])
        rec["bolt_id"] = int(bolt.get("id", -1))
        return rec

    _log("anchoring + length filters…")
    anchored: list[dict[str, Any]] = []
    for l in stage1_lines:
        rec = _anchor_or_reject(_assemble(l))
        if rec is not None:
            anchored.append(rec)
    _log(f"anchoring: {len(anchored)} survived")

    # Wire-class extension: emit trajectories from unmatched bolt CCs whose
    # connected metal extends into the brain. Catches shanks where contacts
    # saturate into a continuous wire (clinical CTs, lateral / short
    # entries — e.g. AMC099 L_4) that the contact-pitch walker can't
    # resolve into discrete LoG peaks.
    used_bolt_ids = {
        int(rec.get("bolt_id", -1))
        for rec in anchored
        if int(rec.get("bolt_id", -1)) >= 0
    }
    wire_recs: list[dict[str, Any]] = []
    for bolt in bolts:
        if int(bolt["id"]) in used_bolt_ids:
            continue
        if int(bolt["n_vox"]) < WIRE_CLASS_MIN_VOXELS:
            continue
        if float(bolt["dist_max_mm"]) < WIRE_CLASS_MIN_DEPTH_MM:
            continue
        pts = np.asarray(bolt["pts_ras"], dtype=float)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = (centered.T @ centered) / max(1, len(pts) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        total_var = float(eigvals.sum())
        if total_var <= 1e-9:
            continue
        elongation = float(eigvals[-1]) / total_var
        if elongation < WIRE_CLASS_MIN_ELONGATION:
            continue
        axis_dir = eigvecs[:, -1]
        proj = centered @ axis_dir
        span = float(proj.max() - proj.min())
        if span < WIRE_CLASS_MIN_SPAN_MM:
            continue
        p_min = centroid + float(proj.min()) * axis_dir
        p_max = centroid + float(proj.max()) * axis_dir
        s_ras, e_ras = orient_shallow_to_deep(
            p_min, p_max, dist_arr, ras_to_ijk_mat,
        )
        rec = dict(
            start_ras=np.asarray(s_ras, dtype=float),
            end_ras=np.asarray(e_ras, dtype=float),
            shallow_endpoint_name="start",
            deep_endpoint_name="end",
            length_mm=float(np.linalg.norm(e_ras - s_ras)),
            n_inliers=0,
            contact_span_mm=span,
            original_span_mm=span,
            original_median_pitch_mm=0.0,
            dist_min_mm=float(bolt["dist_min_mm"]),
            dist_max_mm=float(bolt["dist_max_mm"]),
            dist_mean_mm=float(np.mean(bolt["pts_dist"])),
            bolt_source="metal_cc",
            bolt_n_vox=int(bolt["n_vox"]),
            bolt_dist_min_mm=float(bolt["dist_min_mm"]),
            bolt_id=int(bolt["id"]),
            wire_class=True,
            inlier_idx=[],
        )
        new_fmean, new_fmed = frangi_along_line_stats(
            rec["start_ras"], rec["end_ras"], frangi_s1, ras_to_ijk_mat,
        )
        rec["frangi_mean_mm"] = float(new_fmean)
        rec["frangi_median_mm"] = float(new_fmed)
        rec["frac_strong_metal"] = frac_strong_metal_along_line(
            rec["start_ras"], rec["end_ras"],
            log1, ct_arr_kji, ras_to_ijk_mat,
        )
        rec["skull_entry_ras"] = np.asarray(s_ras, dtype=float)
        rec["intracranial_length_mm"] = float(np.linalg.norm(e_ras - s_ras))
        wire_recs.append(rec)
    if wire_recs:
        _log(f"wire-class: emitting {len(wire_recs)} from unmatched bolt CCs")
        anchored.extend(wire_recs)

    # Dedup keeps the longer line of each cluster.
    anchored.sort(key=lambda rec: -float(rec.get("length_mm", 0.0)))
    anchored = dedup_trajectories(anchored)
    _log(f"final dedup: {len(anchored)} trajectories")

    # Axis-directed deep-end refinement. The 3D regional-minima blob
    # extractor misses contacts when the per-contact LoG wells merge
    # into one continuous CC (seen on T2 X06 / RAI, where the deep 3–4
    # contacts sit inside a single long bright shaft and don't produce
    # distinct 3D minima). Sample the LoG profile 1-dimensionally along
    # the trajectory axis and push ``end_ras`` out to the last real
    # contact peak.
    for ri, rec in enumerate(anchored):
        others = [r for j, r in enumerate(anchored) if j != ri]
        new_end = refine_deep_end_via_axis_log(
            rec, log1, ras_to_ijk_mat,
            others=others,
        )
        if new_end is not None:
            rec["end_ras"] = new_end
        # Hard cap: end must sit within DEEP_END_MARGIN_PAST_LAST_CONTACT_MM
        # of the deepest walker inlier. No SEEG electrode has a long gap
        # past its last contact; anything further is over-reach.
        clipped = clip_deep_end_to_inliers(rec)
        if clipped is not None:
            rec["end_ras"] = clipped
        if new_end is not None or clipped is not None:
            rec["length_mm"] = float(
                np.linalg.norm(
                    np.asarray(rec["end_ras"]) - np.asarray(rec["start_ras"])
                )
            )

    # Crossing-tip retreat: after all trajectories have been extended,
    # pull back any tip that lives inside another's contact-acceptance
    # tube. Runs only at the final stage so every axis has settled
    # before we decide which tip is the intruder. Passing the LoG
    # volume lets the retreat additionally snap the pulled-back tip to
    # the deep edge of the last real contact instead of floating in the
    # gap between contacts.
    _log("crossing-tip retreat…")
    retreat_crossing_tips(
        anchored,
        log_arr=log1,
        ras_to_ijk_mat=ras_to_ijk_mat,
        min_length_mm=bounds.min_post_anchor_len_mm,
        logger=_log,
    )

    # Intracranial-only length (skull entry → deep tip). The existing
    # ``length_mm`` is bolt-tip → deep-tip and includes ~15–25 mm of bolt
    # protrusion outside the skull; downstream displays/clinical reporting
    # want the part that actually sits inside the brain.
    for rec in anchored:
        entry = np.asarray(
            rec.get("skull_entry_ras", rec.get("start_ras")),
            dtype=float,
        )
        end = np.asarray(rec["end_ras"], dtype=float)
        rec["intracranial_length_mm"] = float(np.linalg.norm(end - entry))

    # Electrode-model suggestion intentionally NOT computed here. The
    # canonical picker is the matched filter in
    # ``rosa_core.contact_placement.stage_d_pick.pick_matched_filter``,
    # which runs as part of ``place_seeg`` whenever contacts are placed.
    # The detection-time per-trajectory classifier (formerly invoked here
    # via ``rosa_core.electrode_classifier.classify_electrode_model``)
    # was removed 2026-05-11: it wrote ``Rosa.BestModelId`` to MRML node
    # attributes that no Slicer module reads (CTV explicitly defaults its
    # dropdowns to empty per the 2026-05-10 user-feedback comment in
    # ``ContactsTrajectoryView._populate_contact_table``), and the CLI
    # placement path never consumed it. Keeping it would double-classify
    # every trajectory (detection + placement) for no downstream effect.

    # Confidence score (v1). Each survivor gets a continuous physical-
    # evidence score in [0, 1] plus a coarse confidence label.
    # ``confidence`` is the numeric score (canonical engine schema
    # expects a float there); ``confidence_label`` carries the band.
    for rec in anchored:
        score, label, components = compute_trajectory_score(rec, bounds=bounds)
        rec["confidence"] = score
        rec["confidence_label"] = label
        rec["score_components"] = components

    # Convert to JSON-safe dicts (tuples of floats).
    trajectories: list[dict[str, Any]] = []
    for rec in anchored:
        out = dict(rec)
        out["start_ras"] = [float(x) for x in rec["start_ras"]]
        out["end_ras"] = [float(x) for x in rec["end_ras"]]
        if "skull_entry_ras" in rec:
            out["skull_entry_ras"] = [float(x) for x in rec["skull_entry_ras"]]
        trajectories.append(out)

    # Per-trajectory fingerprint — compact trace makes it easy to diff
    # Slicer-run results against a CLI run to spot which specific
    # trajectory disappeared / shifted when subject-level totals don't match.
    try:
        _log(f"trajectory summary ({len(trajectories)} kept):")
        for _i, _t in enumerate(trajectories):
            _se = _t.get("skull_entry_ras") or _t.get("start_ras") or [0.0, 0.0, 0.0]
            _en = _t.get("end_ras") or [0.0, 0.0, 0.0]
            _bolt = str(_t.get("bolt_source") or "?")
            _n = int(_t.get("n_inliers") or 0)
            _sp = float(_t.get("contact_span_mm") or 0.0)
            _sc = float(_t.get("confidence") or 0.0)
            _conf = str(_t.get("confidence_label") or "?")
            _log(
                f"  [{_i:02d}] bolt={_bolt} n={_n} span={_sp:.1f}mm "
                f"score={_sc:.2f}({_conf}) "
                f"skull_entry=({_se[0]:+.1f},{_se[1]:+.1f},{_se[2]:+.1f}) "
                f"deep_tip=({_en[0]:+.1f},{_en[1]:+.1f},{_en[2]:+.1f})"
            )
    except Exception:
        pass

    if return_features:
        features = {
            "log_sigma1": log1,
            "frangi_sigma1": frangi_s1,
            "head_distance": dist_arr,
            "intracranial": intracranial.astype(np.uint8),
            "hull": hull.astype(np.uint8),
            "bolt_mask": bolt_mask,
            # IJK->RAS matrix for the grid the feature arrays live on.
            # Differs from the input volume's matrix when canonical-1mm
            # resampling fired (raw sub-mm input). Slicer must use this
            # to position the feature volumes correctly in the scene.
            "ijk_to_ras_mat": np.asarray(ijk_to_ras_mat, dtype=float),
        }
        return trajectories, features
    return trajectories


__all__ = ["run_two_stage_detection"]
