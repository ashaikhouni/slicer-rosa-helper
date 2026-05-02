"""Skull-entry placement audit.

For each metal-anchored trajectory in the dataset, compare the
existing ``skull_entry_ras`` (placed by ``anchor_trajectory_to_bolt``
as the deepest bolt-CC tube voxel still in the hull-band) against
two candidate refiner positions:

  - Hull-crossing point: walk axis from end_ras toward start_ras, find
    the first sample where ``hull_dist <= 5 mm``. This is the scalp
    surface, NOT the bone surface — useful as a fast / cheap proxy.

  - HU-ring bone-transition: walk axis end -> start, sample a
    perpendicular ring (radii 2.0 / 3.0 mm) at each step, find the
    first sample where ring-median HU >= 500. This is the brain ->
    bone transition along the axis = where the wire enters the
    skull. The refiner the user proposed.

The probe reports |existing - candidate| in mm for both candidates,
broken down by ``bolt_source``. If the offsets are small (< 2 mm
median), the existing skull_entry is fine and the refiner is overkill.
If they're large (> 5 mm median), the refiner would meaningfully
relocate skull_entry.

Usage:
    cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper && \\
        /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_skull_entry_offset.py [SUBJECT|ALL]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

from rosa_detect import contact_pitch_v1_fit as cpfit
from rosa_detect.service import run_contact_pitch_v1
from eval_seeg_localization import (
    build_detection_context, iter_subject_rows,
    load_reference_ground_truth_shanks,
)


def _greedy_match(gt_shanks, trajs):
    """Same matcher used elsewhere — angle <= 10 deg AND midpoint
    perp <= 8 mm."""
    pairs = []
    for gi, g in enumerate(gt_shanks):
        gs = np.asarray(g.start_ras, dtype=float)
        ge = np.asarray(g.end_ras, dtype=float)
        gn = float(np.linalg.norm(ge - gs))
        ga = (ge - gs) / gn if gn > 1e-9 else np.array([0.0, 0.0, 1.0])
        gm = 0.5 * (gs + ge)
        for ti, t in enumerate(trajs):
            ts = np.asarray(t["start_ras"], dtype=float)
            te = np.asarray(t["end_ras"], dtype=float)
            tn = float(np.linalg.norm(te - ts))
            ta = (te - ts) / tn if tn > 1e-9 else np.array([0.0, 0.0, 1.0])
            tm = 0.5 * (ts + te)
            ang = float(np.degrees(np.arccos(min(1.0,
                                                    abs(float(np.dot(ga, ta)))))))
            d = gm - tm
            mid = float(np.linalg.norm(d - (d @ ta) * ta))
            if ang <= 10 and mid <= 8:
                pairs.append((ang + mid, gi, ti))
    pairs.sort()
    used_g, used_t = set(), set()
    m = {}
    for _s, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        m[ti] = str(gt_shanks[gi].shank)
    return m

DATASET_ROOT = Path("/Users/ammar/Dropbox/thalamus_subjects/seeg_localization")
EXCLUDE = {"T17", "T19", "T21"}

STEP_MM = 0.5
MAX_WALK_MM = 80.0
HULL_CROSS_THRESHOLD_MM = 5.0
# Bone state machine: a ring sample with HU >= BONE_MIN_HU counts as
# "in bone". Once in_bone is True, the first sample with HU < BRAIN_MAX_HU
# is the dura/bone interface (= where wire enters brain).
BONE_MIN_HU = 400.0
BRAIN_MAX_HU = 200.0
# Ring radii. Wider than my first attempt (2.0/3.0 mm) which got
# polluted by contact metal halo bleeding into the perp tube at deep
# axial positions. 4-5 mm clears the typical 2-3 mm contact partial-
# volume halo while staying inside a 5-7 mm wide skull cross-section.
RING_RADII_MM = (4.0, 5.0)
RING_N_AZIMUTHAL = 8


def _hook_volumes():
    """Capture dist_arr + ijk_to_ras_mat from the bolt-extract call.
    The ``ct_arr`` argument is the unified metal-evidence volume
    (0-2 range), NOT raw HU — caller computes raw HU separately.
    """
    state = {}
    real = cpfit.extract_bolt_candidates

    def patched(log_arr, dist_arr, ijk_to_ras_mat, spacing_xyz, **kw):
        state["log_arr"] = log_arr
        state["dist_arr"] = dist_arr
        state["ijk_to_ras_mat"] = np.asarray(ijk_to_ras_mat, dtype=float)
        return real(log_arr, dist_arr, ijk_to_ras_mat, spacing_xyz, **kw)

    cpfit.extract_bolt_candidates = patched

    def restore():
        cpfit.extract_bolt_candidates = real

    return state, restore


def _canonical_ct_and_frangi(raw_img):
    """Resample to canonical 1 mm spacing + HU clamp; return
    (ct_arr_kji, frangi_arr_kji). Matches the volumes the pipeline
    uses internally for HU/Frangi sampling.
    """
    import SimpleITK as sitk
    spacing = raw_img.GetSpacing()
    if min(float(s) for s in spacing) < cpfit.CANONICAL_SPACING_MM * 0.95:
        size_in = raw_img.GetSize()
        target = (cpfit.CANONICAL_SPACING_MM,) * 3
        target_size = [
            max(1, int(round(size_in[i] * float(spacing[i]) /
                              cpfit.CANONICAL_SPACING_MM)))
            for i in range(3)
        ]
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(target)
        rs.SetSize(target_size)
        rs.SetOutputOrigin(raw_img.GetOrigin())
        rs.SetOutputDirection(raw_img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024)
        img = rs.Execute(raw_img)
        sigma = getattr(cpfit, "RAW_RESAMPLE_GAUSSIAN_SIGMA_MM", 0.0)
        if sigma > 0:
            img = sitk.SmoothingRecursiveGaussian(img, sigma)
    else:
        img = raw_img
    img_clamped = sitk.Clamp(img, lowerBound=-1024.0,
                              upperBound=cpfit.HU_CLIP_MAX)
    ct_arr = sitk.GetArrayFromImage(img_clamped).astype(np.float32)
    frangi_arr = cpfit.frangi_single(img_clamped, cpfit.FRANGI_STAGE1_SIGMA)
    return ct_arr, frangi_arr


def _perpendicular_basis(axis):
    """Two unit vectors perpendicular to axis (and to each other)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    # Pick any non-parallel vector.
    if abs(a[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    v1 = np.cross(a, ref)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(a, v1)
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2


def _sample_at(arr, ras_to_ijk, p):
    K, J, I = arr.shape
    h = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
    ijk = (ras_to_ijk @ h)[:3]
    i = int(np.clip(round(ijk[0]), 0, I - 1))
    j = int(np.clip(round(ijk[1]), 0, J - 1))
    k = int(np.clip(round(ijk[2]), 0, K - 1))
    return float(arr[k, j, i])


def _ring_median_hu(p, axis, ct_arr, ras_to_ijk, radii_mm=RING_RADII_MM,
                    n_azimuthal=RING_N_AZIMUTHAL):
    """Median HU sampled on a perpendicular ring around p (excludes
    the axis center where the electrode metal sits)."""
    v1, v2 = _perpendicular_basis(axis)
    samples = []
    for r in radii_mm:
        for k in range(n_azimuthal):
            theta = 2.0 * np.pi * k / n_azimuthal
            offset = r * (np.cos(theta) * v1 + np.sin(theta) * v2)
            samples.append(_sample_at(ct_arr, ras_to_ijk, p + offset))
    return float(np.median(samples))


def _walk_for_hull_cross(start, end, dist_arr, ras_to_ijk,
                          step_mm=STEP_MM, max_walk_mm=MAX_WALK_MM,
                          threshold_mm=HULL_CROSS_THRESHOLD_MM):
    """Walk axis from end (deep) toward start (shallow). Return first
    sample with hull_dist <= threshold_mm."""
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(s - e))
    if L < 1e-3:
        return None
    axis_shallow = (s - e) / L
    n = int(min(max_walk_mm, L) / step_mm) + 1
    for k in range(n):
        p = e + k * step_mm * axis_shallow
        d = _sample_at(dist_arr, ras_to_ijk, p)
        if d <= threshold_mm:
            return p
    return None


def _walk_for_bone_legacy_deep_to_shallow(
    start, end, ct_arr, ras_to_ijk,
    step_mm=STEP_MM, max_walk_mm=MAX_WALK_MM,
    bone_threshold=BONE_MIN_HU,
):
    """LEGACY (kept for reference). Walks deep -> shallow looking for
    first ring-median HU >= bone_threshold. Suffers from contact-metal
    halo bleed into a 2-3 mm perp ring at the deepest contact, giving
    spurious "bone" hits 50-80 mm too deep."""
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(s - e))
    if L < 1e-3:
        return None
    axis_shallow = (s - e) / L
    n = int(min(max_walk_mm, L) / step_mm) + 1
    for k in range(n):
        p = e + k * step_mm * axis_shallow
        m = _ring_median_hu(p, axis_shallow, ct_arr, ras_to_ijk,
                              radii_mm=(2.0, 3.0))
        if m >= bone_threshold:
            return p
    return None


def _walk_for_dura_via_hu(start, end, ct_arr, ras_to_ijk,
                            step_mm=STEP_MM, max_walk_mm=MAX_WALK_MM,
                            bone_min=BONE_MIN_HU,
                            brain_max=BRAIN_MAX_HU,
                            radii_mm=RING_RADII_MM,
                            dist_arr=None, max_hull_dist_mm=25.0):
    """Walk axis SHALLOW -> DEEP from ``start`` (bolt-tip). State
    machine on ring-median HU finds the dura/bone interface = first
    sample where bone exits to brain.

      bone_indicator := HU >= bone_min   (bone or metal — both
        non-brain materials count as "in bone region")
      brain_indicator := HU < brain_max

      Each step:
        - If bone_indicator: in_bone := True.
        - Elif in_bone and brain_indicator: return position
          (= dura interface).

    When ``dist_arr`` is provided, the search is capped at
    ``hull_dist <= max_hull_dist_mm`` — keeps the search in the
    typical bone/dura band so contacts deeper than dura can't pollute
    the state machine.
    """
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return None
    axis_deep = (e - s) / L
    n = int(min(max_walk_mm, L) / step_mm) + 1
    in_bone = False
    for k in range(n):
        p = s + k * step_mm * axis_deep
        if dist_arr is not None:
            d = _sample_at(dist_arr, ras_to_ijk, p)
            if d > max_hull_dist_mm:
                return None
        m = _ring_median_hu(p, axis_deep, ct_arr, ras_to_ijk,
                              radii_mm=radii_mm)
        if m >= bone_min:
            in_bone = True
        elif in_bone and m < brain_max:
            return p
    return None


def _ring_max_frangi(p, axis, frangi_arr, ras_to_ijk,
                      radii_mm=(0.0, 2.0, 4.0), n_azimuthal=8):
    """Max Frangi response on a small ring + center around p."""
    samples = [_sample_at(frangi_arr, ras_to_ijk, p)]
    if max(radii_mm) > 0.0:
        v1, v2 = _perpendicular_basis(axis)
        for r in radii_mm:
            if r <= 0.0:
                continue
            for k in range(n_azimuthal):
                theta = 2.0 * np.pi * k / n_azimuthal
                offset = r * (np.cos(theta) * v1 + np.sin(theta) * v2)
                samples.append(_sample_at(frangi_arr, ras_to_ijk, p + offset))
    return float(np.max(samples))


def _walk_for_frangi_drop(start, end, frangi_arr, ras_to_ijk,
                            step_mm=STEP_MM, max_walk_mm=MAX_WALK_MM,
                            high_threshold=15.0, low_threshold=5.0):
    """Walk axis SHALLOW -> DEEP. Frangi tubular response is HIGH on
    bolt material and on contacts but drops in the brief gap between
    bolt-end and first contact (= dura/bone interface region).

    State machine:
      IN_METAL := False.
      Each step, sample max-Frangi over a small ring + center.
        - If sample >= high_threshold: IN_METAL := True.
        - Elif IN_METAL and sample < low_threshold: this is the
          metal -> non-metal transition (bolt-end). Return position.
    """
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return None
    axis_deep = (e - s) / L
    n = int(min(max_walk_mm, L) / step_mm) + 1
    in_metal = False
    for k in range(n):
        p = s + k * step_mm * axis_deep
        m = _ring_max_frangi(p, axis_deep, frangi_arr, ras_to_ijk)
        if m >= high_threshold:
            in_metal = True
        elif in_metal and m < low_threshold:
            return p
    return None


# Annulus around the axis (hollow cylinder). Inner radius needs to
# clear the LoG / HU halo of the electrode + contacts: a Slicer
# measurement showed the LoG halo extending ~8 mm perpendicular from
# the axis, so radii 3-7 mm is inside the halo bleed and pollutes
# any tissue measurement. Pushed inner to 9 mm and outer to 13 mm —
# samples real brain tissue cleanly at the cost of catching slightly
# more distant anatomy (sulci/ventricles) on some shanks.
SLAB_INNER_RADIUS_MM = 9.0
SLAB_OUTER_RADIUS_MM = 13.0
SLAB_RADII_MM = (9.0, 10.0, 11.0, 12.0, 13.0)
SLAB_N_AZIMUTHAL = 8
METAL_MASK_HU = 500.0  # residual safety net (rarely fires now that
                        # the annulus is well outside the halo).
LOG_PEAK_THRESHOLD = 250.0  # |LoG| floor at brain/bone transition;
                             # tunable. Brain interior: |LoG| ~ 30-80;
                             # contact halo: |LoG| 500+; bone/brain
                             # boundary: |LoG| 200-500 typically.


def _slab_sample(p, axis, ct_arr, log_arr, ras_to_ijk,
                  radii=SLAB_RADII_MM, n_az=SLAB_N_AZIMUTHAL,
                  metal_hu_threshold=METAL_MASK_HU):
    """Sample a perpendicular ANNULUS around p (radii×azimuth grid).
    Inner cylinder (radius < min(radii)) is excluded by geometry — no
    samples taken there, so the wire / contact halo doesn't enter the
    statistic. ``metal_hu_threshold`` provides a residual mask in case
    bolt threads reach the inner edge of the annulus on shallow
    samples. Returns (median_hu, median_abs_log) over surviving
    voxels. Empty result (everything was metal) returns (nan, nan).
    """
    v1, v2 = _perpendicular_basis(axis)
    hu_vals = []
    log_vals = []
    for r in radii:
        for k in range(n_az):
            theta = 2.0 * np.pi * k / n_az
            offset = r * (np.cos(theta) * v1 + np.sin(theta) * v2)
            q = p + offset
            hu = _sample_at(ct_arr, ras_to_ijk, q)
            if hu > metal_hu_threshold:
                continue
            hu_vals.append(hu)
            log_vals.append(abs(_sample_at(log_arr, ras_to_ijk, q)))
    if not hu_vals:
        return float("nan"), float("nan")
    return float(np.median(hu_vals)), float(np.median(log_vals))


def _walk_for_dura_via_log_peak(start, end, log_arr, ct_arr, ras_to_ijk,
                                  step_mm=STEP_MM, max_walk_mm=MAX_WALK_MM,
                                  peak_threshold=LOG_PEAK_THRESHOLD,
                                  dist_arr=None, min_hull_dist_mm=-5.0):
    """Walk axis DEEP -> SHALLOW. At each step, sample a perpendicular
    slab around the axis (radii 1-5 mm), excluding electrode metal
    voxels (HU > METAL_MASK_HU). Compute slab-median |LoG|.

    The deep portion of the trajectory is in brain — uniform tissue,
    median |LoG| stays low (~30-80). The brain/bone transition has a
    sharp tissue boundary — slab-median |LoG| peaks. Return the FIRST
    axial position where slab-median |LoG| crosses the peak threshold
    (going deep -> shallow) = the dura/bone interface.

    Stops walking once hull_dist <= min_hull_dist_mm (we've reached
    the scalp surface or beyond — past the bone region).
    """
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(s - e))
    if L < 1e-3:
        return None
    axis_shallow = (s - e) / L
    n = int(min(max_walk_mm, L) / step_mm) + 1
    for k in range(n):
        p = e + k * step_mm * axis_shallow
        if dist_arr is not None:
            d = _sample_at(dist_arr, ras_to_ijk, p)
            if d < min_hull_dist_mm:
                return None
        _, m_log = _slab_sample(p, axis_shallow, ct_arr, log_arr, ras_to_ijk)
        if m_log == m_log and m_log >= peak_threshold:  # not nan, above threshold
            return p
    return None


def _slab_brain_likeness(start, end, ct_arr, log_arr, ras_to_ijk,
                          step_mm=1.0):
    """Median slab-HU across the trajectory's intracranial portion.
    See docstring above (kept for compatibility)."""
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return float("nan")
    axis_deep = (e - s) / L
    n = max(2, int(L / step_mm) + 1)
    medians = []
    for k in range(n):
        t = k * step_mm if k < n - 1 else L
        p = s + t * axis_deep
        m_hu, _ = _slab_sample(p, axis_deep, ct_arr, log_arr, ras_to_ijk)
        if m_hu == m_hu:
            medians.append(m_hu)
    if not medians:
        return float("nan")
    return float(np.median(medians))


def _slab_fraction_in_head(p, axis, dist_arr, ras_to_ijk,
                              radii=SLAB_RADII_MM, n_az=SLAB_N_AZIMUTHAL):
    """Fraction of perpendicular-annulus voxels at ``hull_dist > 0``
    (inside the head). Real intracranial trajectories: annulus
    fully inside head -> fraction = 1.0. Hull-skim trajectories at
    the head boundary: ~half the annulus is outside (hull_dist <= 0)
    -> fraction ≈ 0.5. Geometric discriminator that doesn't depend
    on tissue HU.
    """
    v1, v2 = _perpendicular_basis(axis)
    n_total = 0
    n_inside = 0
    for r in radii:
        for k in range(n_az):
            theta = 2.0 * np.pi * k / n_az
            offset = r * (np.cos(theta) * v1 + np.sin(theta) * v2)
            q = p + offset
            d = _sample_at(dist_arr, ras_to_ijk, q)
            n_total += 1
            if d > 0.0:
                n_inside += 1
    if n_total == 0:
        return float("nan")
    return float(n_inside) / float(n_total)


def _slab_fraction_in_head_along_line(start, end, dist_arr, ras_to_ijk,
                                        step_mm=1.0):
    """Median fraction-in-head over the trajectory's intracranial
    portion. Real intracranial shanks: median ~ 1.0. Hull-skim FPs:
    median ~ 0.5 (annulus straddles the head boundary throughout).
    """
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    L = float(np.linalg.norm(e - s))
    if L < 1e-3:
        return float("nan")
    axis_deep = (e - s) / L
    n = max(2, int(L / step_mm) + 1)
    fracs = []
    for k in range(n):
        t = k * step_mm if k < n - 1 else L
        p = s + t * axis_deep
        f = _slab_fraction_in_head(p, axis_deep, dist_arr, ras_to_ijk)
        if f == f:
            fracs.append(f)
    if not fracs:
        return float("nan")
    return float(np.median(fracs))


def _proj_along_axis(p, axis, ref):
    """Signed scalar projection of (p - ref) onto axis (unit)."""
    return float(np.dot(np.asarray(p, dtype=float) - np.asarray(ref, dtype=float),
                          np.asarray(axis, dtype=float)))


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    subjects_filter = None if arg == "ALL" else {arg}
    rows = iter_subject_rows(DATASET_ROOT, subjects_filter)
    rows = [r for r in rows if str(r["subject_id"]) not in EXCLUDE]
    rows.sort(key=lambda r: int(str(r["subject_id"]).lstrip("T")))

    state, restore = _hook_volumes()
    records = []
    try:
        for row in rows:
            sid = str(row["subject_id"])
            ctx, raw_img = build_detection_context(
                row["ct_path"], run_id=f"probe_skull_entry_{sid}",
                config={}, extras={},
            )
            ctx["contact_pitch_v1_pitch_strategy"] = "auto"
            result = run_contact_pitch_v1(ctx)
            trajs = list(result.get("trajectories") or [])
            try:
                gt, _ = load_reference_ground_truth_shanks(row)
                matched_map = _greedy_match(gt, trajs)
            except Exception:
                matched_map = {}

            dist_arr = state.get("dist_arr")
            ijk_to_ras = state.get("ijk_to_ras_mat")
            if dist_arr is None or ijk_to_ras is None:
                print(f"  {sid}: WARNING — no captured volumes, skipping")
                continue
            ras_to_ijk = np.linalg.inv(ijk_to_ras)
            # Compute raw HU + Frangi separately — the hooked call
            # passes the normalized metal-evidence volume, not raw HU.
            ct_arr, frangi_arr = _canonical_ct_and_frangi(raw_img)

            for ti, t in enumerate(trajs):
                bs = str(t.get("bolt_source", "?"))
                start = np.asarray(t["start_ras"], dtype=float)
                end = np.asarray(t["end_ras"], dtype=float)
                axis_shallow = (start - end) / np.linalg.norm(start - end)
                existing_se = t.get("skull_entry_ras")

                # Hull-cross candidate (deep -> shallow, hull_dist <= 5 mm)
                hull_p = _walk_for_hull_cross(start, end, dist_arr, ras_to_ijk)
                # HU dura-interface candidate (shallow -> deep ring HU
                # state machine; kept for comparison with the LoG-peak
                # candidate)
                hu_p = _walk_for_dura_via_hu(
                    start, end, ct_arr, ras_to_ijk, dist_arr=dist_arr,
                )
                # Frangi-gap candidate (shallow -> deep; metal-end
                # transition; kept for comparison)
                frangi_p = _walk_for_frangi_drop(
                    start, end, frangi_arr, ras_to_ijk,
                )
                # LoG-peak candidate (DEEP -> SHALLOW slab walk).
                # Slab-median |LoG| stays low through uniform brain,
                # peaks at the brain/bone transition. Metal voxels
                # (HU > METAL_MASK_HU) are masked out so contacts
                # don't dominate.
                log_peak_p = _walk_for_dura_via_log_peak(
                    start, end, log_arr=state.get("log_arr"),
                    ct_arr=ct_arr, ras_to_ijk=ras_to_ijk,
                    dist_arr=dist_arr,
                )

                rec = {"subject": sid, "ti": ti, "bolt_source": bs}
                rec["matched"] = ti in matched_map
                rec["existing_se_set"] = existing_se is not None
                rec["hull_cross_found"] = hull_p is not None
                rec["hu_dura_found"] = hu_p is not None
                rec["frangi_gap_found"] = frangi_p is not None
                rec["log_peak_found"] = log_peak_p is not None

                if existing_se is not None:
                    se = np.asarray(existing_se, dtype=float)
                    rec["existing_hull_d"] = _sample_at(dist_arr, ras_to_ijk, se)
                    rec["existing_ring_hu"] = _ring_median_hu(
                        se, axis_shallow, ct_arr, ras_to_ijk,
                    )
                    if hull_p is not None:
                        rec["dist_existing_to_hull_cross"] = float(
                            np.linalg.norm(se - hull_p)
                        )
                    if hu_p is not None:
                        rec["dist_existing_to_hu_dura"] = float(
                            np.linalg.norm(se - hu_p)
                        )
                    if frangi_p is not None:
                        rec["dist_existing_to_frangi_gap"] = float(
                            np.linalg.norm(se - frangi_p)
                        )
                    if log_peak_p is not None:
                        rec["dist_existing_to_log_peak"] = float(
                            np.linalg.norm(se - log_peak_p)
                        )
                if hu_p is not None and frangi_p is not None:
                    rec["dist_hu_to_frangi"] = float(
                        np.linalg.norm(hu_p - frangi_p)
                    )
                # Hull dist at each candidate: a valid dura/bone interface
                # sits at hull_d ~ 5-15 mm (skull thickness varies).
                if hu_p is not None:
                    rec["hu_dura_hull_d"] = _sample_at(
                        dist_arr, ras_to_ijk, hu_p,
                    )
                if frangi_p is not None:
                    rec["frangi_gap_hull_d"] = _sample_at(
                        dist_arr, ras_to_ijk, frangi_p,
                    )
                if log_peak_p is not None:
                    rec["log_peak_hull_d"] = _sample_at(
                        dist_arr, ras_to_ijk, log_peak_p,
                    )
                # Brain-likeness score: median slab HU across the
                # intracranial portion. Uses existing skull_entry_ras
                # when available (matches what production would do
                # without a refiner); falls back to start_ras.
                anchor_for_brain = None
                if existing_se is not None:
                    anchor_for_brain = np.asarray(existing_se, dtype=float)
                else:
                    anchor_for_brain = start
                rec["brain_likeness_median_hu"] = _slab_brain_likeness(
                    anchor_for_brain, end,
                    ct_arr, state.get("log_arr"), ras_to_ijk,
                )
                rec["frac_annulus_in_head"] = (
                    _slab_fraction_in_head_along_line(
                        anchor_for_brain, end, dist_arr, ras_to_ijk,
                    )
                )
                records.append(rec)
            print(f"  {sid:>4s}: {len(trajs)} trajectories")
    finally:
        restore()

    # --- summary by bolt_source ----------------------------------------
    print(f"\n=== {len(records)} trajectories total ===")

    by_src = defaultdict(list)
    for r in records:
        by_src[r["bolt_source"]].append(r)

    print(f"\nExisting skull_entry placement + candidate find rates:")
    for src in sorted(by_src):
        recs = by_src[src]
        n_set = sum(1 for r in recs if r["existing_se_set"])
        n_hull = sum(1 for r in recs if r["hull_cross_found"])
        n_hu = sum(1 for r in recs if r["hu_dura_found"])
        n_frangi = sum(1 for r in recs if r["frangi_gap_found"])
        n_log = sum(1 for r in recs if r.get("log_peak_found"))
        print(f"  bolt_source={src:>11s} (n={len(recs):>3d}):  "
              f"existing={n_set}  hull_cross={n_hull}  "
              f"hu_dura={n_hu}  frangi_gap={n_frangi}  "
              f"log_peak={n_log}")

    def _percentiles(values, label):
        if not values:
            print(f"  {label}: (empty)")
            return
        a = np.array(values, dtype=float)
        print(f"  {label} (n={len(a)}): "
              f"p10={np.percentile(a,10):.2f}  "
              f"p25={np.percentile(a,25):.2f}  "
              f"p50={np.percentile(a,50):.2f}  "
              f"p75={np.percentile(a,75):.2f}  "
              f"p90={np.percentile(a,90):.2f}  "
              f"max={a.max():.2f}")

    # --- offset distributions per candidate ---------------------------
    print(f"\n=== |existing skull_entry - candidate| (mm) — "
          f"how far each refiner would shift skull_entry ===\n")

    def _section(title, key):
        print(f"  {title}:")
        for src in sorted(by_src):
            vals = [r[key] for r in by_src[src] if key in r]
            _percentiles(vals, f"    bolt_source={src:>11s}")

    _section(
        f"hull-cross (deep->shallow walk; first hull_d <= "
        f"{HULL_CROSS_THRESHOLD_MM:.0f} mm)",
        "dist_existing_to_hull_cross",
    )
    print()
    _section(
        f"HU dura-interface (shallow->deep; ring HU enters bone "
        f">={BONE_MIN_HU:.0f} then exits to brain <{BRAIN_MAX_HU:.0f})",
        "dist_existing_to_hu_dura",
    )
    print()
    _section(
        "Frangi gap (shallow->deep; ring max-Frangi enters metal "
        ">=15 then drops to <5)",
        "dist_existing_to_frangi_gap",
    )
    print()
    _section(
        f"LoG peak (deep->shallow slab walk; first slab-median "
        f"|LoG| >= {LOG_PEAK_THRESHOLD:.0f}; metal voxels HU > "
        f"{METAL_MASK_HU:.0f} masked out)",
        "dist_existing_to_log_peak",
    )

    print(f"\n=== HU vs Frangi candidate agreement (|hu - frangi|) ===")
    vals = [r["dist_hu_to_frangi"] for r in records
            if "dist_hu_to_frangi" in r]
    _percentiles(vals, f"  all (n={len(vals)})")

    print(f"\n=== hull_dist at each candidate "
          f"(valid dura/bone interface ~= 5-15 mm) ===\n")
    print(f"  HU dura-interface candidate:")
    for src in sorted(by_src):
        vals = [r["hu_dura_hull_d"] for r in by_src[src]
                if "hu_dura_hull_d" in r]
        _percentiles(vals, f"    bolt_source={src:>11s}")
    print(f"\n  Frangi gap candidate:")
    for src in sorted(by_src):
        vals = [r["frangi_gap_hull_d"] for r in by_src[src]
                if "frangi_gap_hull_d" in r]
        _percentiles(vals, f"    bolt_source={src:>11s}")
    print(f"\n  LoG peak candidate:")
    for src in sorted(by_src):
        vals = [r["log_peak_hull_d"] for r in by_src[src]
                if "log_peak_hull_d" in r]
        _percentiles(vals, f"    bolt_source={src:>11s}")

    print(f"\n=== brain-likeness median slab HU across intracranial "
          f"portion (anchor=existing skull_entry; metal masked HU>"
          f"{METAL_MASK_HU:.0f}) ===")
    print(f"  Real intracranial brain ~ 30-50 HU; FPs through bone/teeth/"
          f"scalp >> 200 HU\n")
    for src in sorted(by_src):
        vals = [r["brain_likeness_median_hu"] for r in by_src[src]
                if "brain_likeness_median_hu" in r
                and r["brain_likeness_median_hu"] ==
                    r["brain_likeness_median_hu"]]
        _percentiles(vals, f"  bolt_source={src:>11s} (all)")

    print(f"\n  --- matched vs orphan breakdown (the discrimination test) ---")
    matched_vals = [r["brain_likeness_median_hu"] for r in records
                    if r.get("matched") and "brain_likeness_median_hu" in r
                    and r["brain_likeness_median_hu"]
                    == r["brain_likeness_median_hu"]]
    orphan_vals = [r["brain_likeness_median_hu"] for r in records
                   if not r.get("matched") and "brain_likeness_median_hu" in r
                   and r["brain_likeness_median_hu"]
                   == r["brain_likeness_median_hu"]]
    _percentiles(matched_vals, f"  matched (n={len(matched_vals)})")
    _percentiles(orphan_vals, f"  orphan  (n={len(orphan_vals)})")

    # How well would `brain_likeness <= T` separate matched from orphan?
    print(f"\n  --- simulated gate at varying brain-HU cutoffs ---")
    print(f"  {'cutoff':>6s} {'matched_pass':>13s} {'orphan_pass':>12s}  "
          f"({'TP rate':>8s} / {'FP rate':>7s})")
    if matched_vals and orphan_vals:
        for cutoff in (50, 80, 100, 150, 200, 300):
            m_pass = sum(1 for v in matched_vals if v <= cutoff)
            o_pass = sum(1 for v in orphan_vals if v <= cutoff)
            print(f"  {cutoff:>6.0f} "
                  f"{m_pass:>5d}/{len(matched_vals):<5d} ({100*m_pass/len(matched_vals):>5.1f}%)  "
                  f"{o_pass:>3d}/{len(orphan_vals):<3d} ({100*o_pass/max(1,len(orphan_vals)):>5.1f}%)")

    # --- fraction-in-head: geometric discriminator -----------------
    print(f"\n=== fraction-in-head (median over axis): annulus voxels"
          f" with hull_dist > 0 ===")
    print(f"  Real intracranial: ~ 1.0 (annulus fully inside head)")
    print(f"  Hull-skim FPs: ~ 0.5 (annulus straddles head boundary)\n")
    matched_fih = [r["frac_annulus_in_head"] for r in records
                    if r.get("matched")
                    and "frac_annulus_in_head" in r
                    and r["frac_annulus_in_head"]
                    == r["frac_annulus_in_head"]]
    orphan_fih = [r["frac_annulus_in_head"] for r in records
                   if not r.get("matched")
                   and "frac_annulus_in_head" in r
                   and r["frac_annulus_in_head"]
                   == r["frac_annulus_in_head"]]
    _percentiles(matched_fih, f"  matched (n={len(matched_fih)})")
    _percentiles(orphan_fih, f"  orphan  (n={len(orphan_fih)})")
    print(f"\n  --- simulated gate at varying frac-in-head cutoffs ---")
    print(f"  {'cutoff':>6s} {'matched_pass':>13s} {'orphan_pass':>12s}")
    if matched_fih and orphan_fih:
        for cutoff in (0.55, 0.65, 0.75, 0.85, 0.95):
            m_pass = sum(1 for v in matched_fih if v >= cutoff)
            o_pass = sum(1 for v in orphan_fih if v >= cutoff)
            print(f"  >={cutoff:.2f}  "
                  f"{m_pass:>5d}/{len(matched_fih):<5d} ({100*m_pass/len(matched_fih):>5.1f}%)  "
                  f"{o_pass:>3d}/{len(orphan_fih):<3d} ({100*o_pass/max(1,len(orphan_fih)):>5.1f}%)")

    print(f"\n=== HU-ring at existing skull_entry (>500 = in bone) ===")
    for src in sorted(by_src):
        vals = [r["existing_ring_hu"] for r in by_src[src]
                if "existing_ring_hu" in r]
        _percentiles(vals, f"  bolt_source={src:>11s}")

    print(f"\n=== hull_dist at existing skull_entry ===")
    for src in sorted(by_src):
        vals = [r["existing_hull_d"] for r in by_src[src]
                if "existing_hull_d" in r]
        _percentiles(vals, f"  bolt_source={src:>11s}")

    # Outliers where HU and Frangi disagree by > 5 mm — reveals where
    # one signal is more reliable than the other.
    print(f"\n=== HU vs Frangi disagreements > 5 mm (sorted by gap desc) ===")
    print(f"{'subj':>4s} {'ti':>3s} {'bolt':>11s}  "
          f"{'|HU-Frangi|':>11s}  {'HU_d':>5s} {'Fr_d':>5s} "
          f"{'existing_d':>10s}")
    big = sorted(
        (r for r in records
         if "dist_hu_to_frangi" in r and r["dist_hu_to_frangi"] > 5.0),
        key=lambda r: -r["dist_hu_to_frangi"],
    )[:30]
    for r in big:
        hu_d = r.get("hu_dura_hull_d", float("nan"))
        fr_d = r.get("frangi_gap_hull_d", float("nan"))
        ex_d = r.get("existing_hull_d", float("nan"))
        print(f"{r['subject']:>4s} {r['ti']:>3d} {r['bolt_source']:>11s}  "
              f"{r['dist_hu_to_frangi']:>11.2f}  "
              f"{hu_d:>5.1f} {fr_d:>5.1f} {ex_d:>10.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
