"""CC labeling probe for the ``cross-shank merging'' fix in swap #1.

Question
--------
If we threshold the LoG field and label connected components, how do
real shanks' contacts distribute across CCs?

Two axes:
  * **Cross-shank merges** (bad for us): do any two shanks' contacts
    share a CC? That's the T4-X11-style merging — geometric walker
    currently stitches across the gap, a CC check would reject it.
  * **Same-shank fragmentation** (cost of CC check): does one real
    shank's contacts split across multiple CCs? If yes, a naive
    ``all-inliers-must-share-one-CC'' rule kills real detections.
    CC-graph chaining is needed.

Both behaviours depend on threshold. Too low: contacts + skull + other
shanks all merge into one megalithic CC. Too high: contacts fragment
along the wire.

Run
---
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
        tests/deep_core/probe_cc_labeling.py [subject...]

Default subjects: T4 (primary X11 case), T1, T22 (regression pins).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper")
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "PostopCTLocalization"))

import numpy as np
import SimpleITK as sitk

from rosa_detect.contact_pitch_v1_fit import (
    build_masks, log_sigma,
    LOG_SIGMA_MM, HU_CLIP_MAX,
)
from shank_core.io import image_ijk_ras_matrices


def _cc_label_26(binary_kji: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 3D connected components with 26-connectivity using SITK.
    Returns (labels_kji, n_ccs).
    """
    mask_img = sitk.GetImageFromArray(binary_kji.astype(np.uint8))
    cc_img = sitk.ConnectedComponent(mask_img, True)
    labels = sitk.GetArrayFromImage(cc_img).astype(np.int32)
    n_ccs = int(labels.max())
    return labels, n_ccs

DATASET_ROOT = Path(os.environ.get(
    "ROSA_SEEG_DATASET",
    "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
))
CT_DIR = DATASET_ROOT / "contact_label_dataset" / "ct"
LABELS_DIR = DATASET_ROOT / "contact_label_dataset" / "labels"

# LoG thresholds to test (signed values LoG <= -T).
LOG_THRESHOLDS = [300.0, 500.0, 800.0, 1000.0]


def _load_gt_contacts(subject: str) -> list[dict]:
    """Read GT contacts for one subject. Returns list of dicts with
    keys ``shank`` (str) and ``ras`` (np.ndarray[3])."""
    p = LABELS_DIR / f"{subject}_contacts.tsv"
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            try:
                xyz = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=float,
                )
            except (KeyError, ValueError):
                continue
            out.append({"shank": row["shank"], "ras": xyz})
    return out


def _ras_to_voxel(ras_pts: np.ndarray, ras_to_ijk: np.ndarray,
                   shape_kji: tuple) -> np.ndarray:
    """RAS -> integer (k,j,i) indices clipped to volume shape.

    ``ras_to_ijk`` is the 4x4 canonical matrix returned by
    ``shank_core.io.image_ijk_ras_matrices`` (matches the convention
    the fit pipeline uses). It maps RAS -> IJK where the output
    components are (i, j, k) in that order.
    """
    h = np.concatenate([ras_pts, np.ones((ras_pts.shape[0], 1))], axis=1)
    ijk = (ras_to_ijk @ h.T).T[:, :3]
    K, J, I = shape_kji
    ii = np.clip(np.round(ijk[:, 0]).astype(int), 0, I - 1)
    jj = np.clip(np.round(ijk[:, 1]).astype(int), 0, J - 1)
    kk = np.clip(np.round(ijk[:, 2]).astype(int), 0, K - 1)
    return np.stack([kk, jj, ii], axis=1)


def _probe_subject(subject: str) -> None:
    ct_path = CT_DIR / f"{subject}_ct.nii.gz"
    if not ct_path.exists():
        print(f"{subject}: CT not found at {ct_path}")
        return

    t0 = time.perf_counter()
    print(f"\n{'=' * 80}")
    print(f"SUBJECT {subject}")
    print("=" * 80)

    img = sitk.ReadImage(str(ct_path))
    img = sitk.Clamp(img, lowerBound=-1024.0, upperBound=HU_CLIP_MAX)
    hull, intracranial, dist_arr = build_masks(img)
    log1 = log_sigma(img, sigma_mm=LOG_SIGMA_MM)
    _ijk_to_ras, ras_to_ijk = image_ijk_ras_matrices(img)
    ras_to_ijk = np.asarray(ras_to_ijk, dtype=float)

    gt_contacts = _load_gt_contacts(subject)
    if not gt_contacts:
        print(f"{subject}: NO GT CONTACTS")
        return
    by_shank = defaultdict(list)
    for c in gt_contacts:
        by_shank[c["shank"]].append(c["ras"])
    print(
        f"GT: {len(gt_contacts)} contacts across "
        f"{len(by_shank)} shanks — "
        + ", ".join(f"{s}({len(p)})" for s, p in sorted(by_shank.items()))
    )
    print(f"preprocess wall: {time.perf_counter() - t0:.1f}s")

    # Map every GT contact to its voxel — then snap each voxel to
    # the local LoG minimum within SNAP_RADIUS_MM. Manual GT labels
    # (e.g., T4) can sit 1-3 mm off the true metal peak; without this
    # snap, most "unchanged" nominal contacts land in the LoG halo and
    # the CC lookup returns 0 (background).
    SNAP_RADIUS_MM = 3
    ras_pts = np.array([c["ras"] for c in gt_contacts], dtype=float)
    nominal_voxels = _ras_to_voxel(ras_pts, ras_to_ijk, log1.shape)
    contact_shanks = [c["shank"] for c in gt_contacts]

    K, J, I = log1.shape
    voxels = np.zeros_like(nominal_voxels)
    snap_dists_mm = np.zeros(len(gt_contacts), dtype=float)
    for n, (kk, jj, ii) in enumerate(nominal_voxels):
        r = SNAP_RADIUS_MM
        k0, k1 = max(0, kk - r), min(K, kk + r + 1)
        j0, j1 = max(0, jj - r), min(J, jj + r + 1)
        i0, i1 = max(0, ii - r), min(I, ii + r + 1)
        nb = log1[k0:k1, j0:j1, i0:i1]
        flat_idx = int(np.argmin(nb))
        dk, dj, di = np.unravel_index(flat_idx, nb.shape)
        voxels[n] = (k0 + dk, j0 + dj, i0 + di)
        snap_dists_mm[n] = float(np.linalg.norm([
            (k0 + dk) - kk, (j0 + dj) - jj, (i0 + di) - ii,
        ]))

    # Diagnostic: distribution of LoG + hull-distance at GT contact
    # voxels after snap-to-minimum.
    logs_at_contacts = -log1[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    dists_at_contacts = dist_arr[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    hull_at_contacts = hull[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    print(
        f"\nDiagnostic: GT-contact voxel stats after snap-to-min "
        f"(radius={SNAP_RADIUS_MM} mm):\n"
        f"  snap distance (mm): min={snap_dists_mm.min():.1f}  "
        f"median={np.median(snap_dists_mm):.1f}  "
        f"mean={snap_dists_mm.mean():.1f}  "
        f"max={snap_dists_mm.max():.1f}  "
        f"(#no-snap={int((snap_dists_mm == 0).sum())})\n"
        f"  A @ contacts ( = -LoG):  min={logs_at_contacts.min():+6.0f}  "
        f"median={np.median(logs_at_contacts):+6.0f}  "
        f"mean={logs_at_contacts.mean():+6.0f}  max={logs_at_contacts.max():+6.0f}\n"
        f"  dist @ contacts (hull distance mm):  "
        f"min={dists_at_contacts.min():+6.1f}  "
        f"median={np.median(dists_at_contacts):+6.1f}  "
        f"mean={dists_at_contacts.mean():+6.1f}  "
        f"max={dists_at_contacts.max():+6.1f}\n"
        f"  contacts in hull mask:         "
        f"{int(hull_at_contacts.sum()):>4d}/{len(gt_contacts)}\n"
        f"  contacts in intracranial mask: "
        f"{int((dists_at_contacts >= 10).sum()):>4d}/{len(gt_contacts)}\n"
        f"  contacts with LoG <= -300:     "
        f"{int((logs_at_contacts >= 300).sum()):>4d}/{len(gt_contacts)}\n"
        f"  contacts with LoG <= -500:     "
        f"{int((logs_at_contacts >= 500).sum()):>4d}/{len(gt_contacts)}"
    )

    print(
        f"\n{'mask':>6s} {'thr':>4s}  {'tot_ccs':>8s} {'max_vox':>8s} {'avg_vox':>7s}  "
        f"{'frag_shanks':>12s} {'avg_ccs/shank':>14s}  "
        f"{'merged_pairs':>12s}"
    )
    print("-" * 90)

    mask_variants = [("hull", hull), ("intra", intracranial)]
    for mask_name, base_mask in mask_variants:
      for thr in LOG_THRESHOLDS:
        mask = (log1 <= -thr) & base_mask
        labels, n_ccs = _cc_label_26(mask)

        # CC label per GT contact voxel.
        contact_cc = labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]

        # Per-shank CC tally, excluding background (0 = not in any CC).
        shank_to_ccs: dict[str, set[int]] = defaultdict(set)
        cc_to_shanks: dict[int, set[str]] = defaultdict(set)
        miss = 0
        for s, c in zip(contact_shanks, contact_cc):
            if int(c) == 0:
                miss += 1
                continue
            shank_to_ccs[s].add(int(c))
            cc_to_shanks[int(c)].add(s)

        frag_shanks = sum(1 for cs in shank_to_ccs.values() if len(cs) > 1)
        avg_ccs_per_shank = (
            float(np.mean([len(cs) for cs in shank_to_ccs.values()]))
            if shank_to_ccs else 0.0
        )
        merged_ccs = [cc for cc, shanks in cc_to_shanks.items() if len(shanks) > 1]
        merged_pairs = set()
        for cc in merged_ccs:
            sorted_shanks = sorted(cc_to_shanks[cc])
            for i in range(len(sorted_shanks)):
                for j in range(i + 1, len(sorted_shanks)):
                    merged_pairs.add((sorted_shanks[i], sorted_shanks[j]))

        # CC size stats (only non-zero).
        if n_ccs > 0:
            sizes = np.bincount(labels.ravel())[1:]  # drop background
            sz_max = int(sizes.max()) if sizes.size else 0
            sz_mean = float(sizes.mean()) if sizes.size else 0.0
        else:
            sz_max = 0
            sz_mean = 0.0

        print(
            f"{mask_name:>6s} {thr:4.0f}  {n_ccs:>8d} {sz_max:>8d} {sz_mean:>7.1f}  "
            f"{frag_shanks:>4d}/{len(shank_to_ccs):<3d}      "
            f"{avg_ccs_per_shank:>6.2f}    "
            f"  {len(merged_pairs):>10d}   "
            f"(contacts_in_bg={miss})"
        )
        if merged_pairs:
            print(f"         merged_pairs: {sorted(merged_pairs)}")
        # Detail on fragmented shanks.
        frag_detail = {
            s: len(cs) for s, cs in shank_to_ccs.items() if len(cs) > 1
        }
        if frag_detail:
            print(
                "         fragmented: "
                + ", ".join(f"{s}({n})" for s, n in sorted(frag_detail.items()))
            )


def main():
    subjects = sys.argv[1:] if len(sys.argv) > 1 else ["T4", "T1", "T22"]
    for s in subjects:
        _probe_subject(s)


if __name__ == "__main__":
    main()
