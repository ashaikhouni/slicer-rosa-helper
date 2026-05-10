"""Integration tests pinning the notebook's multi-subject numbers.

These tests are **gated on dataset env vars** — they're skipped on machines
without the SEEG/AMC datasets. On CI / dev boxes that have them, they're
the primary regression guard for the staged pipeline.

Tests:

* AMC88 mode-4 seeded → 8/8 LoG band ≥ medium, ≥7/8 in `high`.
* T18  mode-4 seeded → 13/13 LoG model picks correct (curated GT only).

Both use ``place_seeg`` mode 4 — the only mode wired in Session 1.

Numbers come from ``project_v3_staged_scoring_2026-05-09.md`` and the
``handoff_v3_production_lift_2026-05-09.md`` summary.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

AMC_ROOT = Path(os.environ.get("ROSA_AMC_TESTING_ROOT", "/Users/ammar/Documents/testing"))
SEEG_ROOT = Path(os.environ.get("ROSA_SEEG_DATASET",
                                 "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization"))


def _amc_subject_available(sid: str) -> bool:
    d = AMC_ROOT / sid
    if not d.is_dir():
        return False
    ct = next(iter(d.glob("*_CT.nii.gz")), None) or next(iter(d.glob("*.nii.gz")), None)
    elec = d / "Electrodes"
    if not elec.is_dir():
        elec = d / "electrodes"
    return ct is not None and elec.is_dir()


def _curated_gt_available(sid: str) -> bool:
    p = SEEG_ROOT / "contact_label_dataset" / "rosa_helper_import" / sid / f"{sid}_GT_aligned_world_coords.txt"
    return p.exists()


def _gt_axis_from_contacts(contacts):
    pts = np.asarray(contacts, dtype=float)
    cm = pts.mean(axis=0)
    cn = pts - cm
    _, _, vh = np.linalg.svd(cn, full_matrices=False)
    d = vh[0] / np.linalg.norm(vh[0])
    pr = cn @ d
    if pr[0] > pr[-1]:
        d = -d; pr = -pr
    return cm + d * float(pr.min()), cm + d * float(pr.max())


def _load_dat(path: Path):
    out = []
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 3:
            break
        try:
            out.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            break
    return np.asarray(out, dtype=float)


def _parse_curated_gt(path: Path) -> list[dict]:
    """Mirrors ``resolve_subject`` from the notebook builder."""
    out_by_traj: dict[str, list[dict]] = {}
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) < 9:
            continue
        try:
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
        except ValueError:
            continue
        traj = parts[0]
        model = parts[12] if len(parts) >= 13 else None
        out_by_traj.setdefault(traj, []).append({"x": x, "y": y, "z": z, "model": model})
    out = []
    for name, contacts in sorted(out_by_traj.items()):
        if len(contacts) < 2:
            continue
        pts = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], dtype=float)
        s, e = _gt_axis_from_contacts(pts)
        out.append({
            "name": name, "contacts_ras": pts,
            "start_ras": s, "end_ras": e,
            "model_id": contacts[0].get("model"),
        })
    return out


def _amc_subject(sid: str):
    """Resolve AMC subject CT + GT shanks. Returns (ct_path, gt_list, strategy)."""
    d = AMC_ROOT / sid
    ct = next(iter(d.glob("*_CT.nii.gz")), None) or next(iter(d.glob("*.nii.gz")), None)
    elec = d / "Electrodes"
    if not elec.is_dir():
        elec = d / "electrodes"
    gt = []
    for f in sorted(elec.glob("*.dat")):
        if f.stem.lower() == "elecpointset":
            continue
        pts = _load_dat(f)
        if pts.shape[0] >= 2:
            s, e = _gt_axis_from_contacts(pts)
            gt.append({"name": f.stem, "contacts_ras": pts,
                       "start_ras": s, "end_ras": e})
    strategy = "dixi" if sid == "T22" else "pmt_35"
    return str(ct), gt, strategy


def _t_subject_curated(sid: str):
    """Resolve T-series subject with curated GT.

    Reuses the ``eval_seeg_localization`` manifest reader; tries to import it
    lazily so the test still skips cleanly when the helper isn't installed.
    """
    try:
        from eval_seeg_localization import iter_subject_rows  # type: ignore
    except ImportError:
        return None, None, None
    rows = iter_subject_rows(SEEG_ROOT, {sid})
    if not rows:
        return None, None, None
    row = rows[0]
    curated = (SEEG_ROOT / "contact_label_dataset" / "rosa_helper_import"
               / sid / f"{sid}_GT_aligned_world_coords.txt")
    if not curated.exists():
        return None, None, None
    ct_path = row.get("source_ct_file") or row["ct_path"]
    return ct_path, _parse_curated_gt(curated), "dixi"


def _build_features(ct_path: str):
    """Mirrors the notebook's ``compute_features + bolts`` block."""
    import SimpleITK as sitk
    from shank_core.io import image_ijk_ras_matrices
    from rosa_detect import guided_fit_engine as gfe
    from rosa_detect import contact_pitch_v1_fit as f1
    from rosa_detect.primitives.bolt_anchor import (
        BOLT_HULL_PROXIMITY_MM, METAL_BOLT_THRESHOLD, extract_bolt_candidates,
    )

    img = sitk.ReadImage(ct_path)
    i2r_in, r2i_in = image_ijk_ras_matrices(img)
    features = gfe.compute_features(img, np.asarray(i2r_in), np.asarray(r2i_in))
    i2r = np.asarray(features["ijk_to_ras_mat"])
    r2i = np.asarray(features["ras_to_ijk_mat"])

    metal_evidence = f1.compute_metal_evidence_volume(
        features["log"], features["ct_arr_kji"],
    )
    bolts, _ = extract_bolt_candidates(
        features["log"], features["head_distance"], i2r, img.GetSpacing(),
        ras_to_ijk_mat=r2i, ct_arr=metal_evidence,
        hu_threshold=METAL_BOLT_THRESHOLD, hull_proximity_mm=BOLT_HULL_PROXIMITY_MM,
    )
    return features, bolts


def _seeds_from_gt(gt: list[dict]) -> list:
    from rosa_core.placement_modes import Seed
    return [
        Seed(name=g["name"], start_ras=g["start_ras"], end_ras=g["end_ras"],
             model_id=g.get("model_id"))
        for g in gt
    ]


def _greedy_gt_match(
    gt: list[dict],
    trajectories,
    *,
    angle_tol_deg: float = 12.0,
    perp_tol_mm: float = 8.0,
) -> dict[int, int]:
    """Match each GT shank to the best emission by axis angle + perp.

    Returns ``{emission_idx: gt_idx}``. Same algorithm the notebook uses.
    """
    pairs = []
    for gi, g in enumerate(gt):
        ax_g = g["end_ras"] - g["start_ras"]
        ax_g = ax_g / max(np.linalg.norm(ax_g), 1e-9)
        mid_g = 0.5 * (g["start_ras"] + g["end_ras"])
        for ei, t in enumerate(trajectories):
            ax_e = t.end_ras - t.start_ras
            ax_e = ax_e / max(np.linalg.norm(ax_e), 1e-9)
            mid_e = 0.5 * (t.start_ras + t.end_ras)
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ax_g, ax_e)))))))
            v = mid_g - mid_e
            perp = v - float(np.dot(v, ax_e)) * ax_e
            d_perp = float(np.linalg.norm(perp))
            if ang <= angle_tol_deg and d_perp <= perp_tol_mm:
                pairs.append((ang + d_perp, gi, ei))
    pairs.sort(key=lambda p: p[0])
    used_g, used_e = set(), set()
    out: dict[int, int] = {}
    for _s, gi, ei in pairs:
        if gi in used_g or ei in used_e:
            continue
        used_g.add(gi); used_e.add(ei)
        out[ei] = gi
    return out


# ---------------------------------------------------------------------
# AMC88 — mode 5 (seeded, library-match) AND mode 1 (full auto)
# ---------------------------------------------------------------------


@unittest.skipUnless(_amc_subject_available("AMC88"),
                     f"AMC88 not found at {AMC_ROOT}")
class Amc88LogTests(unittest.TestCase):
    """AMC88 with LoG sampler — notebook says 8/8 in 'high' band.

    Two angles:
      * Mode 5 — seeded with GT axes, library matches each shank.
      * Mode 1 — full auto (candidate-seed gen + two-pass placement).

    Notebook number: 8/8 GT-matched in 'high', 0 LoG orphans in 'high'.
    """

    def setUp(self):
        from rosa_core.contact_placement import sample_neg_log_max
        from rosa_core.electrode_classifier import filter_models_for_strategy
        from rosa_core import load_electrode_library

        ct, gt, strategy = _amc_subject("AMC88")
        self.ct = ct
        self.gt = gt
        self.strategy = strategy
        self.features, self.bolts = _build_features(ct)
        self.library = filter_models_for_strategy(
            load_electrode_library()["models"], strategy,
        )
        self.sample_fn = sample_neg_log_max

    def test_amc88_mode5_snaps_seeds_to_v1_candidates(self):
        """Mode 5 (per user 2026-05-09 spec): seed should snap to closest
        mode-1 emission; unmatched seeds drop; unmatched candidates drop.

        AMC88 has 8 GT shanks → ≥8 v1 emissions → all 8 GT seeds should
        snap. Inheriting v1's bolt-anchored geometry, the matched filter
        gets the same model picks as mode 1.
        """
        from rosa_core.placement_modes import place_seeg

        seeds = _seeds_from_gt(self.gt)
        batch = place_seeg(
            self.ct, seeds=seeds,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
        )
        self.assertEqual(batch.diagnostics["mode"], 5)
        diag = batch.diagnostics["mode5"]
        self.assertEqual(diag["n_input_seeds"], len(self.gt))
        # All 8 GT seeds should snap to a v1 candidate.
        self.assertEqual(
            diag["n_seeds_matched"], len(self.gt),
            f"all {len(self.gt)} GT seeds should snap to a v1 candidate; "
            f"got {diag['n_seeds_matched']}. diag={diag}",
        )
        # Each emission should have a model_id and placed contacts —
        # snapped from mode 1, so geometry is bolt-anchored.
        for t in batch.trajectories:
            self.assertIsNotNone(t.model_id, f"snap should produce model_id: {t.name}")
            self.assertGreater(
                len(t.contacts_ras), 0,
                f"snapped emission should have contacts: {t.name}",
            )
        # Snap-to-v1 inherits mode-1's geometry, so band counts match
        # mode 1 exactly: 8/8 in 'high' on AMC88.
        bands = [t.band for t in batch.trajectories]
        n_high = bands.count("high")
        self.assertEqual(
            n_high, len(self.gt),
            f"snap-to-v1 mode 5 should match mode-1 number {len(self.gt)}/"
            f"{len(self.gt)} in 'high'; got bands={bands}",
        )

    def test_amc88_mode1_no_orphans_in_high(self):
        """Mode 1 full auto: generate seeds + place. Notebook: 0 LoG orphans
        in 'high', 8/8 GT matched in 'high'.

        Pin: at least 8 emissions reach 'high' (the GT shanks); orphan check
        requires GT-axis matching which we mirror inline.
        """
        from rosa_core.placement_modes import place_seeg

        batch = place_seeg(
            self.ct,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
            band_floor="low",  # don't filter so we can see orphans
        )
        self.assertEqual(batch.diagnostics["mode"], 1)

        # Greedy-match GT to the dispatcher's emissions on axis angle + mid
        # offset. Same matching rules as the notebook.
        gt_axis_pairs = []
        for gi, g in enumerate(self.gt):
            axis_g = (g["end_ras"] - g["start_ras"])
            axis_g = axis_g / max(np.linalg.norm(axis_g), 1e-9)
            mid_g = 0.5 * (g["start_ras"] + g["end_ras"])
            for ei, t in enumerate(batch.trajectories):
                axis_e = (t.end_ras - t.start_ras)
                axis_e = axis_e / max(np.linalg.norm(axis_e), 1e-9)
                mid_e = 0.5 * (t.start_ras + t.end_ras)
                ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(axis_g, axis_e)))))))
                v = mid_g - mid_e
                perp = v - float(np.dot(v, axis_e)) * axis_e
                d_perp = float(np.linalg.norm(perp))
                if ang <= 12.0 and d_perp <= 8.0:
                    gt_axis_pairs.append((ang + d_perp, gi, ei))
        gt_axis_pairs.sort(key=lambda p: p[0])
        used_g, used_e = set(), set()
        gt_for_emission = {}
        for _s, gi, ei in gt_axis_pairs:
            if gi in used_g or ei in used_e:
                continue
            used_g.add(gi); used_e.add(ei)
            gt_for_emission[ei] = self.gt[gi]["name"]

        n_matched = len(gt_for_emission)
        n_matched_high = sum(
            1 for ei, _name in gt_for_emission.items()
            if batch.trajectories[ei].band == "high"
        )
        n_orphan_high = sum(
            1 for ei, t in enumerate(batch.trajectories)
            if ei not in gt_for_emission and t.band == "high"
        )
        # Strict notebook parity: 8/8 in 'high', 0 LoG orphans in 'high'.
        self.assertEqual(n_matched, len(self.gt),
                         f"all {len(self.gt)} GT should match an emission; got {n_matched}")
        self.assertEqual(n_matched_high, len(self.gt),
                         f"all {len(self.gt)} GT should land in 'high' "
                         f"(notebook number); got {n_matched_high}")
        self.assertEqual(n_orphan_high, 0,
                         f"notebook reports 0 LoG orphans in 'high'; got {n_orphan_high}")

    def test_amc88_mode2_top_n_covers_gt_shanks(self):
        """Mode 2 (count constraint): n_expected = 8 GT shanks → top 8
        emissions by compound score should include all 8 GT-matched ones.
        Real-data check: AMC88's GT shanks dominate the high-band, so
        top-8 covers all of them.
        """
        from rosa_core.placement_modes import place_seeg
        n_gt = len(self.gt)
        batch = place_seeg(
            self.ct, n_expected=n_gt,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
        )
        self.assertEqual(batch.diagnostics["mode"], 2)
        self.assertLessEqual(len(batch.trajectories), n_gt,
                             f"mode 2 should return at most n_expected={n_gt}")
        e_to_g = _greedy_gt_match(self.gt, batch.trajectories)
        self.assertEqual(
            len(e_to_g), n_gt,
            f"top-{n_gt} should cover all {n_gt} GT shanks; "
            f"got {len(e_to_g)}/{n_gt} matched.",
        )


# ---------------------------------------------------------------------
# T18 — modes 1/2/3/4/5 against curated GT (model_id labels available)
# ---------------------------------------------------------------------


@unittest.skipUnless(_curated_gt_available("T18"),
                     f"T18 curated GT not found under {SEEG_ROOT}")
class T18LogTests(unittest.TestCase):
    """T18 with LoG sampler — notebook says 13/13 model picks (mode 5).

    Mode dispatch: T18 curated GT MAY include per-shank model_id. If it
    does, this is mode 4 (model vouched). If not, mode 5 (library match).
    Either way, pin ≥12/13 correct picks (notebook: 13/13).

    Skips if ``eval_seeg_localization`` isn't installed.
    """

    def setUp(self):
        ct, gt, strategy = _t_subject_curated("T18")
        if ct is None:
            self.skipTest("eval_seeg_localization or T18 manifest unavailable")
        self.ct = ct
        self.gt = gt
        self.strategy = strategy
        self.features, self.bolts = _build_features(ct)

        from rosa_core.electrode_classifier import filter_models_for_strategy
        from rosa_core import load_electrode_library
        from rosa_core.contact_placement import sample_neg_log_max

        self.library = filter_models_for_strategy(
            load_electrode_library()["models"], strategy,
        )
        self.sample_fn = sample_neg_log_max

    def test_t18_mode5_snap_matches_notebook_picks(self):
        """Mode 5 (snap-to-v1): T18 GT seeds snap to v1 emissions, then
        each snapped emission's matched-filter pick is compared against GT
        model_id. With snap, mode 5 should match mode 1's notebook number
        (13/13 picks correct on T18 LoG side).

        Without snap (per-seed placement on raw GT axes), mode 5 gets
        ~9/13 — see Session 2 commit message.
        """
        from rosa_core.placement_modes import place_seeg

        gt_with_models = [g for g in self.gt if g.get("model_id")]
        if not gt_with_models:
            self.skipTest("T18 curated GT lacks per-shank model_id labels")

        # Strip model_id so the dispatcher routes to mode 5 (library match
        # via snap), not mode 4 (model vouched).
        seeds = [
            type(s)(name=s.name, start_ras=s.start_ras, end_ras=s.end_ras,
                    model_id=None,
                    seeder_label=s.seeder_label,
                    seeder_confidence=s.seeder_confidence,
                    seeder_model=s.seeder_model)
            for s in _seeds_from_gt(self.gt)
        ]
        batch = place_seeg(
            self.ct, seeds=seeds,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
        )
        self.assertEqual(batch.diagnostics["mode"], 5)

        # Map snapped seed name → GT model_id.
        gt_model_by_name = {g["name"]: g.get("model_id") for g in self.gt}
        n_correct = sum(
            1 for t in batch.trajectories
            if gt_model_by_name.get(t.name) and t.model_id == gt_model_by_name[t.name]
        )
        n_with_model = len(gt_with_models)
        # Snap-to-v1 inherits mode-1's geometry → same picks as mode 1.
        # Strict notebook parity: 13/13.
        self.assertEqual(
            n_correct, n_with_model,
            f"mode-5 snap should match notebook {n_with_model}/{n_with_model}; "
            f"got {n_correct}. picks: "
            f"{[(t.name, t.model_id, gt_model_by_name.get(t.name)) for t in batch.trajectories]}",
        )

    def test_t18_mode1_model_picks_at_least_12_of_13(self):
        """Mode 1 (full auto via candidate-seeds + two-pass): notebook 13/13.

        This matches the notebook's actual flow — v1 emission seeds carry
        bolt-anchored centerlines + confidence labels. We then GT-match by
        axis to compare placer-picked model_id against curated GT model_id.

        Why not mode 5: GT-derived axes (PCA over contacts) are NOT
        equivalent to v1-emitted seeds — the placer's anchor walker
        assumes start_ras is at the bolt, which v1 guarantees but GT-PCA
        does not. Mode 5 with GT axes systematically underperforms.
        """
        from rosa_core.placement_modes import place_seeg

        # Curated GT must have model labels for this test to mean anything.
        gt_model_count = sum(1 for g in self.gt if g.get("model_id"))
        if gt_model_count == 0:
            self.skipTest("T18 curated GT lacks per-shank model_id labels")

        batch = place_seeg(
            self.ct,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
            band_floor="low",  # see all emissions for GT match
        )
        self.assertEqual(batch.diagnostics["mode"], 1)

        # Greedy axis-match each GT to a placer emission.
        gt_to_emission = {}
        used_e = set()
        pairs = []
        for gi, g in enumerate(self.gt):
            ax_g = (g["end_ras"] - g["start_ras"])
            ax_g = ax_g / max(np.linalg.norm(ax_g), 1e-9)
            mid_g = 0.5 * (g["start_ras"] + g["end_ras"])
            for ei, t in enumerate(batch.trajectories):
                ax_e = (t.end_ras - t.start_ras)
                ax_e = ax_e / max(np.linalg.norm(ax_e), 1e-9)
                mid_e = 0.5 * (t.start_ras + t.end_ras)
                ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ax_g, ax_e)))))))
                v = mid_g - mid_e
                perp = v - float(np.dot(v, ax_e)) * ax_e
                d_perp = float(np.linalg.norm(perp))
                if ang <= 12.0 and d_perp <= 8.0:
                    pairs.append((ang + d_perp, gi, ei))
        pairs.sort(key=lambda p: p[0])
        for _s, gi, ei in pairs:
            if gi in gt_to_emission or ei in used_e:
                continue
            gt_to_emission[gi] = ei
            used_e.add(ei)

        # For matched GT shanks, compare picked model_id to GT model_id.
        n_correct = 0
        details = []
        for gi, ei in gt_to_emission.items():
            gt_model = self.gt[gi].get("model_id")
            picked = batch.trajectories[ei].model_id
            details.append((self.gt[gi]["name"], picked, gt_model))
            if gt_model and picked == gt_model:
                n_correct += 1
        # Strict: all GT model picks should be correct (notebook 13/13 on T18).
        self.assertEqual(
            n_correct, gt_model_count,
            f"all {gt_model_count}/{gt_model_count} model picks should match "
            f"GT (notebook number); got {n_correct}. picks: {details}",
        )

    def test_t18_mode2_top_n_covers_gt_shanks(self):
        """Mode 2: pass n_expected = 13 GT shanks → top-13 emissions by
        compound score should include all 13 GT-matched ones (real-data
        check: T18 GT shanks dominate compound rankings)."""
        from rosa_core.placement_modes import place_seeg
        n_gt = len(self.gt)
        batch = place_seeg(
            self.ct, n_expected=n_gt,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
        )
        self.assertEqual(batch.diagnostics["mode"], 2)
        self.assertLessEqual(len(batch.trajectories), n_gt)
        e_to_g = _greedy_gt_match(self.gt, batch.trajectories)
        self.assertEqual(
            len(e_to_g), n_gt,
            f"top-{n_gt} should cover all {n_gt} GT shanks; "
            f"got {len(e_to_g)}/{n_gt} matched.",
        )

    def test_t18_mode3_named_assignment_matches_gt_models(self):
        """Mode 3: ``expected = [(name, model_id)]`` from curated GT →
        the dispatcher assigns the best matching candidate per name.

        Slack of 1 here is REAL: T18 has 4 duplicate models in expected
        (DIXI-15AM × 6, DIXI-18AM × 3, DIXI-18CM × 2, DIXI-8AM × 2). When
        multiple expected entries share a model, the greedy assignment
        picks the highest-scoring matching candidate first, then the next,
        etc. — but the per-name correspondence among same-model entries
        is arbitrary, so the "correct picks" count can flip ±1 across
        candidate orderings without changing the underlying algorithm.

        On the current run we get 13/13; we pin ≥12/13 to insulate from
        ordering noise (the duplicate-model warning surfaces this caveat
        in batch.diagnostics["mode3"]["duplicate_models"]).
        """
        from rosa_core.placement_modes import place_seeg
        gt_with_models = [g for g in self.gt if g.get("model_id")]
        if not gt_with_models:
            self.skipTest("T18 curated GT lacks per-shank model_id labels")

        expected = [(g["name"], g["model_id"]) for g in gt_with_models]
        batch = place_seeg(
            self.ct, expected=expected,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
        )
        self.assertEqual(batch.diagnostics["mode"], 3)
        self.assertLessEqual(
            len(batch.trajectories), len(expected),
            "mode 3 returns at most one trajectory per expected entry",
        )
        want_by_name = dict(expected)
        n_correct = sum(
            1 for t in batch.trajectories
            if t.model_id == want_by_name.get(t.name)
        )
        self.assertGreaterEqual(
            n_correct, len(expected) - 1,
            f"mode 3 should match ≥{len(expected) - 1}/{len(expected)} "
            f"requested models; got {n_correct}. picks: "
            f"{[(t.name, t.model_id, want_by_name.get(t.name)) for t in batch.trajectories]}",
        )

    def test_t18_mode4_seeded_with_model_id(self):
        """Mode 4 (snap-to-v1 + force vouched model): GT seeds with
        ``model_id`` set → snap to closest v1 candidate (inheriting bolt-
        anchored geometry), then force the matched filter to the user-
        vouched model template.

        Strict notebook parity: 13/13 placed. The previous per-seed mode 4
        rejected X09 (GT-PCA axis was reversed + too short for the matched
        filter to fit DIXI-15AM in the leftover contact zone). The snap
        path uses v1's canonical bolt → tip axis instead.
        """
        from rosa_core.placement_modes import place_seeg
        gt_with_models = [g for g in self.gt if g.get("model_id")]
        if not gt_with_models:
            self.skipTest("T18 curated GT lacks per-shank model_id labels")

        seeds = _seeds_from_gt(gt_with_models)
        for s in seeds:
            self.assertIsNotNone(s.model_id, f"mode-4 prep error: {s.name}")

        batch = place_seeg(
            self.ct, seeds=seeds,
            features=self.features, bolts=self.bolts,
            library=self.library, sample_fn=self.sample_fn,
        )
        self.assertEqual(batch.diagnostics["mode"], 4)
        self.assertEqual(len(batch.trajectories), len(seeds))
        n_seeds = len(seeds)
        n_placed = sum(
            1 for t, s in zip(batch.trajectories, seeds)
            if t.model_id == s.model_id and len(t.contacts_ras) > 0
        )
        self.assertEqual(
            n_placed, n_seeds,
            f"mode 4 (snap-to-v1) should place all {n_seeds}/{n_seeds} "
            f"GT-seeded vouched models; got {n_placed}. mode4_diag="
            f"{batch.diagnostics.get('mode4', {})}. picks: "
            f"{[(t.name, t.model_id, s.model_id, len(t.contacts_ras)) for t, s in zip(batch.trajectories, seeds)]}",
        )


if __name__ == "__main__":
    unittest.main()
