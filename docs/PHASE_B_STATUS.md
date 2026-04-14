# Deep Core Phase B Status — handoff doc

Last updated: 2026-04-13.
Replaces the previous template-matching status (commits c119c69 and earlier).

## What this document is

You are Claude Code continuing work on the `deep_core` SEEG electrode detection module in the `rosa_viewer` project. Phase B was redesigned in commits `cf455cb..18c1dde`. This doc tells you the current architecture, current numbers, and what to look at next. After reading it, run `git log --oneline -10` and `ls PostopCTLocalization/postop_ct_localization/deep_core_*.py` to orient.

Memory files at `~/.claude/projects/-Users-ammar-Dropbox-rosa-viewer/memory/` have user profile and feedback.

## Phase B in one sentence

For each proposal coming out of annulus rejection, fit a PCA axis to its atom point cloud, prune outlier atoms whose points sit far off the line, reabsorb other colinear atoms from the full pool (no axial window), orient the axis so `+axis` points toward the bolt by metal-mask scalp-exit walking, extend along the metal mask in both directions, reject any fit whose axis doesn't physically reach the scalp, and emit an intracranial trajectory plus a `bolt_ras` anchor for downstream Phase C contact placement. Proposals are processed in a priority order and each accepted fit *claims* its atoms so later proposals can't reabsorb them — this is what prevents bridged proposals from surviving.

The library of electrode models is a soft recognizer (span gate), not a generator. No contacts are placed in Phase B; that is Phase C's job.

## Architecture

### Files

- `PostopCTLocalization/postop_ct_localization/deep_core_axis_reconstruction.py` — new helper module, pure numpy, no Slicer/vtk dependencies. All the per-fit primitives.
  - `refine_axis_from_cloud(points_ras, seed_axis) -> AxisFit` — PCA line fit
  - `prune_outlier_atoms(...)` — per-atom residual filter to defend against bridged proposals
  - `reabsorb_colinear_atoms(fit, atom_pool, cfg, already_absorbed_ids)` — pulls colinear atoms by perpendicular distance only (no axial window — the whole point is to recover atoms outside the cluster)
  - `walk_metal_mask(fit, start_t, direction, ...)` — extension walk along a perpendicular tube; tracks `last_metal_t`, termination `reason` (`exit_scalp`, `exit_gap`, `exit_volume`), and `min_head_distance` reached during the walk
  - `orient_axis_by_scalp_exit(fit, ...) -> (fit_or_flipped, has_scalp_exit)` — flips axis so `+axis` is the bolt direction; returns has_scalp_exit=False when no walk reaches the scalp (used as a hard rejection gate)
  - `classify_tissue_along_axis(fit, t_values, arr_kji, ras_to_ijk_fn, cfg)` — lateral HU ring classification (air/brain/bone/metal). Used for the `brain_span` rejection gate.
  - `find_intracranial_exit_by_head_distance(...)` — current shallow endpoint helper (head_distance threshold, 15mm). Brittle; flagged for replacement.
  - `find_brain_entry_from_outside(...)` — alternative tissue-based brain-entry helper. Currently unused by Phase B because the median ring classification picks brain too shallow at burr holes; included so a Phase C stage can combine it with bolt_ras and an axis-center HU walk.
  - `library_span_match`, `library_model_contact_span_mm`, `library_model_first_contact_offset_mm` — library recognizer helpers
- `PostopCTLocalization/postop_ct_localization/deep_core_model_fit.py` — entry point used by `deep_core_v1.py`.
  - `run_model_fit_group(proposals, ..., support_atoms, blob_sample_points_ras)` — orchestration: priority sort, claim-based loop, conflict-free output
  - `_fit_one_proposal(prop, ..., blocked_atom_ids)` — per-proposal pipeline: gather cloud → PCA fit → outlier prune → reabsorb → orient by scalp exit (hard gate) → extend → classify → find shallow interface → rejection gates → emit
  - `_gather_initial_cloud(prop, atom_by_id, blob_sample_points_ras, blocked_atom_ids)` — cloud assembly from the proposal's atoms (filtering out claimed atoms) plus the proposal's tokenized blob points
  - `filter_models_by_family(library, families, min_contacts)` — DIXI filter (kept for call-site compatibility)
- `PostopCTLocalization/postop_ct_localization/deep_core_config.py` — `DeepCoreModelFitConfig` dataclass
- `CommonLib/shank_engine/pipelines/deep_core_v1.py` — `_run_model_fit` stage wrapper, `_proposals_to_trajectories` surfaces `start_ras`/`end_ras`/`bolt_ras`/`intracranial_span_mm`/`bolt_extent_mm`/`explained_atom_ids`/`axis_ras`
- `tests/deep_core/test_axis_reconstruction.py` — 18 unit tests for the helpers
- `tests/deep_core/test_pipeline_dataset.py` — end-to-end regression on T1 and T22, baselines locked at the post-redesign numbers

### Per-proposal fit pipeline (`_fit_one_proposal`)

1. **Gather initial cloud** from the proposal's `atom_id_list` (excluding `blocked_atom_ids` claimed by higher-priority fits) plus the proposal's tokenized blob points.
2. **PCA fit** the cloud → `AxisFit(center, axis, residual_rms_mm, residual_median_mm, ...)`.
3. **Outlier-atom prune**: for each atom in the proposal, compute the median perpendicular distance of its points from the current fit line. If the worst outlier is above `axis_fit_max_residual_mm` *and* substantially worse than the median, drop it and refit. Iterate until stable. This is what saves bridged proposals like RAMC where Phase A's chaining grabbed a stray atom from a parallel shank.
4. **Residual gate**: reject if `residual_median_mm > axis_fit_max_residual_mm` (1.8mm).
5. **Reabsorption**: sweep the full atom pool (minus claimed and minus initial), reabsorb any atom within `reabsorb_radial_tol_mm` (1.5mm) perpendicular of the line. For line atoms with a reliable own axis, also enforce the angular tolerance. **No axial window** — colinear contact atoms 30mm+ outside the cluster get pulled in (this is what fixes DIXI-15CM cases like LPMC).
6. **Refit** on the combined cloud if the residual stays under `axis_fit_max_residual_mm`.
7. **Cloud-extent override**: pull `fit.t_min`/`t_max` from the cloud projection bounds (slightly tighter than the PCA frame extent).
8. **Orient by scalp exit**: walk both directions, flip axis so `+axis` is the direction whose walk reaches the scalp. **If neither walk reaches the scalp, reject the fit outright** — every real electrode has a bolt that crosses the skin, so an axis that stays inside brain in both directions is either not an electrode or is a bridged proposal whose tilted axis doesn't aim at any bolt.
9. **Extend** in both directions; record `t_deep_ext` and `t_shallow_ext`.
10. **Classify tissue** along `t_values = arange(t_deep_ext, t_shallow_ext, step_mm)` for the `brain_span` rejection gate.
11. **Shallow interface**: walk `head_distance` from deep to shallow, find the first sample where it drops below `intracranial_exit_head_distance_mm` (15mm). This is `t_interface`. Brittle — see "Known weakness" below.
12. **Rejection gates**:
    - `intracranial_span = t_interface - t_deep_intra` must be ≥ `min_intracranial_span_mm` (15mm)
    - `brain_span` (longest contiguous brain run) must be ≥ same threshold
    - intracranial span must be within `[lib_min - tol, lib_max + tol]` (DIXI library, tol = 5mm). If over, **trim** to `lib_max + tol` rather than rejecting.
13. **Library span match**: closest library model by total exploration length (informational only).
14. **Emit**: `start_ras = center + axis * t_deep_intra`, `end_ras = center + axis * t_interface`, `bolt_ras = center + axis * t_shallow_ext`, plus axis, span, intracranial_span_mm, bolt_extent_mm, explained_atom_ids, best_model_id, axis residuals.

### Claim-based assignment (`run_model_fit_group`)

```
ordered = sort proposals by (-len(atom_id_list), -span)
claimed: set[int] = {}
for prop in ordered:
    fitted = _fit_one_proposal(prop, blocked_atom_ids=claimed)
    if fitted is not None:
        accepted.append(fitted)
        claimed |= set(fitted.explained_atom_ids)
return accepted  # no separate conflict resolution step
```

This replaces the segment-based perpendicular-distance conflict resolver from earlier iterations. The advantage: a long clean shank fits first, claims all its atoms, and a bridged proposal spanning that shank plus a neighbour now sees its shared atoms blocked, fails to reach a clean fit on the residual atoms, and is rejected by the scalp-exit gate.

## Current numbers

Run with `/Users/ammar/miniforge3/envs/shankdetect/bin/python3 -m unittest tests.deep_core.test_pipeline_dataset` from the helper root. The local Python venv lacks SimpleITK and vtk; you must use the `shankdetect` conda env for the dataset regression.

| Subject | Pre-redesign | Now |
|---|---|---|
| T1 loose (12 GT) | 8 | **11** |
| T1 strict (12 GT) | 3 | **5** |
| T22 loose (9 GT) | 4 | **8** |
| T22 strict (9 GT) | 1 | **2** |

Helper unit tests: 18, all passing.

### Plan-predicted fixes

| Shank | Predicted fix | Status |
|---|---|---|
| LPMC (DIXI-15CM, 68mm) | Metal-mask extension finds the shallow 38mm of bright metal beyond the deep_core shrink rind | **Fixed** by reabsorption-without-axial-window pulling deep contact atoms into the cluster |
| RPMC (DIXI-15CM) | Same pattern | **Fixed** same way |
| LAMC (close to LCMN) | Smaller group-conflict radius lets it survive | **Fixed** by claim-based assignment + scalp-exit gate |
| RAMC (drift in tip) | Endpoint from observed metal | **Borderline** — predicted with start=1.5mm, ang=1.8°, end=10.5mm (0.5mm over loose gate). Outlier-pruning recovered the axis; the residual end_error is the calibrated head_distance threshold mismatch. Phase C will fix it. |
| RHH | Held over from session | **Fixed** by `min_head_distance` walk-tracking for scalp exit (rescues sparse-mask bolts) |
| RCMN, RAI end errors | Side effect of head_distance threshold landing | **Fixed** for RCMN/RAI/LHH/LAI by switching from bone/brain HU interface to head_distance threshold |
| P11 false positive | Library span gate + bone/brain gate should reject | **Fixed** (no false-positive rows in the per-shank tally) |

### Remaining unpaired

- **T1 RAMC**: the trajectory IS predicted (`P03`, end_error 10.49mm) — it just fails the loose 10mm end_error gate by 0.5mm. Will be recovered when Phase C contact placement refines the shallow endpoint from `bolt_ras`.
- **T22 RSFG**: never proposed by Phase A. Nothing for Phase B to do.

## Config

`DeepCoreModelFitConfig` in `deep_core_config.py`. The relevant knobs:

- `axis_fit_max_residual_mm` = 1.8 (was 1.2; bumped after empirical calibration on bridged-proposal residuals)
- `reabsorb_radial_tol_mm` = 1.5
- `reabsorb_angle_tol_deg` = 5.0 (only enforced for line atoms with `axis_reliable=True`)
- `reabsorb_axial_window_mm` = 5.0 (**no longer enforced** — present for back-compat; safe to remove)
- `extension_step_mm` = 0.5
- `extension_tube_radius_mm` = 1.5
- `extension_max_gap_mm` = 3.0
- `extension_termination_gap_mm` = 5.0
- `extension_head_distance_floor_mm` = -1.0
- `scalp_exit_detect_head_distance_mm` = 5.0 (new — a walk "found the bolt" if its sampled head_distance dropped below this at any point)
- `lateral_hu_ring_radius_mm` = 3.5
- `lateral_hu_ring_samples` = 8
- `hu_air_max` / `hu_brain_max` / `hu_bone_max` = -500 / 150 / 1800
- `intracranial_exit_head_distance_mm` = 15.0 (**brittle calibration**, see below)
- `min_intracranial_span_mm` = 15.0
- `library_span_tolerance_mm` = 5.0
- `axis_conflict_radius_mm` = 1.5 (**unused** — claim-based assignment replaced segment conflict resolution; safe to remove)

## Known weakness

**`end_ras` is computed from a calibrated `head_distance` threshold (15mm).** This was empirically tuned to match where DIXI shallowest contacts land in T1/T22 (their head_distance is consistently 12–20mm), but it is not a physical signal — change electrode model, change drive depth, change subject head shape, and the 15mm calibration breaks.

The right replacement is to detect the shallowest contact directly via on-axis HU peaks (contact metal → 1900+ HU spikes spaced at known library offsets), use that to anchor the GT-side endpoint, AND emit the bolt anchor (`bolt_ras`, already done) so the contact-placement stage can refine. This is what Phase C should do. The helper `find_brain_entry_from_outside` is in place in the axis-reconstruction module ready to combine with a bolt-tip walk and an axis-center HU peak detector.

The user has explicitly tagged this as Phase C work — Phase B is *good enough* in the current loose-gate evaluation as long as `end_ras` is in the right neighborhood and `bolt_ras` is accurate. Don't try to fix it again inside Phase B; the previous session burned multiple iterations on this and ended up reverting to the calibrated threshold.

## What to look at next

In rough order of value:

1. **Phase A bridged-proposal investigation** for RAMC: Phase A's `deep_core_candidate_generation.py` produces a 4-atom proposal `[90, 91, 92, 100]` where atom 100 is a stray from a neighbouring shank and atoms 90/91/92 are clean RAMC line atoms. Phase B's outlier prune now drops atom 100 cleanly, so this isn't blocking Phase B numbers — but it's worth fixing upstream so the proposal looks right in Slicer regardless.
2. **Phase A coverage** for T22 RSFG: Phase A never proposes it. Outside Phase B's scope but a clear next target.
3. **Phase C contact placement**: the cleanest path to closing the RAMC loose gap and eliminating the head_distance calibration. Inputs from Phase B: `start_ras`, `end_ras`, `bolt_ras`, `axis_ras`, `explained_atom_ids`, `best_model_id`. Algorithm: walk on-axis HU sampled at `extension_step_mm`, detect peaks above 1500 HU, fit them against library contact-offset patterns, output refined contact positions and corrected start/end.
4. **Remove dead config fields**: `axis_conflict_radius_mm` and `reabsorb_axial_window_mm` are no longer enforced. Keep them if they're surfaced in the UI to avoid breaking saved configs; remove otherwise.
5. **Helper module unit tests for the new pieces**: `prune_outlier_atoms`, `walk_metal_mask` termination reasons, `orient_axis_by_scalp_exit` flipping logic, `min_head_distance` rescue path. Currently only the original 18 tests cover the pre-prune/pre-claim helpers.

## Probes that exist

In `tmp/` (root of `rosa_viewer`):

- `tmp/probe_phase_b_eval.py` — runs T1 and T22 through the pipeline, prints per-shank `start_error`/`end_error`/`angle_deg` and a tally categorizing failures by which gate they trip. Use this as the primary tuning lens. Run with the shankdetect env Python.
- `tmp/probe_rcmn.py` — prints per-shank axis-frame coordinates of GT vs predicted endpoints, lateral HU classification counts, head_distance profile along axis, scalp-exit walk reasons, orientation flip status. Edit the `for name in (...)` line at the bottom to target the shanks you care about.
- `tmp/probe_rhh.py` — finds proposals "in the area" of a target shank (perp/angle thresholds), traces them through every gate and reports the first rejection reason. Optional deep-dive prints colinear atoms in the pool and the metal-mask density along the axis.
- `tmp/probe_lpmc_atoms.py` — same idea, scoped to LPMC. Useful as a template for "list all atoms in proposal X plus all colinear atoms in the pool plus the metal density."
- `tmp/calibrate_phase_b.py` — original calibration script from the design phase. Has plumbing for sweeping config knobs across the dataset.

Don't be afraid to write new throwaway probes in `tmp/` — they're not committed and they're how the previous session resolved every single failure.

## Things this session validated empirically — do not re-derive

- HU bands `[-500, 150, 1800]` cleanly separate air/brain/bone on T1. Defaults are correct.
- `head_distance ≤ 0` reliably marks scalp exit into external air. `extension_head_distance_floor_mm = -1` gives a clean buffer.
- `scalp_exit_detect_head_distance_mm = 5` is the right value for the walk-min-hd rescue. Below 5, it misses sparse bolts; above 5, it fires inside brain proper for some shanks.
- Cross-section widening as a bolt-onset signal is unreliable across shanks (LPMC widens, LAI doesn't). Don't reintroduce it.
- The PCA outlier-atom prune is a 2× factor over the median residual + an absolute floor at `axis_fit_max_residual_mm`. Tighter prunes drop legitimate atoms; looser prunes leave bridged proposals tilted.
- Reabsorption with **no axial window** does not run away on T1/T22. The radial 1.5mm gate is sufficient because parallel shanks are typically 3–10mm apart in DIXI implants.
- Priority sort by `-len(atom_id_list)` then `-span` is enough — no need for residual or fit-quality terms.
