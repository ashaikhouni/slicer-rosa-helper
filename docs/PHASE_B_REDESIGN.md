# Phase B Redesign — Trajectory Reconstruction via Mask Extension

Status: planning checkpoint, 2026-04-13.
Supersedes: the template-matching design implicit in `deep_core_model_fit.py` today.
Related: `PHASE_B_STATUS.md` (current baseline and failure modes).

## What the previous design got wrong

Phase B today treats the problem as "slide each library model along each proposal axis and score per-contact HU samples." That breaks because:

- **Wrong objective.** Scoring by `n_in_brain_hits` as the primary key rewards short models that fit entirely in brain and penalizes long models whose shallow contacts legitimately extend into bolt hardware. LPMC and RPMC are real 15CM / 68mm electrodes that get fit with 15AM / 49mm purely because of this bias.
- **Wrong source of endpoints.** Output endpoints come from library contact offsets applied at a ±5mm anchor shift. That means the shallow endpoint can't exceed the longest model span, and the deep endpoint drifts wherever maximum HU happens to cluster — RAMC fails with 12.7mm tip error even though the correct axis was right there.
- **No handling of `deep_core_shrink_mm`.** Candidate generation searches only inside the shrunk deep_core mask (20mm rind excluded). For CM/BM electrodes the shallow contact groups live inside that excluded rind, so Phase A literally cannot see them and hands Phase B proposals that are correct but incomplete (LPMC: 30mm proposal for a 68mm real electrode).
- **Fragile rejection.** Bone/dental/brace rejection relies on the annulus-rejection stage alone. Phase B doesn't do its own bone-vs-brain check on the extended axis.

## Guiding principle for the redesign

**Two-phase role of masks, matching the user's mental model:**

> Find seeds in the region with the *least* skull contamination (deep_core shrunk mask → atoms → proposals → Phase B input). Once a seed axis exists and is trustworthy, extend it using the *full* raw metal mask — we now have enough axis information that skull metal can't fool us, because we only accept voxels that lie on the known line.

The library is a **recognizer, not a generator.** We output observed endpoints; library spans are a soft plausibility gate. We do not place contacts; that belongs to a later stage.

## Inputs the new Phase B consumes

From prior stages, already in the existing pipeline contract:

- `ctx["arr_kji"]` — raw CT HU volume.
- `ctx["ras_to_ijk_fn"]` — coordinate transform.
- `mask_result["metal_mask_kji"]` — raw thresholded metal mask (un-shrunk).
- `mask_result["metal_grown_mask_kji"]` — dilated metal mask (for tolerant matching).
- `mask_result["hull_mask_kji"]` — head hull binary.
- `support_result["head_distance_map_kji"]` — signed distance from hull surface (positive inside).
- `support_result["support_atoms"]` — full 105-atom pool with `support_points_ras`, `axis_ras`, `start_ras`/`end_ras`, `span_mm`, `diameter_mm`, `parent_blob_id`, `kind`.
- `proposals["proposals"]` — from annulus rejection, with `atom_id_list`, `token_indices`, `axis_ras`, `start_ras`/`end_ras`, plus annulus profile fields.

The mask stage must continue to pass `metal_mask_kji` through the pipeline so Phase B can see it. Verify it survives the support-stage return; add a pass-through if it doesn't.

## Algorithm

### Step 1 — Axis refinement per proposal

For each proposal:
1. Gather the point cloud: union of `support_points_ras` across the atoms in `atom_id_list`, plus the tokenized points at `token_indices` (via `support_result["blob_sample_points_ras"]`).
2. Fit a straight line through the cloud using the existing PCA-based line-fit primitive in `deep_core_atoms` (no RANSAC — proposals are already reasonably clean). Record `center_ras`, `axis_ras` (unit), and per-point radial residual.
3. Reject the proposal at this step if the PCA elongation is too low (not line-like) or if the fit's median radial residual exceeds ~1.2mm (axis untrustworthy).

### Step 2 — Colinear atom reabsorption

Sweep the **full atom pool** (all 105 atoms on T1, not just the proposal's own list). An atom is reabsorbed into the current proposal if:
- its `center_ras` is within ~1.5mm perpendicular distance of the refined axis, AND
- its own `axis_ras` makes an angle ≤ ~5° with the refined axis (or atom is a `contact` kind where axis is ill-defined — use center-only test), AND
- its axis-projected center falls within some window of the refined line.

This is how Phase B recovers atoms that were dropped during colinear dedup, annulus rejection, or the kissing-contact-blob scenario. After reabsorption, recompute the axis fit from the union cloud.

Mark reabsorbed atoms in a per-proposal `explained_atom_ids` list so downstream conflict resolution can enforce "no two proposals share atoms."

### Step 3 — 1D metal profile along the refined axis

Build a dense boolean profile `metal_present(t)` for `t ∈ [t_min, t_max]` at a step of ~0.5mm. Two signals:

- **From point cloud:** project every absorbed atom's points onto the axis, mark `metal_present` true where the projected point density is above a small floor.
- **From `metal_mask_kji`:** at each `t`, sample the mask in a short perpendicular line of length ~2mm centered on `axis(t)`. Mark `metal_present` true if any voxel is set.

Use the union. This picks up the un-shrunk metal that Phase A couldn't see.

### Step 4 — Extension via raw metal mask (the critical step)

From both ends of the current atom cluster's axial extent, extend outward in 0.5mm steps, sampling `metal_mask_kji` in a narrow perpendicular tube (~1.5mm radius) around `axis(t)`. Continue extending while metal is present within the tube, within some max-gap tolerance (e.g., allow up to 2-3mm contiguous dark to cross a CM inter-group gap).

Stop extending on each end when **any** of the following is true:

- A continuous run of `extension_termination_gap_mm` has zero metal in the tube (genuine deep-tip on the deep side; end of bright metal on the shallow side), OR
- The axis leaves the volume, OR
- `head_distance(axis(t)) ≤ extension_head_distance_floor_mm` — the axis has crossed the scalp into external air. Head distance is the distance from any air interface (positive inside the non-air hull, zero at the scalp, negative outside); crossing zero marks the exit from the skin into external space. Empirically validated on T1 — every shank's axis crosses 0 cleanly 15-25mm past the bone/brain interface. A small negative buffer (~-1mm) absorbs axis jitter at the boundary.

Cross-section widening as a bolt-onset signal was evaluated empirically and rejected: it's inconsistent across shanks (LPMC's bolt widens to ~5.85mm radius but LAI's stays at ~2.5mm all the way out, never widening). Head distance is a cleaner, already-computed, single-signal substitute.

Head distance is noisy near intracranial sinuses (local drops toward zero), but on a straight electrode axis this doesn't trigger the floor — it only fires at the final exit past the skin. For bone↔brain interface detection *inside* the hull, head distance cannot distinguish bone from brain (both are positive); that's Step 5's job.

Record `t_deep_extended` and `t_shallow_extended` as the refined endpoint parameters.

### Step 5 — Bone↔brain interface from lateral HU classification

The electrode axis passes through four tissue regions from deep to shallow: brain → bone (skull thickness) → bolt hardware / external. Head distance can't separate these. Instead, walk the refined axis at 0.5mm steps and at each `t` sample HU at a **lateral ring**: 8 points at perpendicular offset ~3.5mm from `axis(t)` in the plane normal to the axis, avoiding the metal itself. Take the median HU of the ring and classify:

| Median lateral HU | Class | Notes |
|---|---|---|
| `< −500` | **air** | sinus, ventricle with outside communication, or external. Not counted as brain. |
| `[−500, 150]` | **brain** | gray matter ~35, white ~25, CSF ~0, fat ~−100. |
| `[150, 1800]` | **bone** | cortical bone 700-3000, trabecular 150-700. |
| `> 1800` | **metal-contaminated** | surrounding voxels dominated by metal bloom; skip this step and interpolate neighbors. |

Walk from deep to shallow. Smooth the classification with a small window (3 consecutive samples must agree) to absorb single-voxel noise. The **bone/brain interface** is the first `t` where the class transitions from `brain` to `bone`. That's `end_ras`.

**Sinus / intracranial air handling**: if a stretch of the axis has class = `air` while still deep in brain (a sinus intrusion), skip those samples when building the run but do **not** treat them as brain or bone. The interface is still defined by the first brain→bone transition that is *sustained* shallow-ward.

**Rejection: "line in bone"**: if fewer than 15mm of contiguous axis has class = `brain`, the proposal is rejected. This catches dental fillings, bone bridges, and brace wires (which live entirely inside bone or outside the skull and have no brain-surrounded extent).

**Interface sanity check**: the annulus rejection stage's `annulus_profile_brain_fraction` gives a coarser brain-surround fraction. If our Step 5 classification disagrees with that (e.g., we say "fully in brain" but the annulus stage said <0.3 brain fraction), log a warning — the axis may be off or the lateral ring radius is wrong.

### Step 6 — Rejection gates

A proposal passes only if:

1. **Intracranial extent ≥ 15mm.** Short line atoms in bone (dental, bridges) fail here.
2. **Brain surround fraction ≥ ~0.75.** Most of the intracranial extent must be surrounded by brain-HU tissue at 2-3mm lateral (annulus profile does this).
3. **Total extended length (intracranial + bolt) within library-span + bolt-tolerance.** Longest allowed length ≈ `max(library_span) + 40mm` ≈ 78.5 + 40 = 118.5mm. Rejects very long skull bridges.
4. **Intracranial extent matches *some* library model span within ±5mm.** Soft recognizer gate: "looks like a DIXI-family electrode by length." If intracranial extent < min library span − 5mm or > max library span + 5mm, reject.
5. **Axis fit quality ok** (from step 1).

### Step 7 — Group conflict resolution

After per-proposal scoring:

1. Sort accepted fits by a simple key: `(intracranial_extent_mm, -axis_residual_rms_mm)`.
2. Greedy accept non-conflicting fits. Two fits conflict if:
   - They share any `explained_atom_ids`, OR
   - Their refined axes come within 1.5mm perpendicular distance over any shared axial region.

The conflict radius drops from today's 2mm (on contact positions) to 1.5mm (on axes), specifically to let LAMC survive against LCMN.

### Step 8 — Emit

Output trajectory per accepted proposal:
- `start_ras` = `axis(t_deep_extended)` (deep tip from observed metal)
- `end_ras` = `axis(t_interface)` (bone/brain transition — the clinically meaningful shallow endpoint)
- `axis_ras` = refined axis unit vector
- `span_mm` = `t_interface - t_deep_extended`
- `intracranial_span_mm` = same (for clarity)
- `bolt_extent_mm` = `t_shallow_extended - t_interface` (may be 0 if the electrode is short and doesn't reach the skull exit with its shallow end)
- `closest_library_model_id` = model whose span minimizes `|library_span - intracranial_span_mm|` — purely informational
- `explained_atom_ids` — consumed atom IDs
- `model_fit_passed` = True

Drop `model_contact_positions_ras`, `model_in_brain_mask`, `best_model_n_hits`, etc. Contact placement is a later stage.

## Downstream impact

- `_proposals_to_trajectories` in `deep_core_v1.py` needs to stop populating `best_model_n_hits` / `model_contact_positions_ras` (or tolerate their absence).
- The regression tests in `tests/deep_core/test_pipeline_dataset.py` lock Phase B loose/strict baselines on `match_distance_mm=10` / `match_angle_deg=25` / `match_start_mm=20`. Expected to improve, but we should re-lock after the rewrite lands.
- Slicer UI that reads trajectory metadata may reference contact positions — audit `PostopCTLocalization` widgets for consumers of the dropped fields and gate them on presence.

## Expected effect on known T1 failures

| Shank | Current | After redesign (expected) |
|---|---|---|
| LPMC | missed (30mm proposal, library fit gives 49mm span ending 19mm short) | metal-mask extension finds the shallow 38mm of bright metal → full 68mm intracranial span → accepted |
| RPMC | missed (same pattern) | same fix |
| RAMC | tip error 12.7mm (library anchor search drifts) | endpoint comes from observed metal profile, not anchor search → tip at real deepest voxel |
| LAMC | missed (group conflict kills the single 52.7mm atom) | smaller conflict radius on refined axes → LAMC survives next to LCMN |
| P11 false positive | DIXI-10AM at a phantom location | axis fit quality + library span gate + bone/brain gate should reject |

Assumption: the 4 loose misses and 1 false positive are attributable to the failures above. Won't know for sure until implementation + regression run.

## Open questions / risks

1. **Conflict radius calibration.** 1.5mm on axes is a guess. May need per-case tuning. If it's too loose, adjacent electrodes (e.g., LAMC beside LCMN) get merged; too tight, parallel shanks don't deduplicate.

2. ~~Bolt-onset cross-section widening thresholds.~~ Evaluated empirically on T1 and rejected in favor of `head_distance_map_kji ≤ floor` as the sole extension stop past the scalp. See commit log / `tmp/calibrate_phase_b.py`.

7. **Lateral HU classification thresholds.** `[150, 1800]` as bone is a standard Hounsfield range; empirically validated on T1 (brain samples 11-111, bone samples 824-1092, air samples -720 to -1100 — clean separation). Sampling radius 3.5mm needs to clear contact bloom (~2mm) while staying in local tissue — worked on T1 without modification.

3. **Reabsorption tolerance.** 1.5mm perpendicular distance + 5° angular are guesses. Too loose and we pull in atoms from parallel shanks; too tight and we miss the atoms we're trying to reabsorb. Empirical calibration on T1 first.

4. **No atom-reassembly of split CM groups.** The plan assumes metal-mask extension alone will bridge the gap. If the mask is also broken in that region, we'll miss the shallow groups. Checked empirically for LPMC before finalizing Step 4.

5. **Port fit_electrode_axis_and_tip's slab-centroid logic?** The guided-fit pipeline has a mature line fitter. Whether to reuse it or use a simpler PCA primitive in deep_core depends on code dependencies (guided_fit.py imports Slicer's `ctk/qt` at module level, so direct reuse isn't trivial). Plan for now: use the PCA primitive in `deep_core_atoms`, port specific helpers if needed.

6. **Metal mask availability through `support_result`.** Need to verify the mask stage output actually propagates `metal_mask_kji` to the model_fit stage. If not, add a pass-through in `_run_support`.

## Implementation order

1. Verify `metal_mask_kji` and `metal_grown_mask_kji` are accessible from `_run_model_fit` (short read/print test, no code changes).
2. Add a new helper module `deep_core_axis_reconstruction.py` with: `refine_axis_from_cloud`, `reabsorb_colinear_atoms`, `build_metal_profile`, `extend_axis_along_mask`, `classify_bone_brain_interface`. All pure-Python / numpy, no Slicer dependencies.
3. Unit-test the helpers in isolation on a synthetic volume + synthetic atom list to confirm each does what it should before wiring them in.
4. Rewrite `deep_core_model_fit.py` to call the new helpers in sequence. Keep the module name and public function `run_model_fit_group` so `deep_core_v1.py` doesn't need to change imports.
5. Regenerate T1 and T22 outputs, compare against locked baselines, adjust thresholds.
6. Re-lock the regression test baselines in `test_pipeline_dataset.py` when we're satisfied.
7. Update `PHASE_B_STATUS.md` to reflect the new design and current numbers.

Commit boundary: each of steps 2, 4, 6 is a commit so you can check Slicer between them.
