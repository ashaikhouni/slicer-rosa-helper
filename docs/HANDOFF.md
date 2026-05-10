# contact_pitch_v1 handoff

Last updated: 2026-05-10 — post-Phase-B refactor complete.

This is the single sticky reference. Session-by-session detail lives
in the memory files (auto-loaded into Claude Code context). For full
historical narrative, read those rather than expanding this doc.

## Current state

| metric | value |
|---|---|
| pipeline version | 1.0.29 (algorithm unchanged since pre-refactor) |
| dataset recall | 295 / 295 (22 subjects, T17 / T19 / T21 excluded) |
| dataset orphans | see `test_dataset_full` for the asserted budget |
| AMC099 | 16 / 16 (wire-class extension recovered L_4) |
| S56 | 16 / 16 (anisotropic σ recovered L_2 / L_3 horizontal shanks) |
| ct88 | 8 / 8 |
| code shape | monolith deleted 2026-05-10; algorithm now lives in `rosa_detect.candidate_seeds.*` (one file per stage) and `rosa_detect.primitives.*` (preprocessing / bolt-anchor / geometry shared with the placer). Tunable knobs in one file: `candidate_seeds.constants`. |

## Pipeline shape

Auto Fit (`contact_pitch_v1`) is the sole detection pipeline. Direct
shank detection from the postop CT only — no bolt-first stage. Stage 2
(Frangi shaft fallback) was retired 2026-04-27; only the unified
metal-evidence cascade remains.

End-to-end orchestration lives in `CommonLib/rosa_detect/`:

- **Public detection entry**:
  [`rosa_detect.service.run_contact_pitch_v1`](../CommonLib/rosa_detect/service.py)
  (used by Slicer Auto Fit + the `rosa-agent` CLI).
- **Public placement entry**:
  [`rosa_core.placement_modes.place_seeg`](../CommonLib/rosa_core/placement_modes.py)
  (used by ContactsTrajectoryView + `rosa-agent place` for all five modes).
- **v1 detector**:
  [`rosa_detect.candidate_seeds.orchestrator.run_two_stage_detection`](../CommonLib/rosa_detect/candidate_seeds/orchestrator.py)
  composes the per-stage modules (`blob_extraction`, `walker`,
  `stage1_runner`, `pitch_library`, `dedup`, `crossing_tips`,
  `deep_end_refine`, `synth_anchor`, `confidence_score`,
  `axis_peak_refine`).
- **Staged contact placement internals**: `rosa_core.contact_placement`.
- **Slicer→headless adapter**:
  [`rosa_scene.sitk_volume_adapter`](../CommonLib/rosa_scene/sitk_volume_adapter.py)
  (bridges `vtkMRMLScalarVolumeNode` to the SITK image inputs the
  algorithm consumes; only place vtk + slicer get imported on the
  algorithm-call path).

For the algorithm walkthrough (preprocessing, walker, bolt anchor,
scoring, dedup, refine), the authoritative source is the inline
docstrings + comments in the per-stage modules under
`rosa_detect/candidate_seeds/`. Past doc copies of the algorithm
description rotted within weeks of being written; they are no longer
maintained here.

## Key adjacent pieces

- **Guided Fit** — Slicer-side UI is
  [`guided_fit.py`](../PostopCTLocalization/postop_ct_localization/guided_fit.py);
  the algorithm itself lives in
  [`CommonLib/rosa_detect/guided_fit_engine.py`](../CommonLib/rosa_detect/guided_fit_engine.py).
  Phase 1 + phase 2 landed 2026-04-29. Phase 2 inherits Auto Fit's
  geometry + score on a match; phase 1 PCA fit is the fallback.
- **Manual Fit** ([`manual_fit.py`](../PostopCTLocalization/postop_ct_localization/manual_fit.py))
  — orientation rule combo + bulk swap + edit/delete UI landed 2026-04-28.
- **Electrode-model classifier**
  ([`rosa_core/electrode_classifier.py`](../CommonLib/rosa_core/electrode_classifier.py))
  — unified picker (PaCER → walker-signature → length-only). Joint
  pitch + count + span + length scoring; primary electrode-model
  picker for Auto / Guided / Manual Fit + Contacts & Trajectory View.
- **Headless CLI**
  ([`cli/rosa_agent/`](../cli/rosa_agent/),
  [`cli/README.md`](../cli/README.md)) — same algorithm runnable
  outside Slicer. `pip install .` at the repo root creates a
  `rosa-agent` console script. T22 dataset run produces the same
  9 trajectories / 129 contacts as the Slicer regression. First
  real ROSA case (s57) tested 2026-05-02: 16/16 ground-truth
  trajectories recovered.

## Score-band policy (do not relax without re-reading)

`feedback_score_band_policy.md` (memory). Summary:
- **High** = pitch + REAL bolt (`bolt_source = "metal"`).
- **Medium** = synthesized bolt, wire-class fallback, or no-bolt accept.
- **Low** = currently unused band.

Capping is categorical, not gradient — future score-component work
must not let weak-evidence cases earn high.

## Tests

```sh
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 -m unittest \
  tests.deep_core.test_pipeline_dataset_contact_pitch_v1 \
  tests.deep_core.test_walker_signature_classifier \
  tests.rosa_core.test_contact_peak_fit
```

`test_pipeline_dataset_contact_pitch_v1` includes:
- `test_T22` / `test_T2` / `test_T2_auto_strategy` — quick gates for
  iteration (~15 s for the three).
- `test_dataset_full` — runs all 22 subjects, asserts recall +
  orphan budget. Slower (~3 min). This is the gate for the upcoming
  refactor work; do not relax its asserts to make a refactor pass.

## How to resume cold

1. Read this file.
2. Skim, in order:
   - `feedback_cli_slicer_parity.md` — parity invariant (P0 if violated)
   - `feedback_score_band_policy.md` — high-band policy
   - `feedback_gt_completeness.md` — orphans are FPs (authoritative)
   - `feedback_concept_over_threshold.md` — no per-subject magic numbers
3. Read the most recent state memory:
   `project_contact_pitch_v1_2026-04-29_state.md`.
4. If structural / refactor work is on the table, read
   `project_contact_pitch_v1_risks_2026-04-29.md`.
5. Run the regression above and confirm green before any change.

## Pending structural work

See `project_contact_pitch_v1_risks_2026-04-29.md` for the five
identified risks with location, fix, and tradeoff.

| # | risk | status |
|---|---|---|
| 5 | silent except-pass in score paths | **landed** `f699e2e` |
| 4 | full-dataset regression + handoff consolidation | **landed** `360c95c` |
| 3 | coordinate naming silent LPS/RAS sign-flip | **landed** `f140d07` |
| 1 | Auto Fit ↔ Guided Fit preprocessing drift (extract `prepare_volume`) | **landed** `0cecf9a` |
| 2 | `contact_pitch_v1_fit.py` monolith split | **landed** 2026-05-10 (module deleted; algorithm in `candidate_seeds.*` + `primitives.*`) |

All five identified risks are now closed.
