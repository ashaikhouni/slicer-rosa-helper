# contact_pitch_v1 handoff

Last updated: 2026-04-29 — pipeline 1.0.29

This is the single sticky reference. Session-by-session detail lives
in the memory files (auto-loaded into Claude Code context). For full
historical narrative, read those rather than expanding this doc.

## Current state

| metric | value |
|---|---|
| pipeline version | 1.0.29 |
| dataset recall | 295 / 295 (22 subjects, T17 / T19 / T21 excluded) |
| dataset orphans | see `test_dataset_full` for the asserted budget |
| AMC099 | 16 / 16 (wire-class extension recovered L_4) |
| S56 | 16 / 16 (anisotropic σ recovered L_2 / L_3 horizontal shanks) |
| ct88 | 8 / 8 |
| code monolith | `contact_pitch_v1_fit.py` is currently 3411 lines — pending split, see `project_contact_pitch_v1_risks_2026-04-29.md` |

## Pipeline shape

Auto Fit (`contact_pitch_v1`) is the sole detection pipeline. Direct
shank detection from the postop CT only — no bolt-first stage. Stage 2
(Frangi shaft fallback) was retired 2026-04-27; only the unified
metal-evidence cascade remains.

End-to-end orchestration is in
[`PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py`](../PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py).
The pipeline registers via
[`CommonLib/shank_engine/pipelines/contact_pitch_v1.py`](../CommonLib/shank_engine/pipelines/contact_pitch_v1.py).

For the algorithm walkthrough (preprocessing, walker, bolt anchor,
scoring, dedup, refine), the authoritative source is the inline
docstrings + comments in `contact_pitch_v1_fit.py` itself. Past doc
copies of the algorithm description rotted within weeks of being
written; they are no longer maintained here.

## Key adjacent pieces

- **Guided Fit** ([`guided_fit.py`](../PostopCTLocalization/postop_ct_localization/guided_fit.py),
  [`guided_fit_engine.py`](../PostopCTLocalization/postop_ct_localization/guided_fit_engine.py))
  — phase 1 + phase 2 landed 2026-04-29. Phase 2 inherits Auto Fit's
  geometry + score on a match; phase 1 PCA fit is the fallback.
- **Manual Fit** ([`manual_fit.py`](../PostopCTLocalization/postop_ct_localization/manual_fit.py))
  — orientation rule combo + bulk swap + edit/delete UI landed 2026-04-28.
- **Walker-signature classifier** ([`rosa_core/walker_signature_classifier.py`](../CommonLib/rosa_core/walker_signature_classifier.py))
  — joint pitch + count + span + length scoring; primary electrode-model
  picker.

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
| 2 | `contact_pitch_v1_fit.py` monolith split | deferred |

Item 2 (monolith split into `preprocess` / `pitch_walker` /
`bolt_anchor` / `scoring` / `model_suggestion` modules) is the only
open structural item. The `prepare_volume` extract from item 1 is
the natural first slice of that split — when the monolith work
starts, it has a working precedent.
