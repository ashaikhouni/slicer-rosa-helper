<!--
  Reviewer-friendly PR template. Keep sections that apply, delete the
  rest. The Test plan box is mandatory — use [x] for what you ran
  locally and [ ] for what's expected to run in CI.
-->

## Summary

<!-- 1-3 bullets on what changed and why. Lead with the user-visible
     impact, not the implementation details. -->

-

## Motivation / context

<!-- Optional. Link to the issue, memory note, or prior PR that
     prompted this. Skip for trivial changes. -->

## Test plan

- [ ] `pytest tests/rosa_core tests/rosa_agent tests/shank_core -q` (mirrors the CI gate)
- [ ] dataset-gated: `python -m unittest tests.deep_core.test_pipeline_dataset_contact_pitch_v1` (when touching detector / placer)
- [ ] Slicer-side smoke: load a real ROSA case, run Auto Fit, eyeball Contacts & Trajectory View (when touching Slicer modules)
- [ ] manual end-to-end (describe):

## Migration notes

<!-- Optional. Required when:
       - a public API signature changes (rosa_core.* exports, CLI flags,
         TSV column contracts)
       - a workflow role name or MRML attribute changes
       - default behavior changes for an existing flag / mode
     Otherwise delete this section. -->

## Risk / blast radius

<!-- One line. Examples:
       - low: docs only
       - low: new opt-in CLI subcommand, no existing behavior touched
       - medium: detector parameter retuning; full-dataset regression green
       - high: changes a public TSV column or workflow role -->

## Docs touched

<!-- Tick what was updated alongside the code change. If the change
     warrants a doc update and you skipped one, say why. -->

- [ ] `README.md` (overview / capabilities)
- [ ] `cli/README.md` (CLI flags or Library API)
- [ ] `docs/USER_GUIDE.md` (cross-surface workflow)
- [ ] `docs/SLICER_GUIDE.md` (per-module reference)
- [ ] `docs/PIPELINE_CONSTANTS.md` (added / changed a tunable knob)
- [ ] `docs/DEVELOPER_GUIDE.md` (architecture / parity / extension recipes)
- [ ] inline docstrings only / no docs needed
