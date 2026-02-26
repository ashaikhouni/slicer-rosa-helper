# ROSA Toolkit Progress Log

## Snapshot
- **Timestamp (UTC)**: 2026-02-26 21:26:57Z
- **Current phase**: Phase 4 (next) — `Contacts & Trajectory View` module extraction
- **Last stable pushed commit**: `8f4337b` (pre-push baseline)
- **Working branch**: `main`
- **Open worktree state**:
  - Ahead of `origin/main` by local phase commits pending push.

## Completed Phases
### Phase 1 — Shared MRML Workflow Contract Integration
- **Status**: Closed
- **Date closed**: 2026-02-26
- **Commit range**: `8f4337b..3b854f3`
- **Acceptance checks passed**:
  - User reported end-to-end workflows running correctly in Slicer.
  - Export manifest generation verified and profile semantics validated.
  - Frame-based coordinate outputs and CSV schemas validated on sample case.

### Phase 2 — `CommonLib` Extraction and Import Rewiring
- **Status**: Closed (local validation complete)
- **Date closed**: 2026-02-26
- **Commit range**: `3b854f3..474ffbc`
- **Acceptance checks passed**:
  - Added extension-level `CommonLib/rosa_core` and `CommonLib/rosa_workflow`.
  - Rewired `RosaHelper` and `ShankDetect` to shared import paths.
  - Added one-release compatibility bridge at `RosaHelper/Lib/rosa_slicer/workflow/*`.
  - Updated CMake to install `CommonLib` at extension scope.
  - User smoke-test logs confirm:
    - contacts generate/update + QC + bundle export with manifest
    - ShankDetect detect/view-align/contact generation
    - no runtime import regressions observed.

### Phase 3 — `ExportCenter` Module Extraction
- **Status**: Closed (local validation complete)
- **Date closed**: 2026-02-26
- **Commit range**: `7675dc1..211146c`
- **Acceptance checks passed**:
  - Added dedicated `ExportCenter` module wired to shared workflow contract.
  - Export logic moved into shared `CommonLib/rosa_workflow/export_bundle.py` service.
  - `RosaHelper` export path now delegates to shared service via compatibility bridge.
  - `contacts_only` profile validated from `ExportCenter` with correct manifest + outputs.
  - Output directory default behavior changed to explicit/manual selection (no forced case default).

## Active Phase
### Phase 4 — `Contacts & Trajectory View` Module Extraction
- **Objective**:
  - Extract contact generation and trajectory-aligned view workflows into a dedicated module.
- **In scope**:
  - New module for contact generation/update and trajectory view alignment controls.
  - Publish generated markups/models/tables through shared workflow roles.
  - Keep behavior parity with existing `RosaHelper` controls during transition.
- **Out of scope**:
  - CT auto-fit algorithm changes.
  - Atlas assignment or burn workflow changes.
- **Exit criteria**:
  - Contact generation/update runs from new module without `RosaHelper` UI dependency.
  - Trajectory slice-view alignment runs from new module and matches existing behavior.
  - Generated contacts/models/QC remain consumable by `ExportCenter` and Atlas workflows.

## Open Issues / Decisions
### Blocking Items
- None currently.

### Deferred Items
- Full module extraction (Phases 3–7) after `CommonLib` stabilization.
- Compatibility bridge removal deferred to Phase 8.

### Decision Log
- **D-001**: Tracking format locked to `ROADMAP.md + PROGRESS.md`.
- **D-002**: Progress updates occur at **end of each phase**, not each commit.
- **D-003**: Prior master plan remains authoritative; phase decomposition is execution structure.
- **D-004**: Next implementation target locked to **Phase 2 only** (no simultaneous module split).
- **D-005**: Phase 2 validated locally before push; advance implementation target to Phase 3.
- **D-006**: Phase 3 validated locally before push; advance implementation target to Phase 4.

## Maintenance Rules
- Update this file at phase boundaries with:
  - commit hash range
  - acceptance checklist status
  - next phase pointer
- Keep this file execution-focused; keep scope changes in `ROADMAP.md`.
