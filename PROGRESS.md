# Progress Log

Last updated: 2026-03-01

## Snapshot

- Current phase: **Phase 8** (compatibility bridge removal and cleanup)
- Working branch: `main`
- Last stable pushed commit (recorded): `7675dc1`
- Worktree: active refactor/doc updates in progress

## Completed Phases

### Phase 1 — Shared Workflow Contract
- Status: Closed
- Date closed: 2026-02-26
- Commit range: `8f4337b..3b854f3`
- Result: workflow roles/registries integrated and validated in end-to-end usage.

### Phase 2 — CommonLib Extraction
- Status: Closed
- Date closed: 2026-02-26
- Commit range: `3b854f3..474ffbc`
- Result: shared libs centralized; module imports rewired; temporary bridge added.

### Phase 3 — ExportCenter Extraction
- Status: Closed
- Date closed: 2026-02-26
- Commit range: `7675dc1..211146c`
- Result: dedicated export module with profile-driven bundle output.

### Phase 4 — Contacts & Trajectory View Extraction
- Status: Closed
- Date closed: 2026-02-27
- Commit range: `70c4966..e2a7b2f`
- Result: contact generation/QC/slice alignment moved to dedicated module.

### Phase 5 — Postop CT Localization Extraction
- Status: Closed
- Date closed: 2026-02-27
- Commit range: `e2a7b2f..(phase-5-close)`
- Result: guided fit + de novo detect unified under one localization module.

### Phase 6 — Atlas Module Extraction
- Status: Closed
- Date closed: 2026-02-27
- Commit range: `7675dc1..(phase-6-close)`
- Result: atlas loading, labeling, and burn split cleanly into dedicated modules.

### Phase 7 — Contact Import Extraction
- Status: Closed
- Date closed: 2026-03-01
- Commit range: `8c9cdc1..b3da923`
- Result: external contact/trajectory import with strict schema and workflow publishing.

## Active Phase

### Phase 8 — Compatibility Bridge Removal and Cleanup

Objective:
- remove remaining bridge couplings
- finalize cleanup checks
- preserve functional parity

In scope:
- stale bridge path removal
- dead-code cleanup
- docs and architecture stabilization
- smoke/sanity verification

Out of scope:
- net-new product features unrelated to cleanup target

Exit criteria:
- all modules import from `CommonLib`
- cleanup sanity script passes
- cross-module smoke flow passes
- docs updated to current architecture

## Open Issues / Decisions

Blocking:
- none currently

Deferred:
- optional `SEEGWorkbench` orchestration shell (future version)

Decision log (selected):
- D-001: tracking system fixed to `ROADMAP.md` + `PROGRESS.md`
- D-008: keep two CT engines but expose unified localization UX
- D-010: split atlas workflows by responsibility, not by atlas type
- D-016: contact import schema requires `trajectory_name,index,x,y,z`
- D-017: `RosaHelper` serves loader role with ROSA load + custom import tabs
- D-024: standalone `ShankDetect` retired; reusable `shank_core` remains in `CommonLib`

## Maintenance Rules

- Update this file at phase boundaries with commit ranges and acceptance status.
- Keep this file execution-focused.
- Use `ROADMAP.md` for scope/boundary changes only.
