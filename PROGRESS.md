# ROSA Toolkit Progress Log

## Snapshot
- **Timestamp (UTC)**: 2026-02-27 20:05:51Z
- **Current phase**: Phase 6 (next) — `Atlas Labeling` and `Navigation Burn` extraction
- **Last stable pushed commit**: `7675dc1`
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

### Phase 4 — `Contacts & Trajectory View` Module Extraction
- **Status**: Closed (local validation complete)
- **Date closed**: 2026-02-27
- **Commit range**: `70c4966..e2a7b2f`
- **Acceptance checks passed**:
  - Added dedicated `ContactsTrajectoryView` module for contact generation/update, QC, and trajectory slice alignment.
  - Moved contact/model/trajectory scene operations into shared `CommonLib/rosa_scene` services.
  - Rewired `RosaHelper` contact/trajectory handlers to shared services and reduced duplicated logic.
  - Made `Contacts & Trajectory View` the primary UI by removing legacy contact/trajectory panels from `RosaHelper`.
  - Verified atlas labeling still works in `RosaHelper` by recovering contacts from workflow `ContactFiducials` when needed.

### Phase 5 — `Postop CT Localization` Module Extraction
- **Status**: Closed (local validation complete)
- **Date closed**: 2026-02-27
- **Commit range**: `e2a7b2f..(phase-close commit)`
- **Acceptance checks passed**:
  - Added dedicated `PostopCTLocalization` module with unified UX:
    - Guided Fit (planned trajectory refinement on CT)
    - De Novo Detect (CT-only trajectory detection wrapper over `shank_core.pipeline`)
  - Added shared trajectory scene helpers in `CommonLib/rosa_scene/trajectory_scene.py` for create/update/collect operations.
  - Registered `PostopCTLocalization` in extension root CMake.
  - Published both guided-fit and de-novo outputs to workflow `WorkingTrajectoryLines` role.
  - User smoke-test logs confirm:
    - Guided fit candidates/fit/apply workflow runs.
    - De novo detection runs and publishes trajectories consumable by contact generation.
    - End-to-end interop with contacts/QC/export remains functional.

## Active Phase
### Phase 6 — `AtlasSources`, `Atlas Labeling`, and `Navigation Burn` Module Extraction
- **Objective**:
  - Extract atlas source loading, labeling, and burn workflows from `RosaHelper` into dedicated modules while preserving output compatibility.
- **In scope**:
  - Create module-level UIs/services for:
    - `AtlasSources`: FreeSurfer + THOMAS load/registration/publish
    - `AtlasLabeling`: contact assignment to selected atlas sources
    - `NavigationBurn`: burn-to-volume and DICOM export workflow
  - Use tabbed UI where one module hosts multiple atlas workflows:
    - `AtlasSources`: `FreeSurfer`, `THOMAS`, `Registry`
    - `NavigationBurn`: `Burn Volume`, `DICOM Export`
  - Reuse shared workflow roles/tables and `CommonLib` services.
  - Preserve current atlas CSV semantics and burn output behavior.
- **Out of scope**:
  - New atlas algorithms or segmentation methods.
  - ANTs/ANTsPy integration.
- **Exit criteria**:
  - Atlas source loading runs from `AtlasSources` without `RosaHelper` UI dependency.
  - Atlas labeling runs from `AtlasLabeling` without `RosaHelper` UI dependency.
  - Navigation burn runs from `NavigationBurn` and exports valid DICOM series.
  - Outputs remain consumable by `ExportCenter`.
  - `RosaHelper` no longer contains primary atlas/burn control surfaces.

### Phase 6 Refactor Sequence
1. Extract atlas source loading/registration into new `AtlasSources` module.
2. Extract assignment UI/handlers into new `AtlasLabeling` module (consume published sources only).
3. Extract burn workflow UI/handlers into new `NavigationBurn` module.
4. Rewire all three modules to shared workflow roles/tables only.
5. Remove primary atlas/burn panels from `RosaHelper` (keep compatibility fallbacks if needed).
6. Smoke-test end-to-end:
   - load -> localize -> contacts -> atlas labels -> burn -> export.

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
- **D-007**: Phase 4 validated locally before push; advance implementation target to Phase 5.
- **D-008**: Keep two CT localization engines, but consolidate UX into one module (`Postop CT Localization`).
- **D-009**: Phase 5 closed with unified postop localization module; advance implementation target to Phase 6.
- **D-010**: Atlas workflows are split by responsibility, not by atlas type.
- **D-011**: Lock module split for Phase 6: `AtlasSources` + `AtlasLabeling` + `NavigationBurn`.
- **D-012**: Lock tabbed UI policy for multi-workflow atlas modules.

## Maintenance Rules
- Update this file at phase boundaries with:
  - commit hash range
  - acceptance checklist status
  - next phase pointer
- Keep this file execution-focused; keep scope changes in `ROADMAP.md`.
