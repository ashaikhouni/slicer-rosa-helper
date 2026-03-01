# ROSA Toolkit Progress Log

## Snapshot
- **Timestamp (UTC)**: 2026-03-01 03:50:00Z
- **Current phase**: Phase 8 — compatibility bridge removal and cleanup (in progress)
- **Last stable pushed commit**: `7675dc1`
- **Working branch**: `main`
- **Open worktree state**:
  - Removed legacy `RosaHelper` widget-mixin dependency and stale wrapper modules.
  - Extended `tools/phase8_sanity.py` to assert stale monolith files are removed.
  - Final cross-module interactive smoke validation still pending.

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

### Phase 6 — `AtlasSources`, `Atlas Labeling`, and `Navigation Burn` Module Extraction
- **Status**: Closed (local validation complete)
- **Date closed**: 2026-02-27
- **Commit range**: `7675dc1..(phase-6-close commit)`
- **Acceptance checks passed**:
  - Added dedicated `AtlasSources` module with tabbed workflows:
    - `FreeSurfer` (register/load/publish, include existing volume publish)
    - `THOMAS` (register/load/publish)
    - `Registry` (image + transform visibility)
  - Added dedicated `AtlasLabeling` module consuming workflow contacts and atlas sources.
  - Added dedicated `NavigationBurn` module:
    - burn THOMAS nucleus from workflow-aligned segmentations
    - import/register nav DICOM MRI for burn/export context
    - export burned result as DICOM series
  - Registered new modules in extension CMake.
  - Removed primary atlas/burn control surfaces from `RosaHelper` UI.
  - Fixed workflow registry string normalization and base/postop flag sync.
  - User confirmed end-to-end Phase 6 flow works.

### Phase 7 — `Contact Import` Module Extraction
- **Status**: Closed (local validation complete)
- **Date closed**: 2026-03-01
- **Commit range**: `8c9cdc1..b3da923`
- **Acceptance checks passed**:
  - Added dedicated `ContactImport` module for CSV/TSV/XLSX/POM ingestion.
  - Enforced strict contact/trajectory schemas and required reference volume.
  - Imported contacts now auto-create grouped external trajectories (`ImportedExternal`) and publish to workflow roles.
  - Added generic example templates under `CommonLib/resources/examples/contact_import/`.
  - User smoke-tested import templates and confirmed downstream interop.
  - Loader refactor delivered tabbed `ROSA Load` + `Custom Import` flow with base/postop role assignment.
  - Transform provenance now retained and organized under `RosaWorkflow/Transforms`.

## Active Phase
### Phase 8 — Compatibility Bridge Removal and Cleanup Release
- **Objective**:
  - Complete bridge-removal cleanup validation and release-readiness checks.
- **In scope**:
  - Confirm module behavior after source-sync changes.
  - Final cross-module smoke + export verification.
  - Close phase and push once validated.
- **Out of scope**:
  - New feature work unrelated to Phase 8 cleanup.

## Open Issues / Decisions
### Blocking Items
- None currently.

### Deferred Items
- Optional additional cleanup/refactors after Phase 8 closure.

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
- **D-013**: Lock atlas source adapter/output contract in `ATLAS_SOURCE_CONTRACT.md` for future source additions.
- **D-014**: Phase 6 extraction implemented as standalone modules; phase close gated on smoke validation.
- **D-015**: Removed Navigation Burn auto-register fallback to reduce user confusion; burn assumes workflow-aligned THOMAS segmentations.
- **D-016**: Contact import contract locks `trajectory_name,index,x,y,z` as required contacts schema to avoid grouping ambiguity.
- **D-017**: `RosaHelper` now acts as `Loader` with tabs (`ROSA Load`, `Custom Import`), and workflow transform nodes are organized under `RosaWorkflow/Transforms` while preserving native-to-base provenance.
- **D-018**: Compatibility bridge removal started by deleting legacy `RosaHelper/Lib/rosa_slicer/workflow/*`; modules must import workflow services from `CommonLib`.
- **D-019**: Atlas modules now use a shared Loader-core bridge helper (`rosa_scene.loader_core_bridge`) instead of duplicating per-module dynamic import code.
- **D-020**: Added `tools/phase8_sanity.py` as a repeatable guardrail for cleanup release checks.
- **D-021**: Phase 8 closure is deferred until final validation sign-off.
- **D-022**: Active trajectory source is now shared and persisted across Postop CT and Contacts modules.
- **D-023**: Removed `RosaHelper/Lib/rosa_slicer/widget_mixin.py` and obsolete wrapper bridges (`electrode_scene.py`, `trajectory_scene.py`) as Phase 8 cleanup.
- **D-024**: Retired standalone `ShankDetect` module; de-novo CT detection stays in `PostopCTLocalization`, and reusable `shank_core` moved to `CommonLib/shank_core`.

## Maintenance Rules
- Update this file at phase boundaries with:
  - commit hash range
  - acceptance checklist status
  - next phase pointer
- Keep this file execution-focused; keep scope changes in `ROADMAP.md`.
