# ROSA Toolkit Roadmap

## Summary
This file is the canonical roadmap for the ROSA Toolkit refactor.  
Execution status is tracked separately in `PROGRESS.md`.

## Scope and Architecture Contract
- Packaging: one Slicer extension containing multiple focused modules.
- Shared contract: one `RosaWorkflow` MRML parameter node and fixed role names.
- Shared libraries: move reusable code into extension-level `CommonLib`.
- Interop model: modules exchange state via MRML roles/registry tables, not Python widget memory.
- Export model: coordinates are emitted in a user-selected output frame, while atlas semantics come from atlas-native sampling.

## Phase Map (Locked Boundaries)
1. **Phase 1**: Shared MRML workflow contract integration in existing modules.
2. **Phase 2**: `CommonLib` extraction and import rewiring.
3. **Phase 3**: `ExportCenter` module extraction.
4. **Phase 4**: `Contacts & Trajectory View` module extraction.
5. **Phase 5**: `CT Fit` module extraction.
6. **Phase 6**: `Atlas Labeling` and `Navigation Burn` module extraction.
7. **Phase 7**: `Contact Import` module extraction.
8. **Phase 8**: Compatibility bridge removal and cleanup release.

## Acceptance Criteria Per Phase
- **Phase 1**
  - Workflow roles/registries are published from `RosaHelper` and `ShankDetect`.
  - Export frame/profile behavior works and writes manifest.
  - Existing ROSA + ShankDetect workflows continue to run.
- **Phase 2**
  - Shared workflow/core code is hosted in extension-level `CommonLib`.
  - `RosaHelper` and `ShankDetect` import only from shared libs (no cross-module `Lib` dependency).
  - One-release compatibility bridge exists for old import paths.
- **Phase 3**
  - Export workflow moved into dedicated `ExportCenter` module with no behavior regressions.
- **Phase 4**
  - Contact generation + trajectory-view workflow moved into dedicated module.
- **Phase 5**
  - Postop CT fit workflow moved into dedicated module.
- **Phase 6**
  - Atlas labeling and burn workflows separated cleanly and interoperate through MRML contract.
- **Phase 7**
  - Contact import from CSV/TSV/XLSX/POM implemented and published to shared roles.
- **Phase 8**
  - Compatibility bridge removed.
  - Legacy import paths and monolithic couplings retired.

## Risks and Compatibility Policy
- Main risk: regressions from import-path/layout changes during module extraction.
- Mitigation: phase-gated smoke checks with clear rollback commits.
- Compatibility policy:
  - Keep old import paths via bridge for one release cycle after `CommonLib` extraction.
  - Remove bridge only in Phase 8 after all module extractions are stable.

## Current Phase Pointer
- **Current phase target**: **Phase 2** (`CommonLib` extraction and import rewiring).
- Phase boundaries are locked unless this file is explicitly updated.

## Update Policy
- Update `ROADMAP.md` only when scope, phase boundaries, or acceptance criteria change.
- Track day-to-day execution details in `PROGRESS.md`.
