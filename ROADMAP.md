# Roadmap

Last updated: 2026-03-01

This is the canonical scope/phase plan.
Execution status belongs in `PROGRESS.md`.

## Scope and Architecture Contract

- One Slicer extension, multiple focused modules.
- One shared workflow contract (`RosaWorkflow` MRML node + fixed roles).
- Reusable logic lives in extension-level `CommonLib`.
- Modules interoperate through workflow roles/tables, not widget-local memory.
- Export supports user-selected output frame for coordinates.
- Atlas semantics are sampled in atlas-native space.

Atlas split is locked:
- `AtlasSources`: loading/registration/publish
- `AtlasLabeling`: contact-to-atlas assignment
- `NavigationBurn`: THOMAS burn + DICOM export

Trajectory ownership is locked:
- grouped by producer (`planned_rosa`, `imported_rosa`, `manual`, `guided_fit`, `de_novo`)
- each module updates only its own group
- contact generation can target selected source group

## Phase Map

1. Phase 1: shared MRML workflow contract integration.
2. Phase 2: `CommonLib` extraction and import rewiring.
3. Phase 3: `ExportCenter` extraction.
4. Phase 4: `Contacts & Trajectory View` extraction.
5. Phase 5: `Postop CT Localization` extraction.
6. Phase 6: atlas module extraction (`AtlasSources`, `AtlasLabeling`, `NavigationBurn`).
7. Phase 7: `ContactImport` extraction.
8. Phase 8: compatibility bridge removal and cleanup release.

## Acceptance Criteria by Phase

Phase 1:
- workflow roles/registries published and consumed across load/localization/export paths.

Phase 2:
- shared logic moved to `CommonLib`; modules no longer rely on module-local bridge paths.

Phase 3:
- exports run from dedicated `ExportCenter` with no behavior regressions.

Phase 4:
- contact generation/QC/trajectory viewing runs from dedicated module.

Phase 5:
- guided and de novo CT localization run from one dedicated module and publish shared outputs.

Phase 6:
- atlas loading, labeling, and burn workflows split into dedicated modules with role-based interop.

Phase 7:
- contact/trajectory file import runs from dedicated module with schema validation.

Phase 8:
- compatibility bridge removed, stale monolith leftovers cleaned, release sanity checks pass.

## Risks and Compatibility Policy

- Main risk: regressions from refactor and import-path changes.
- Mitigation: phase-gated smoke checks + repeatable sanity script.
- Compatibility policy: keep bridge only for one release cycle after extraction, then remove in Phase 8.

## Current Phase Pointer

- Current target: **Phase 8**.
- Update this file only if scope, boundaries, or acceptance criteria change.
