# Atlas Source Contract

Last updated: 2026-03-01

## Purpose

Define the extension contract for atlas labeling so new sources can be added without rewriting core assignment policy.

## Canonical Files

- `CommonLib/rosa_scene/atlas_provider_types.py`
- `CommonLib/rosa_scene/atlas_provider_registry.py`
- `CommonLib/rosa_scene/atlas_providers.py`
- `CommonLib/rosa_scene/atlas_assignment_service.py`
- `CommonLib/rosa_scene/atlas_assignment_policy.py`

## Provider Interface Requirements

Each provider must expose:

1. `source_id: str`
- Stable lowercase source key (`thomas`, `freesurfer`, `wm`, `ants`, ...).

2. `display_name: str`
- Human-readable source name.

3. `is_ready() -> bool`
- Returns `True` only when provider has valid data to sample.

4. `sample_contact(point_world_ras) -> AtlasSampleResult | None`
- Input point is world RAS.
- Provider owns world->native transform handling.
- Return object must include:
  - `source`
  - `label`
  - `label_value`
  - `distance_to_voxel_mm`
  - `distance_to_centroid_mm`
  - `native_ras` (`[x, y, z]`)

## Responsibility Split

Provider owns:
- source-native indexing and nearest-label search
- native coordinate reporting

Assignment service owns:
- iterating contacts
- combining source results
- computing `closest_*` and `primary_*` rows
- publishing assignment table

## Space/Interpolation Rules

1. Assignment sampling must run in atlas-native space.
2. Display overlays may remain in base/ROSA-aligned space.
3. Labelmap resampling/hardening must use nearest-neighbor interpolation.

## Workflow Publication Rules

Atlas nodes published to workflow should include:
- `Rosa.Managed=1`
- `Rosa.Source=<source_id>`
- `Rosa.Role=<workflow role>`
- `Rosa.Space=<space name>`

## Adding a New Atlas Source

1. Implement provider class matching `AtlasProvider` protocol.
2. Register provider construction in provider registry.
3. Load/register/publish source in `AtlasSources`.
4. Ensure `AtlasLabeling` can request this provider.
5. Add/extend tests for provider readiness and assignment schema stability.
6. If needed, extend export columns for source-specific fields.

## Acceptance Criteria

1. New provider works without changing assignment-policy core.
2. Existing THOMAS/FreeSurfer/WM behavior remains unchanged.
3. Output still includes valid `closest_*` and `primary_*` fields.
