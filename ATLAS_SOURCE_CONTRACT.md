# Atlas Source Contract

## Purpose
Define the extension contract for atlas labeling so new sources (for example ANTs) can be added without changing core assignment policy.

## Canonical Code Contract
Typed protocol and sample record live in:
- `/CommonLib/rosa_scene/atlas_provider_types.py`

Current provider registry lives in:
- `/CommonLib/rosa_scene/atlas_provider_registry.py`

Current concrete providers live in:
- `/CommonLib/rosa_scene/atlas_providers.py`

## Required Provider Interface
Every atlas provider must expose:

1. `source_id: str`
- Stable lowercase source key (example: `thomas`, `freesurfer`, `wm`, `ants`).

2. `display_name: str`
- Human-readable source name.

3. `is_ready() -> bool`
- Returns True when the provider has enough data to sample contacts.

4. `sample_contact(point_world_ras) -> AtlasSampleResult | None`
- Input point is world RAS.
- Provider internally handles world->native conversion.
- Returns standardized fields:
  - `source`
  - `label`
  - `label_value`
  - `distance_to_voxel_mm`
  - `distance_to_centroid_mm`
  - `native_ras` (`[x, y, z]`)

## Ownership Split
Provider owns:
- source-native indexing
- nearest-label query
- native coordinate reporting

`AtlasAssignmentService` owns:
- iterating contacts
- comparing sources (`closest_*`)
- primary assignment fields (`primary_*`, currently mirrors `closest_*`)
- publishing assignment table

## Space and Transform Rules
1. Label sampling is done in atlas-native space.
2. Display overlays can remain in base/ROSA-aligned space.
3. Labelmap resampling/hardening must use nearest-neighbor interpolation.

## Workflow Publication Rules
Published atlas nodes should include:
- `Rosa.Managed=1`
- `Rosa.Source=<source_id>`
- `Rosa.Role=<workflow role>`
- `Rosa.Space=<space name>`

## How To Add a New Source (for example ANTs)
1. Implement a provider class that satisfies `AtlasProvider`.
2. Add provider construction to `AtlasProviderRegistry`.
3. Publish source nodes/transform in `AtlasSources`.
4. Ensure `AtlasLabeling` selector can pass source nodes into `AtlasAssignmentService`.
5. (Optional) Extend export schema with `<source>_*` columns if source-specific columns are desired in CSV.

## Acceptance Criteria
1. New source integrates without changing assignment-policy code.
2. Existing THOMAS/FreeSurfer/WM behavior remains unchanged.
3. Assignment table still contains valid `closest_*` and `primary_*` fields.
