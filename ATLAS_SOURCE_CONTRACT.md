# Atlas Source Contract

## Purpose
Define a stable contract for adding new atlas sources without changing contact-labeling logic.

This contract is for:
- `AtlasSources` module (loading/registration/publishing)
- `AtlasLabeling` module (assignment/export)
- future atlas integrations

## Design Rule
Add new atlas types by implementing a source adapter and publishing standard outputs.
Do not add source-specific logic directly inside generic labeling/export pipelines.

## Required Source Adapter Interface
Each atlas source must expose an adapter with these capabilities:

1. `source_id`  
   - Stable lowercase ID (example: `thomas`, `freesurfer`, `wmparc`, `customatlas`).

2. `display_name`  
   - Human-readable name for UI.

3. `workflow_roles`  
   - One or more workflow roles used to publish source nodes.
   - Existing examples:
     - `THOMASSegmentations`
     - `FSParcellationVolumes`
     - `WMParcellationVolumes`

4. `collect_source_nodes(workflow_node)`  
   - Return nodes currently published for this source.

5. `query_at_ras(point_ras_world)`  
   - Sample source in native space and return:
     - `label`
     - `label_value`
     - `distance_to_voxel_mm`
     - `distance_to_centroid_mm`
     - `native_x_ras`, `native_y_ras`, `native_z_ras`

6. `is_valid_label(label, label_value)`  
   - Source-level filtering (for example: exclude generic masks when nuclei exist).

## Space and Transform Rules
1. Label assignment must be computed in atlas-native space.
2. Visualization can use aligned copies in base/ROSA space.
3. Source adapter must own world->native transform usage internally.
4. Labelmaps/segmentations must use nearest-neighbor interpolation when resampled.

## Node Metadata Requirements
Published atlas nodes should include:
- `Rosa.Managed=1`
- `Rosa.Source=<source_id>`
- `Rosa.Role=<workflow role>`
- `Rosa.Space=<space name>`
- optional:
  - `Rosa.AtlasSource=<source_id>`
  - `Rosa.AtlasLUT=<lut name/path>`
  - `Rosa.AtlasVariant=<variant id>`

## Atlas Assignment Table Contract
For every enabled source `<src>`, `AtlasLabeling` writes:
- `<src>_label`
- `<src>_label_value`
- `<src>_distance_to_voxel_mm`
- `<src>_distance_to_centroid_mm`
- `<src>_native_x_ras`
- `<src>_native_y_ras`
- `<src>_native_z_ras`

Plus global fields:
- `closest_*` (nearest across selected sources)
- `primary_*` (policy-selected final assignment)

## Acceptance Criteria for New Sources
1. Source appears in `AtlasSources` UI and can be published into workflow roles.
2. Source can be selected in `AtlasLabeling`.
3. Per-source columns appear in `AtlasAssignmentTable` and export CSV.
4. Export works without changing existing source behavior.
5. Existing sources (THOMAS/FreeSurfer/WM) remain regression-free.
