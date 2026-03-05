"""Atlas contact labeling and assignment table publishing service."""

from __future__ import annotations

from rosa_core import lps_to_ras_point
from rosa_workflow import WorkflowState

from .atlas_assignment_policy import (
    build_assignment_row,
    choose_closest_sample,
    collect_provider_samples,
)
from .atlas_provider_registry import AtlasProviderRegistry
from .electrode_scene import ElectrodeSceneService


class AtlasAssignmentService:
    """Assign contacts to atlas labels and publish workflow table rows."""

    def __init__(self, utils, workflow_state=None):
        """Initialize atlas assignment orchestration with reusable services."""
        self.utils = utils
        self.workflow_state = workflow_state or WorkflowState()
        self.electrode_scene = ElectrodeSceneService(workflow_state=self.workflow_state)
        self.provider_registry = AtlasProviderRegistry(utils=self.utils)

    def assign_contacts_to_atlases(
        self,
        contacts,
        freesurfer_volume_node=None,
        thomas_segmentation_nodes=None,
        wm_volume_node=None,
        reference_volume_node=None,
    ):
        """Sample configured atlas providers for each contact and return row dictionaries."""
        providers = self.provider_registry.build_default_providers(
            freesurfer_volume_node=freesurfer_volume_node,
            thomas_segmentation_nodes=thomas_segmentation_nodes,
            wm_volume_node=wm_volume_node,
            reference_volume_node=reference_volume_node,
        )
        active_providers = {source_id: provider for source_id, provider in providers.items() if provider is not None and provider.is_ready()}

        rows = []
        for contact in contacts or []:
            point_ras = lps_to_ras_point(contact.get("position_lps", [0.0, 0.0, 0.0]))
            samples = collect_provider_samples(point_ras, active_providers)
            closest_source, closest = choose_closest_sample(samples)
            rows.append(build_assignment_row(contact, point_ras, samples, closest_source, closest))

        rows.sort(key=lambda r: (str(r.get("trajectory", "")), int(r.get("contact_index", 0))))
        return rows

    def publish_atlas_assignment_rows(self, atlas_rows, workflow_node=None):
        """Publish atlas assignment rows to `Rosa_AtlasAssignments` and workflow role."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        columns = [
            "trajectory", "contact_label", "contact_index", "x_ras", "y_ras", "z_ras",
            "closest_source", "closest_label", "closest_label_value", "closest_distance_to_voxel_mm", "closest_distance_to_centroid_mm",
            "primary_source", "primary_label", "primary_label_value", "primary_distance_to_voxel_mm", "primary_distance_to_centroid_mm",
            "thomas_label", "thomas_label_value", "thomas_distance_to_voxel_mm", "thomas_distance_to_centroid_mm",
            "freesurfer_label", "freesurfer_label_value", "freesurfer_distance_to_voxel_mm", "freesurfer_distance_to_centroid_mm",
            "wm_label", "wm_label_value", "wm_distance_to_voxel_mm", "wm_distance_to_centroid_mm",
            "thomas_native_x_ras", "thomas_native_y_ras", "thomas_native_z_ras",
            "freesurfer_native_x_ras", "freesurfer_native_y_ras", "freesurfer_native_z_ras",
            "wm_native_x_ras", "wm_native_y_ras", "wm_native_z_ras",
        ]
        rows = []
        for row in atlas_rows or []:
            p_ras = row.get("contact_ras", [0.0, 0.0, 0.0])
            th_native = row.get("thomas_native_ras", [0.0, 0.0, 0.0])
            fs_native = row.get("freesurfer_native_ras", [0.0, 0.0, 0.0])
            wm_native = row.get("wm_native_ras", [0.0, 0.0, 0.0])
            rows.append(
                {
                    "trajectory": row.get("trajectory", ""),
                    "contact_label": row.get("contact_label", ""),
                    "contact_index": row.get("contact_index", 0),
                    "x_ras": float(p_ras[0]),
                    "y_ras": float(p_ras[1]),
                    "z_ras": float(p_ras[2]),
                    "closest_source": row.get("closest_source", ""),
                    "closest_label": row.get("closest_label", ""),
                    "closest_label_value": row.get("closest_label_value", 0),
                    "closest_distance_to_voxel_mm": row.get("closest_distance_to_voxel_mm", 0.0),
                    "closest_distance_to_centroid_mm": row.get("closest_distance_to_centroid_mm", 0.0),
                    "primary_source": row.get("primary_source", ""),
                    "primary_label": row.get("primary_label", ""),
                    "primary_label_value": row.get("primary_label_value", 0),
                    "primary_distance_to_voxel_mm": row.get("primary_distance_to_voxel_mm", 0.0),
                    "primary_distance_to_centroid_mm": row.get("primary_distance_to_centroid_mm", 0.0),
                    "thomas_label": row.get("thomas_label", ""),
                    "thomas_label_value": row.get("thomas_label_value", 0),
                    "thomas_distance_to_voxel_mm": row.get("thomas_distance_to_voxel_mm", 0.0),
                    "thomas_distance_to_centroid_mm": row.get("thomas_distance_to_centroid_mm", 0.0),
                    "freesurfer_label": row.get("freesurfer_label", ""),
                    "freesurfer_label_value": row.get("freesurfer_label_value", 0),
                    "freesurfer_distance_to_voxel_mm": row.get("freesurfer_distance_to_voxel_mm", 0.0),
                    "freesurfer_distance_to_centroid_mm": row.get("freesurfer_distance_to_centroid_mm", 0.0),
                    "wm_label": row.get("wm_label", ""),
                    "wm_label_value": row.get("wm_label_value", 0),
                    "wm_distance_to_voxel_mm": row.get("wm_distance_to_voxel_mm", 0.0),
                    "wm_distance_to_centroid_mm": row.get("wm_distance_to_centroid_mm", 0.0),
                    "thomas_native_x_ras": float(th_native[0]),
                    "thomas_native_y_ras": float(th_native[1]),
                    "thomas_native_z_ras": float(th_native[2]),
                    "freesurfer_native_x_ras": float(fs_native[0]),
                    "freesurfer_native_y_ras": float(fs_native[1]),
                    "freesurfer_native_z_ras": float(fs_native[2]),
                    "wm_native_x_ras": float(wm_native[0]),
                    "wm_native_y_ras": float(wm_native[1]),
                    "wm_native_z_ras": float(wm_native[2]),
                }
            )

        table = self.electrode_scene.create_or_update_table_node(
            node_name="Rosa_AtlasAssignments",
            columns=columns,
            rows=rows,
        )
        self.workflow_state.set_single_role("AtlasAssignmentTable", table, workflow_node=wf)
        self.workflow_state.tag_node(
            table,
            role="AtlasAssignmentTable",
            source="atlas",
            space="ROSA_BASE",
            signature=table.GetID(),
            workflow_node=wf,
        )
