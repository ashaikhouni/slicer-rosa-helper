"""Compatibility facade for atlas, burn, and labeling services.

This class keeps the historical `AtlasCoreService` API stable while delegating
implementation to focused services.
"""

from __future__ import annotations

from rosa_workflow import WorkflowState

from .atlas_assignment_service import AtlasAssignmentService
from .atlas_registration_service import AtlasRegistrationService
from .atlas_utils import AtlasUtils
from .dicom_io_service import DicomIOService
from .thomas_service import ThomasService


class AtlasCoreService:
    """Facade over focused atlas-related services."""

    def __init__(self, module_dir=None, workflow_state=None):
        self.workflow_state = workflow_state or WorkflowState()
        self.utils = AtlasUtils()
        self.registration_service = AtlasRegistrationService(module_dir=module_dir)
        self.thomas_service = ThomasService(utils=self.utils)
        self.dicom_service = DicomIOService()
        self.assignment_service = AtlasAssignmentService(utils=self.utils, workflow_state=self.workflow_state)

    def show_volume_in_all_slice_views(self, volume_node):
        return self.utils.show_volume_in_all_slice_views(volume_node)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        return self.registration_service.run_brainsfit_rigid_registration(
            fixed_volume_node=fixed_volume_node,
            moving_volume_node=moving_volume_node,
            output_transform_node=output_transform_node,
            initialize_mode=initialize_mode,
            logger=logger,
        )

    def list_freesurfer_parcellation_candidates(self, subject_dir):
        return self.registration_service.list_freesurfer_parcellation_candidates(subject_dir)

    def load_freesurfer_parcellation_volumes(
        self,
        subject_dir,
        selected_names=None,
        color_lut_path=None,
        apply_color_table=True,
        create_3d_geometry=False,
        logger=None,
    ):
        return self.registration_service.load_freesurfer_parcellation_volumes(
            subject_dir=subject_dir,
            selected_names=selected_names,
            color_lut_path=color_lut_path,
            apply_color_table=apply_color_table,
            create_3d_geometry=create_3d_geometry,
            logger=logger,
        )

    def apply_transform_to_nodes(self, nodes, transform_node, harden=False):
        return self.registration_service.apply_transform_to_nodes(nodes=nodes, transform_node=transform_node, harden=harden)

    def load_thomas_thalamus_masks(self, thomas_dir, logger=None, replace_existing=True, node_name_prefix="THOMAS_"):
        return self.thomas_service.load_thomas_thalamus_masks(
            thomas_dir=thomas_dir,
            logger=logger,
            replace_existing=replace_existing,
            node_name_prefix=node_name_prefix,
        )

    def load_dicom_scalar_volume_from_directory(self, dicom_dir, logger=None):
        return self.dicom_service.load_dicom_scalar_volume_from_directory(dicom_dir=dicom_dir, logger=logger)

    def place_node_under_same_study(self, node, reference_node, logger=None):
        return self.dicom_service.place_node_under_same_study(node=node, reference_node=reference_node, logger=logger)

    def export_scalar_volume_to_dicom_series(
        self,
        volume_node,
        reference_volume_node,
        export_dir,
        series_description,
        modality="MR",
        logger=None,
    ):
        return self.dicom_service.export_scalar_volume_to_dicom_series(
            volume_node=volume_node,
            reference_volume_node=reference_volume_node,
            export_dir=export_dir,
            series_description=series_description,
            modality=modality,
            logger=logger,
        )

    def collect_thomas_nuclei(self, segmentation_nodes):
        return self.thomas_service.collect_thomas_nuclei(segmentation_nodes)

    def burn_thomas_nucleus_to_volume(
        self,
        segmentation_nodes,
        input_volume_node,
        nucleus,
        side="Both",
        fill_value=1200.0,
        output_name="THOMAS_Burned_MRI",
        logger=None,
    ):
        return self.thomas_service.burn_thomas_nucleus_to_volume(
            segmentation_nodes=segmentation_nodes,
            input_volume_node=input_volume_node,
            nucleus=nucleus,
            side=side,
            fill_value=fill_value,
            output_name=output_name,
            logger=logger,
        )

    def assign_contacts_to_atlases(
        self,
        contacts,
        freesurfer_volume_node=None,
        thomas_segmentation_nodes=None,
        wm_volume_node=None,
        reference_volume_node=None,
        prefer_thomas=True,
    ):
        return self.assignment_service.assign_contacts_to_atlases(
            contacts=contacts,
            freesurfer_volume_node=freesurfer_volume_node,
            thomas_segmentation_nodes=thomas_segmentation_nodes,
            wm_volume_node=wm_volume_node,
            reference_volume_node=reference_volume_node,
            prefer_thomas=prefer_thomas,
        )

    def publish_atlas_assignment_rows(self, atlas_rows, workflow_node=None):
        return self.assignment_service.publish_atlas_assignment_rows(atlas_rows=atlas_rows, workflow_node=workflow_node)
