"""Registration and FreeSurfer-parcellation helpers for atlas workflows."""

from __future__ import annotations

from __main__ import slicer

from .freesurfer_service import FreeSurferService


class AtlasRegistrationService:
    """Registration and FreeSurfer loading operations."""

    def __init__(self, module_dir=None):
        self.fs_service = FreeSurferService(module_dir=module_dir)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        return self.fs_service.run_brainsfit_rigid_registration(
            fixed_volume_node=fixed_volume_node,
            moving_volume_node=moving_volume_node,
            output_transform_node=output_transform_node,
            initialize_mode=initialize_mode,
            logger=logger,
        )

    def list_freesurfer_parcellation_candidates(self, subject_dir):
        return self.fs_service.freesurfer_parcellation_candidates(subject_dir)

    def load_freesurfer_parcellation_volumes(
        self,
        subject_dir,
        selected_names=None,
        color_lut_path=None,
        apply_color_table=True,
        create_3d_geometry=False,
        logger=None,
    ):
        return self.fs_service.load_freesurfer_parcellation_volumes(
            subject_dir=subject_dir,
            selected_names=selected_names,
            color_lut_path=color_lut_path,
            apply_color_table=apply_color_table,
            create_3d_geometry=create_3d_geometry,
            logger=logger,
        )

    def load_freesurfer_surfaces(
        self,
        subject_dir,
        surface_set="pial",
        annotation_name=None,
        color_lut_path=None,
        logger=None,
    ):
        return self.fs_service.load_freesurfer_surfaces(
            subject_dir=subject_dir,
            surface_set=surface_set,
            annotation_name=annotation_name,
            color_lut_path=color_lut_path,
            logger=logger,
        )

    def decimate_model_nodes(self, model_nodes, reduction=0.6):
        return self.fs_service.decimate_model_nodes(model_nodes=model_nodes, reduction=reduction)

    def create_surface_from_parcellation_volume(self, volume_node, output_name=None):
        return self.fs_service.create_surface_from_parcellation_volume(
            volume_node=volume_node,
            output_name=output_name,
        )

    def apply_transform_to_nodes(self, nodes, transform_node, harden=False):
        if transform_node is None:
            raise ValueError("Transform node is required.")
        transform_id = transform_node.GetID()
        for node in nodes or []:
            if node is None:
                continue
            if hasattr(node, "SetAndObserveTransformNodeID"):
                node.SetAndObserveTransformNodeID(transform_id)
                if bool(harden):
                    slicer.vtkSlicerTransformLogic().hardenTransform(node)
        return nodes
