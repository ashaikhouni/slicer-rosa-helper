"""FreeSurfer-parcellation/surface helpers + thin registration facade.

Registration internals live in `RegistrationService`. This class keeps a
single object handle (`logic.registration_service`) so callers don't need
to wire two services.
"""

from __future__ import annotations

from .freesurfer_service import FreeSurferService
from .registration_service import RegistrationService


class AtlasRegistrationService:
    """Combined facade over RegistrationService + FreeSurferService."""

    def __init__(self, module_dir=None):
        self.registration = RegistrationService(module_dir=module_dir)
        self.fs_service = FreeSurferService(module_dir=module_dir)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        """Run rigid BRAINSFit registration via shared RegistrationService."""
        return self.registration.run_brainsfit_rigid_registration(
            fixed_volume_node=fixed_volume_node,
            moving_volume_node=moving_volume_node,
            output_transform_node=output_transform_node,
            initialize_mode=initialize_mode,
            logger=logger,
        )

    def list_freesurfer_parcellation_candidates(self, subject_dir):
        """List available FreeSurfer parcellation volumes in one subject directory."""
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
        """Load selected FreeSurfer parcellation volumes and optional 3D models."""
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
        """Load FreeSurfer surfaces and optional annotation overlays."""
        return self.fs_service.load_freesurfer_surfaces(
            subject_dir=subject_dir,
            surface_set=surface_set,
            annotation_name=annotation_name,
            color_lut_path=color_lut_path,
            logger=logger,
        )

    def decimate_model_nodes(self, model_nodes, reduction=0.6):
        """Decimate a list of model nodes by target reduction ratio."""
        return self.fs_service.decimate_model_nodes(model_nodes=model_nodes, reduction=reduction)

    def create_surface_from_parcellation_volume(self, volume_node, output_name=None):
        """Create one closed-surface model from label-volume voxels."""
        return self.fs_service.create_surface_from_parcellation_volume(
            volume_node=volume_node,
            output_name=output_name,
        )

    def apply_transform_to_nodes(self, nodes, transform_node, harden=False):
        """Apply one transform to nodes and optionally harden in-place."""
        return self.registration.apply_transform_to_nodes(
            nodes=nodes, transform_node=transform_node, harden=harden,
        )
