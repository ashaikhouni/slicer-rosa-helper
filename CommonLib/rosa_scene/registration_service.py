"""Image-registration service (BRAINSFit + transform application).

BRAINSFit is a Slicer/ITK module, not FreeSurfer-specific. Keeping it here
prevents the misleading "registration via FreeSurferService" indirection.
"""

from __future__ import annotations

from __main__ import slicer, vtk


class RegistrationService:
    """Rigid registration helpers used across ROSA modules."""

    def __init__(self, module_dir=None):
        self.module_dir = module_dir

    @staticmethod
    def _cli_success(cli_node):
        if cli_node is None:
            return False
        status = (cli_node.GetStatusString() or "").lower()
        return ("completed" in status) and ("error" not in status)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        """Run BRAINSFit rigid registration (moving -> fixed) and return the transform node.

        Tries four parameter variants for compatibility across Slicer versions.
        """

        def log(msg):
            if logger:
                logger(msg)

        if fixed_volume_node is None or moving_volume_node is None:
            raise ValueError("Fixed and moving volumes are required.")
        if output_transform_node is None:
            raise ValueError("Output transform node is required.")
        if not hasattr(slicer.modules, "brainsfit"):
            raise RuntimeError("BRAINSFit module is not available in this Slicer install.")

        identity = vtk.vtkMatrix4x4()
        output_transform_node.SetMatrixTransformToParent(identity)

        base = {
            "fixedVolume": fixed_volume_node.GetID(),
            "movingVolume": moving_volume_node.GetID(),
            "initializeTransformMode": initialize_mode or "useGeometryAlign",
            "samplingPercentage": 0.02,
            "minimumStepLength": 0.001,
            "maximumStepLength": 0.2,
        }
        variants = [
            dict(base, linearTransform=output_transform_node.GetID(), useRigid=True),
            dict(base, outputTransform=output_transform_node.GetID(), useRigid=True),
            dict(base, linearTransform=output_transform_node.GetID(), transformType="Rigid"),
            dict(base, outputTransform=output_transform_node.GetID(), transformType="Rigid"),
        ]

        last_error = "Unknown BRAINSFit error"
        for i, params in enumerate(variants, start=1):
            cli_node = None
            try:
                cli_node = slicer.cli.runSync(slicer.modules.brainsfit, None, params)
                if self._cli_success(cli_node):
                    log(f"[reg] BRAINSFit variant {i}/{len(variants)} succeeded")
                    return output_transform_node
                status = cli_node.GetStatusString() if cli_node is not None else "no status"
                err_text = ""
                if cli_node is not None and hasattr(cli_node, "GetErrorText"):
                    err_text = cli_node.GetErrorText() or ""
                last_error = f"{status} {err_text}".strip()
                log(f"[reg] BRAINSFit variant {i}/{len(variants)} failed: {last_error}")
            except Exception as exc:
                last_error = str(exc)
                log(f"[reg] BRAINSFit variant {i}/{len(variants)} exception: {last_error}")
            finally:
                if cli_node is not None and cli_node.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(cli_node)

        raise RuntimeError(f"BRAINSFit rigid registration failed: {last_error}")

    @staticmethod
    def apply_transform_to_nodes(nodes, transform_node, harden=False):
        """Apply one transform to nodes and optionally harden in-place."""
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
