"""Pure-Python rigid image registration via SimpleITK.

Mirrors the parameter set the Slicer-side ``RegistrationService`` passes to
BRAINSFit (rigid + Mattes mutual information + 2% random sampling +
geometry-align init), so the CLI and the Slicer modules produce
behaviorally equivalent transforms on the same input pair. BRAINSFit is
itself a Slicer CLI wrapper around the same ITK registration framework
SimpleITK exposes — running the same algorithm with the same parameters
on the two sides keeps Slicer/CLI parity per
``feedback_cli_slicer_parity.md``.

No Slicer / VTK / Qt imports — sits in ``rosa_core`` so the headless CLI
agent and any future tooling can use it.

Public surface:

    register_rigid_mi(fixed, moving, *, ...) -> RegistrationResult
    apply_transform_to_points_ras(points_ras, transform_ras_to_ras) -> ndarray
    transform_to_4x4_ras(itk_transform) -> ndarray   # SITK is LPS internally;
                                                    # this returns the RAS form
                                                    # callers actually want.
    resample_volume(moving, transform, *, reference=None, interp="linear") -> sitk.Image
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Output of ``register_rigid_mi``.

    Carries both the SITK transform object (for direct reuse with
    ``sitk.Resample``) and a 4×4 RAS-frame numpy matrix (for the rest of
    the codebase, which works in RAS).

    Attributes:
        transform: ``sitk.Transform`` (Versor3D rigid) mapping fixed
            physical-space points to moving physical-space points
            in SITK's LPS convention. Pass to ``sitk.Resample`` directly.
        matrix_ras_4x4: 4×4 RAS-frame numpy matrix mapping a *moving*
            RAS point to its *fixed* RAS counterpart. Use this to push
            seeds / contacts between the two RAS frames the rest of the
            codebase speaks.
        final_metric: optimizer's final mutual-information value.
            More negative is a better fit.
        n_iterations: optimizer iterations actually consumed.
        converged_reason: optimizer's stop-condition string.
    """

    transform: Any  # sitk.Transform — kept loose so this module stays import-free for the docstring
    matrix_ras_4x4: np.ndarray
    final_metric: float
    n_iterations: int
    converged_reason: str


# ---------------------------------------------------------------------
# RAS <-> LPS conversion for transforms
# ---------------------------------------------------------------------


_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
_RAS_TO_LPS = _LPS_TO_RAS  # involution


def _itk_versor_to_4x4_lps(transform):
    """Convert a SITK rigid transform into a 4×4 LPS matrix.

    The matrix maps fixed-space points → moving-space points in SITK
    physical space (LPS). Combine with the LPS↔RAS flip to get the RAS
    form callers want.
    """
    matrix_3x3 = np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3)
    translation = np.asarray(transform.GetTranslation(), dtype=float).reshape(3)
    center = np.asarray(transform.GetCenter(), dtype=float).reshape(3)
    # ITK rigid form: y = R * (x - c) + c + t
    #              => y = R*x + (c - R*c + t)
    offset = center - matrix_3x3 @ center + translation
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix_3x3
    out[:3, 3] = offset
    return out


def transform_to_4x4_ras(transform) -> np.ndarray:
    """Convert a SITK rigid transform into a 4×4 RAS-frame matrix.

    The returned matrix maps a *moving-frame RAS point* to its *fixed-frame
    RAS point*. (SITK's transform direction is "fixed → moving in physical
    space"; combined with the LPS↔RAS flip and the registration's
    moving→fixed alignment intent, the RAS-frame result moves points the
    direction callers expect: from the moving volume's RAS frame into the
    fixed volume's RAS frame.)
    """
    lps = _itk_versor_to_4x4_lps(transform)
    # RAS = flip @ LPS @ flip  (involution; conjugating the LPS transform
    # back into RAS frame).
    return _LPS_TO_RAS @ lps @ _RAS_TO_LPS


def apply_transform_to_points_ras(
    points_ras,
    transform_ras_4x4,
) -> np.ndarray:
    """Apply a 4×4 RAS-frame transform to N×3 RAS points.

    Returns an N×3 numpy array. Use the inverse 4×4 to push points the
    other direction (numpy.linalg.inv).
    """
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return pts.copy()
    h = np.ones((pts.shape[0], 4), dtype=float)
    h[:, :3] = pts
    out = (transform_ras_4x4 @ h.T).T
    return out[:, :3]


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------

# Defaults match BRAINSFit's RegistrationService.run_brainsfit_rigid_registration
# parameter set (samplingPercentage=0.02, minimumStepLength=0.001,
# maximumStepLength=0.2, init=useGeometryAlign). Diverging here is a
# parity hazard.
DEFAULT_SAMPLING_PERCENTAGE = 0.02
DEFAULT_MIN_STEP_MM = 0.001
DEFAULT_MAX_STEP_MM = 0.2
DEFAULT_NUM_ITERATIONS = 200
DEFAULT_NUM_HISTOGRAM_BINS = 50
DEFAULT_SHRINK_FACTORS = (4, 2, 1)
DEFAULT_SMOOTHING_SIGMAS_MM = (2.0, 1.0, 0.0)


def register_rigid_mi(
    fixed,
    moving,
    *,
    sampling_percentage: float = DEFAULT_SAMPLING_PERCENTAGE,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    min_step_mm: float = DEFAULT_MIN_STEP_MM,
    max_step_mm: float = DEFAULT_MAX_STEP_MM,
    num_histogram_bins: int = DEFAULT_NUM_HISTOGRAM_BINS,
    shrink_factors: tuple[int, ...] = DEFAULT_SHRINK_FACTORS,
    smoothing_sigmas_mm: tuple[float, ...] = DEFAULT_SMOOTHING_SIGMAS_MM,
    init_mode: str = "geometry",
    metal_clip_hu: float | None = None,
    seed: int = 1,
    logger: Optional[Callable[[str], None]] = None,
) -> RegistrationResult:
    """Rigid Versor3D registration with Mattes mutual information.

    Mirrors BRAINSFit (Slicer-side default). Registers ``moving`` to
    ``fixed``; the returned transform aligns moving to fixed in physical
    space.

    Args:
        fixed: SITK image — typically the reference volume (e.g. ROSA
            base, or the postop CT in the atlas-registration case).
        moving: SITK image to be aligned to ``fixed``.
        sampling_percentage: fraction of voxels used for MI sampling.
            BRAINSFit default 0.02. Increase for noisy or low-contrast
            pairs; decrease for speed.
        num_iterations: optimizer iterations per resolution level.
        min_step_mm / max_step_mm: regular-step gradient-descent bounds.
        num_histogram_bins: Mattes MI bin count.
        shrink_factors / smoothing_sigmas_mm: multi-resolution pyramid
            (defaults: 3 levels, 4× → 2× → 1× shrink, 2 → 1 → 0 mm
            smoothing). Length must match.
        init_mode: ``"geometry"`` (image-center alignment, BRAINSFit's
            ``useGeometryAlign``) or ``"moments"`` (center-of-mass
            alignment, BRAINSFit's ``useMomentsAlign``).
        metal_clip_hu: optional HU ceiling for the fixed and moving
            images BEFORE registration. When set, voxels above this
            value are clamped down. Postop CTs have metal artifact that
            can dominate MI; clamping at e.g. 1500 prevents the
            electrode shafts from steering the gradient. None = no clip.
        seed: random sampling seed for reproducibility.
        logger: optional ``logger(str)`` callback for per-iteration
            progress (cheap; called ~num_iterations × len(shrink_factors)
            times).

    Returns:
        ``RegistrationResult`` carrying both the SITK transform and a
        4×4 RAS matrix (use the latter when pushing points / seeds
        between the two volumes' RAS frames).
    """
    import SimpleITK as sitk

    if len(shrink_factors) != len(smoothing_sigmas_mm):
        raise ValueError(
            f"shrink_factors ({len(shrink_factors)}) and "
            f"smoothing_sigmas_mm ({len(smoothing_sigmas_mm)}) must align"
        )

    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    if metal_clip_hu is not None:
        clip = float(metal_clip_hu)
        fixed_f = sitk.Threshold(fixed_f, lower=-1e6, upper=clip, outsideValue=clip)
        moving_f = sitk.Threshold(moving_f, lower=-1e6, upper=clip, outsideValue=clip)

    # Initial transform: align image centers (geometry) or center of mass.
    if str(init_mode).lower() in ("moments", "moment", "useMomentsAlign".lower()):
        init_filter = sitk.CenteredTransformInitializerFilter.MOMENTS
    else:
        init_filter = sitk.CenteredTransformInitializerFilter.GEOMETRY
    initial_tx = sitk.CenteredTransformInitializer(
        fixed_f,
        moving_f,
        sitk.VersorRigid3DTransform(),
        init_filter,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=int(num_histogram_bins),
    )
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling_percentage), int(seed))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(max_step_mm),
        minStep=float(min_step_mm),
        numberOfIterations=int(num_iterations),
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(list(smoothing_sigmas_mm))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # inPlace=True so Execute modifies and returns the rigid transform
    # directly. With inPlace=False the optimizer wraps the result in a
    # CompositeTransform, which doesn't expose GetMatrix/GetVersor and
    # breaks transform_to_4x4_ras.
    reg.SetInitialTransform(initial_tx, inPlace=True)

    if logger is not None:
        def _log_event():
            try:
                logger(
                    f"[reg] iter={reg.GetOptimizerIteration():>3} "
                    f"metric={reg.GetMetricValue():+.5f}"
                )
            except Exception:
                pass

        reg.AddCommand(sitk.sitkIterationEvent, _log_event)

    final_tx = reg.Execute(fixed_f, moving_f)
    matrix_ras = transform_to_4x4_ras(final_tx)
    return RegistrationResult(
        transform=final_tx,
        matrix_ras_4x4=matrix_ras,
        final_metric=float(reg.GetMetricValue()),
        n_iterations=int(reg.GetOptimizerIteration()),
        converged_reason=str(reg.GetOptimizerStopConditionDescription()),
    )


# ---------------------------------------------------------------------
# Resampling helper (for atlas labelmaps + optional CT → ref output)
# ---------------------------------------------------------------------


def resample_volume(
    moving,
    transform,
    *,
    reference=None,
    interp: str = "linear",
    default_value: float = 0.0,
):
    """Resample ``moving`` through ``transform`` onto the ``reference`` grid.

    Args:
        moving: SITK image to resample.
        transform: SITK transform (mapping fixed → moving in physical
            space) returned by ``register_rigid_mi``.
        reference: SITK image whose grid (size, spacing, origin,
            direction) is the output grid. Defaults to ``moving``'s own
            grid (useful for affine-only relabel where you want to keep
            the moving voxel layout but shift its physical placement).
        interp: ``"linear"`` (CT, T1 — anything continuous) or
            ``"nearest"`` (labelmap — preserves integer label values).
        default_value: fill value for samples outside moving's domain.
    """
    import SimpleITK as sitk

    if reference is None:
        reference = moving
    interp_id = sitk.sitkNearestNeighbor if interp == "nearest" else sitk.sitkLinear
    return sitk.Resample(
        moving,
        reference,
        transform,
        interp_id,
        float(default_value),
        moving.GetPixelID(),
    )
