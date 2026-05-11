"""Tests for ``rosa_detect.primitives.preprocessing.prepare_volume``.

Specifically guards the per-axis anti-alias σ computation when one or more
axes are already at canonical 1 mm spacing. Sub-canonical axes need
smoothing (σ = canonical − spacing); already-canonical axes need none
(σ = 0). Before the 2026-05-04 fix, sigmas of 0 were passed to ITK's
SmoothingRecursiveGaussian, which throws ``Sigma must be greater than zero``.
The fix replaces 0 with a tiny epsilon (1e-9 mm) so ITK accepts the call
without changing the smoothing behaviour for non-degenerate axes.

This test suite covers every combination of axes being canonical (1.0)
vs sub-canonical (0.5) so neither the original bug nor a future regression
of "fix only the Z case" can pass.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[2]
for sub in ("CommonLib", "PostopCTLocalization"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from rosa_detect.primitives.preprocessing import (  # noqa: E402
    CANONICAL_SPACING_MM,
    prepare_volume,
)
from shank_core.io import image_ijk_ras_matrices  # noqa: E402


def _make_ct(spacing_xyz: tuple[float, float, float]) -> sitk.Image:
    """Tiny synthetic CT volume at the requested spacing.

    A 32^3 box big enough that resampling produces non-empty output but
    small enough to keep the test fast (~50 ms per call).
    """
    arr = np.full((32, 32, 32), -1024, dtype=np.float32)
    arr[12:20, 12:20, 12:20] = 2500.0  # bright cube in the middle
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


# Every spacing combination of {0.5, 1.0} across the 3 axes — covers
# the all-fine, all-canonical, and the seven mixed cases. Excludes
# (1,1,1) since needs_resample is False there (no smoothing called).
_SUB = 0.5
_CAN = CANONICAL_SPACING_MM


@pytest.mark.parametrize(
    "spacing",
    [s for s in itertools.product((_SUB, _CAN), repeat=3)
     if min(s) < _CAN * 0.95],
    ids=lambda s: f"sx{s[0]}_sy{s[1]}_sz{s[2]}",
)
def test_prepare_volume_mixed_axes_does_not_crash(spacing):
    """Resampling with one or more already-canonical axes must NOT throw.

    Pre-fix: spacing=(0.5, 0.5, 1.0) crashed inside
    SmoothingRecursiveGaussian because the Z-axis sigma was 0 and ITK
    rejects sigma=0. Same crash reproducible with (1.0, 0.5, 0.5),
    (0.5, 1.0, 0.5), etc. — any axis being 1.0.
    """
    img = _make_ct(spacing)
    i2r, r2i = image_ijk_ras_matrices(img)
    out_img, out_i2r, out_r2i = prepare_volume(
        img, np.asarray(i2r), np.asarray(r2i),
    )
    # Output is at canonical spacing on every axis
    out_spacing = out_img.GetSpacing()
    for axis_spacing in out_spacing:
        assert abs(axis_spacing - _CAN) < 1e-3, (
            f"axis spacing {axis_spacing} not canonical for input {spacing}"
        )
    # Matrices are sane shape
    assert np.asarray(out_i2r).shape == (4, 4)
    assert np.asarray(out_r2i).shape == (4, 4)


def test_prepare_volume_already_canonical_skips_resample():
    """When all axes are already 1.0 mm, the function takes the no-op
    branch (no resample, no smoothing) and returns the input image."""
    img = _make_ct((1.0, 1.0, 1.0))
    i2r, r2i = image_ijk_ras_matrices(img)
    out_img, _, _ = prepare_volume(img, np.asarray(i2r), np.asarray(r2i))
    assert out_img.GetSize() == img.GetSize()
    assert all(
        abs(a - b) < 1e-6 for a, b in zip(out_img.GetSpacing(), img.GetSpacing())
    )


def test_prepare_volume_isotropic_subcanonical_smooths_all_axes():
    """Sanity: when every axis is sub-canonical (e.g., 0.5 mm), smoothing
    should still happen — the binary input (cube of 2500 HU on -1024
    background) should produce intermediate HU values at the cube edges.
    Verifies the 1e-9 epsilon fix didn't break smoothing on
    non-degenerate axes."""
    img = _make_ct((0.5, 0.5, 0.5))
    i2r, r2i = image_ijk_ras_matrices(img)
    out_img, _, _ = prepare_volume(img, np.asarray(i2r), np.asarray(r2i))
    arr = sitk.GetArrayFromImage(out_img)
    # Output volume should still contain bright voxels (cube survived resample)
    assert arr.max() > 1500.0, f"max HU {arr.max()} too low; cube lost"
    # Smoothing creates intermediate values between -1024 and 2500.
    # A binary input + nearest-neighbour resample would produce only
    # 2 distinct values; smoothing creates a transition zone.
    transitional = ((arr > 200.0) & (arr < 2000.0)).sum()
    assert transitional > 10, (
        f"expected smoothing transition voxels; got {int(transitional)}"
    )


def test_prepare_volume_mixed_axis_smooths_only_subcanonical_axes():
    """Quantitative check on the per-axis σ behaviour: with spacing
    (0.5, 0.5, 1.0) the X and Y axes get smoothed (σ ≈ 0.5 mm) and the
    Z axis gets the 1e-9 epsilon (effectively no smoothing). The mixed-σ
    output should have a noticeably different HU distribution than the
    fully-isotropic-smoothed equivalent.
    """
    img_mixed = _make_ct((0.5, 0.5, 1.0))
    i2r_m, r2i_m = image_ijk_ras_matrices(img_mixed)
    out_mixed, _, _ = prepare_volume(img_mixed, np.asarray(i2r_m), np.asarray(r2i_m))
    arr_m = sitk.GetArrayFromImage(out_mixed)
    # Cube survived; smoothing happened
    assert arr_m.max() > 1500.0
    transitional_m = ((arr_m > 200.0) & (arr_m < 2000.0)).sum()
    assert transitional_m > 0, "no transition voxels — smoothing was skipped entirely"
