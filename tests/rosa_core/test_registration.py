"""Pin the rigid + Mattes MI registration helper.

The CLI agent uses this to align an external CT to a ROSA reference
volume (or an external T1 to the postop CT for atlas labeling). It must
mirror BRAINSFit's parameter set so Slicer-side and CLI-side runs on the
same input pair land in the same place — see feedback_cli_slicer_parity.

Two tests:

  * Synthetic round-trip — translate + small rotation by a known
    transform, recover within sub-mm. Catches a basic algorithm /
    convention break.
  * RAS-frame transform application — feed a known 4×4 to
    apply_transform_to_points_ras and verify the points move where they
    should. Catches axis-flip / convention regressions in the LPS<->RAS
    bridge.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports():
    try:
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_core import registration  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/rosa_core not importable in this environment.",
)
class RegisterRigidMITests(unittest.TestCase):
    """Round-trip: applying a known transform should be recoverable."""

    def _make_phantom(self):
        """Return a SITK image with structure the registration can latch onto.

        Sized to survive the 4× shrink of the default multi-resolution
        pyramid (80³ → 20³ at the coarsest level). Non-symmetric blob
        positions force the optimizer to find rotation gradient
        information on every axis — symmetric phantoms can let it settle
        on a degenerate solution.
        """
        import numpy as np
        import SimpleITK as sitk

        size = (80, 80, 80)
        arr = np.zeros(size, dtype=np.float32)
        arr[16:28, 16:28, 16:28] = 1000.0
        arr[48:64, 24:36, 44:56] = 1500.0
        arr[40:52, 56:72, 12:24] = 800.0
        arr[60:70, 10:20, 60:70] = 1200.0

        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection((1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0))
        return img

    def _apply_known_transform(self, fixed_img, *, translation_lps, rotation_axis, angle_rad):
        """Resample ``fixed_img`` through a known rigid transform → ``moving_img``."""
        import SimpleITK as sitk

        tx = sitk.VersorRigid3DTransform()
        tx.SetCenter([
            0.5 * (sz - 1) * sp
            for sz, sp in zip(fixed_img.GetSize(), fixed_img.GetSpacing())
        ])
        tx.SetRotation(rotation_axis, float(angle_rad))
        tx.SetTranslation([float(v) for v in translation_lps])

        # Resample fixed THROUGH the inverse of the known transform to
        # synthesize a moving image such that registering moving→fixed
        # should recover ``tx``.
        return sitk.Resample(
            fixed_img,
            fixed_img,
            tx.GetInverse(),
            sitk.sitkLinear,
            0.0,
            fixed_img.GetPixelID(),
        )

    def test_recovers_pure_translation(self):
        import numpy as np
        from rosa_core.registration import register_rigid_mi

        fixed = self._make_phantom()
        # A pure 3-mm-each-axis LPS translation.
        true_t = (3.0, -2.0, 4.0)
        moving = self._apply_known_transform(
            fixed,
            translation_lps=true_t,
            rotation_axis=(0.0, 0.0, 1.0),
            angle_rad=0.0,
        )

        result = register_rigid_mi(fixed, moving)
        recovered_t = np.asarray(result.transform.GetTranslation(), dtype=float)
        # Sub-millimeter recovery on a clean synthetic phantom is a
        # safe regression bar — anything looser would let real bugs
        # (axis flips, sign errors) slip through.
        np.testing.assert_allclose(
            recovered_t, np.asarray(true_t),
            atol=0.5,
            err_msg=(
                f"translation recovery drifted: expected ~{true_t}, "
                f"got {recovered_t}, metric={result.final_metric}, "
                f"reason={result.converged_reason!r}"
            ),
        )

    def test_recovers_translation_plus_small_rotation(self):
        import math
        import numpy as np
        from rosa_core.registration import register_rigid_mi

        fixed = self._make_phantom()
        true_t = (2.0, 1.5, -1.0)
        true_axis = (0.0, 0.0, 1.0)
        true_angle = math.radians(5.0)
        moving = self._apply_known_transform(
            fixed,
            translation_lps=true_t,
            rotation_axis=true_axis,
            angle_rad=true_angle,
        )

        result = register_rigid_mi(
            fixed, moving,
            num_iterations=400,  # rotation is harder; give the optimizer headroom.
        )
        # Recovered angle is the magnitude of the versor rotation.
        versor = np.asarray(result.transform.GetVersor(), dtype=float)
        recovered_angle = 2.0 * math.atan2(
            float(np.linalg.norm(versor[:3])), float(versor[3]),
        )
        recovered_t = np.asarray(result.transform.GetTranslation(), dtype=float)
        # Loose-but-not-toothless bars: 1 mm translation, 1° rotation.
        # Tighter than this on a 40^3 phantom is hard for any rigid
        # registration — the synthetic moving image was itself
        # interpolated, so there's a noise floor.
        np.testing.assert_allclose(
            recovered_t, np.asarray(true_t), atol=1.0,
            err_msg=(
                f"translation drifted: got {recovered_t}, expected {true_t}"
            ),
        )
        self.assertLess(
            abs(math.degrees(recovered_angle - true_angle)),
            1.0,
            f"rotation drifted: got {math.degrees(recovered_angle):.3f}°, "
            f"expected {math.degrees(true_angle):.3f}°",
        )


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/SimpleITK/rosa_core not importable in this environment.",
)
class TransformPointsTests(unittest.TestCase):
    """Pin the LPS<->RAS bridge in ``transform_to_4x4_ras``."""

    def test_apply_identity_4x4_is_a_noop(self):
        import numpy as np
        from rosa_core.registration import apply_transform_to_points_ras

        pts = np.array([[1.1, 2.2, 3.3], [-4.0, 5.5, -6.6]], dtype=float)
        out = apply_transform_to_points_ras(pts, np.eye(4))
        np.testing.assert_allclose(out, pts)

    def test_apply_pure_ras_translation(self):
        import numpy as np
        from rosa_core.registration import apply_transform_to_points_ras

        # +5 mm in R, -3 mm in A, +2 mm in S.
        m = np.eye(4)
        m[:3, 3] = [5.0, -3.0, 2.0]
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
        out = apply_transform_to_points_ras(pts, m)
        np.testing.assert_allclose(out[0], [5.0, -3.0, 2.0])
        np.testing.assert_allclose(out[1], [6.0, -2.0, 3.0])

    def test_translation_round_trip_through_lps_conversion(self):
        """When the registration recovers a pure LPS translation, the
        4×4 RAS form must mirror it across the L/P axes (sign flip).
        """
        import numpy as np
        import SimpleITK as sitk
        from rosa_core.registration import transform_to_4x4_ras

        tx = sitk.VersorRigid3DTransform()
        tx.SetCenter([0.0, 0.0, 0.0])
        tx.SetTranslation([5.0, -3.0, 2.0])  # LPS
        m_ras = transform_to_4x4_ras(tx)
        # LPS (+L=+5, +P=-3, +S=+2) ↔ RAS (+R=-5, +A=+3, +S=+2)
        np.testing.assert_allclose(
            m_ras[:3, 3], [-5.0, 3.0, 2.0], atol=1e-9,
            err_msg=(
                "LPS->RAS sign flip wrong in transform_to_4x4_ras: "
                f"got translation {m_ras[:3, 3]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
