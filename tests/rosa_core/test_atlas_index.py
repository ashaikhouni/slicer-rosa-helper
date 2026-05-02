"""Pin the rosa_core.atlas_index helpers shared by both atlas providers.

Both Slicer's ``atlas_providers`` and the CLI's ``atlas_provider_headless``
delegate centroid math, sample-result formatting, and FreeSurfer LUT
parsing here. A regression in any of these would silently break atlas
label assignment on one or both sides.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


class AtlasIndexTests(unittest.TestCase):
    def test_compute_label_centroids_basic(self):
        import numpy as np
        from rosa_core.atlas_index import compute_label_centroids

        pts = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [50.0, 0.0, 0.0],
        ], dtype=float)
        labels = np.array([1, 1, 2, 2, 3], dtype=np.int32)
        centroids = compute_label_centroids(pts, labels)
        self.assertEqual(set(centroids.keys()), {1, 2, 3})
        np.testing.assert_allclose(centroids[1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(centroids[2], [11.0, 10.0, 10.0])
        np.testing.assert_allclose(centroids[3], [50.0, 0.0, 0.0])

    def test_compute_label_centroids_empty(self):
        import numpy as np
        from rosa_core.atlas_index import compute_label_centroids

        out = compute_label_centroids(
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=np.int32),
        )
        self.assertEqual(out, {})

    def test_distance_to_centroid_mm(self):
        from rosa_core.atlas_index import distance_to_centroid_mm

        self.assertEqual(distance_to_centroid_mm([0, 0, 0], None), 0.0)
        self.assertAlmostEqual(
            distance_to_centroid_mm([3.0, 4.0, 0.0], [0.0, 0.0, 0.0]),
            5.0,
        )

    def test_format_atlas_sample_shape(self):
        from rosa_core.atlas_index import format_atlas_sample

        s = format_atlas_sample(
            source_id="thomas",
            label_value=42,
            label_name="LEFT_VLA",
            distance_to_voxel_mm=1.25,
            distance_to_centroid_mm=4.5,
            native_ras=[1.1, 2.2, 3.3],
        )
        # Pin every key the policy layer reads.
        self.assertEqual(s["source"], "thomas")
        self.assertEqual(s["label"], "LEFT_VLA")
        self.assertEqual(s["label_value"], 42)
        self.assertAlmostEqual(s["distance_to_voxel_mm"], 1.25)
        self.assertAlmostEqual(s["distance_to_centroid_mm"], 4.5)
        self.assertEqual(s["native_ras"], [1.1, 2.2, 3.3])
        # Numeric fields must be plain Python floats / ints (json-safe).
        self.assertIs(type(s["distance_to_voxel_mm"]), float)
        self.assertIs(type(s["label_value"]), int)

    def test_parse_freesurfer_lut(self):
        from rosa_core.atlas_index import parse_freesurfer_lut

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lut.txt"
            path.write_text(
                "# header comment\n"
                "\n"
                "0   Unknown                          0   0   0   0\n"
                "10  Left-Thalamus-Proper             0   118 14  255\n"
                "garbage line\n"
                "11  Left-Caudate                     122 186 220 255\n"
            )
            out = parse_freesurfer_lut(path)
        self.assertEqual(out[0], "Unknown")
        self.assertEqual(out[10], "Left-Thalamus-Proper")
        self.assertEqual(out[11], "Left-Caudate")
        self.assertNotIn("garbage", out)


if __name__ == "__main__":
    unittest.main()
