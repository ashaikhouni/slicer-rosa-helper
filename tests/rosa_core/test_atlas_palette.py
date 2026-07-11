"""Unit tests for rosa_core.atlas_palette — meaningful per-atlas region colors.

Pure Python (no nibabel/torch); locks each resolution tier: publisher LUT,
Yeo-network coloring, FreeSurfer-id mapping, and the golden-ratio fallback,
plus the FreeSurfer-style LUT parser.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from rosa_core.atlas_palette import (  # noqa: E402
    build_atlas_palette, is_network_atlas, network_palette,
    parse_color_lut, golden_hue,
)


class AtlasPaletteTests(unittest.TestCase):
    def test_parse_freesurfer_style_color_lut(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lut.txt"
            p.write_text("# comment\n\n0  Unknown        0   0   0   0\n"
                         "1  LH-Pulvinar    255 0   0   0\n"
                         "2  LH-Anterior    0   255 0   0\n")
            lut = parse_color_lut(p)
            self.assertEqual(lut[1], (255, 0, 0))
            self.assertEqual(lut[2], (0, 255, 0))
            self.assertIn(0, lut)          # Unknown parsed (id 0)

    def test_publisher_lut_used_verbatim_with_golden_gaps(self):
        names = {1: "A", 2: "B", 3: "C"}
        pub = {1: (10, 20, 30), 2: (40, 50, 60)}   # 3 not covered
        pal = build_atlas_palette(names, publisher_lut=pub)
        self.assertEqual(pal[1], (10, 20, 30))
        self.assertEqual(pal[2], (40, 50, 60))
        self.assertEqual(pal[3], golden_hue(3))    # gap → golden fallback

    def test_network_atlas_detected_and_grouped_by_network(self):
        names = {
            1: "17Networks_LH_VisCent_Striate_1",
            2: "17Networks_LH_VisCent_Striate_2",
            3: "17Networks_RH_SomMotA_1",
            4: "7Networks_LH_Default_3",
        }
        self.assertTrue(is_network_atlas(names))
        pal = network_palette(names)
        # Same network (VisCent) → same base hue family; different networks differ.
        self.assertEqual(len(pal), 4)
        vis1, vis2 = pal[1], pal[2]
        som = pal[3]
        # VisCent parcels share a hue family (close), SomMot is a different family.
        import colorsys
        h1 = colorsys.rgb_to_hsv(*[c / 255 for c in vis1])[0]
        h2 = colorsys.rgb_to_hsv(*[c / 255 for c in vis2])[0]
        hs = colorsys.rgb_to_hsv(*[c / 255 for c in som])[0]
        self.assertAlmostEqual(h1, h2, places=5)        # same network → same hue
        self.assertNotAlmostEqual(h1, hs, places=2)     # different network → different hue

    def test_build_routes_network_atlas_to_network_palette(self):
        names = {i: f"17Networks_LH_VisCent_{i}" for i in range(1, 6)}
        pal = build_atlas_palette(names)
        self.assertEqual(set(pal), set(names))
        self.assertTrue(all(len(v) == 3 for v in pal.values()))

    def test_freesurfer_ids_used_when_majority_match(self):
        names = {17: "Left-Hippocampus", 53: "Right-Hippocampus", 999999: "novel"}
        fs = {17: (220, 20, 20), 53: (220, 20, 20), 18: (1, 2, 3)}
        pal = build_atlas_palette(names, freesurfer_lut=fs)
        self.assertEqual(pal[17], (220, 20, 20))
        self.assertEqual(pal[53], (220, 20, 20))
        self.assertEqual(pal[999999], golden_hue(999999))   # unmatched → golden

    def test_golden_fallback_when_no_structure(self):
        names = {1: "RegionOne", 2: "RegionTwo", 3: "RegionThree"}
        pal = build_atlas_palette(names)   # no publisher, not network, no fs
        self.assertEqual(pal[1], golden_hue(1))
        self.assertEqual(pal[2], golden_hue(2))
        self.assertNotEqual(pal[1], pal[2])            # distinct

    def test_empty_names(self):
        self.assertEqual(build_atlas_palette({}), {})


if __name__ == "__main__":
    unittest.main()
