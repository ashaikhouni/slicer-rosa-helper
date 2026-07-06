"""Tests for the bundled-atlas registry (rosa_core.bundled_atlases).

Pure-Python (no SimpleITK / nibabel): manifest integrity, CerebrA LUT
parsing, path resolution, and — importantly — that the committed atlas
binaries still match the sha256s pinned in atlases.json (a corrupted or
silently-replaced atlas file is caught here, not at label time).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rosa_core import bundled_atlases as ba


class ManifestTests(unittest.TestCase):
    def test_manifest_loads_and_default_is_valid(self):
        m = ba.load_manifest()
        self.assertIn(m["default"], m["atlases"])
        self.assertEqual(m["default"], "cerebra")

    def test_default_atlas_template_exists_in_manifest(self):
        m = ba.load_manifest()
        for entry in m["atlases"].values():
            self.assertIn(entry["template"], m["templates"])


class CerebraLutTests(unittest.TestCase):
    def setUp(self):
        self.assets = ba.resolve("cerebra")
        self.lut = ba.parse_lut(self.assets.lut_path, self.assets.lut_format)

    def test_background_is_unknown(self):
        self.assertEqual(self.lut[0], "Unknown")

    def test_hemisphere_split_thalamus(self):
        # CerebrA row "Thalamus,40,91" -> RH=40, LH=91.
        self.assertEqual(self.lut[40], "Right Thalamus")
        self.assertEqual(self.lut[91], "Left Thalamus")

    def test_deep_structures_present(self):
        vals = set(self.lut.values())
        for name in ("Left Hippocampus", "Right Amygdala", "Left Brainstem"):
            self.assertIn(name, vals)

    def test_all_102_labels_plus_background(self):
        # 51 structures x 2 hemispheres + background.
        self.assertEqual(len(self.lut), 103)


class ResolveTests(unittest.TestCase):
    def test_resolve_default_returns_existing_files(self):
        a = ba.resolve()  # default
        self.assertEqual(a.atlas_id, "cerebra")
        self.assertEqual(a.transform_kind, "affine")
        for p in (a.labelmap_path, a.template_path, a.lut_path):
            self.assertTrue(p.is_file(), p)

    def test_unknown_atlas_raises_keyerror(self):
        with self.assertRaises(KeyError):
            ba.resolve("does-not-exist")

    def test_list_atlases_reports_cerebra_available(self):
        rows = {r["id"]: r for r in ba.list_atlases()}
        self.assertIn("cerebra", rows)
        self.assertTrue(rows["cerebra"]["available"])
        self.assertTrue(rows["cerebra"]["is_default"])
        self.assertTrue(rows["cerebra"]["bundled"])


class IntegrityTests(unittest.TestCase):
    """The committed atlas binaries must match the manifest's pinned sha256s."""

    def test_bundled_files_match_pinned_checksums(self):
        # Every bundled atlas (labelmap + lut) and every template it uses.
        root = ba.default_resource_root()
        m = ba.load_manifest(root)
        checked = 0
        for entry in m["atlases"].values():
            if not entry.get("bundled"):
                continue
            tmpl = m["templates"][entry["template"]]
            for path, expected in [
                (root / entry["labelmap"], entry["labelmap_sha256"]),
                (root / entry["lut"], entry["lut_sha256"]),
                (root / tmpl["file"], tmpl["sha256"]),
            ]:
                self.assertEqual(ba.sha256_of(path), expected, f"checksum drift: {path.name}")
                checked += 1
        self.assertGreaterEqual(checked, 6)   # cerebra + thalamus_mial

    def test_ensure_available_bundled_verifies_without_download(self):
        for atlas_id in ("cerebra", "thalamus_mial"):
            a = ba.ensure_available(atlas_id, allow_download=False)
            self.assertEqual(a.atlas_id, atlas_id)

    def test_thalamic_atlas_is_distance_gated(self):
        a = ba.resolve("thalamus_mial")
        self.assertEqual(a.max_label_distance_mm, 2.0)   # thalamus-only → gated
        self.assertIsNone(ba.resolve("cerebra").max_label_distance_mm)  # whole-brain → ungated
        lut = ba.parse_lut(a.lut_path, a.lut_format)
        self.assertEqual(lut[0], "Unknown")
        self.assertIn("Pulvinar", lut[1])                # LH-Pulvinar

    def test_tsv_atlases_parse(self):
        # Harvard-Oxford + Schaefer ship BIDS index-name TSV LUTs.
        ho = ba.resolve("harvard_oxford")
        self.assertEqual(ho.lut_format, "tsv")
        ho_lut = ba.parse_lut(ho.lut_path, ho.lut_format)
        self.assertEqual(ho_lut[0], "Unknown")
        self.assertIn("Right Thalamus", ho_lut.values())    # subcortical merged in
        sch_lut = ba.parse_lut(*[(x.lut_path, x.lut_format) for x in [ba.resolve("schaefer")]][0])
        self.assertEqual(len(sch_lut), 401)                 # 400 parcels + background
        self.assertTrue(sch_lut[1].startswith("7Networks"))

    def test_all_bundled_atlases_available(self):
        rows = {r["id"]: r for r in ba.list_atlases()}
        for aid in ("cerebra", "thalamus_mial", "harvard_oxford", "schaefer",
                    "thalamus_iglesias", "suit_cerebellum"):
            self.assertIn(aid, rows)
            self.assertTrue(rows[aid]["available"], aid)

    def test_license_tiers(self):
        rows = {r["id"]: r for r in ba.list_atlases()}
        self.assertEqual(rows["cerebra"]["license_tier"], "permissive")
        self.assertEqual(rows["suit_cerebellum"]["license_tier"], "noncommercial")
        self.assertEqual(rows["thalamus_iglesias"]["license_tier"], "noncommercial")
        # every tier is one of the known values
        self.assertTrue(all(r["license_tier"] in ("permissive", "noncommercial", "fetch")
                            for r in rows.values()))


class ChecksumHelperTests(unittest.TestCase):
    def test_verify_mismatch_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("hello")
            p = f.name
        with self.assertRaises(ValueError):
            ba.verify_checksum(p, "0" * 64)

    def test_verify_unpinned_is_skipped(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("hello")
            p = f.name
        ba.verify_checksum(p, None)   # no raise
        ba.verify_checksum(p, "")     # no raise

    def test_ensure_missing_without_download_raises(self):
        # A manifest that points at a non-existent labelmap, downloads off.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "templates").mkdir()
            (root / "templates" / "t.nii.gz").write_bytes(b"x")
            manifest = {
                "schema_version": 1, "default": "ghost",
                "templates": {"tmpl": {"file": "templates/t.nii.gz", "sha256": ""}},
                "atlases": {"ghost": {
                    "name": "Ghost", "bundled": False,
                    "labelmap": "ghost/labels.nii.gz", "lut": "ghost/lut.csv",
                    "lut_format": "cerebra_csv", "template": "tmpl",
                    "transform_kind": "affine",
                }},
            }
            (root / "atlases.json").write_text(json.dumps(manifest))
            with self.assertRaises(FileNotFoundError):
                ba.ensure_available("ghost", root=root, allow_download=False)


if __name__ == "__main__":
    unittest.main()
