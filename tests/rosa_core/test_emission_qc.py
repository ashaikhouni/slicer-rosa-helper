"""Unit tests for ``rosa_core.emission_qc``.

Heavy-volume integration is exercised by probe scripts and notebooks.
These tests pin:
  * the lazy-export contract (no eager rosa_detect / SITK pull)
  * dataclass shape (EmissionQC, PerModelCorr)
  * emission_qc_to_dict round-trip via JSON
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMONLIB = REPO_ROOT / "CommonLib"


class LazyExportTests(unittest.TestCase):
    def test_import_rosa_core_does_not_load_emission_qc(self):
        """Importing rosa_core shouldn't pull rosa_detect or matplotlib."""
        code = (
            "import sys\n"
            "import rosa_core\n"
            "loaded_emission_qc = 'rosa_core.emission_qc' in sys.modules\n"
            "loaded_rosa_detect = any(m == 'rosa_detect' or m.startswith('rosa_detect.')\n"
            "                          for m in sys.modules)\n"
            "loaded_mpl = 'matplotlib' in sys.modules\n"
            "print('emqc' if loaded_emission_qc else 'no_emqc')\n"
            "print('rd'   if loaded_rosa_detect else 'no_rd')\n"
            "print('mpl'  if loaded_mpl else 'no_mpl')\n"
        )
        env = {"PYTHONPATH": str(COMMONLIB), "PATH": ""}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        out = proc.stdout.splitlines()
        # rosa_core import is lazy: emission_qc not loaded until accessed.
        self.assertIn("no_emqc", out)
        self.assertIn("no_rd", out)
        self.assertIn("no_mpl", out)

    def test_emission_qc_lazy_attrs_resolve(self):
        """Accessing the public names triggers emission_qc import."""
        code = (
            "import rosa_core\n"
            "print(rosa_core.EmissionQC.__name__)\n"
            "print(rosa_core.PerModelCorr.__name__)\n"
            "print(rosa_core.build_emission_qc.__name__)\n"
            "print(rosa_core.build_subject_qc.__name__)\n"
            "print(rosa_core.emission_qc_to_dict.__name__)\n"
            "print(rosa_core.render_emission_figure.__name__)\n"
        )
        env = {"PYTHONPATH": str(COMMONLIB), "PATH": ""}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        out = proc.stdout.splitlines()
        expected = [
            "EmissionQC",
            "PerModelCorr",
            "build_emission_qc",
            "build_subject_qc",
            "emission_qc_to_dict",
            "render_emission_figure",
        ]
        self.assertEqual(out, expected)


class DataclassShapeTests(unittest.TestCase):
    """Smoke-construct the dataclasses; pin the public field surface."""

    def setUp(self):
        sys.path.insert(0, str(COMMONLIB))
        from rosa_core import EmissionQC, PerModelCorr, emission_qc_to_dict
        self.EmissionQC = EmissionQC
        self.PerModelCorr = PerModelCorr
        self.to_dict = emission_qc_to_dict

    def tearDown(self):
        cl = str(COMMONLIB)
        if cl in sys.path:
            sys.path.remove(cl)

    def test_per_model_corr_minimal(self):
        pm = self.PerModelCorr(model_id="DIXI-12AM", n_slots=12, n_covered=12, corr=0.9)
        self.assertEqual(pm.model_id, "DIXI-12AM")
        self.assertEqual(pm.n_slots, 12)
        self.assertEqual(pm.n_covered, 12)
        self.assertAlmostEqual(pm.corr, 0.9)

    def test_emission_qc_minimal(self):
        qc = self.EmissionQC(
            emission_idx=0,
            model_id="DIXI-12AM", corr=0.9, n_placed=12, bolt_source="metal",
            bolt_end_arc_mm=20.0,
            seed_start_ras=[0.0, 0.0, 0.0],
            seed_end_ras=[40.0, 0.0, 0.0],
            centerline_ras=[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]],
            placed_ras=[[3.0 * i, 0.0, 0.0] for i in range(1, 13)],
            placed_contact_arcs_mm=[3.0 * i for i in range(1, 13)],
            placed_hu=[2000.0] * 12, placed_hu_mean=2000.0, placed_hu_min=2000.0,
            walker_arcs_mm=list(range(0, 41)),
            walker_max_hu=[1500.0] * 41,
            per_model_corr=[
                self.PerModelCorr("DIXI-12AM", 12, 12, 0.9),
                self.PerModelCorr("DIXI-10AM", 10, 10, 0.7),
            ],
        )
        self.assertEqual(qc.model_id, "DIXI-12AM")
        self.assertEqual(qc.n_placed, 12)
        self.assertEqual(len(qc.per_model_corr), 2)
        self.assertEqual(qc.gt_match_name, None)
        self.assertEqual(qc.diagnostics, {})

    def test_emission_qc_to_dict_json_roundtrip(self):
        qc = self.EmissionQC(
            emission_idx=3,
            model_id="PMT-12", corr=0.55, n_placed=12, bolt_source="metal",
            bolt_end_arc_mm=15.0,
            seed_start_ras=[1.0, 2.0, 3.0],
            seed_end_ras=[40.0, 5.0, 6.0],
            centerline_ras=[[1.0, 2.0, 3.0], [20.0, 3.5, 4.5], [40.0, 5.0, 6.0]],
            placed_ras=[[float(i), 0.0, 0.0] for i in range(12)],
            placed_contact_arcs_mm=[float(i) for i in range(12)],
            placed_hu=[1800.0] * 12, placed_hu_mean=1800.0, placed_hu_min=1800.0,
            walker_arcs_mm=[0.25 * i for i in range(160)],
            walker_max_hu=[2000.0] * 160,
            per_model_corr=[self.PerModelCorr("PMT-12", 12, 12, 0.55)],
            gt_match_name="LH", expected_n_contacts=12,
        )
        d = self.to_dict(qc)
        # round-trip via JSON; this is what the CLI dumps.
        s = json.dumps(d)
        d2 = json.loads(s)
        self.assertEqual(d2["model_id"], "PMT-12")
        self.assertEqual(d2["gt_match_name"], "LH")
        self.assertEqual(d2["expected_n_contacts"], 12)
        # Nested dataclasses serialize as dicts via dataclasses.asdict.
        self.assertEqual(d2["per_model_corr"][0]["model_id"], "PMT-12")


if __name__ == "__main__":
    unittest.main()
