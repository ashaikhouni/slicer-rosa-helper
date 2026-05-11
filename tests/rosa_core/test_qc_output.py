"""Unit tests for ``rosa_core.qc_output`` — manifest + TSV writers.

Schema-only — uses synthetic ``PlacedTrajectory`` objects so the tests
run fast without requiring a CT or features dict.
"""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
import tempfile

import numpy as np

from rosa_core.placement_modes import PlacedTrajectory, PlacementBatch
from rosa_core.qc_output import (
    CONTACT_COLUMNS,
    PIPELINE_VERSION,
    QC_TRAJECTORY_COLUMNS,
    trajectory_to_qc_row,
    write_contacts_tsv,
    write_diagnostics_cmp_tsv,
    write_manifest_json,
    write_qc_directory,
    write_trajectories_tsv,
)


def _make_traj(name="L1", model="DIXI-15CM", band="high", compound=0.85):
    return PlacedTrajectory(
        name=name,
        start_ras=np.array([10, 20, 30], dtype=float),
        end_ras=np.array([60, 22, 30], dtype=float),
        centerline_ras=np.array([[10, 20, 30], [60, 22, 30]], dtype=float),
        contacts_ras=[np.array([15 + 3.5 * k, 21, 30], dtype=float) for k in range(15)],
        model_id=model,
        compound_score=compound,
        band=band,
        bolt_source="metal",
        bolt_end_arc_mm=10.5,
        score_components={
            "model_id": model,
            "band": band,
            "compound_score": compound,
            "corr": 0.85,
            "n_slots": 15,
            "n_covered": 14,
            "n_covered_frac": 14 / 15,
            "n_placed": 15,
            "tube_like_frac": 0.92,
            "model_corr_margin": 0.21,
            "model_corr_uniformity": 0.55,
            "zone_cv": 0.12,
            "zone_ptp_mod": 1.45,
            "bolt_zone_frac": 0.18,
            "pitch_power_frac": 0.6,
            "fft_n_segments": 1,
            "fft_n_reliable_segments": 1,
            "fft_subject_norm": 0.95,
            "pitch_mm": 3.5,
            "model_uniform_pitch": True,
            "placed_hu_mean": 2401.5,
            "placed_hu_min": 1850.2,
            "cc_match": "BOLT-3",
            "cc_dist_mm": 1.85,
            "cc_arc_along_mm": 8.5,
            "cc_overlap_score": 0.63,
            "cc_n_voxels": 142,
            "subscores": {
                "s_corr": 0.85,
                "s_fft": 0.95,
                "s_tube": 0.92,
                "s_margin": 1.0,
                "s_walker": 1.0,
                "s_cc_overlap": 0.63,
                "s_seeder": 1.0,
                "bolt_only_penalty": 0.0,
            },
            "per_model_corr": [
                ("DIXI-15CM", 15, 14, 0.85),
                ("DIXI-12CM", 12, 12, 0.78),
            ],
        },
        diagnostics={"signal_kind": "neg_log_max", "n_slots": 15,
                     "n_covered": 14, "fft_reliable": True},
    )


def _make_batch(*, mode=1, trajectories=None):
    if trajectories is None:
        trajectories = [
            _make_traj("L1", band="high", compound=0.85),
            _make_traj("R1", band="medium", compound=0.55),
        ]
    return PlacementBatch(
        trajectories=trajectories,
        diagnostics={
            "mode": mode,
            "n_input_seeds": 0,
            "n_emitted": len(trajectories),
            "n_library_models": 5,
            "subject_fft_normalized": True,
            "band_floor": "medium",
        },
    )


class TrajectoryRowTests(unittest.TestCase):
    def test_back_compat_columns_first(self):
        traj = _make_traj()
        row = trajectory_to_qc_row(traj)
        # First 12 columns mirror TRAJECTORY_COLUMNS.
        for col in ("name", "start_x", "start_y", "start_z",
                     "end_x", "end_y", "end_z",
                     "confidence", "confidence_label", "electrode_model",
                     "bolt_source", "length_mm"):
            self.assertIn(col, row)

    def test_back_compat_aliases(self):
        traj = _make_traj(model="DIXI-15CM", band="high", compound=0.85)
        row = trajectory_to_qc_row(traj)
        self.assertEqual(row["confidence_label"], "high")
        self.assertEqual(row["electrode_model"], "DIXI-15CM")
        self.assertAlmostEqual(float(row["confidence"]), 0.85, places=4)
        # New columns also present.
        self.assertEqual(row["model_id"], "DIXI-15CM")
        self.assertEqual(row["band"], "high")
        self.assertAlmostEqual(float(row["compound_score"]), 0.85, places=4)

    def test_subscore_breakdown_present(self):
        row = trajectory_to_qc_row(_make_traj())
        for k in ("s_corr", "s_fft", "s_tube", "s_margin",
                  "s_walker", "s_cc_overlap", "s_seeder", "bolt_only_penalty"):
            self.assertIn(k, row)
            self.assertNotEqual(row[k], "")

    def test_length_mm_computed(self):
        row = trajectory_to_qc_row(_make_traj())
        # start (10,20,30) → end (60,22,30): distance ≈ sqrt(50²+2²) ≈ 50.04
        self.assertAlmostEqual(float(row["length_mm"]), 50.04, places=1)


class WriteTrajectoriesTsvTests(unittest.TestCase):
    def test_writes_with_qc_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectories.tsv"
            n = write_trajectories_tsv(path, [_make_traj("A"), _make_traj("B")])
            self.assertEqual(n, 2)
            text = path.read_text()
            header = text.splitlines()[0].split("\t")
            self.assertEqual(tuple(header), QC_TRAJECTORY_COLUMNS)
            self.assertEqual(len(text.splitlines()), 3)  # header + 2 rows


class WriteContactsTsvTests(unittest.TestCase):
    def test_one_row_per_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contacts.tsv"
            n = write_contacts_tsv(path, [_make_traj("L1"), _make_traj("R1")])
            self.assertEqual(n, 30)  # 2 traj × 15 contacts each
            text = path.read_text()
            header = text.splitlines()[0].split("\t")
            self.assertEqual(tuple(header), CONTACT_COLUMNS)
            # Spot-check first data row.
            r1 = text.splitlines()[1].split("\t")
            d = dict(zip(header, r1))
            self.assertEqual(d["trajectory"], "L1")
            self.assertEqual(d["label"], "L11")
            self.assertEqual(d["contact_index"], "1")
            self.assertEqual(d["electrode_model"], "DIXI-15CM")


class WriteManifestJsonTests(unittest.TestCase):
    def test_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            batch = _make_batch(mode=1)
            write_manifest_json(
                path, batch,
                ct_path="/data/AMC88.nii.gz",
                subject_id="AMC88",
                library_id="pmt_35",
                mode_args={"library": "pmt_35"},
                runtime_seconds=14.3,
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["subject_id"], "AMC88")
            self.assertEqual(data["ct_path"], "/data/AMC88.nii.gz")
            self.assertEqual(data["library_id"], "pmt_35")
            self.assertEqual(data["mode"], 1)
            self.assertEqual(data["pipeline_version"], PIPELINE_VERSION)
            self.assertEqual(data["n_trajectories"], 2)
            self.assertEqual(data["n_high"], 1)
            self.assertEqual(data["n_medium"], 1)
            self.assertEqual(data["n_low"], 0)
            self.assertEqual(data["runtime_seconds"], 14.3)
            self.assertIn("timestamp", data)

    def test_per_mode_diag_block_passes_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            batch = PlacementBatch(
                trajectories=[_make_traj("S1")],
                diagnostics={
                    "mode": 5,
                    "n_input_seeds": 1, "n_emitted": 1,
                    "n_library_models": 5,
                    "subject_fft_normalized": True,
                    "band_floor": None,
                    "mode5": {"n_seeds_matched": 1, "n_candidates_dropped": 0,
                              "snap_angle_tol_deg": 12.0, "snap_perp_tol_mm": 8.0},
                },
            )
            write_manifest_json(path, batch)
            data = json.loads(path.read_text())
            self.assertEqual(data["mode"], 5)
            self.assertIsNotNone(data["mode5"])
            self.assertEqual(data["mode5"]["n_seeds_matched"], 1)

    def test_handles_numpy_in_diag(self):
        # Diagnostics blocks may contain numpy scalars (from per-emission ctx
        # snapshots). The custom JSON encoder must coerce them.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            batch = PlacementBatch(
                trajectories=[_make_traj("S1")],
                diagnostics={
                    "mode": 1,
                    "n_input_seeds": 0, "n_emitted": 1,
                    "n_library_models": np.int64(5),  # numpy int
                    "subject_fft_normalized": True,
                    "band_floor": "medium",
                    "mode1": {"value": np.float32(0.85)},  # numpy float
                },
            )
            write_manifest_json(path, batch)  # should not raise
            data = json.loads(path.read_text())
            self.assertEqual(data["n_library_models"], 5)


class WriteDiagnosticsCmpTsvTests(unittest.TestCase):
    def test_includes_subscores_with_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "cmp.tsv"
            n = write_diagnostics_cmp_tsv(path, [_make_traj()])
            self.assertEqual(n, 1)
            text = path.read_text()
            header = text.splitlines()[0].split("\t")
            # Sub-prefixed sub-scores present.
            self.assertIn("sub_s_corr", header)
            self.assertIn("sub_s_seeder", header)
            self.assertIn("sub_bolt_only_penalty", header)
            # Stable leading columns.
            self.assertEqual(header[:6], [
                "name", "model_id", "band", "compound_score",
                "bolt_source", "bolt_end_arc_mm",
            ])

    def test_handles_empty_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics" / "cmp.tsv"
            n = write_diagnostics_cmp_tsv(path, [])
            self.assertEqual(n, 0)
            self.assertTrue(path.exists())  # writes header even when empty


class WriteQcDirectoryTests(unittest.TestCase):
    def test_writes_full_directory_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qc_out"
            batch = _make_batch()
            written = write_qc_directory(
                batch, out,
                ct_path="/data/AMC88.nii.gz",
                subject_id="AMC88",
                library_id="pmt_35",
                runtime_seconds=14.3,
                write_figures=False,  # no features supplied
            )
            self.assertEqual(written, out)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "trajectories.tsv").exists())
            self.assertTrue((out / "contacts.tsv").exists())
            self.assertTrue((out / "diagnostics" / "cmp.tsv").exists())
            # No figures dir written when features=None.
            self.assertFalse((out / "figures").exists())


if __name__ == "__main__":
    unittest.main()
