"""App-side tests for importing a localization computed elsewhere (CLI batch).

Covers the three backend pieces:
  * ``import_check.check_localization`` — the parity content check (contacts on
    metal / in bounds), green on a matching CT, red on a mismatched one.
  * the ``import`` job kind — stages the TSVs and runs view-results ONLY (no
    detect/contacts), so an imported case reviews like a pipeline one.
  * ``POST /jobs/import`` — rejects a missing file / red pairing with 422 and
    creates a case on a good pairing.

Needs numpy+nibabel (the check loads the CT) and the app [test] extra; skips
cleanly otherwise. Local only (``pytest app/tests``), not the engine CI.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import nibabel as nib
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def _make_localization(root: Path, *, origin=(-20.0, -20.0, -15.0)):
    """A CT with a bright electrode streak + matching contacts/trajectories TSVs,
    laid out like a CLI batch output dir. Returns (ct, contacts, trajectories)."""
    out = root / "cli_out"
    out.mkdir(parents=True, exist_ok=True)
    aff = np.eye(4)
    aff[:3, 3] = origin
    arr = np.zeros((40, 40, 30), np.int16)
    arr[5:35, 15, 10] = 2500                            # the electrode metal
    ct = out / "ct.nii.gz"
    nib.save(nib.Nifti1Image(arr, aff), str(ct))

    def ras(i, j, k):
        return (i + origin[0], j + origin[1], k + origin[2])

    traj = out / "trajectories.tsv"
    with open(traj, "w") as f:
        f.write("name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\t"
                "confidence\tconfidence_label\telectrode_model\tbolt_source\tlength_mm\n")
        e, t = ras(5, 15, 10), ras(34, 15, 10)
        f.write(f"T01\t{e[0]}\t{e[1]}\t{e[2]}\t{t[0]}\t{t[1]}\t{t[2]}\t0.9\thigh\tDIXI-12AM\tmetal\t29\n")
    contacts = out / "contacts.tsv"
    with open(contacts, "w") as f:
        f.write("trajectory\tlabel\tcontact_index\tx\ty\tz\tpeak_detected\telectrode_model\n")
        for i in range(12):
            p = ras(5 + i * 2.6, 15, 10)
            f.write(f"T01\tT01{i+1}\t{i+1}\t{p[0]}\t{p[1]}\t{p[2]}\t1\tDIXI-12AM\n")
    return ct, contacts, traj


@unittest.skipUnless(HAVE, "numpy/nibabel/fastapi unavailable")
class ContentCheckTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_matching_ct_is_green(self):
        from rosa_service.import_check import check_localization
        ct, contacts, _ = _make_localization(self.root)
        r = check_localization(ct, contacts)
        self.assertEqual(r["verdict"], "green", r)
        self.assertEqual(r["n"], 12)
        self.assertGreaterEqual(r["frac_on_metal"], 0.7)

    def test_wrong_ct_out_of_bounds_is_red(self):
        from rosa_service.import_check import check_localization
        # contacts built for origin (-20,-20,-15); score them against a CT whose
        # world origin is far away -> every contact maps outside the volume.
        _, contacts, _ = _make_localization(self.root)
        other = _make_localization(self.root / "other", origin=(-500.0, -500.0, -500.0))[0]
        r = check_localization(other, contacts)
        self.assertEqual(r["verdict"], "red", r)
        self.assertLess(r["frac_in_bounds"], 0.5)


@unittest.skipUnless(HAVE, "numpy/nibabel/fastapi unavailable")
class ImportKindTests(unittest.TestCase):
    def test_import_kind_stages_then_views_no_detect(self):
        from rosa_service.jobs import build_command
        from rosa_service.models import JobSpec
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            spec = JobSpec(kind="import", params={
                "ct": "/x/ct.nii.gz", "contacts": "/x/contacts.tsv",
                "trajectories": "/x/trajectories.tsv", "label": "case"})
            steps = build_command(spec, wd)
        flat = [" ".join(s) for s in steps]
        self.assertIn("shutil", flat[0])                       # first: stage the TSVs
        self.assertTrue(any("view-results" in s for s in flat))
        self.assertFalse(any(" detect " in f" {s} " for s in flat))   # no re-detection
        self.assertFalse(any(s.strip().endswith("contacts") or " contacts " in s
                             for s in flat if "view-results" not in s and "shutil" not in s))

    def test_import_kind_requires_trajectories(self):
        from rosa_service.jobs import build_command
        from rosa_service.models import JobSpec
        with tempfile.TemporaryDirectory() as td:
            spec = JobSpec(kind="import", params={
                "ct": "/x/ct.nii.gz", "contacts": "/x/contacts.tsv"})
            with self.assertRaises(ValueError):
                build_command(spec, Path(td))


@unittest.skipUnless(HAVE, "numpy/nibabel/fastapi unavailable")
class ImportRouteTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        os.environ["ROSA_APP_WORKDIR"] = str(self.root)
        from rosa_service.app import create_app
        app = create_app()

        # Don't spawn the real view-results engine in a unit test: stub the
        # runner's step executor so create() registers a job without subprocesses.
        async def _noop_run(job):
            from rosa_service.models import JobState
            job.state = JobState.succeeded
        app.state.runner._run = _noop_run
        self.client = TestClient(app)

    def tearDown(self):
        self._td.cleanup()
        os.environ.pop("ROSA_APP_WORKDIR", None)

    def test_missing_file_is_422(self):
        ct, contacts, traj = _make_localization(self.root)
        r = self.client.post("/api/v1/jobs/import", json={
            "ct": str(ct), "contacts": str(self.root / "nope.tsv"),
            "trajectories": str(traj)})
        self.assertEqual(r.status_code, 422)

    def test_red_pairing_is_422_no_job(self):
        _, contacts, traj = _make_localization(self.root)
        other = _make_localization(self.root / "other", origin=(-500.0, -500.0, -500.0))[0]
        r = self.client.post("/api/v1/jobs/import", json={
            "ct": str(other), "contacts": str(contacts), "trajectories": str(traj)})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.client.get("/api/v1/jobs").json(), [])   # nothing created

    def test_good_pairing_creates_case(self):
        ct, contacts, traj = _make_localization(self.root)
        r = self.client.post("/api/v1/jobs/import", json={
            "ct": str(ct), "contacts": str(contacts), "trajectories": str(traj),
            "label": "batch-case"})
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["check"]["verdict"], "green")
        self.assertEqual(body["job"]["kind"], "import")
        self.assertEqual(body["job"]["label"], "batch-case")
        jobs = self.client.get("/api/v1/jobs").json()
        self.assertEqual(len(jobs), 1)


if __name__ == "__main__":
    unittest.main()
