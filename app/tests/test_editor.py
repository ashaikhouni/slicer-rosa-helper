"""App-side test for the trajectory-editor payload + routes.

Synthesizes a tiny case (CT + trajectories/contacts + manifest) in a temp
workdir, rehydrates it as a job, and checks the editor plan/volume/page routes.
Needs numpy+nibabel+scipy (the payload builder) and the app [test] extra; skips
cleanly otherwise. Runs locally via ``pytest app/tests`` (not the engine CI).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import nibabel as nib
    import scipy.ndimage  # noqa: F401 — payload builder resamples with map_coordinates
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def _make_case(root: Path) -> str:
    jid = "testjob"
    jd = root / jid
    jd.mkdir(parents=True)
    up = root / "_uploads"
    up.mkdir(exist_ok=True)
    aff = np.eye(4)
    aff[:3, 3] = [-20.0, -20.0, -15.0]                 # RAS = index + origin
    arr = np.zeros((40, 40, 30), np.int16)
    arr[5:35, 15, 10] = 2500                           # a bright "electrode" streak
    ct = up / "ct.nii.gz"
    nib.save(nib.Nifti1Image(arr, aff), str(ct))

    def ras(i, j, k):
        return (i + aff[0, 3], j + aff[1, 3], k + aff[2, 3])

    with open(jd / "trajectories.tsv", "w") as f:
        f.write("name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\t"
                "confidence\tconfidence_label\telectrode_model\tbolt_source\tlength_mm\n")
        e, t = ras(5, 15, 10), ras(34, 15, 10)
        f.write(f"T01\t{e[0]}\t{e[1]}\t{e[2]}\t{t[0]}\t{t[1]}\t{t[2]}\t0.9\thigh\tDIXI-12AM\tmetal\t29\n")
    with open(jd / "contacts.tsv", "w") as f:
        f.write("trajectory\tlabel\tcontact_index\tx\ty\tz\tpeak_detected\telectrode_model\n")
        for i in range(12):
            p = ras(5 + i * 2.6, 15, 10)
            f.write(f"T01\tT01{i+1}\t{i+1}\t{p[0]}\t{p[1]}\t{p[2]}\t1\tDIXI-12AM\n")
    (jd / "manifest.json").write_text(json.dumps(
        {"id": jid, "kind": "pipeline", "state": "succeeded", "params": {"ct": str(ct)}}))
    return jid


@unittest.skipUnless(HAVE, "numpy/nibabel/scipy/fastapi unavailable")
class EditorRouteTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.jid = _make_case(self.root)
        os.environ["ROSA_APP_WORKDIR"] = str(self.root)
        from rosa_service.app import create_app
        self.client = TestClient(create_app())

    def tearDown(self):
        self._td.cleanup()
        os.environ.pop("ROSA_APP_WORKDIR", None)

    def test_plan_geometry_and_library(self):
        r = self.client.get(f"/api/v1/jobs/{self.jid}/editor/plan")
        self.assertEqual(r.status_code, 200)
        p = r.json()
        self.assertEqual(len(p["trajectories"]), 1)
        self.assertEqual(len(p["dims"]), 3)
        self.assertIn("DIXI-12AM", p["models"])           # canonical library present
        self.assertIn("DIXI-18CM", p["models"])
        # geometry is in crop-index space (>= 0, within dims)
        t = p["trajectories"][0]
        self.assertTrue(all(0 <= t["entry"][k] <= p["dims"][k] for k in range(3)))

    def test_volume_is_int16_matching_dims(self):
        p = self.client.get(f"/api/v1/jobs/{self.jid}/editor/plan").json()
        r = self.client.get(f"/api/v1/jobs/{self.jid}/editor/volume")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/octet-stream")
        self.assertEqual(len(r.content), int(np.prod(p["dims"])) * 2)   # int16 == 2 bytes/vox

    def test_page_serves(self):
        r = self.client.get(f"/api/v1/jobs/{self.jid}/editor/", follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("In-line", r.text)                  # the reslicer page

    def test_unknown_job_404(self):
        self.assertEqual(self.client.get("/api/v1/jobs/nope/editor/plan").status_code, 404)


@unittest.skipUnless(HAVE, "numpy/nibabel/scipy/fastapi unavailable")
class EditorWritebackTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.jid = _make_case(self.root)
        os.environ["ROSA_APP_WORKDIR"] = str(self.root)
        from rosa_service.app import create_app
        app = create_app()
        async def _noop(job):                      # don't spawn the rebuild engine
            from rosa_service.models import JobState
            job.state = JobState.succeeded
        app.state.runner._run = _noop
        self.client = TestClient(app)

    def tearDown(self):
        self._td.cleanup()
        os.environ.pop("ROSA_APP_WORKDIR", None)

    def test_save_rewrites_tsvs_and_regenerates_contacts(self):
        import csv
        plan = self.client.get(f"/api/v1/jobs/{self.jid}/editor/plan").json()
        self.assertIn("origin", plan)               # needed to invert index→world
        t = plan["trajectories"][0]
        t["name"] = "RENAMED"; t["tipOffset"] = 1.0
        r = self.client.post(f"/api/v1/jobs/{self.jid}/editor/plan", json=plan)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["n_trajectories"], 1)
        self.assertIsNotNone(body["rebuild_job"])   # viewer rebuild kicked off
        traj = (self.root / self.jid / "trajectories.tsv").read_text()
        self.assertIn("RENAMED", traj)
        con = list(csv.DictReader(open(self.root / self.jid / "contacts.tsv"), delimiter="\t"))
        self.assertTrue(con and all(c["trajectory"] == "RENAMED" for c in con))
        self.assertEqual(len(con), plan["models"][t["model"]]["n"])   # contacts == comb length

    def test_removed_shank_drops_from_tsv(self):
        plan = self.client.get(f"/api/v1/jobs/{self.jid}/editor/plan").json()
        plan["trajectories"] = []                   # remove everything
        r = self.client.post(f"/api/v1/jobs/{self.jid}/editor/plan", json=plan)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["n_contacts"], 0)

    def test_bad_plan_is_422(self):
        r = self.client.post(f"/api/v1/jobs/{self.jid}/editor/plan",
                             json={"trajectories": []})   # no origin/models
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
