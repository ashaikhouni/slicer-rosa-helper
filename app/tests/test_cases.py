"""App-side tests for the Cases list endpoint (GET /cases).

Checks that a finished case is summarized with electrode/contact counts + MRI /
label state, and that non-case jobs (label/selftest, or unfinished) are excluded.
Local only (``pytest app/tests``), not the engine CI.
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
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


def _make_case(root: Path, jid: str, *, n_contacts=12, n_shanks=1, kind="pipeline",
               state="succeeded", label="case", t1=None, region=None) -> None:
    jd = root / jid
    jd.mkdir(parents=True)
    with open(jd / "trajectories.tsv", "w") as f:
        f.write("name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\telectrode_model\n")
        for s in range(n_shanks):
            f.write(f"T0{s+1}\t0\t0\t0\t1\t1\t1\tDIXI-12AM\n")
    with open(jd / "contacts.tsv", "w") as f:
        f.write("trajectory\tlabel\tcontact_index\tx\ty\tz\n")
        for i in range(n_contacts):
            f.write(f"T0{i % n_shanks + 1}\tc{i}\t{i}\t{i}\t0\t0\n")
    params = {"label": label, "ct": str(jd / "ct.nii.gz")}
    if t1:
        params["t1"] = t1
    (jd / "manifest.json").write_text(json.dumps(
        {"id": jid, "kind": kind, "state": state, "params": params}))
    if region is not None:
        (jd / "review.json").write_text(json.dumps(
            {"shanks": [{"name": "T01", "contacts": [{"index": 0, "region": region}]}]}))


@unittest.skipUnless(HAVE, "fastapi unavailable")
class CasesListTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        os.environ["ROSA_APP_WORKDIR"] = str(self.root)

    def tearDown(self):
        self._td.cleanup()
        os.environ.pop("ROSA_APP_WORKDIR", None)

    def _client(self):
        from rosa_service.app import create_app
        return TestClient(create_app())

    def test_case_is_summarized_with_counts(self):
        _make_case(self.root, "aaaa1111", n_contacts=15, n_shanks=3,
                   label="LTP-case", t1="/some/t1.nii.gz", region="Hippocampus")
        cases = self._client().get("/api/v1/cases").json()
        self.assertEqual(len(cases), 1)
        c = cases[0]
        self.assertEqual(c["label"], "LTP-case")
        self.assertEqual(c["n_contacts"], 15)
        self.assertEqual(c["n_shanks"], 3)
        self.assertTrue(c["has_mri"])
        self.assertTrue(c["labeled"])
        self.assertEqual(c["kind"], "pipeline")

    def test_unlabeled_ct_only_case(self):
        _make_case(self.root, "bbbb2222", n_contacts=8, n_shanks=2, label="CT-only")
        c = self._client().get("/api/v1/cases").json()[0]
        self.assertFalse(c["has_mri"])
        self.assertFalse(c["labeled"])
        self.assertEqual(c["n_shanks"], 2)

    def test_import_kind_counts_too(self):
        _make_case(self.root, "cccc3333", n_contacts=10, n_shanks=1, kind="import")
        c = self._client().get("/api/v1/cases").json()[0]
        self.assertEqual(c["kind"], "import")
        self.assertEqual(c["n_contacts"], 10)

    def test_excludes_noncase_and_unfinished(self):
        _make_case(self.root, "dddd4444", kind="pipeline", state="succeeded")   # a case
        _make_case(self.root, "eeee5555", kind="label", state="succeeded")      # not a case
        _make_case(self.root, "ffff6666", kind="pipeline", state="running")     # unfinished
        cases = self._client().get("/api/v1/cases").json()
        self.assertEqual([c["id"] for c in cases], ["dddd4444"])


if __name__ == "__main__":
    unittest.main()
