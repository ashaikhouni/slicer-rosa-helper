"""Tests for the ReviewDoc — loader, edits, export, and the job→review flow.

Unit tests drive the pure logic over a synthetic contacts.tsv; endpoint tests
run a ``selftest-emit`` job (which writes a contacts.tsv) then review + export
over it. Needs the app ``[test]`` extra AND the engine importable (review.py
uses the engine's TSV reader). Runs via ``pytest app/tests``, not engine CI.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))                 # app/
sys.path.insert(0, str(_HERE.parents[2] / "CommonLib"))   # engine
sys.path.insert(0, str(_HERE.parents[2] / "cli"))         # engine

try:
    import httpx
    from httpx import ASGITransport
    from rosa_service.app import create_app
    from rosa_service.models import ReviewEdit, ReviewOp
    from rosa_service.review import apply_edit, build_review_doc, export_contacts
    from rosa_agent.io.trajectory_io import read_tsv_rows
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

API = "/api/v1"


def _write_contacts(path: Path, *, with_region: bool = True) -> None:
    cols = ["trajectory", "label", "contact_index", "x", "y", "z",
            "peak_detected", "electrode_model"] + (["region"] if with_region else [])
    lines = ["\t".join(cols)]
    for shank, region in (("LAC", "Amygdala"), ("LPC", "Hippocampus")):
        for i in range(1, 4):
            row = [shank, f"{shank}{i}", str(i), f"{i}.0", f"{i * 2}.0", "5.0",
                   "1", "DIXI-15AM"] + ([region] if with_region else [])
            lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@unittest.skipUnless(HAVE_DEPS, "app [test] extra + engine unavailable")
class ReviewLogicTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        _write_contacts(self.dir / "contacts.tsv")

    def tearDown(self):
        self.td.cleanup()

    def test_build_groups_shanks_and_contacts(self):
        doc = build_review_doc(self.dir)
        self.assertEqual([s.name for s in doc.shanks], ["LAC", "LPC"])
        lac = doc.shanks[0]
        self.assertEqual(len(lac.contacts), 3)
        self.assertEqual(lac.model, "DIXI-15AM")
        self.assertEqual(lac.contacts[0].name, "LAC1")
        self.assertEqual(lac.contacts[0].region, "Amygdala")
        self.assertTrue(all(c.accepted for c in lac.contacts))

    def test_build_without_region_is_none(self):
        _write_contacts(self.dir / "contacts.tsv", with_region=False)
        doc = build_review_doc(self.dir)
        self.assertIsNone(doc.shanks[0].contacts[0].region)

    def test_edits_reject_relabel_and_shank(self):
        doc = build_review_doc(self.dir)
        apply_edit(doc, ReviewEdit(op=ReviewOp.reject_contact, shank="LAC", index=2))
        apply_edit(doc, ReviewEdit(op=ReviewOp.relabel_contact, shank="LPC", index=1, region="Thalamus"))
        apply_edit(doc, ReviewEdit(op=ReviewOp.reject_shank, shank="LAC"))
        lac = doc.shanks[0]
        self.assertFalse(lac.accepted)
        self.assertFalse(next(c for c in lac.contacts if c.index == 2).accepted)
        lpc1 = doc.shanks[1].contacts[0]
        self.assertEqual(lpc1.region, "Thalamus")

    def test_invalid_edits_raise(self):
        doc = build_review_doc(self.dir)
        with self.assertRaises(ValueError):
            apply_edit(doc, ReviewEdit(op=ReviewOp.reject_shank, shank="NOPE"))
        with self.assertRaises(ValueError):
            apply_edit(doc, ReviewEdit(op=ReviewOp.reject_contact, shank="LAC", index=99))
        with self.assertRaises(ValueError):
            apply_edit(doc, ReviewEdit(op=ReviewOp.reject_contact, shank="LAC"))  # no index
        with self.assertRaises(ValueError):
            apply_edit(doc, ReviewEdit(op=ReviewOp.relabel_contact, shank="LAC", index=1))  # no region

    def test_export_only_accepted(self):
        doc = build_review_doc(self.dir)
        apply_edit(doc, ReviewEdit(op=ReviewOp.reject_contact, shank="LAC", index=2))
        apply_edit(doc, ReviewEdit(op=ReviewOp.reject_shank, shank="LPC"))
        out = self.dir / "reviewed.tsv"
        n = export_contacts(doc, out)
        self.assertEqual(n, 2)  # LAC has 3, minus rejected #2 → 2; LPC shank rejected → 0
        rows = read_tsv_rows(out)
        self.assertEqual({r["trajectory"] for r in rows}, {"LAC"})
        self.assertEqual({r["contact_index"] for r in rows}, {"1", "3"})


@unittest.skipUnless(HAVE_DEPS, "app [test] extra + engine unavailable")
class ReviewEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.app = create_app(work_root=self.td.name)
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.td.cleanup()

    async def _run_emit_job(self) -> str:
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-emit"})).json()["id"]
        for _ in range(200):
            if (await self.client.get(f"{API}/jobs/{jid}")).json()["state"] in ("succeeded", "failed", "cancelled"):
                break
            await asyncio.sleep(0.05)
        return jid

    async def test_review_flow_end_to_end(self):
        jid = await self._run_emit_job()

        doc = (await self.client.get(f"{API}/jobs/{jid}/review")).json()
        self.assertEqual([s["name"] for s in doc["shanks"]], ["LAC", "LPC"])
        self.assertEqual(len(doc["shanks"][0]["contacts"]), 3)

        patch = {"ops": [
            {"op": "reject_contact", "shank": "LAC", "index": 2},
            {"op": "relabel_contact", "shank": "LPC", "index": 1, "region": "Thalamus"},
        ]}
        patched = (await self.client.patch(f"{API}/jobs/{jid}/review", json=patch)).json()
        lac2 = next(c for c in patched["shanks"][0]["contacts"] if c["index"] == 2)
        self.assertFalse(lac2["accepted"])
        self.assertEqual(patched["shanks"][1]["contacts"][0]["region"], "Thalamus")

        exp = (await self.client.post(f"{API}/jobs/{jid}/review/export")).json()
        self.assertEqual(exp["n_contacts"], 5)  # 6 total − 1 rejected
        out = Path(self.td.name) / jid / "contacts_reviewed.tsv"
        self.assertTrue(out.is_file())

    async def test_review_missing_job_404(self):
        r = await self.client.get(f"{API}/jobs/deadbeef/review")
        self.assertEqual(r.status_code, 404)

    async def test_review_no_contacts_409(self):
        # a plain selftest job produces no contacts.tsv
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        for _ in range(200):
            if (await self.client.get(f"{API}/jobs/{jid}")).json()["state"] != "queued":
                if (await self.client.get(f"{API}/jobs/{jid}")).json()["state"] in ("succeeded", "failed"):
                    break
            await asyncio.sleep(0.05)
        r = await self.client.get(f"{API}/jobs/{jid}/review")
        self.assertEqual(r.status_code, 409)

    async def test_patch_invalid_op_422(self):
        jid = await self._run_emit_job()
        await self.client.get(f"{API}/jobs/{jid}/review")   # build first
        r = await self.client.patch(f"{API}/jobs/{jid}/review",
                                    json={"ops": [{"op": "reject_shank", "shank": "NOPE"}]})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
