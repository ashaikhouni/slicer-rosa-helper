"""Tests for the anatomical-labeling flow (atlas picker → propose → approve).

The real ``label`` job runs SITK registration (slow, needs the engine); here we
exercise the *service* wiring with the synthetic ``selftest-label`` kind, which
drops a ``contacts_labeled.tsv`` matching ``selftest-emit``'s contacts. That
covers: /atlases, proposed-labels parsing, and approve → the parent ReviewDoc's
regions actually change. Driven with httpx.AsyncClient (see test_jobs).
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

sys_path = str(Path(__file__).resolve().parents[1])
import sys  # noqa: E402
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    import httpx
    from httpx import ASGITransport
    from rosa_service.app import create_app
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

API = "/api/v1"


@unittest.skipUnless(HAVE_DEPS, "fastapi/httpx (app [test] extra) unavailable")
class LabelingFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.app = create_app(work_root=self.td.name)
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.td.cleanup()

    async def _wait_terminal(self, jid, n=200):
        for _ in range(n):
            body = (await self.client.get(f"{API}/jobs/{jid}")).json()
            if body["state"] in ("succeeded", "failed", "cancelled"):
                return body
            await asyncio.sleep(0.05)
        raise AssertionError(f"job {jid} never finished (last={body})")

    async def _emit_parent(self) -> str:
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-emit"})).json()["id"]
        await self._wait_terminal(jid)
        return jid

    async def test_atlases_lists_cerebra(self):
        r = await self.client.get(f"{API}/atlases")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["default"], "cerebra")
        ids = {a["id"] for a in body["atlases"]}
        self.assertIn("cerebra", ids)

    async def test_propose_then_approve_sets_regions(self):
        parent = await self._emit_parent()
        # Synthetic label job carrying the parent link.
        label = (await self.client.post(
            f"{API}/jobs",
            json={"kind": "selftest-label", "params": {"parent": parent}})).json()["id"]
        await self._wait_terminal(label)

        # Proposed labels are readable but NOT yet in the parent review.
        prop = (await self.client.get(f"{API}/jobs/{label}/labels")).json()
        self.assertEqual(prop["n_contacts"], 6)
        self.assertEqual(prop["n_labeled"], 6)
        self.assertEqual(prop["parent"], parent)

        before = (await self.client.get(f"{API}/jobs/{parent}/review")).json()
        regions_before = {c["region"] for s in before["shanks"] for c in s["contacts"]}
        self.assertNotIn("Left Amygdala", regions_before)   # selftest-emit used "Amygdala"

        # Approve → regions land on the parent's ReviewDoc.
        appr = await self.client.post(f"{API}/jobs/{label}/labels/approve")
        self.assertEqual(appr.status_code, 200)
        doc = appr.json()
        lac = next(s for s in doc["shanks"] if s["name"] == "LAC")
        self.assertTrue(all(c["region"] == "Left Amygdala" for c in lac["contacts"]))
        lpc = next(s for s in doc["shanks"] if s["name"] == "LPC")
        self.assertTrue(all(c["region"] == "Left Hippocampus" for c in lpc["contacts"]))

    async def test_geometry_change_invalidates_proposed_labels(self):
        import hashlib
        parent = await self._emit_parent()
        label = (await self.client.post(
            f"{API}/jobs",
            json={"kind": "selftest-label", "params": {"parent": parent}})).json()["id"]
        await self._wait_terminal(label)
        contacts = self.app.state.runner.get(parent).workdir / "contacts.tsv"
        job = self.app.state.runner.get(label)
        job.params["contacts_sha256"] = hashlib.sha256(contacts.read_bytes()).hexdigest()
        self.assertFalse((await self.client.get(f"{API}/jobs/{label}/labels")).json()["stale"])
        contacts.write_text(contacts.read_text() + "\n")
        self.assertTrue((await self.client.get(f"{API}/jobs/{label}/labels")).json()["stale"])
        result = await self.client.post(f"{API}/jobs/{label}/labels/approve")
        self.assertEqual(result.status_code, 409)
        self.assertIn("Contacts changed", result.json()["detail"])

    async def test_label_requires_parent_contacts(self):
        # A parent with no contacts.tsv → 409.
        bare = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        await self._wait_terminal(bare)
        r = await self.client.post(f"{API}/jobs/{bare}/label",
                                   json={"t1": "/nope/t1.nii.gz", "atlas": "cerebra"})
        self.assertEqual(r.status_code, 409)

    async def test_labels_missing_output_is_409(self):
        # A label endpoint on a job that produced no contacts_labeled.tsv.
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        await self._wait_terminal(jid)
        r = await self.client.get(f"{API}/jobs/{jid}/labels")
        self.assertEqual(r.status_code, 409)


if __name__ == "__main__":
    unittest.main()
