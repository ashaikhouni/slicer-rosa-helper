"""Tests for the web UI serving + upload + file-download routes.

The SPA itself (index.html/app.js) is served statically; these check the
service side of it — static serving with API precedence, the drag-drop upload
endpoint (incl. filename sanitisation), and job-file download. Needs the app
``[test]`` extra.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import httpx
    from httpx import ASGITransport
    from fastapi.testclient import TestClient
    from rosa_service.app import create_app
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

API = "/api/v1"


@unittest.skipUnless(HAVE_DEPS, "app [test] extra unavailable")
class StaticAndUploadTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(work_root=self.td.name))

    def tearDown(self):
        self.client.close()
        self.td.cleanup()

    def test_serves_spa_at_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("ROSA", r.text)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertEqual(self.client.get("/style.css").status_code, 200)

    def test_api_routes_take_precedence_over_spa_mount(self):
        # The catch-all "/" mount must not shadow the API.
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get(f"{API}/jobs").status_code, 200)

    def test_upload_saves_file_and_returns_path(self):
        r = self.client.post(f"{API}/uploads",
                             files={"file": ("ct.nii.gz", b"FAKECT", "application/gzip")})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "ct.nii.gz")
        self.assertEqual(body["bytes"], 6)
        p = Path(body["path"])
        self.assertTrue(p.is_file())
        self.assertIn("_uploads", p.parts)

    def test_upload_sanitises_filename(self):
        # A path-y filename must not escape the uploads dir.
        r = self.client.post(f"{API}/uploads",
                             files={"file": ("../../evil.nii", b"x", "application/octet-stream")})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "evil.nii")
        self.assertNotIn("..", Path(body["path"]).name)


@unittest.skipUnless(HAVE_DEPS, "app [test] extra unavailable")
class JobFileDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.app = create_app(work_root=self.td.name)
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.td.cleanup()

    async def _emit_job(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-emit"})).json()["id"]
        for _ in range(200):
            if (await self.client.get(f"{API}/jobs/{jid}")).json()["state"] == "succeeded":
                break
            await asyncio.sleep(0.05)
        return jid

    async def test_downloads_job_file(self):
        jid = await self._emit_job()
        r = await self.client.get(f"{API}/jobs/{jid}/files/contacts.tsv")
        self.assertEqual(r.status_code, 200)
        self.assertIn("trajectory", r.text)   # the TSV header

    async def test_missing_file_404_and_traversal_400(self):
        jid = await self._emit_job()
        self.assertEqual((await self.client.get(f"{API}/jobs/{jid}/files/nope.tsv")).status_code, 404)
        # An encoded traversal must not escape the job dir.
        r = await self.client.get(f"{API}/jobs/{jid}/files/..%2f..%2fsecret", follow_redirects=False)
        self.assertIn(r.status_code, (400, 404))


if __name__ == "__main__":
    unittest.main()
