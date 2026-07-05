"""Tests for multi-step jobs, the pipeline step chain, and viewer serving.

Multi-step execution is exercised with synthetic ``selftest-multi*`` kinds; the
``pipeline`` chain is checked structurally (its real run needs a CT + minutes,
verified manually, not here). Viewer serving is tested against a fake viewer dir
dropped into a job's workdir. Needs the app ``[test]`` extra.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

try:
    import httpx
    from httpx import ASGITransport
    from rosa_service.app import create_app
    from rosa_service.jobs import build_command
    from rosa_service.models import JobSpec
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

API = "/api/v1"


@unittest.skipUnless(HAVE_DEPS, "app [test] extra unavailable")
class PipelineBuildCommandTests(unittest.TestCase):
    def test_pipeline_maps_to_detect_contacts_viewresults(self):
        steps = build_command(
            JobSpec(kind="pipeline", params={"ct": "/data/ct.nii.gz", "label": "S1"}),
            Path("/tmp/jobdir"))
        self.assertEqual(len(steps), 3)
        self.assertIn("detect", steps[0])
        self.assertIn("contacts", steps[1])
        self.assertIn("view-results", steps[2])
        # The CT flows into every step; the label reaches view-results.
        for step in steps:
            self.assertIn("/data/ct.nii.gz", step)
        self.assertIn("S1", steps[2])

    def test_pipeline_requires_ct(self):
        with self.assertRaises(ValueError):
            build_command(JobSpec(kind="pipeline", params={}), Path("/tmp/jobdir"))

    def test_single_step_kinds_return_one_step(self):
        steps = build_command(JobSpec(kind="selftest"), Path("/tmp/jobdir"))
        self.assertEqual(len(steps), 1)


@unittest.skipUnless(HAVE_DEPS, "app [test] extra unavailable")
class MultiStepRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.app = create_app(work_root=self.td.name)
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.td.cleanup()

    async def _wait(self, jid):
        for _ in range(200):
            body = (await self.client.get(f"{API}/jobs/{jid}")).json()
            if body["state"] in ("succeeded", "failed", "cancelled"):
                return body
            await asyncio.sleep(0.05)
        raise AssertionError(f"job {jid} never finished (last={body})")

    async def _logs(self, jid):
        chunks = []
        async with self.client.stream("GET", f"{API}/jobs/{jid}/logs") as r:
            async for c in r.aiter_text():
                chunks.append(c)
        return "".join(chunks)

    async def test_multi_step_runs_all_steps_in_order(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-multi"})).json()["id"]
        final = await self._wait(jid)
        self.assertEqual(final["state"], "succeeded")
        body = await self._logs(jid)
        self.assertIn("step-A done", body)
        self.assertIn("step-B done", body)
        self.assertIn("[step 1/2]", body)   # step markers emitted for multi-step
        self.assertIn("[step 2/2]", body)

    async def test_multi_step_fails_fast(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-multi-fail"})).json()["id"]
        final = await self._wait(jid)
        self.assertEqual(final["state"], "failed")
        self.assertEqual(final["exit_code"], 2)
        body = await self._logs(jid)
        self.assertIn("step-A", body)
        self.assertNotIn("step-B should not run", body)   # 2nd step never ran


@unittest.skipUnless(HAVE_DEPS, "app [test] extra unavailable")
class ViewerServingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.app = create_app(work_root=self.td.name)
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.td.cleanup()

    async def _job_with_viewer(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        for _ in range(200):
            if (await self.client.get(f"{API}/jobs/{jid}")).json()["state"] == "succeeded":
                break
            await asyncio.sleep(0.05)
        vdir = Path(self.td.name) / jid / "viewer"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "index.html").write_text("<html>viewer</html>", encoding="utf-8")
        (vdir / "scene.glb").write_bytes(b"glTF\x02\x00\x00\x00")
        return jid

    async def test_serves_index_and_assets(self):
        jid = await self._job_with_viewer()
        # /viewer redirects to /viewer/ (so relative asset fetches resolve)
        r = await self.client.get(f"{API}/jobs/{jid}/viewer")
        self.assertEqual(r.status_code, 307)
        self.assertTrue(r.headers["location"].endswith(f"/jobs/{jid}/viewer/"))
        # index + asset served
        idx = await self.client.get(f"{API}/jobs/{jid}/viewer/")
        self.assertEqual(idx.status_code, 200)
        self.assertIn("viewer", idx.text)
        glb = await self.client.get(f"{API}/jobs/{jid}/viewer/scene.glb")
        self.assertEqual(glb.status_code, 200)
        self.assertEqual(glb.content[:4], b"glTF")

    async def test_missing_asset_404_and_no_viewer_404(self):
        jid = await self._job_with_viewer()
        r = await self.client.get(f"{API}/jobs/{jid}/viewer/nope.txt")
        self.assertEqual(r.status_code, 404)
        # a fresh job with no viewer dir
        jid2 = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        r2 = await self.client.get(f"{API}/jobs/{jid2}/viewer/")
        self.assertEqual(r2.status_code, 404)

    async def test_viewer_for_missing_job_404(self):
        r = await self.client.get(f"{API}/jobs/deadbeef/viewer/")
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
