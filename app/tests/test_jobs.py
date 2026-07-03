"""Tests for the job runner + endpoints — lifecycle, logs (SSE), and cancel.

Exercised with fast synthetic ``selftest*`` job kinds (real subprocesses, no
data), so the full runner path is covered without a pipeline run.

Driven with ``httpx.AsyncClient`` over ``ASGITransport`` inside a single event
loop (``IsolatedAsyncioTestCase``) — the job runs as a background
``asyncio.create_task`` on that same loop, and ``await asyncio.sleep`` lets it
progress. (Starlette's sync ``TestClient`` tears the loop down after each
request, so the background subprocess would never advance.)

Needs the app ``[test]`` extra (fastapi + httpx); runs via ``pytest app/tests``,
not the engine's CI.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import httpx
    from httpx import ASGITransport
    from rosa_service.app import create_app
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False

API = "/api/v1"


@unittest.skipUnless(HAVE_DEPS, "fastapi/httpx (app [test] extra) unavailable")
class JobEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.app = create_app(work_root=self.td.name)
        self.client = httpx.AsyncClient(
            transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.td.cleanup()

    async def _wait_terminal(self, jid, timeout_iters=200):
        for _ in range(timeout_iters):
            body = (await self.client.get(f"{API}/jobs/{jid}")).json()
            if body["state"] in ("succeeded", "failed", "cancelled"):
                return body
            await asyncio.sleep(0.05)
        raise AssertionError(f"job {jid} never finished (last={body})")

    async def test_selftest_job_runs_to_success(self):
        r = await self.client.post(f"{API}/jobs", json={"kind": "selftest", "params": {"steps": 3}})
        self.assertEqual(r.status_code, 201)
        jid = r.json()["id"]
        final = await self._wait_terminal(jid)
        self.assertEqual(final["state"], "succeeded")
        self.assertEqual(final["exit_code"], 0)
        self.assertIsNotNone(final["started_at"])
        self.assertIsNotNone(final["ended_at"])

    async def test_logs_stream_replays_output(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest", "params": {"steps": 2}})).json()["id"]
        await self._wait_terminal(jid)
        chunks = []
        async with self.client.stream("GET", f"{API}/jobs/{jid}/logs") as resp:
            self.assertEqual(resp.status_code, 200)
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
        body = "".join(chunks)
        self.assertIn("done", body)
        self.assertIn("step", body)

    async def test_failing_job_reports_failure(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-fail"})).json()["id"]
        final = await self._wait_terminal(jid)
        self.assertEqual(final["state"], "failed")
        self.assertEqual(final["exit_code"], 3)

    async def test_cancel_running_job(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest-hang"})).json()["id"]
        for _ in range(100):  # wait until it's actually running
            if (await self.client.get(f"{API}/jobs/{jid}")).json()["state"] == "running":
                break
            await asyncio.sleep(0.05)
        resp = await self.client.delete(f"{API}/jobs/{jid}")
        self.assertEqual(resp.status_code, 200)
        final = await self._wait_terminal(jid)
        self.assertEqual(final["state"], "cancelled")

    async def test_unknown_kind_is_422(self):
        r = await self.client.post(f"{API}/jobs", json={"kind": "does-not-exist"})
        self.assertEqual(r.status_code, 422)

    async def test_missing_job_is_404(self):
        r = await self.client.get(f"{API}/jobs/deadbeef")
        self.assertEqual(r.status_code, 404)

    async def test_list_reflects_created_jobs(self):
        a = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        b = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        ids = {row["id"] for row in (await self.client.get(f"{API}/jobs")).json()}
        self.assertIn(a, ids)
        self.assertIn(b, ids)
        await self._wait_terminal(a)
        await self._wait_terminal(b)

    async def test_manifest_written_to_job_dir(self):
        jid = (await self.client.post(f"{API}/jobs", json={"kind": "selftest"})).json()["id"]
        await self._wait_terminal(jid)
        manifest = Path(self.td.name) / jid / "manifest.json"
        self.assertTrue(manifest.is_file())
        data = json.loads(manifest.read_text())
        self.assertEqual(data["id"], jid)
        self.assertEqual(data["state"], "succeeded")
        self.assertEqual(data["exit_code"], 0)


@unittest.skipUnless(HAVE_DEPS, "fastapi/httpx (app [test] extra) unavailable")
class RehydrateTests(unittest.IsolatedAsyncioTestCase):
    async def test_finished_jobs_survive_a_new_runner(self):
        from rosa_service.jobs import JobRunner
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        app = create_app(work_root=td.name)
        client = httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
        jid = (await client.post(f"{API}/jobs", json={"kind": "selftest-emit"})).json()["id"]
        for _ in range(200):
            if (await client.get(f"{API}/jobs/{jid}")).json()["state"] == "succeeded":
                break
            await asyncio.sleep(0.05)
        await client.aclose()

        # A fresh runner on the same work dir rebuilds the finished job.
        r2 = JobRunner(td.name)
        statuses = {s.id: s.state.value for s in r2.list()}
        self.assertIn(jid, statuses)
        self.assertEqual(statuses[jid], "succeeded")


if __name__ == "__main__":
    unittest.main()
