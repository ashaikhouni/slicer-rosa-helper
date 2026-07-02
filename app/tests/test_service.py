"""App-side smoke test — the service starts and the engine link resolves.

Needs the app ``[test]`` extra (fastapi + httpx). Not run by the engine's
hosted CI (which installs only the engine test deps); runs locally / in the
app's own CI via ``pytest app/tests``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient
    from rosa_service.app import create_app
    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "fastapi/httpx (app [test] extra) unavailable")
class HealthzTests(unittest.TestCase):
    def test_healthz_ok(self):
        client = TestClient(create_app())
        resp = client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["api"], "v1")

    def test_engine_link_resolves(self):
        # The whole point of the scaffold: the app can import the engine.
        client = TestClient(create_app())
        body = client.get("/healthz").json()
        self.assertTrue(body["engine_import_ok"],
                        "rosa_core (engine) is not importable from the app")


if __name__ == "__main__":
    unittest.main()
