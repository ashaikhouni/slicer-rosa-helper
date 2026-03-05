import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from shank_engine import (  # noqa: E402
    DIAGNOSTICS_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    DetectionError,
    default_result,
    sanitize_result,
)


class ContractsTests(unittest.TestCase):
    def test_default_result_has_required_sections(self):
        out = default_result(
            pipeline_id="blob_ransac_v1",
            pipeline_version="1.0.0",
            run_id="run-1",
            status="ok",
            params={"threshold": 1800},
        )
        self.assertEqual(out["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(out["diagnostics"]["schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
        for key in ("counts", "timing", "reason_codes", "params", "notes", "extras"):
            self.assertIn(key, out["diagnostics"])

    def test_sanitize_result_is_json_serializable(self):
        out = default_result(
            pipeline_id="blob_ransac_v1",
            pipeline_version="1.0.0",
            run_id="run-2",
            status="ok",
            params={"threshold": 1800},
        )
        out["diagnostics"]["timing"]["runtime_ms"] = np.float32(12.5)
        out["meta"]["array"] = np.asarray([[1, 2], [3, 4]], dtype=np.int16)

        clean = sanitize_result(out)
        payload = json.dumps(clean)
        self.assertIsInstance(payload, str)

    def test_schema_coercion_adds_warning(self):
        out = default_result(
            pipeline_id="blob_ransac_v1",
            pipeline_version="1.0.0",
            run_id="run-3",
            status="ok",
            params={},
        )
        out["schema_version"] = "old.v0"
        out["diagnostics"]["schema_version"] = "old.diag.v0"
        clean = sanitize_result(out)
        self.assertEqual(clean["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(clean["diagnostics"]["schema_version"], DIAGNOSTICS_SCHEMA_VERSION)
        self.assertGreaterEqual(len(clean["warnings"]), 2)

    def test_strict_schema_raises(self):
        out = default_result(
            pipeline_id="blob_ransac_v1",
            pipeline_version="1.0.0",
            run_id="run-4",
            status="ok",
            params={},
        )
        out["schema_version"] = "old.v0"
        with self.assertRaises(DetectionError):
            sanitize_result(out, strict_schema=True)


if __name__ == "__main__":
    unittest.main()
