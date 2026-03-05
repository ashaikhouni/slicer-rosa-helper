import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from shank_engine import default_result, sanitize_result  # noqa: E402


EXPECTED_RESULT_KEYS = {
    "schema_version",
    "pipeline_id",
    "pipeline_version",
    "run_id",
    "status",
    "trajectories",
    "contacts",
    "diagnostics",
    "artifacts",
    "warnings",
    "error",
    "meta",
}

EXPECTED_DIAGNOSTICS_KEYS = {
    "schema_version",
    "pipeline_id",
    "run_id",
    "counts",
    "timing",
    "reason_codes",
    "params",
    "notes",
    "extras",
}


class SchemaSnapshotTests(unittest.TestCase):
    def test_result_schema_keys_snapshot(self):
        out = default_result(
            pipeline_id="snapshot",
            pipeline_version="1",
            run_id="snap-1",
            status="ok",
            params={},
        )
        clean = sanitize_result(out)
        self.assertEqual(set(clean.keys()), EXPECTED_RESULT_KEYS)
        self.assertEqual(set(clean["diagnostics"].keys()), EXPECTED_DIAGNOSTICS_KEYS)


if __name__ == "__main__":
    unittest.main()
