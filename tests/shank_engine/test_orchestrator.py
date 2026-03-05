import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))

from shank_engine.diagnostics import StageExecutionError  # noqa: E402
from shank_engine.pipelines.base import BaseDetectionPipeline  # noqa: E402


class _FailingPipeline(BaseDetectionPipeline):
    pipeline_id = "failing"
    pipeline_version = "1"

    def run(self, ctx):
        result = self.make_result(ctx)
        d = self.diagnostics(result)
        try:
            self.run_stage(
                ctx=ctx,
                result=result,
                diagnostics=d,
                stage_name="boom",
                fn=lambda: (_ for _ in ()).throw(ValueError("bad stage")),
            )
        except StageExecutionError:
            pass
        return self.finalize(result, d, 0.0)


class _ComponentPipeline(BaseDetectionPipeline):
    pipeline_id = "components"
    pipeline_version = "1"
    default_components = {"x": "default"}

    def run(self, ctx):
        result = self.make_result(ctx)
        d = self.diagnostics(result)
        resolved = self.resolve_component(ctx, "x")
        d.set_extra("resolved", resolved)
        return self.finalize(result, d, 0.0)


class OrchestratorTests(unittest.TestCase):
    def test_stage_error_payload(self):
        out = _FailingPipeline().run({"run_id": "x", "config": {}})
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"]["stage"], "boom")
        self.assertIn("pipeline_error", out["diagnostics"]["reason_codes"])

    def test_component_override_precedence(self):
        out = _ComponentPipeline().run(
            {
                "run_id": "x2",
                "config": {},
                "components": {"x": "override"},
            }
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["diagnostics"]["extras"].get("resolved"), "override")


if __name__ == "__main__":
    unittest.main()
