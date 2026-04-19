import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))
sys.path.insert(0, str(REPO_ROOT / "PostopCTLocalization"))

from shank_engine import ComponentRegistry, PipelineRegistry, register_builtin_pipelines  # noqa: E402


class _DummyPipeline:
    pipeline_id = "dummy"
    pipeline_version = "0"

    def run(self, ctx):
        return {"status": "ok"}


class RegistryTests(unittest.TestCase):
    def test_builtin_registration(self):
        reg = PipelineRegistry()
        register_builtin_pipelines(reg)
        keys = reg.keys()
        self.assertIn("contact_pitch_v1", keys)

    def test_duplicate_registration_raises(self):
        reg = PipelineRegistry()
        reg.register_pipeline("x", lambda: _DummyPipeline())
        with self.assertRaises(Exception):
            reg.register_pipeline("x", lambda: _DummyPipeline(), overwrite=False)

    def test_create_pipeline_uses_factory(self):
        reg = PipelineRegistry()
        reg.register_pipeline("dummy", lambda: _DummyPipeline(), overwrite=True)
        pipe = reg.create_pipeline("dummy")
        self.assertEqual(pipe.pipeline_id, "dummy")

    def test_component_registry(self):
        comps = ComponentRegistry()
        comps.register_component("gating", object())
        self.assertIn("gating", comps.keys())
        self.assertIsNotNone(comps.get_component("gating"))


if __name__ == "__main__":
    unittest.main()
