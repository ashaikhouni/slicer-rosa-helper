"""build_command tests for the MRI (T1) intake at case creation.

Exercises the argv mapping directly (no engine / no SITK): the pipeline job
gains an optional ``t1`` that brings the brain surface + mask in from the start,
and the label job's surface flags must stay byte-for-byte after sharing the
``_brain_surface_view_flags`` helper with the pipeline.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys_path = str(Path(__file__).resolve().parents[1])
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from rosa_service import jobs as J
    from rosa_service.models import JobSpec
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

try:
    import httpx
    from httpx import ASGITransport
    from rosa_service.app import create_app
    HAVE_HTTPX = True
except Exception:  # noqa: BLE001
    HAVE_HTTPX = False

API = "/api/v1"


def _sub(seq, sub) -> bool:
    """Is ``sub`` a contiguous subsequence of ``seq``?"""
    n, m = len(seq), len(sub)
    return any(seq[i:i + m] == sub for i in range(n - m + 1))


@unittest.skipUnless(HAVE, "rosa_service (app deps) unavailable")
class PipelineMriBuildTests(unittest.TestCase):
    def setUp(self):
        self.wd = Path("/tmp/case_xyz")   # not touched on disk — argv only
        self.regcache = self.wd / "regcache"
        brainchop = mock.patch.object(J, "_brainchop_available", return_value=False)
        brainchop.start()
        self.addCleanup(brainchop.stop)

    # ---- pipeline WITHOUT an MRI: unchanged (CT MIP-only) ----------------

    def test_pipeline_ct_only_has_no_brain_flags(self):
        spec = JobSpec(kind="pipeline", params={"ct": "/d/ct.nii.gz", "label": "case"})
        with mock.patch.object(J, "_deepbet_available", return_value=True), \
             mock.patch.object(J, "_fastsurfer_available", return_value=True):
            steps = J.build_command(spec, self.wd)
        self.assertEqual(len(steps), 3)                      # detect, contacts, view-results
        flat = [tok for s in steps for tok in s]
        self.assertNotIn("--brain-native-volume", flat)
        self.assertNotIn("brain-extract", flat)
        # CT-only localization path is untouched: no MRI mask, no registration.
        self.assertNotIn("--brain-mask", flat)
        self.assertNotIn("--register-to", flat)
        self.assertNotIn("--brain-to-ct-transform", flat)

    # ---- pipeline WITH an MRI: deepbet mask + surface flags --------------

    def test_pipeline_with_mri_extracts_registers_and_feeds_localization(self):
        spec = JobSpec(kind="pipeline",
                       params={"ct": "/d/ct.nii.gz", "label": "case", "t1": "/d/t1.nii.gz"})
        with mock.patch.object(J, "_deepbet_available", return_value=True), \
             mock.patch.object(J, "_fastsurfer_available", return_value=False), \
             mock.patch.object(J, "_deepmriprep_available", return_value=False):
            steps = J.build_command(spec, self.wd)
        # brain-extract (deepbet + register) → detect → contacts → view-results
        self.assertEqual(len(steps), 5)
        self.assertIn("stamp-mni", steps[-1])
        native = str(self.regcache / "brain_mask_native.nii.gz")
        in_ct = str(self.regcache / "brain_mask_in_ct.nii.gz")
        tfm = str(self.regcache / "t1_to_ct.tfm")
        # 1) brain-extract makes the native mask AND registers it into the CT frame.
        be = steps[0]
        self.assertIn("brain-extract", be)
        self.assertTrue(_sub(be, ["--backend", "deepbet"]))
        self.assertTrue(_sub(be, ["-o", native]))
        self.assertTrue(_sub(be, ["--register-to", "/d/ct.nii.gz"]))
        self.assertTrue(_sub(be, ["--save-transform", tfm]))
        self.assertTrue(_sub(be, ["--mask-in-target", in_ct]))
        # 2) the CONTACTS step consumes the CT-frame MRI mask as the placement anchor.
        contacts = next(s for s in steps if "contacts" in s and "view-results" not in s)
        self.assertTrue(_sub(contacts, ["--brain-mask", in_ct]))
        # 3) view-results meshes the surface from the NATIVE mask and REUSES the
        #    saved transform (no redundant registration).
        view = next(s for s in steps if "view-results" in s)
        self.assertIn("view-results", view)
        self.assertTrue(_sub(view, ["--brain-native-volume", "/d/t1.nii.gz"]))
        self.assertTrue(_sub(view, ["--brain-mask-cache", native]))
        self.assertTrue(_sub(view, ["--brain-surface-cache",
                                    str(self.regcache / "brain_surface.npz")]))
        self.assertTrue(_sub(view, ["--brain-to-ct-transform", tfm]))

    def test_pipeline_with_mri_skips_deepbet_when_unavailable(self):
        # No deepbet → no pre-extract step; view-results still gets surface flags
        # (it extracts the mask via its own auto/SynthStrip backend).
        spec = JobSpec(kind="pipeline",
                       params={"ct": "/d/ct.nii.gz", "t1": "/d/t1.nii.gz"})
        with mock.patch.object(J, "_deepbet_available", return_value=False), \
             mock.patch.object(J, "_fastsurfer_available", return_value=False):
            steps = J.build_command(spec, self.wd)
        self.assertEqual(len(steps), 3)                      # no brain-extract step
        flat = [tok for s in steps for tok in s]
        self.assertNotIn("brain-extract", flat)
        self.assertTrue(_sub(steps[-1], ["--brain-native-volume", "/d/t1.nii.gz"]))
        # No deepbet mask → no CT-frame mask to feed localization, and view-results
        # registers on its own (no reused transform).
        self.assertNotIn("--brain-mask", flat)
        self.assertNotIn("--brain-to-ct-transform", flat)

    def test_pipeline_with_mri_fastsurfer_surface_name_and_flag(self):
        spec = JobSpec(kind="pipeline",
                       params={"ct": "/d/ct.nii.gz", "t1": "/d/t1.nii.gz"})
        with mock.patch.object(J, "_deepbet_available", return_value=False), \
             mock.patch.object(J, "_fastsurfer_available", return_value=True):
            steps = J.build_command(spec, self.wd)
        view = next(s for s in steps if "view-results" in s)
        # FastSurfer available → the FS surface cache name + a build flag.
        self.assertTrue(_sub(view, ["--brain-surface-cache",
                                    str(self.regcache / "brain_surface_fs.npz")]))
        self.assertIn("--fastsurfer", view)

    def test_pipeline_surface_source_deepmriprep(self):
        # Explicit surface=deepmriprep wins even when FastSurfer is available.
        spec = JobSpec(kind="pipeline", params={
            "ct": "/d/ct.nii.gz", "t1": "/d/t1.nii.gz", "surface": "deepmriprep"})
        with mock.patch.object(J, "_deepbet_available", return_value=False), \
             mock.patch.object(J, "_fastsurfer_available", return_value=True), \
             mock.patch.object(J, "_deepmriprep_available", return_value=True):
            steps = J.build_command(spec, self.wd)
        view = next(s for s in steps if "view-results" in s)
        self.assertTrue(_sub(view, ["--brain-surface-cache",
                                    str(self.regcache / "brain_surface_dm.npz")]))
        self.assertIn("--deepmriprep", view)
        self.assertNotIn("--fastsurfer", view)

    def test_pipeline_surface_source_falls_back_when_unavailable(self):
        # surface=deepmriprep but it isn't installed → degrade to auto → FastSurfer.
        spec = JobSpec(kind="pipeline", params={
            "ct": "/d/ct.nii.gz", "t1": "/d/t1.nii.gz", "surface": "deepmriprep"})
        with mock.patch.object(J, "_deepbet_available", return_value=False), \
             mock.patch.object(J, "_fastsurfer_available", return_value=True), \
             mock.patch.object(J, "_deepmriprep_available", return_value=False):
            steps = J.build_command(spec, self.wd)
        view = next(s for s in steps if "view-results" in s)
        self.assertTrue(_sub(view, ["--brain-surface-cache",
                                    str(self.regcache / "brain_surface_fs.npz")]))
        self.assertIn("--fastsurfer", view)
        self.assertNotIn("--deepmriprep", view)

    # ---- label job: surface flags unchanged after the refactor ----------

    def test_label_view_step_surface_flags_include_transform(self):
        contacts = "/cases/parent/contacts.tsv"
        spec = JobSpec(kind="label", params={
            "parent": "parent", "contacts": contacts, "ct": "/cases/parent/ct.nii.gz",
            "t1": "/d/t1.nii.gz", "atlas": "cerebra",
            "regcache": "/cases/parent/regcache"})
        with mock.patch.object(J, "_fastsurfer_available", return_value=False), \
             mock.patch.object(J, "_deepmriprep_available", return_value=False):
            steps = J.build_command(spec, self.wd)
        rc = Path("/cases/parent/regcache")
        view = next(s for s in steps if "view-results" in s)
        # Exact ordered surface block the label job produced before the refactor.
        expect = ["--brain-native-volume", "/d/t1.nii.gz",
                  "--brain-to-ct-transform", str(rc / "t1_to_ct.tfm"),
                  "--brain-mask-cache", str(rc / "brain_mask_native.nii.gz"),
                  "--brain-surface-cache", str(rc / "brain_surface.npz")]
        self.assertTrue(_sub(view, expect), f"surface flags drifted: {view}")
        self.assertTrue(_sub(view, ["--atlas-labelmap",
                                    str(self.wd / "atlas_in_ct.nii.gz"),
                                    "--atlas-name", "cerebra"]))


@unittest.skipUnless(HAVE_HTTPX, "fastapi/httpx (app [test] extra) unavailable")
class LabelMriFallbackTests(unittest.IsolatedAsyncioTestCase):
    """A case created with an MRI can be labeled without re-uploading it: the
    label endpoint falls back to the parent pipeline's ``t1``."""

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
        raise AssertionError(f"job {jid} never finished")

    async def _emit_parent(self, params) -> str:
        # selftest-emit writes a contacts.tsv (so the label endpoint's contacts
        # check passes) and records our ct/t1 params on the job.
        jid = (await self.client.post(
            f"{API}/jobs", json={"kind": "selftest-emit", "params": params})).json()["id"]
        await self._wait_terminal(jid)
        return jid

    async def test_label_uses_case_creation_mri_when_omitted(self):
        parent = await self._emit_parent({"ct": "/x/ct.nii.gz", "t1": "/x/t1.nii.gz"})
        r = await self.client.post(f"{API}/jobs/{parent}/label", json={"atlas": "cerebra"})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["t1"], "/x/t1.nii.gz")   # fell back to parent's MRI

    async def test_label_requires_an_mri_somewhere(self):
        parent = await self._emit_parent({"ct": "/x/ct.nii.gz"})   # no t1 at creation
        r = await self.client.post(f"{API}/jobs/{parent}/label", json={"atlas": "cerebra"})
        self.assertEqual(r.status_code, 409)                       # and none in the request


if __name__ == "__main__":
    unittest.main()
