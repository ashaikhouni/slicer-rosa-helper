"""Synthetic-fixture smoke test for ``rosa-agent export-view``.

Builds a tiny ROSA folder (one display + one planned trajectory) and a
minimal FreeSurfer recon-all subject (T1.mgz + lh.pial + rh.pial + an
aparc+aseg.mgz that's just a uniform label), then exercises the
end-to-end ``run_export_view`` to pin the output contract:

  * ``scene.glb`` exists and parses as a valid glTF 2.0 binary container.
  * ``index.html`` and ``scene_meta.json`` exist.
  * ``view_manifest.json`` records the surfaces that were loaded.

We don't gate on detected trajectory count or surface vertex count — the
toy phantom isn't realistic enough for the SEEG detector. We only pin
the IO contract.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports():
    try:
        import numpy  # noqa: F401
        import nibabel  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_agent.commands import export_view  # noqa: F401
        return True
    except ImportError:
        return False


DEPS_AVAILABLE = _try_imports()


def _write_synthetic_analyze(path: Path, *, size=(20, 20, 20)) -> None:
    import numpy as np
    import SimpleITK as sitk

    arr = np.zeros(size, dtype=np.float32)
    arr[5:9, 5:9, 5:9] = 100.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path.with_suffix(".img")))


def _build_synthetic_rosa_case(case_dir: Path) -> None:
    analyze_root = case_dir / "DICOM" / "uid_a"
    analyze_root.mkdir(parents=True, exist_ok=True)
    _write_synthetic_analyze(analyze_root / "ref_vol")
    ros_text = textwrap.dedent("""
        [TRdicomRdisplay]
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        [VOLUME]
        DICOM/uid_a/ref_vol
        [IMAGERY_NAME]
        ref_vol
        [SERIE_UID]
        uid_a
        [IMAGERY_3DREF]
        0
        [TRAJECTORY]
        traj1
        T1 1 0 0 -1.0 -2.0 3.0 0 -10.0 -20.0 30.0
        [END]
    """).strip()
    (case_dir / "case.ros").write_text(ros_text)


def _build_synthetic_fs_subject(fs_dir: Path) -> None:
    """Minimal recon-all skeleton with T1.mgz + small lh/rh.pial + aparc+aseg.mgz."""
    import numpy as np
    import nibabel as nib

    surf_dir = fs_dir / "surf"
    mri_dir = fs_dir / "mri"
    label_dir = fs_dir / "label"
    for d in (surf_dir, mri_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    # T1.mgz: same shape as the ROSA-folder phantom for fast registration.
    size = 20
    arr = np.zeros((size, size, size), dtype=np.uint8)
    arr[5:9, 5:9, 5:9] = 80
    affine = np.eye(4)
    mgh = nib.MGHImage(arr, affine)
    nib.save(mgh, str(mri_dir / "T1.mgz"))

    # aparc+aseg with one nonzero region.
    parc = np.zeros((size, size, size), dtype=np.int32)
    parc[5:9, 5:9, 5:9] = 17  # Left-Hippocampus per FS LUT
    parc_img = nib.MGHImage(parc.astype(np.int32), affine)
    nib.save(parc_img, str(mri_dir / "aparc+aseg.mgz"))

    # Tiny pial surfaces (4-vertex tetrahedra) — enough for the loader
    # to round-trip; not enough for anything meaningful in 3D.
    def _write_pial(path: Path, offset_x: float):
        verts = np.array([
            [offset_x + 0.0, 0.0, 0.0],
            [offset_x + 2.0, 0.0, 0.0],
            [offset_x + 1.0, 2.0, 0.0],
            [offset_x + 1.0, 1.0, 2.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ], dtype=np.int32)
        nib.freesurfer.io.write_geometry(str(path), verts, faces)

    _write_pial(surf_dir / "lh.pial", offset_x=-3.0)
    _write_pial(surf_dir / "rh.pial", offset_x=+3.0)


def _validate_glb(path: Path) -> dict:
    """Parse the GLB header + JSON chunk; return the glTF document."""
    with open(path, "rb") as f:
        magic, ver, total = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, f"bad GLB magic: {hex(magic)}"
        assert ver == 2
        assert total == path.stat().st_size
        json_len, json_type = struct.unpack("<II", f.read(8))
        assert json_type == 0x4E4F534A, f"first chunk must be JSON, got {hex(json_type)}"
        return json.loads(f.read(json_len).rstrip(b" ").decode("utf-8"))


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/nibabel/SimpleITK/rosa_agent not importable in this environment.",
)
class ExportViewSmokeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.case_dir = self.tmp / "rosa_case"
        self.case_dir.mkdir()
        _build_synthetic_rosa_case(self.case_dir)
        self.fs_dir = self.tmp / "fs_subject"
        _build_synthetic_fs_subject(self.fs_dir)
        self.out_dir = self.tmp / "view_out"

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_glb_and_html(self):
        from rosa_agent.commands.export_view import run_export_view

        try:
            summary = run_export_view(
                target=str(self.case_dir),
                freesurfer_dir=str(self.fs_dir),
                out_dir=self.out_dir,
                ref_volume="ref_vol",
                # No annotation = no vertex coloring; .annot isn't written
                # in this fixture and the loader treats absence as benign.
                annotation="",
            )
        except SystemExit:
            self.fail("export_view raised SystemExit before assembling the GLB")
        except Exception:
            # Detection may not find anything on the toy phantom or the
            # registration may fail to converge — that's OK. We still
            # need the GLB/HTML to either be written, or for the call
            # to fail before the assembly step (which is what would
            # happen if the pipeline call itself raised). In the latter
            # case, scene.glb won't exist and the assertion below will
            # surface the failure precisely.
            pass

        self.assertTrue(
            (self.out_dir / "scene.glb").exists(),
            "scene.glb must be written",
        )
        self.assertTrue((self.out_dir / "index.html").exists())
        self.assertTrue((self.out_dir / "scene_meta.json").exists())
        self.assertTrue((self.out_dir / "view_manifest.json").exists())

        gltf = _validate_glb(self.out_dir / "scene.glb")
        self.assertEqual(gltf["asset"]["version"], "2.0")
        # Surfaces should be present. Names use ``_`` separators so they
        # survive three.js's GLTFLoader.PropertyBinding.sanitizeNodeName
        # (which would strip ``/`` and ``.``).
        node_names = {n["name"] for n in gltf["nodes"]}
        self.assertTrue(any("lh_pial" in n for n in node_names),
                        f"expected lh_pial in scene nodes; got {node_names}")
        self.assertTrue(any("rh_pial" in n for n in node_names))

        # Scene metadata sidecar must list trajectories + contacts arrays
        # (lengths can be zero on the phantom, but the keys must exist).
        meta = json.loads((self.out_dir / "scene_meta.json").read_text())
        self.assertIn("trajectories", meta)
        self.assertIn("contacts", meta)
        # The volume list must include the CT (always) and the T1 (FS mode),
        # with T1 first so the legacy default is preserved.
        vol_ids = [v["id"] for v in meta.get("volumes", [])]
        self.assertIn("ct", vol_ids)
        self.assertIn("t1", vol_ids)
        self.assertEqual(vol_ids[0], "t1", "FS mode must default to T1 (back-compat)")
        self.assertEqual(meta["t1_volume"]["id"], "t1")

    def test_ct_only_no_freesurfer(self):
        """CT-only mode: omit --freesurfer-dir → still assembles a viewer with a
        windowed CT slice/MIP volume, no brain mesh, no atlas labels. This is
        the 'see results without Slicer/FreeSurfer' path."""
        from rosa_agent.commands.export_view import run_export_view

        out = self.tmp / "ct_only_out"
        try:
            run_export_view(
                target=str(self.case_dir),
                freesurfer_dir="",           # <-- no recon
                out_dir=out,
                ref_volume="ref_vol",
            )
        except SystemExit:
            self.fail("CT-only export_view raised SystemExit before assembly")
        except Exception:
            pass  # detection on the toy phantom may raise; we pin the IO contract

        self.assertTrue((out / "scene.glb").exists(), "scene.glb must be written")
        self.assertTrue((out / "index.html").exists())
        self.assertTrue((out / "ct_in_view.nii.gz").exists(),
                        "CT-only mode must export the windowed CT slice volume")

        meta = json.loads((out / "scene_meta.json").read_text())
        vol_ids = [v["id"] for v in meta.get("volumes", [])]
        self.assertEqual(vol_ids, ["ct"], "CT-only mode has exactly the CT volume")
        self.assertEqual(meta["t1_volume"]["id"], "ct",
                         "legacy t1_volume must fall back to the CT")
        self.assertEqual(meta["freesurfer_subject"], "")
        # No FreeSurfer surfaces in the GLB.
        gltf = _validate_glb(out / "scene.glb")
        node_names = {n["name"] for n in gltf.get("nodes", [])}
        self.assertFalse(any("pial" in n for n in node_names),
                         f"CT-only scene must have no surfaces; got {node_names}")

    def test_ct_slice_volume_windows_to_uint8(self):
        """_write_ct_slice_volume: HU window → uint8, metal saturates bright,
        and the returned meta carries id/label/affine the viewer needs."""
        import numpy as np
        import SimpleITK as sitk
        from rosa_agent.commands.export_view import _write_ct_slice_volume

        arr = np.full((10, 10, 10), -1000.0, dtype=np.float32)  # air
        arr[4:6, 4:6, 4:6] = 40.0      # brain
        arr[5, 5, 5] = 3000.0          # a metal contact
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        ct = self.tmp / "synthetic_ct.nii.gz"
        sitk.WriteImage(img, str(ct))

        out = self.tmp / "ct_win.nii.gz"
        m = _write_ct_slice_volume(ct, out, window=(-150.0, 1500.0))
        self.assertTrue(out.exists())
        self.assertEqual(m["id"], "ct")
        self.assertEqual(m["label"], "CT")
        self.assertEqual(m["dtype"], "uint8")
        self.assertEqual(len(m["vox_to_ras"]), 4)
        w = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
        self.assertEqual(w.dtype, np.uint8)
        self.assertEqual(int(w[5, 5, 5]), 255)   # metal saturates
        self.assertEqual(int(w[0, 0, 0]), 0)     # air floors


class WebViewerSyncTests(unittest.TestCase):
    """The GitHub Pages viewer (web/viewer/index.html) is generated from the
    same template as the served viewer (picker mode). Pin that it stays in sync
    so the hosted app never drifts from the engine, and that the two modes are
    correctly gated."""

    def _import(self):
        try:
            from rosa_agent.commands.export_view import render_viewer_html
            return render_viewer_html
        except ImportError:
            self.skipTest("rosa_agent.commands.export_view not importable")

    def test_modes_gated(self):
        render = self._import()
        served = render(title="S", mode="served")
        picker = render(title="P", mode="picker")
        self.assertIn('const VIEWER_MODE = "served"', served)
        self.assertIn('const VIEWER_MODE = "picker"', picker)
        # Served auto-loads; picker waits for a dropped file.
        self.assertIn('if (VIEWER_MODE !== "picker") loadGlb("scene.glb")', served)
        self.assertIn('fetch("scene_meta.json").then(r => r.json()).then(onMeta)', served)
        self.assertIn('id="dropzone"', picker)
        self.assertIn("function __initPicker", picker)
        # Rotatable CT MIP present (both modes share the engine).
        for h in (served, picker):
            self.assertIn("function _makeMipMesh", h)
            self.assertIn('data-control="mip"', h)
            self.assertIn("uniform sampler3D uVolume;", h)

    def test_committed_pages_viewer_in_sync(self):
        render = self._import()
        repo = Path(__file__).resolve().parents[2]
        committed = repo / "web" / "viewer" / "index.html"
        if not committed.exists():
            self.skipTest("web/viewer/index.html not generated")
        html = render(title="ROSA / SEEG viewer", mode="picker")
        # The committed file prepends a generated-note banner before <html>;
        # compare everything from <html> onward (exact).
        body = committed.read_text(encoding="utf-8")

        def _from_html(s):
            return s[s.index("<html"):]

        self.assertEqual(
            _from_html(body), _from_html(html),
            "web/viewer/index.html is stale — re-run `python tools/build_web_viewer.py`",
        )


@unittest.skipUnless(
    DEPS_AVAILABLE,
    "numpy/nibabel/SimpleITK/rosa_agent not importable in this environment.",
)
class ViewResultsTests(unittest.TestCase):
    """`view-results` renders ALREADY-COMPUTED results (no pipeline re-run), and
    reads fit-rosa's entry_/tip_/predicted_model schema as well as the standard
    start_/end_/electrode_model contract."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_ct(self, path: Path):
        import numpy as np
        import SimpleITK as sitk
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        arr[8, 8, 8] = 3000.0  # a bright "contact"
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(path))

    def test_schema_tolerant_trajectory_reader(self):
        from rosa_agent.commands.export_view import _read_pipeline_trajectories
        std = self.tmp / "std.tsv"
        std.write_text(
            "name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\telectrode_model\n"
            "A\t0\t0\t0\t0\t0\t10\tDIXI-8AM\n"
        )
        fitrosa = self.tmp / "fr.tsv"
        fitrosa.write_text(
            "name\tstatus\tpredicted_model\tentry_x\tentry_y\tentry_z\ttip_x\ttip_y\ttip_z\n"
            "B\tok\tDIXI-10AM\t1\t0\t0\t1\t0\t12\n"
        )
        a = _read_pipeline_trajectories(std)
        b = _read_pipeline_trajectories(fitrosa)
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["electrode_model"], "DIXI-8AM")
        self.assertEqual(len(b), 1, "fit-rosa entry_/tip_ schema must be read")
        self.assertEqual(b[0]["start"], (1.0, 0.0, 0.0))
        self.assertEqual(b[0]["end"], (1.0, 0.0, 12.0))
        self.assertEqual(b[0]["electrode_model"], "DIXI-10AM")  # from predicted_model

    def test_dir_scan_renders_without_pipeline(self):
        from rosa_agent.commands.view_results import main as vr_main
        rd = self.tmp / "fitrosa_qc"
        (rd / "work").mkdir(parents=True)
        self._write_ct(rd / "work" / "postop_ct.nii.gz")
        (rd / "trajectories.tsv").write_text(
            "name\tstatus\tpredicted_model\tentry_x\tentry_y\tentry_z\ttip_x\ttip_y\ttip_z\n"
            "LAM\tok\tDIXI-8AM\t0\t0\t0\t0\t0\t12\n"
            "RHH\tok\tDIXI-10AM\t5\t0\t0\t5\t0\t12\n"
        )
        (rd / "contacts.tsv").write_text(
            "# reference_frame: ROSA::ct sha256:abc\n"
            "trajectory\tlabel\tcontact_index\tx\ty\tz\tpeak_detected\telectrode_model\n"
            "LAM\tLAM1\t1\t0\t0\t2\t1\tDIXI-8AM\n"
            "RHH\tRHH1\t1\t5\t0\t2\t1\tDIXI-10AM\n"
        )
        out = self.tmp / "view"
        rc = vr_main([str(rd), "-o", str(out)])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "scene.glb").exists())
        self.assertTrue((out / "ct_in_view.nii.gz").exists())
        self.assertTrue((out / "index.html").exists())
        meta = json.loads((out / "scene_meta.json").read_text())
        self.assertEqual(len(meta["trajectories"]), 2)
        self.assertEqual(len(meta["contacts"]), 2)
        self.assertEqual([v["id"] for v in meta["volumes"]], ["ct"])

    def test_missing_inputs_exit_2(self):
        from rosa_agent.commands.view_results import main as vr_main
        # Empty dir → no contacts → exit 2 (no traceback).
        empty = self.tmp / "empty"
        empty.mkdir()
        rc = vr_main([str(empty), "-o", str(self.tmp / "o1")])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
