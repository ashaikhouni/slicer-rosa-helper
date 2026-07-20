"""``app.rosa_service.editor_payload`` — the trajectory editor's CT-volume cache.

Regression for the "Failed to load the case: byte length of Int16Array should be
a multiple of 2" error seen on a *recipient's* machine: a truncated/partial
``editor_ct.i16`` (an interrupted build, or a case copied between machines
mid-transfer) must be detected by ``ensure_cache`` and rebuilt — never served —
otherwise the browser's ``new Int16Array(oddBuffer)`` throws.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "CommonLib"))


def _deps() -> bool:
    try:
        import numpy  # noqa: F401
        import nibabel  # noqa: F401
        import scipy  # noqa: F401
        from rosa_service import editor_payload  # noqa: F401
        return True
    except Exception:
        return False


DEPS = _deps()


def _make_job(root: Path) -> Path:
    import numpy as np
    import nibabel as nib
    job = root / "job"
    job.mkdir()
    arr = np.zeros((40, 40, 40), np.int16); arr[5:35, 5:35, 5:35] = 500
    ct = job / "ct.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(ct))            # RAS == voxel
    (job / "manifest.json").write_text(json.dumps({"params": {"ct": str(ct)}}))
    (job / "trajectories.tsv").write_text(
        "name\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\telectrode_model\n"
        "A\t10\t10\t10\t10\t10\t30\telectrode\n")
    rows = "trajectory\tx\ty\tz\telectrode_model\n" + "".join(
        f"A\t10\t10\t{z}\telectrode\n" for z in (12, 16, 20, 24, 28))
    (job / "contacts.tsv").write_text(rows)
    return job


@unittest.skipUnless(DEPS, "numpy/nibabel/scipy/rosa_service not importable.")
class EditorCacheTests(unittest.TestCase):
    def _expected_size(self, job: Path) -> int:
        dims = json.loads((job / "editor_plan.json").read_text())["dims"]
        return dims[0] * dims[1] * dims[2] * 2               # int16 → 2 bytes/voxel

    def test_build_is_even_and_matches_dims(self):
        from rosa_service.editor_payload import ensure_cache
        job = _make_job(Path(tempfile.mkdtemp()))
        ensure_cache(job)
        size = (job / "editor_ct.i16").stat().st_size
        self.assertEqual(size, self._expected_size(job))
        self.assertEqual(size % 2, 0)

    def test_truncated_i16_is_rebuilt(self):
        from rosa_service.editor_payload import ensure_cache
        job = _make_job(Path(tempfile.mkdtemp()))
        ensure_cache(job)
        vol = job / "editor_ct.i16"
        expected = self._expected_size(job)

        # simulate a partial write / mid-transfer truncation (odd byte length),
        # while the plan stays "fresh" (newer than trajectories.tsv)
        vol.write_bytes(vol.read_bytes()[:-1])
        self.assertEqual(vol.stat().st_size, expected - 1)

        ensure_cache(job)                                    # must detect + rebuild
        self.assertEqual(vol.stat().st_size, expected)
        self.assertEqual(vol.stat().st_size % 2, 0)


@unittest.skipUnless(DEPS, "numpy/nibabel/scipy/rosa_service not importable.")
class ProbePatchTests(unittest.TestCase):
    def test_native_probe_samples_correct_ras_point(self):
        """probe_patch centered on a known bright RAS voxel must read that value
        at the patch centre (locks the RAS→native-voxel mapping)."""
        import numpy as np
        import nibabel as nib
        from rosa_service.editor_payload import probe_patch
        root = Path(tempfile.mkdtemp())
        arr = np.zeros((40, 40, 40), np.int16)
        arr[20, 21, 22] = 3000                               # bright voxel at index (20,21,22)
        ct = root / "ct.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(ct))   # identity → RAS == index
        size = 64
        buf = np.frombuffer(
            probe_patch(ct, [20.0, 21.0, 22.0], [1, 0, 0], [0, 1, 0], 8.0, size),
            dtype="<i2").reshape(size, size)
        self.assertEqual(int(buf[size // 2, size // 2]), 3000)   # centre hits the bright voxel
        self.assertEqual(int(buf[0, 0]), 0)                       # corners are background


if __name__ == "__main__":
    unittest.main()
