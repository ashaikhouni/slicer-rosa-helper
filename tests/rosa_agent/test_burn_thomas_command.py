"""``rosa-agent burn-thomas`` — burn a THOMAS thalamic structure into a DICOM
series' pixel data and export it as a new DICOM series.

The headless/CLI equivalent of the Slicer NavigationBurn module. These tests
build a synthetic DICOM + a synthetic THOMAS dir sharing one frame, run the
command with ``--no-register`` (deterministic — registration itself is covered
by the import-thomas tests), and verify the round-trip: the right voxels get the
fill, the source patient/study identity is carried onto a *new* series, and the
geometry is preserved.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT / "CommonLib"))


def _try_imports() -> bool:
    try:
        import numpy  # noqa: F401
        import SimpleITK  # noqa: F401
        from rosa_agent.commands import burn_thomas  # noqa: F401
        from rosa_core import dicom_burn  # noqa: F401
        return True
    except Exception:
        return False


DEPS_AVAILABLE = _try_imports()

# Synthetic geometry: (z, y, x). Left/right VA blobs sit in disjoint x-ranges so
# both a left label (4) and a right label (104) survive into the combined map.
Z, Y, X = 8, 16, 16
_LEFT_BLOB = (slice(3, 5), slice(6, 10), slice(3, 6))
_RIGHT_BLOB = (slice(3, 5), slice(6, 10), slice(10, 13))


def _ref_image(arr):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 2.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _write_input_dicom(image, out_dir: Path, patient_name="TEST^PATIENT", patient_id="RID001"):
    """Write a minimal but valid multi-slice DICOM series carrying patient/study
    identity, and return ``(study_uid, series_uid)``."""
    import SimpleITK as sitk
    out_dir.mkdir(parents=True, exist_ok=True)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    study = f"2.25.{uuid.uuid4().int}"
    series = f"2.25.{uuid.uuid4().int}"
    d = image.GetDirection()
    orient = "\\".join(str(v) for v in (d[0], d[3], d[6], d[1], d[4], d[7]))
    for i in range(image.GetDepth()):
        sl = image[:, :, i]
        sl.SetMetaData("0010|0010", patient_name)
        sl.SetMetaData("0010|0020", patient_id)
        sl.SetMetaData("0020|000d", study)
        sl.SetMetaData("0020|000e", series)
        sl.SetMetaData("0008|0018", f"2.25.{uuid.uuid4().int}")
        sl.SetMetaData("0008|0060", "CT")
        sl.SetMetaData("0028|1052", "0")
        sl.SetMetaData("0028|1053", "1")
        sl.SetMetaData("0020|0037", orient)
        sl.SetMetaData("0020|0032", "\\".join(
            str(v) for v in image.TransformIndexToPhysicalPoint((0, 0, i))))
        sl.SetMetaData("0020|0013", str(i + 1))
        w.SetFileName(str(out_dir / f"{i + 1:04d}.dcm"))
        w.Execute(sl)
    return study, series


def _make_case(root: Path):
    """Build a synthetic DICOM + THOMAS dir sharing one frame. Returns a dict of
    paths + reference counts."""
    import numpy as np
    import SimpleITK as sitk

    ct = np.full((Z, Y, X), -1000, np.int16)
    ct[:, 0, :] = 800                        # a bright band, so the CT isn't uniform
    ct_img = _ref_image(ct)
    dcm_in = root / "dcm_in"
    study_uid, in_series = _write_input_dicom(ct_img, dcm_in)

    thomas = root / "THOMAS"
    (thomas / "left").mkdir(parents=True)
    (thomas / "right").mkdir(parents=True)
    labL = np.zeros((Z, Y, X), np.int16); labL[_LEFT_BLOB] = 4      # left VA
    labR = np.zeros((Z, Y, X), np.int16); labR[_RIGHT_BLOB] = 4     # right VA
    sitk.WriteImage(_ref_image(labL), str(thomas / "left" / "thomasfull_L.nii.gz"))
    sitk.WriteImage(_ref_image(labR), str(thomas / "right" / "thomasfull_R.nii.gz"))
    sitk.WriteImage(_ref_image(np.zeros((Z, Y, X), np.float32)), str(thomas / "T1.nii.gz"))

    return {
        "dcm_in": dcm_in, "thomas": thomas, "ct_img": ct_img,
        "study_uid": study_uid, "in_series": in_series,
        "n_left": int((labL == 4).sum()), "n_right": int((labR == 4).sum()),
        "n_band": int((ct == 800).sum()), "n_bg": int((ct == -1000).sum()),
    }


@unittest.skipUnless(DEPS_AVAILABLE, "numpy/SimpleITK/rosa_agent not importable.")
class LabelResolutionTests(unittest.TestCase):
    def test_named_nucleus_both_sides(self):
        from rosa_agent.commands.burn_thomas import _resolve_labels
        labels, names = _resolve_labels(["VA"], "both", False)
        self.assertEqual(labels, {4, 104})
        self.assertEqual(names, ["VA"])

    def test_number_and_side(self):
        from rosa_agent.commands.burn_thomas import _resolve_labels
        labels, _ = _resolve_labels(["4"], "left", False)
        self.assertEqual(labels, {4})

    def test_all_covers_every_nucleus(self):
        from rosa_agent.commands.burn_thomas import _resolve_labels
        from rosa_core.thomas_import import THOMAS_NUCLEI
        labels, _ = _resolve_labels([], "right", True)
        self.assertEqual(labels, {n + 100 for n in THOMAS_NUCLEI})

    def test_unknown_nucleus_raises(self):
        from rosa_agent.commands.burn_thomas import _resolve_labels
        with self.assertRaises(ValueError):
            _resolve_labels(["NOPE"], "both", False)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy/SimpleITK/rosa_agent not importable.")
class BurnEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.case = _make_case(self.tmp)

    def _run(self, extra):
        from rosa_agent.commands.burn_thomas import main
        out = self.tmp / "dcm_out"
        rc = main([str(self.case["dcm_in"]), str(self.case["thomas"]),
                   "--out-dir", str(out), "--no-register", *extra])
        self.assertEqual(rc, 0)
        return out

    def test_burns_structure_and_preserves_identity(self):
        import SimpleITK as sitk
        from rosa_core import dicom_burn
        out = self._run(["--nucleus", "VA", "--side", "both", "--fill", "1200"])

        burned, reader = dicom_burn.read_series(out)
        arr = sitk.GetArrayFromImage(burned)
        expected = self.case["n_left"] + self.case["n_right"]
        # exactly the structure voxels became the fill; the band is untouched and
        # background dropped by exactly the burned count.
        self.assertEqual(int((arr == 1200).sum()), expected)
        self.assertEqual(int((arr == 800).sum()), self.case["n_band"])
        self.assertEqual(int((arr == -1000).sum()), self.case["n_bg"] - expected)

        # patient + study identity carried forward; series identity is new.
        self.assertEqual(reader.GetMetaData(0, "0010|0010").strip(), "TEST^PATIENT")
        self.assertEqual(reader.GetMetaData(0, "0010|0020").strip(), "RID001")
        self.assertEqual(reader.GetMetaData(0, "0020|000d").strip(), self.case["study_uid"])
        self.assertNotEqual(reader.GetMetaData(0, "0020|000e").strip(), self.case["in_series"])
        self.assertEqual(reader.GetMetaData(0, "0008|103e").strip(), "THOMAS_BURNED")

        # geometry preserved
        self.assertEqual(burned.GetSpacing(), self.case["ct_img"].GetSpacing())
        self.assertEqual(burned.GetSize(), self.case["ct_img"].GetSize())

    def test_side_left_only_burns_left(self):
        import SimpleITK as sitk
        from rosa_core import dicom_burn
        out = self._run(["--nucleus", "VA", "--side", "left", "--fill", "1500"])
        arr = sitk.GetArrayFromImage(dicom_burn.read_series(out)[0])
        self.assertEqual(int((arr == 1500).sum()), self.case["n_left"])

    def test_custom_series_description(self):
        from rosa_core import dicom_burn
        out = self._run(["--all", "--series-description", "THAL_NAV"])
        _, reader = dicom_burn.read_series(out)
        self.assertEqual(reader.GetMetaData(0, "0008|103e").strip(), "THAL_NAV")

    def test_distinct_intensities_and_legend(self):
        """--distinct gives each nucleus its own fill (fill, fill+step) and writes
        a legend, so multiple structures stay separable in the grayscale series."""
        import numpy as np
        import SimpleITK as sitk
        from rosa_core import dicom_burn
        # add a second nucleus (CM = 11), disjoint from the VA blobs, to both maps
        for hemi in ("left", "right"):
            p = self.case["thomas"] / hemi / f"thomasfull_{hemi[0].upper()}.nii.gz"
            im = sitk.ReadImage(str(p)); a = sitk.GetArrayFromImage(im)
            a[5:7, 2:5, 6:9] = 11
            im2 = sitk.GetImageFromArray(a); im2.CopyInformation(im)
            sitk.WriteImage(im2, str(p))
        out = self._run(["--nucleus", "VA", "--nucleus", "CM", "--side", "both",
                         "--fill", "1000", "--distinct", "--distinct-step", "500"])
        arr = sitk.GetArrayFromImage(dicom_burn.read_series(out)[0])
        self.assertGreater(int((arr == 1000).sum()), 0)   # VA at the base fill
        self.assertGreater(int((arr == 1500).sum()), 0)   # CM at fill + step
        legend = (out / "burn_legend.tsv").read_text()
        self.assertIn("VA\tboth\t1000", legend)
        self.assertIn("CM\tboth\t1500", legend)


@unittest.skipUnless(DEPS_AVAILABLE, "numpy/SimpleITK/rosa_agent not importable.")
class ArgValidationTests(unittest.TestCase):
    def test_requires_nucleus_or_all(self):
        from rosa_agent.commands.burn_thomas import main
        tmp = Path(tempfile.mkdtemp())
        case = _make_case(tmp)
        rc = main([str(case["dcm_in"]), str(case["thomas"]),
                   "--out-dir", str(tmp / "out"), "--no-register"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
