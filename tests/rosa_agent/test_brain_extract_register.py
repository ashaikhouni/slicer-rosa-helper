"""Tests for brain-extract's optional register-to-target / mask-in-target path.

The registration + resample are mocked (no real SITK pass), so these are fast and
deterministic — they lock the flag gating and the call wiring, not the numerics.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "cli", REPO / "CommonLib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from rosa_agent.commands import brain_extract as be
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


class _Args:
    """Minimal stand-in for the argparse Namespace fields _register_and_emit reads."""
    def __init__(self, **kw):
        self.register_to = None
        self.save_transform = None
        self.mask_in_target = None
        self.__dict__.update(kw)


@unittest.skipUnless(HAVE, "rosa_agent unavailable")
class RegisterEmitTests(unittest.TestCase):
    def _log(self, _m):  # noqa: D401
        pass

    def test_noop_when_no_flags(self):
        rc = be._register_and_emit(_Args(), Path("in.nii.gz"), Path("m.nii.gz"), self._log)
        self.assertEqual(rc, 0)

    def test_target_frame_outputs_require_register_to(self):
        for kw in ({"save_transform": "/x/t.tfm"}, {"mask_in_target": "/x/o.nii.gz"}):
            rc = be._register_and_emit(_Args(**kw), Path("in.nii.gz"),
                                       Path("m.nii.gz"), self._log)
            self.assertEqual(rc, 2, kw)

    def test_missing_target_volume_errors(self):
        rc = be._register_and_emit(
            _Args(register_to="/nope/ct.nii.gz", mask_in_target="/x/o.nii.gz"),
            Path("in.nii.gz"), Path("m.nii.gz"), self._log)
        self.assertEqual(rc, 2)

    def test_registers_and_writes_transform_and_mask(self):
        import rosa_core.registration as reg_mod
        fake = mock.Mock()
        fake.transform = "TF"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ct = td / "ct.nii.gz"; ct.write_bytes(b"stub")   # exists → passes is_file()
            mask = td / "mask.nii.gz"; mask.write_bytes(b"stub")
            tfm = td / "t1_to_ct.tfm"
            out = td / "mask_in_ct.nii.gz"
            args = _Args(register_to=str(ct), save_transform=str(tfm),
                         mask_in_target=str(out))
            with mock.patch.object(reg_mod, "register_rigid_mi", return_value=fake) as m_reg, \
                 mock.patch.object(reg_mod, "save_transform") as m_save, \
                 mock.patch.object(reg_mod, "resample_volume", return_value="WARPED") as m_res, \
                 mock.patch("SimpleITK.ReadImage", return_value="IMG"), \
                 mock.patch("SimpleITK.WriteImage") as m_write:
                rc = be._register_and_emit(args, td / "t1.nii.gz", mask, self._log)
        self.assertEqual(rc, 0)
        # fixed = target (CT), moving = input (T1)
        _, kwargs = m_reg.call_args
        self.assertEqual(m_reg.call_count, 1)
        m_save.assert_called_once()                       # transform persisted
        m_res.assert_called_once()                        # mask resampled to target grid
        self.assertEqual(m_res.call_args.kwargs.get("interp"), "nearest")
        m_write.assert_called_once()                      # mask-in-target written

    def test_registration_failure_returns_nonzero(self):
        import rosa_core.registration as reg_mod
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ct = td / "ct.nii.gz"; ct.write_bytes(b"stub")
            args = _Args(register_to=str(ct), mask_in_target=str(td / "o.nii.gz"))
            with mock.patch("SimpleITK.ReadImage", return_value="IMG"), \
                 mock.patch.object(reg_mod, "register_rigid_mi",
                                   side_effect=RuntimeError("boom")):
                rc = be._register_and_emit(args, td / "t1.nii.gz", td / "m.nii.gz", self._log)
        self.assertEqual(rc, 4)


if __name__ == "__main__":
    unittest.main()
