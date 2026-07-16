"""brainchop GM/WM tissue segmentation, torch-free — MeshNet run via onnxruntime.

brainchop's gray-white-matter MeshNet (a 9-layer dilated 3D CNN, ~23 k params)
was rebuilt from its published tf.js weights and exported to ONNX. This module
replicates brainchop's preprocessing — FastSurfer-style **conform** to 256³ @1 mm
in LIA orientation + robust intensity rescale + min-max — runs the net via
``onnxruntime`` (CPU, torch-free), and maps the tissue back to the native T1.

The output feeds ``gyral_surface_from_mri``'s ``brain_tissue`` support: a learned
GM+WM region → a crisp, dura-free brain surface with **no FreeSurfer/FastSurfer
and no torch**, a step up from the Otsu grayscale-iso fallback.

**T1 only.** Bundleable + fast (a few seconds CPU).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def default_model_dir() -> Path:
    import rosa_core
    return Path(rosa_core.__file__).resolve().parent / "resources" / "brainchop"


def brainchop_available(model_dir: str | Path | None = None) -> bool:
    """True when onnxruntime + the GM/WM ONNX weights are present."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    d = Path(model_dir) if model_dir else default_model_dir()
    return (d / "gmwm.onnx").is_file()


def _conform_intensity(data, f_low: float = 0.0, f_high: float = 0.999):
    """FastSurfer conform intensity: robustly rescale [min, 99.9-pct] → [0, 255]."""
    import numpy as np
    nbins = 1000
    hist, edges = np.histogram(data, bins=nbins)
    cs = np.cumsum(hist).astype(float)
    total = float(cs[-1]) or 1.0
    src_min = edges[0] if f_low <= 0 else edges[int(np.searchsorted(cs, total * f_low))]
    src_max = edges[min(int(np.searchsorted(cs, total * f_high)) + 1, nbins)]
    scale = 255.0 / max(float(src_max - src_min), 1e-6)
    return np.clip((data - src_min) * scale, 0.0, 255.0).astype(np.float32)


def brainchop_gmwm_support(
    mri_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    log: Callable[[str], None] = lambda _m: None,
):
    """Segment GM+WM with brainchop; return a native-T1 ``nib.Nifti1Image`` whose
    voxels are 1 inside the GM+WM tissue (the surface support), 0 elsewhere.
    """
    import numpy as np
    import nibabel as nib
    import nibabel.processing as nibp
    import onnxruntime as ort

    d = Path(model_dir) if model_dir else default_model_dir()
    img = nib.load(str(mri_path))
    # brainchop's prep: FastSurfer conform → 256³ @1 mm, LIA orientation.
    log("[brainchop] conform → 256³ @1mm (LIA) + segment GM/WM …")
    conf = nibp.conform(img, out_shape=(256, 256, 256), voxel_size=(1, 1, 1),
                        order=3, orientation="LIA")
    x = _conform_intensity(np.asanyarray(conf.dataobj).astype(np.float32))
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)          # min-max, as brainchop does

    sess = ort.InferenceSession(str(d / "gmwm.onnx"), providers=["CPUExecutionProvider"])
    lab = sess.run(None, {"x": x[None, None].astype(np.float32)})[0][0].argmax(0)
    support = (lab > 0).astype(np.uint8)                     # GM+WM = both non-background classes

    # back to the native T1 grid (nearest — it's a mask).
    supp_conf = nib.Nifti1Image(support, conf.affine, conf.header)
    native = nibp.resample_from_to(supp_conf, (img.shape[:3], img.affine), order=0)
    out = np.asanyarray(native.dataobj) > 0
    log(f"[brainchop] GM+WM support: {int(out.sum())} voxels")
    return nib.Nifti1Image(out.astype(np.uint8), img.affine, img.header)


__all__ = ["brainchop_gmwm_support", "brainchop_available", "default_model_dir"]
