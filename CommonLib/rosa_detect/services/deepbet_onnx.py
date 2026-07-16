"""deepbet brain extraction, torch-free — ONNX weights run via onnxruntime.

deepbet's two small CNNs (a 128³ bbox locator + a 256³ refiner) are exported to
ONNX; this module replicates deepbet's exact numpy glue (canonical reorient →
nearest-exact resample → percentile normalize → bbox → largest-CC + fill-holes →
reorient back) and runs the nets with ``onnxruntime`` instead of torch.

Verified **bit-identical** to the torch ``deepbet.run_bet`` (Dice 1.00000, exact
voxel count) and faster (~1.7 s CPU). Because ``onnxruntime`` is torch-free and
bundleable, this gives the frozen app a fast + accurate T1 skull-strip with **no
BYO torch and no FreeSurfer** — so the MRI-derived mask route (feeding
``place --brain-mask``) works out of the box instead of falling back to the slow
SynthStrip-on-CT path.

**T1 only** (same as torch deepbet): strip a preop MRI, then register into CT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

# deepbet's fixed model input grids.
_SMALL = (128, 128, 128)   # bbox_model
_FULL = (256, 256, 256)    # model (within the bbox)


def default_model_dir() -> Path:
    """Where the bundled deepbet ONNX weights live (collected in the freeze)."""
    import rosa_core
    return Path(rosa_core.__file__).resolve().parent / "resources" / "deepbet"


def deepbet_onnx_available(model_dir: str | Path | None = None) -> bool:
    """True when onnxruntime is importable AND both ONNX weights are present."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    d = Path(model_dir) if model_dir else default_model_dir()
    return (d / "bbox_model.onnx").is_file() and (d / "model.onnx").is_file()


def _nearest_exact(x, out_shape):
    """Replicate ``torch.nn.functional.interpolate(mode='nearest-exact')``."""
    import numpy as np
    idx = []
    for o, i in zip(out_shape, x.shape):
        src = np.floor((np.arange(o) + 0.5) * (i / o)).astype(np.int64)
        idx.append(np.clip(src, 0, i - 1))
    return x[np.ix_(*idx)]


def _normalize(x, low, high):
    """deepbet.utils.normalize — percentile scale, standardize (unbiased std), shift."""
    import numpy as np
    x = (x - low) / (high - low)
    x = np.clip(x, 0.0, 1.0)
    x = (x - x.mean()) / x.std(ddof=1)          # torch.std is unbiased (ddof=1)
    return (0.226 * x + 0.449).astype(np.float32)


def _largest_cc(mask):
    """Keep the largest 26-connected component (matches deepbet's cc3d default)."""
    import numpy as np
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == int(counts.argmax())


def _bbox_with_margin(mask_small, full_shape, margin=0.1):
    """deepbet.BrainExtraction.get_bbox_with_margin, in numpy."""
    import numpy as np
    rs = [np.where(mask_small.mean(axis=d) > 0.02)[0] for d in [(1, 2), (0, 2), (0, 1)]]
    if any(r.size == 0 for r in rs):
        raise ValueError("deepbet: not enough foreground to locate a brain bbox")
    center = np.array([(r.max() + 1 + r.min()) / 2 for r in rs])
    size = np.array([(r.max() + 1) - r.min() for r in rs], dtype=float)
    scale = np.array(full_shape) / np.array(mask_small.shape)
    center, size = scale * center, (1 + 2 * margin) * (scale * size)
    center, size = np.round(center), np.round(size)
    out = []
    for c, s, fs in zip(center, size, full_shape):
        out.append(slice(max(0, int(c - s / 2)), min(int(fs), int(c + s / 2)), 1))
    return tuple(out)


def run_deepbet_onnx(
    input_path: str | Path,
    mask_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    threshold: float = 0.5,
    log: Callable[[str], None] = lambda _m: None,
) -> Path:
    """Skull-strip a **T1 MRI** with the ONNX deepbet; write the mask, return it.

    Faithful to ``deepbet.run_bet`` (verified Dice 1.0). No torch — runs the two
    CNNs via ``onnxruntime`` (CPU / CoreML).
    """
    import numpy as np
    import nibabel as nib
    import onnxruntime as ort
    from scipy import ndimage as ndi

    input_path = Path(input_path).expanduser().resolve()
    mask_path = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    d = Path(model_dir) if model_dir else default_model_dir()

    # CPU on purpose: CoreML fragments this 3D-UNet (only ~40% of nodes
    # supported → many partitions), which is SLOWER than plain CPU (~1.7 s) and
    # non-deterministic. CPU matches torch deepbet bit-for-bit.
    sess_opts = ort.SessionOptions()
    bbox_sess = ort.InferenceSession(str(d / "bbox_model.onnx"), sess_opts,
                                     providers=["CPUExecutionProvider"])
    main_sess = ort.InferenceSession(str(d / "model.onnx"), sess_opts,
                                     providers=["CPUExecutionProvider"])

    img = nib.load(str(input_path))
    x = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
    if x.ndim == 4:
        x = x[..., 0]
    x = np.nan_to_num(x)
    full = np.zeros_like(x)

    log(f"[deepbet-onnx] stripping {input_path.name} …")
    # coarse bounding-box net @128³
    x_small = _nearest_exact(x, _SMALL)
    low, high = np.quantile(x_small, 0.005), np.quantile(x_small, 0.995)
    ms = bbox_sess.run(None, {"x": _normalize(x_small, low, high)[None, None]})[0][0, 1]
    ms = _largest_cc(ms > 0.5)
    bbox = _bbox_with_margin(ms, x.shape)
    # fine refiner @256³ within the bbox
    xb = _nearest_exact(x[bbox], _FULL)
    mb = main_sess.run(None, {"x": _normalize(xb, low, high)[None, None]})[0][0, 1]
    full[bbox] = _nearest_exact(mb, x[bbox].shape)
    mask = full > threshold
    sub = ndi.binary_fill_holes(_largest_cc(mask[bbox]))
    mask[bbox] = sub

    # reorient the canonical mask back to the input's native orientation + save
    # with the input's geometry (deepbet.utils.reoriented_nifti).
    ornt_ras = [[0, 1], [1, 1], [2, 1]]
    ornt_inv = nib.orientations.ornt_transform(ornt_ras, nib.io_orientation(img.affine))
    out = nib.apply_orientation(mask.astype(np.uint8), ornt_inv)
    out_img = nib.Nifti1Image(out, img.affine, img.header)
    out_img.header.set_data_dtype(np.uint8)
    out_img.to_filename(str(mask_path))

    # match the sform/qform exactly to the input (shared with the other backends).
    from .synthstrip import fix_mask_geometry
    fix_mask_geometry(input_path, mask_path)
    log(f"[deepbet-onnx] wrote mask ({int(mask.sum())} voxels)")
    return mask_path


__all__ = ["run_deepbet_onnx", "deepbet_onnx_available", "default_model_dir"]
