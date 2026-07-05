"""Registration QC — render a CT↔MRI overlay slice as a PNG.

After the labeling step registers the patient MRI to the CT (and warps an atlas
in), the clinician needs to *see* whether the MRI landed on the CT correctly
before trusting the labels. This renders a single orthogonal slice with the two
volumes overlaid so misalignment is obvious:

* ``checker`` — checkerboard of CT / MRI tiles; a good registration keeps edges
  continuous across tile boundaries, a bad one shows them stepping.
* ``blend``   — CT in magenta, MRI in green; aligned structures read gray,
  misaligned ones fringe magenta/green.
* ``ct`` / ``mri`` — either modality alone, to compare.

Pure numpy + nibabel (engine deps) + a tiny zlib PNG encoder, so it adds no
imaging dependency to the app. Both volumes are assumed to share a grid (the
MRI is resampled onto the CT grid by the labeling step), so the same slice
index samples corresponding anatomy.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_AXES = {0: "sagittal", 1: "coronal", 2: "axial"}


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as PNG bytes (no Pillow)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    # Each scanline is prefixed with filter byte 0 (None).
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 = RGB
    idat = zlib.compress(raw, 6)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _window(sl: np.ndarray) -> np.ndarray:
    """Percentile-window a 2D slice to uint8 grayscale (robust to CT air/metal)."""
    sl = np.asarray(sl, dtype=np.float32)
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return np.zeros(sl.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _canonical(img):
    import nibabel as nib
    return nib.as_closest_canonical(img)


def _slice(arr: np.ndarray, axis: int, frac: float) -> np.ndarray:
    n = arr.shape[axis]
    k = int(round(min(max(frac, 0.0), 1.0) * (n - 1)))
    sl = [slice(None)] * 3
    sl[axis] = k
    plane = arr[tuple(sl)]
    # Rotate so superior/anterior reads "up" for a natural view. Consistency
    # between CT and MRI matters more than exact radiological convention here.
    return np.rot90(plane)


def _downsample(a: np.ndarray, max_dim: int) -> np.ndarray:
    step = max(1, int(np.ceil(max(a.shape[:2]) / max_dim)))
    return a[::step, ::step] if step > 1 else a


def render_registration_qc(
    ct_path: str | Path,
    mri_path: str | Path,
    *,
    axis: int = 2,
    frac: float = 0.5,
    mode: str = "color",
    value: float = 0.5,
    direction: str = "h",
    max_dim: int = 512,
    checker: int = 28,
) -> bytes:
    """Render one composited CT↔MRI slice as PNG bytes.

    ``mode``: ``color`` (CT magenta / MRI green) · ``opacity`` (weighted
    blend; ``value`` = MRI weight 0→1) · ``wipe`` (CT one side, MRI the other,
    split at ``value`` along ``direction`` ``h``/``v``, with a marker line) ·
    ``checker`` · ``ct`` · ``mri``. Compositing is done here (server-side) so
    the browser only swaps one image per plane — no fragile client overlay.
    """
    import nibabel as nib

    if axis not in _AXES:
        raise ValueError(f"axis must be 0/1/2, got {axis}")
    ct_img = _canonical(nib.load(str(ct_path)))
    mri_img = _canonical(nib.load(str(mri_path)))
    ct = np.asanyarray(ct_img.dataobj)
    mri = np.asanyarray(mri_img.dataobj)
    if ct.shape != mri.shape:
        raise ValueError(
            f"CT {ct.shape} and MRI {mri.shape} differ — MRI must be resampled "
            f"onto the CT grid (labeling step's --save-registered-mri)")

    cg = _downsample(_window(_slice(ct, axis, frac)), max_dim)
    mg = _downsample(_window(_slice(mri, axis, frac)), max_dim)
    v = float(min(max(value, 0.0), 1.0))

    if mode == "ct":
        rgb = np.stack([cg, cg, cg], axis=-1)
    elif mode == "mri":
        rgb = np.stack([mg, mg, mg], axis=-1)
    elif mode in ("color", "blend"):
        # CT → magenta (R,B), MRI → green (G). Aligned = gray; misaligned fringes.
        rgb = np.stack([cg, mg, cg], axis=-1)
    elif mode == "opacity":
        g = ((1.0 - v) * cg + v * mg).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    elif mode == "wipe":
        h, w = cg.shape
        if direction == "v":
            split = int(round(v * h))
            take_mri = np.arange(h)[:, None] < split      # top = MRI
        else:
            split = int(round(v * w))
            take_mri = np.arange(w)[None, :] < split       # left = MRI
        g = np.where(np.broadcast_to(take_mri, cg.shape), mg, cg)
        rgb = np.stack([g, g, g], axis=-1)
        # bright marker line at the split
        if direction == "v" and 0 < split < h:
            rgb[max(0, split - 1):split + 1, :, :] = (255, 220, 0)
        elif 0 < split < w:
            rgb[:, max(0, split - 1):split + 1, :] = (255, 220, 0)
    else:  # checker
        h, w = cg.shape
        yy, xx = np.mgrid[0:h, 0:w]
        pick_ct = ((yy // checker) + (xx // checker)) % 2 == 0
        g = np.where(pick_ct, cg, mg)
        rgb = np.stack([g, g, g], axis=-1)
    return _png_bytes(rgb)


__all__ = ["render_registration_qc"]
