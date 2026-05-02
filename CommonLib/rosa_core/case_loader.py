"""Case-level helpers for discovering files and composing display transforms."""

from __future__ import annotations

import glob
import os
from typing import Any

from .types import DisplayRecord, Matrix4x4
from .transforms import identity_4x4, lps_to_ras_matrix, matmul_4x4


def find_ros_file(case_dir: str) -> str:
    """Return the single `.ros` file inside a case directory."""
    hits = sorted(glob.glob(os.path.join(case_dir, "*.ros")))
    if not hits:
        raise ValueError(f"No .ros file found in case folder: {case_dir}")
    if len(hits) > 1:
        raise ValueError(f"Multiple .ros files found in case folder: {case_dir}")
    return hits[0]


def resolve_analyze_volume(analyze_root: str, display: DisplayRecord) -> str | None:
    """Resolve `<volume>.img` path for a display record.

    Resolution strategy:
    1. Use `VOLUME` token path from ROS (`DICOM/<uid>/<name>`) when available.
    2. Fallback to recursive name search under `analyze_root`.
    """
    volume_path = display.get("volume_path")
    if volume_path:
        parts = volume_path.strip("/").split("/")
        if len(parts) >= 3:
            uid = parts[-2]
            name = parts[-1]
            candidate = os.path.join(analyze_root, uid, f"{name}.img")
            if os.path.exists(candidate):
                return candidate

    name = display["volume"]
    pattern = os.path.join(analyze_root, "**", f"{name}.img")
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def choose_reference_volume(displays: list[DisplayRecord], preferred: str | None = None) -> str:
    """Choose the root display volume for loading.

    Defaults to the first display in ROS order, or `preferred` if provided.
    """
    if preferred:
        wanted = preferred.lower()
        for d in displays:
            if d["volume"].lower() == wanted:
                return d["volume"]
        available = ", ".join(d["volume"] for d in displays) or "none"
        raise ValueError(f"Reference volume '{preferred}' not found. Available: {available}")

    if not displays:
        raise ValueError("No display volumes found in ROS file")
    return displays[0]["volume"]


def resolve_reference_index(displays: list[DisplayRecord], reference_volume: str | None = None) -> int:
    """Resolve root display index from volume name."""
    if not displays:
        raise ValueError("No display volumes found in ROS file")
    if reference_volume is None:
        return 0
    target = reference_volume.lower()
    for i, disp in enumerate(displays):
        if disp["volume"].lower() == target:
            return i
    available = ", ".join(d["volume"] for d in displays) or "none"
    raise ValueError(f"Reference volume '{reference_volume}' not found. Available: {available}")


def build_effective_matrices(displays: list[DisplayRecord], root_index: int = 0) -> list[Matrix4x4]:
    """Compose effective transforms from each display into the root frame.

    In ROSA, each `TRdicomRdisplay` for display `i` is interpreted as:
    `i -> imagery_3dref(i)`.
    This function composes those parent links to return `i -> root` for every `i`.
    """
    n = len(displays)
    if n == 0:
        return []
    if root_index < 0 or root_index >= n:
        raise ValueError(f"Invalid root_index {root_index} for {n} displays")

    # Normalize references. Missing IMAGERY_3DREF falls back to root.
    refs = []
    for disp in displays:
        ref = disp.get("imagery_3dref")
        if ref is None:
            ref = root_index
        refs.append(ref)

    cache = [None] * n
    active = set()

    def to_root(i):
        """Recursive chain composition with cycle detection."""
        if cache[i] is not None:
            return cache[i]
        if i == root_index:
            cache[i] = identity_4x4()
            return cache[i]
        if i in active:
            raise ValueError(f"Cycle detected in IMAGERY_3DREF chain at display index {i}")
        active.add(i)

        parent = refs[i]
        if not isinstance(parent, int) or parent < 0 or parent >= n:
            raise ValueError(
                f"Invalid IMAGERY_3DREF={parent} for display index {i} (must be 0..{n - 1})"
            )

        parent_to_root = to_root(parent)
        # i -> root = (parent -> root) * (i -> parent)
        composed = matmul_4x4(parent_to_root, displays[i]["matrix"])
        cache[i] = composed
        active.remove(i)
        return composed

    for i in range(n):
        to_root(i)

    return cache


# ---------------------------------------------------------------------
# ROSA volume loading (Analyze .img/.hdr -> in-memory SITK image)
# ---------------------------------------------------------------------
#
# The Slicer-side ``CaseLoaderService.load_case`` (rosa_scene) loads
# every Analyze volume in the .ros file and applies the composed
# ``display_i -> reference_display`` 4×4 to the volume node as a
# hardened linear transform. The CLI agent doesn't have a Slicer scene
# but needs the same effect: read the chosen Analyze volume, return an
# in-memory SITK image whose physical-space metadata reflects the
# composed-display matrix so detection / sampling land in the ROSA
# reference-display frame.
#
# This is purely an affine relabel (origin + direction stamped on the
# SITK image). No voxel resampling — the on-disk pixel grid is
# preserved.


def load_rosa_volume_as_sitk(
    case_dir: str,
    *,
    volume_name: str | None = None,
    invert: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load one ROSA-folder volume into an in-memory SITK image stamped
    with the composed display->reference 4×4 matrix.

    Args:
        case_dir: ROSA case folder (the one containing the .ros file
            and a DICOM/ Analyze tree).
        volume_name: name of the display volume to load (e.g.
            ``"postopCT"``). Defaults to the first display in the .ros
            file (the same default ``CaseLoaderService`` uses).
        invert: invert the composed matrix before stamping. Mirrors
            RosaHelper's "Invert TRdicomRdisplay" UI checkbox; useful
            if the .ros file's transform direction is opposite to what
            the consumer expects.

    Returns:
        ``(sitk_image, metadata)`` where:
          * ``sitk_image`` is a SITK image with origin/direction set so
            its IJK->RAS matrix equals the composed display->reference
            transform (LPS-stamped internally; sample with
            ``shank_core.io.image_ijk_ras_matrices`` to get the RAS
            matrix back).
          * ``metadata`` carries useful provenance for callers:
              - ``"ros_path"`` (str)
              - ``"reference_volume"`` (str — chosen root display name)
              - ``"loaded_volume"`` (str — name of the volume returned)
              - ``"display_index"`` (int — the loaded volume's display index)
              - ``"is_reference"`` (bool — True when loaded == reference)
              - ``"display_to_reference_ras"`` (4×4 nested list)
              - ``"trajectories"`` (list of {name, start_lps, end_lps,
                start_ras, end_ras} — the planned trajectories from the
                .ros file, RAS-converted for downstream guided-fit use)
    """
    import SimpleITK as sitk

    # Local imports keep the heavy pipeline (parser + transforms) lazy.
    from .ros_parser import parse_ros_file
    from .transforms import lps_to_ras_point

    case_dir = os.path.abspath(case_dir)
    ros_path = find_ros_file(case_dir)
    parsed = parse_ros_file(ros_path)
    displays = parsed.get("displays") or []
    if not displays:
        raise ValueError(f"No TRdicomRdisplay/VOLUME entries in {ros_path}")

    analyze_root = os.path.join(case_dir, "DICOM")
    if not os.path.isdir(analyze_root):
        raise ValueError(f"Analyze root not found under case dir: {analyze_root}")

    # Resolve which display the caller wants.
    if volume_name is None:
        loaded = displays[0]
    else:
        target = str(volume_name).lower()
        match = [d for d in displays if d["volume"].lower() == target]
        if not match:
            available = ", ".join(d["volume"] for d in displays)
            raise ValueError(
                f"Display volume {volume_name!r} not found in {ros_path}. "
                f"Available: {available}"
            )
        loaded = match[0]
    loaded_index = int(loaded.get("index", displays.index(loaded)))

    # Compose display->reference matrices using the first display as the
    # reference (matches CaseLoaderService default; can be overridden by
    # passing the desired reference_volume to ``choose_reference_volume``
    # at a higher level later if needed).
    reference_volume = displays[0]["volume"]
    root_index = 0
    effective_lps = build_effective_matrices(displays, root_index=root_index)
    matrix_lps = effective_lps[loaded_index]
    if invert:
        from .transforms import invert_4x4
        matrix_lps = invert_4x4(matrix_lps)
    matrix_ras = lps_to_ras_matrix(matrix_lps)

    # Load the on-disk Analyze volume (.img/.hdr).
    img_path = resolve_analyze_volume(analyze_root, loaded)
    if not img_path:
        raise ValueError(
            f"Analyze .img file for display {loaded['volume']!r} not "
            f"found under {analyze_root}"
        )
    img = sitk.ReadImage(str(img_path))

    # Stamp the IJK->RAS matrix that puts this volume in the ROSA-planning
    # coordinate frame:
    #
    #     centered_native_ijk_to_ras  — voxel center sits at world origin
    #                                   (Slicer's "Center Volume" step)
    #     display_to_reference_ras    — composed TRdicomRdisplay chain
    #     final = display_to_reference_ras @ centered_native_ijk_to_ras
    #
    # This mirrors what CaseLoaderService.load_case does on the Slicer
    # side: load_volume -> center_volume -> apply_transform -> harden.
    # Without the centering step, planned trajectories (which sit in
    # the brain-centered frame ROSA stores) land hundreds of mm away
    # from the imaged content (which sits in the image-corner-origin
    # frame native to the Analyze header).
    #
    # Stamp logic is inlined (not importing
    # ``rosa_detect.service.stamp_ijk_to_ras_on_sitk``) because
    # rosa_core must not depend on rosa_detect — layer goes
    # rosa_core <- rosa_detect, not the reverse.
    import numpy as np
    from shank_core.io import image_ijk_ras_matrices

    ijk_to_ras_native, _ = image_ijk_ras_matrices(img)

    # Center: shift origin so the image-center voxel lands at world (0, 0, 0).
    size = img.GetSize()
    center_ijk_h = np.array([
        (size[0] - 1) * 0.5,
        (size[1] - 1) * 0.5,
        (size[2] - 1) * 0.5,
        1.0,
    ], dtype=float)
    center_ras = ijk_to_ras_native @ center_ijk_h
    centering = np.eye(4, dtype=float)
    centering[:3, 3] = -center_ras[:3]
    ijk_to_ras_centered = centering @ ijk_to_ras_native

    # Compose with display->reference (identity for the reference vol;
    # the chain matrix for everyone else).
    m_xform = np.asarray(matrix_ras, dtype=float)
    ijk_to_ras_final = m_xform @ ijk_to_ras_centered

    spacing = np.array(img.GetSpacing(), dtype=float)
    origin_ras = ijk_to_ras_final[:3, 3].copy()
    dir_ras = np.zeros((3, 3), dtype=float)
    for k in range(3):
        dir_ras[:, k] = ijk_to_ras_final[:3, k] / max(1e-9, float(spacing[k]))
    # SITK lives in LPS: flip X and Y of both origin and direction.
    origin_lps = np.array(
        [-origin_ras[0], -origin_ras[1], origin_ras[2]], dtype=float,
    )
    dir_lps = dir_ras.copy()
    dir_lps[0, :] *= -1.0
    dir_lps[1, :] *= -1.0
    img.SetOrigin(tuple(float(v) for v in origin_lps.tolist()))
    img.SetDirection(tuple(float(v) for v in dir_lps.flatten().tolist()))

    # Convert planned trajectories LPS -> RAS once so downstream callers
    # don't have to repeat the flip per use.
    trajectories = []
    for traj in parsed.get("trajectories", []):
        start_lps = list(traj["start"])
        end_lps = list(traj["end"])
        trajectories.append({
            "name": traj["name"],
            "start_lps": [float(v) for v in start_lps],
            "end_lps": [float(v) for v in end_lps],
            "start_ras": [float(v) for v in lps_to_ras_point(start_lps)],
            "end_ras": [float(v) for v in lps_to_ras_point(end_lps)],
        })

    metadata = {
        "ros_path": ros_path,
        "reference_volume": reference_volume,
        "loaded_volume": loaded["volume"],
        "loaded_volume_path": img_path,
        "display_index": loaded_index,
        "is_reference": loaded_index == root_index,
        "display_to_reference_ras": [
            [float(matrix_ras[r][c]) for c in range(4)] for r in range(4)
        ],
        "trajectories": trajectories,
    }
    return img, metadata
