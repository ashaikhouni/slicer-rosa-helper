"""Case-level helpers for discovering files and composing display transforms."""

import glob
import os

from .transforms import identity_4x4, matmul_4x4


def find_ros_file(case_dir):
    """Return the single `.ros` file inside a case directory."""
    hits = sorted(glob.glob(os.path.join(case_dir, "*.ros")))
    if not hits:
        raise ValueError(f"No .ros file found in case folder: {case_dir}")
    if len(hits) > 1:
        raise ValueError(f"Multiple .ros files found in case folder: {case_dir}")
    return hits[0]


def resolve_analyze_volume(analyze_root, display):
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


def choose_reference_volume(displays, preferred=None):
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


def resolve_reference_index(displays, reference_volume=None):
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


def build_effective_matrices(displays, root_index=0):
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
