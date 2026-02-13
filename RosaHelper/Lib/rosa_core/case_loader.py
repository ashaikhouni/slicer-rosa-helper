import glob
import os

from .transforms import identity_4x4, matmul_4x4


def find_ros_file(case_dir):
    hits = sorted(glob.glob(os.path.join(case_dir, "*.ros")))
    if not hits:
        raise ValueError(f"No .ros file found in case folder: {case_dir}")
    if len(hits) > 1:
        raise ValueError(f"Multiple .ros files found in case folder: {case_dir}")
    return hits[0]


def resolve_analyze_volume(analyze_root, display):
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
    # Each raw display matrix maps volume_i -> imagery_3dref(i).
    # Compose parent chains to express every volume in the root frame.
    n = len(displays)
    if n == 0:
        return []
    if root_index < 0 or root_index >= n:
        raise ValueError(f"Invalid root_index {root_index} for {n} displays")

    # Normalize references. Default to root if missing.
    refs = []
    for disp in displays:
        ref = disp.get("imagery_3dref")
        if ref is None:
            ref = root_index
        refs.append(ref)

    cache = [None] * n
    active = set()

    def to_root(i):
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
