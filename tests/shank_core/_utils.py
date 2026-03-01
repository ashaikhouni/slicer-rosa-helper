import numpy as np


def make_synthetic_ct_with_line(shape=(64, 64, 64)):
    """Create a small synthetic CT with tissue envelope and one metal line."""
    arr = np.full(shape, -1000.0, dtype=np.float32)

    # Soft tissue core.
    k0, j0, i0 = [s // 2 for s in shape]
    kk, jj, ii = np.indices(shape)
    rr = np.sqrt((kk - k0) ** 2 + (jj - j0) ** 2 + (ii - i0) ** 2)
    arr[rr <= 24.0] = 40.0

    # One metal shaft-like line (k fixed, mostly along i).
    k = k0
    j = j0 + 2
    for i in range(i0 - 16, i0 + 17):
        arr[k, j, i] = 2800.0
    return arr


def kji_to_ras_points_identity(ijk_kji):
    """KJI -> RAS conversion for identity-geometry synthetic arrays."""
    idx = np.asarray(ijk_kji, dtype=float).reshape(-1, 3)
    out = np.zeros((idx.shape[0], 3), dtype=float)
    out[:, 0] = idx[:, 2]  # i
    out[:, 1] = idx[:, 1]  # j
    out[:, 2] = idx[:, 0]  # k
    return out


def ras_to_ijk_identity(ras_xyz):
    """RAS -> IJK conversion for identity-geometry synthetic arrays."""
    x, y, z = [float(v) for v in ras_xyz]
    return np.array([x, y, z], dtype=float)
