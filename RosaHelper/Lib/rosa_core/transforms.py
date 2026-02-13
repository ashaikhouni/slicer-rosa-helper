def apply_affine(matrix4x4, point_xyz):
    x, y, z = point_xyz
    vec = [x, y, z, 1.0]
    out = [0.0, 0.0, 0.0, 0.0]
    for r in range(4):
        out[r] = sum(matrix4x4[r][c] * vec[c] for c in range(4))
    return [out[0], out[1], out[2]]


def is_identity_4x4(matrix4x4, tol=1e-4):
    for r in range(4):
        for c in range(4):
            target = 1.0 if r == c else 0.0
            if abs(matrix4x4[r][c] - target) > tol:
                return False
    return True


def invert_4x4(matrix4x4):
    # Gauss-Jordan inversion for small fixed-size matrices.
    a = [[float(matrix4x4[r][c]) for c in range(4)] for r in range(4)]
    inv = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]

    for i in range(4):
        pivot = i
        for r in range(i + 1, 4):
            if abs(a[r][i]) > abs(a[pivot][i]):
                pivot = r

        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")

        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            inv[i], inv[pivot] = inv[pivot], inv[i]

        factor = a[i][i]
        for c in range(4):
            a[i][c] /= factor
            inv[i][c] /= factor

        for r in range(4):
            if r == i:
                continue
            factor = a[r][i]
            for c in range(4):
                a[r][c] -= factor * a[i][c]
                inv[r][c] -= factor * inv[i][c]

    return inv


def lps_to_ras_point(point_xyz):
    return [-point_xyz[0], -point_xyz[1], point_xyz[2]]


def _matmul4(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def lps_to_ras_matrix(matrix4x4_lps):
    flip = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return _matmul4(flip, _matmul4(matrix4x4_lps, flip))


def matmul_4x4(a, b):
    return _matmul4(a, b)


def identity_4x4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def to_itk_affine_text(matrix4x4):
    params = [
        matrix4x4[0][0],
        matrix4x4[0][1],
        matrix4x4[0][2],
        matrix4x4[1][0],
        matrix4x4[1][1],
        matrix4x4[1][2],
        matrix4x4[2][0],
        matrix4x4[2][1],
        matrix4x4[2][2],
        matrix4x4[0][3],
        matrix4x4[1][3],
        matrix4x4[2][3],
    ]
    return (
        "#Insight Transform File V1.0\n"
        "Transform: AffineTransform_double_3_3\n"
        f"Parameters: {' '.join(f'{p:.8f}' for p in params)}\n"
        "FixedParameters: 0 0 0\n"
    )
