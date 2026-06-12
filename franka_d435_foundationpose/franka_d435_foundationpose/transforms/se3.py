"""Minimal SE(3) helpers using only numpy.

Naming convention used throughout the whole project: ``T_target_source``.

A 4x4 homogeneous matrix ``T_A_B`` transforms a point expressed in frame ``B``
(the *source*) into frame ``A`` (the *target*)::

    p_A = T_A_B @ p_B            # p_* are homogeneous 4-vectors

Composition therefore reads left-to-right in target/source pairs::

    T_A_C = T_A_B @ T_B_C

Quaternion convention: ``[x, y, z, w]`` (scalar last), matching ROS / scipy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "invert_T",
    "compose_T",
    "validate_T",
    "quat_to_matrix",
    "matrix_to_quat",
    "make_T",
    "apply_T",
]


def validate_T(T, atol: float = 1e-4, raise_on_error: bool = True) -> bool:
    """Validate that ``T`` is a proper 4x4 homogeneous SE(3) matrix.

    Checks shape, the ``[0, 0, 0, 1]`` bottom row, that the rotation block is
    orthonormal (``R R^T = I``) and that ``det(R) ~= +1``.
    """
    T = np.asarray(T)
    msgs = []
    if T.shape != (4, 4):
        msgs.append(f"expected shape (4, 4), got {T.shape}")
    else:
        if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
            msgs.append(f"bottom row must be [0,0,0,1], got {T[3, :]}")
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=atol):
            msgs.append("rotation block is not orthonormal (R R^T != I)")
        det = float(np.linalg.det(R))
        if not np.isclose(det, 1.0, atol=max(atol, 1e-3)):
            msgs.append(f"det(R) must be +1, got {det:.6f}")
    if msgs:
        if raise_on_error:
            raise ValueError("Invalid SE(3) matrix: " + "; ".join(msgs))
        return False
    return True


def invert_T(T):
    """Return the inverse of a homogeneous transform.

    ``invert_T(T_A_B) == T_B_A``. Uses the closed-form SE(3) inverse rather
    than a generic matrix inverse for numerical stability.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"invert_T expects (4, 4), got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def compose_T(*transforms):
    """Compose a chain of transforms left-to-right: ``compose_T(A, B, C) = A @ B @ C``.

    For the project convention this means ``compose_T(T_A_B, T_B_C) == T_A_C``.
    """
    if not transforms:
        return np.eye(4, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    for T in transforms:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def quat_to_matrix(q):
    """Convert a quaternion ``[x, y, z, w]`` (scalar last) to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    x, y, z, w = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("quat_to_matrix received a near-zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(R):
    """Convert a 3x3 rotation matrix to a quaternion ``[x, y, z, w]`` (scalar last).

    The returned quaternion is normalized and has a non-negative ``w`` to make
    the representation unique.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_quat expects (3, 3) or (4, 4), got {R.shape}")
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


def make_T(R=None, t=None):
    """Build a 4x4 homogeneous transform from a rotation and translation.

    ``R`` may be a 3x3 rotation matrix or a quaternion ``[x, y, z, w]``.
    ``t`` is a length-3 translation (defaults to zero).
    """
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        R = np.asarray(R, dtype=np.float64)
        if R.shape == (4,):
            R = quat_to_matrix(R)
        elif R.shape != (3, 3):
            raise ValueError(f"make_T R must be (3, 3) or quat (4,), got {R.shape}")
        T[:3, :3] = R
    if t is not None:
        t = np.asarray(t, dtype=np.float64).reshape(3)
        T[:3, 3] = t
    return T


def apply_T(T, points):
    """Apply a transform to one or more 3D points.

    ``points`` may be shape ``(3,)`` or ``(N, 3)``. Returns the transformed
    points with the same leading shape.
    """
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)
    if pts.shape[-1] != 3:
        raise ValueError(f"apply_T expects points with last dim 3, got {pts.shape}")
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 4)
    out = (T @ homog.T).T[:, :3]
    return out[0] if single else out
