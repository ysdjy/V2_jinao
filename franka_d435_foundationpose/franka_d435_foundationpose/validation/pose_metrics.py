"""Pose error metrics between a predicted and a ground-truth SE(3) pose.

Convention: poses are 4x4 ``T_target_source``. Errors are computed on the same
target/source pair (e.g. both ``T_base_object``).

Cube symmetry: a cube is invariant under the 24 proper rotations of the
octahedral group, so a single predicted orientation can be "correct" up to any
of those rotations. ``rotation_error_deg_symmetric`` reports the minimum error
over that group. Translation error is unaffected by symmetry and is the more
reliable metric for a cube.
"""

from __future__ import annotations

import numpy as np

CUBE_SYMMETRY_NOTE = (
    "The target is a CUBE, which is symmetric under the 24 proper rotations of "
    "the octahedral group. Therefore: (1) translation_error_m is the reliable "
    "metric; (2) the naive rotation_error_deg can be large purely due to a "
    "symmetry-equivalent orientation — use rotation_error_deg_symmetric instead; "
    "(3) for strict 6D orientation validation, use a NON-symmetric object (e.g. "
    "the knife) or add an asymmetric texture/marker to the cube."
)


def translation_error_m(T_pred, T_gt) -> float:
    """Euclidean translation error (meters) between two 4x4 poses."""
    t_pred = np.asarray(T_pred, dtype=np.float64)[:3, 3]
    t_gt = np.asarray(T_gt, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(t_pred - t_gt))


def _rotation_angle_deg(R) -> float:
    R = np.asarray(R, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def rotation_error_deg(T_pred, T_gt) -> float:
    """Geodesic rotation error (degrees): angle of R_pred @ R_gt^T."""
    R_pred = np.asarray(T_pred, dtype=np.float64)[:3, :3]
    R_gt = np.asarray(T_gt, dtype=np.float64)[:3, :3]
    return _rotation_angle_deg(R_pred @ R_gt.T)


def cube_rotation_group():
    """Return the 24 proper rotation matrices (det=+1) of a cube."""
    import itertools

    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = P * np.array(signs)[None, :]
            if np.isclose(np.linalg.det(M), 1.0):
                mats.append(M)
    # 6 permutations * 8 sign combos -> 48 signed perms; half have det +1 -> 24.
    return mats


def rotation_error_deg_symmetric(T_pred, T_gt, symmetry: str | None = "cube") -> float:
    """Rotation error minimized over the object's symmetry group.

    For ``symmetry="cube"`` (or None default to cube here) the minimum over the
    24 octahedral rotations is returned. For other/None symmetry it equals
    :func:`rotation_error_deg`.
    """
    R_pred = np.asarray(T_pred, dtype=np.float64)[:3, :3]
    R_gt = np.asarray(T_gt, dtype=np.float64)[:3, :3]
    if symmetry == "cube":
        group = cube_rotation_group()
        return min(_rotation_angle_deg(R_pred @ (G @ R_gt).T) for G in group)
    return _rotation_angle_deg(R_pred @ R_gt.T)


def build_validation(
    T_pred,
    T_gt,
    object_name: str,
    frame: str = "base",
    symmetry: str | None = "cube",
    backend: str = "mock",
) -> dict:
    """Assemble a validation dict comparing predicted vs ground-truth pose."""
    T_pred = np.asarray(T_pred, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    trans_err = translation_error_m(T_pred, T_gt)
    rot_err = rotation_error_deg(T_pred, T_gt)
    is_cube = symmetry == "cube" or "cube" in object_name.lower()
    rot_err_sym = rotation_error_deg_symmetric(T_pred, T_gt, "cube" if is_cube else symmetry)

    data = {
        "object_name": object_name,
        "frame": frame,
        "backend": backend,
        "convention": "T_target_source",
        "T_pred": T_pred.tolist(),
        "T_gt": T_gt.tolist(),
        "translation_error_m": trans_err,
        "rotation_error_deg": rot_err,
        "rotation_error_deg_symmetric": rot_err_sym,
        "symmetry": "cube" if is_cube else (symmetry or "none"),
        "warnings": [],
        "notes": [],
    }
    if is_cube:
        data["warnings"].append("cube_symmetry")
        data["notes"].append(CUBE_SYMMETRY_NOTE)
    if backend == "mock":
        data["warnings"].append("mock_backend")
        data["notes"].append(
            "backend is the MOCK estimator: the pose is NOT a real estimate, so "
            "the error values only validate the data/transform plumbing."
        )
    return data


def write_validation_report_md(data: dict, path: str) -> str:
    """Write a human-readable markdown validation report."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = [
        "# Pose validation report",
        "",
        f"- object: `{data['object_name']}`",
        f"- frame: `{data['frame']}`  (poses are T_{data['frame']}_object)",
        f"- backend: `{data['backend']}`",
        f"- convention: `{data['convention']}`",
        "",
        "## Errors",
        "",
        f"- **translation_error_m**: {data['translation_error_m']:.4f} m",
        f"- rotation_error_deg (naive): {data['rotation_error_deg']:.2f} deg",
        f"- rotation_error_deg_symmetric: {data['rotation_error_deg_symmetric']:.2f} deg",
        f"- symmetry: `{data['symmetry']}`",
        "",
        "## Warnings",
        "",
    ]
    L += [f"- ⚠️ `{w}`" for w in data.get("warnings", [])] or ["- (none)"]
    L += ["", "## Notes", ""]
    L += [f"- {n}" for n in data.get("notes", [])] or ["- (none)"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path
