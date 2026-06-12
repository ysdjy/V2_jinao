"""Parse ``configs/camera_mount.yaml`` (light: yaml + numpy only).

Importable on the IsaacLab side and in plain unit tests. Provides the resolved
mount mode, the camera extrinsic as a 4x4 matrix, and helpers shared by the
demo script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..transforms.se3 import make_T
from ..utils.config import load_yaml

CAMERA_MODE_EE = "end_effector"
CAMERA_MODE_FIXED = "fixed_scene"


@dataclass
class MountMode:
    name: str
    raw: dict = field(default_factory=dict)

    # ---- common ----
    @property
    def camera_name(self) -> str:
        return self.raw.get("camera_name", "d435")

    @property
    def camera_frame(self) -> str:
        return self.raw.get("camera_frame", "d435_color_optical_frame")

    @property
    def width(self) -> int:
        return int(self.raw.get("resolution", {}).get("width", 640))

    @property
    def height(self) -> int:
        return int(self.raw.get("resolution", {}).get("height", 480))

    @property
    def sim_optics(self) -> dict:
        return self.raw.get("sim_optics", {}) or {}

    @property
    def want_segmentation(self) -> bool:
        return bool(self.raw.get("sensor_types", {}).get("segmentation", False))

    # ---- end_effector mode ----
    @property
    def parent_link_candidates(self) -> list:
        return list(self.raw.get("parent_link_candidates", []))

    @property
    def prim_path_suffix(self) -> str:
        return self.raw.get("prim_path_suffix", "Sensors/d435_ee")

    def T_ee_camera(self) -> np.ndarray:
        te = self.raw.get("T_ee_camera", {}) or {}
        if "matrix" in te and te["matrix"] is not None:
            return np.asarray(te["matrix"], dtype=np.float64).reshape(4, 4)
        trans = np.asarray(te.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        quat = np.asarray(te.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        return make_T(R=quat, t=trans)

    # ---- fixed_scene mode ----
    @property
    def prim_path(self) -> str:
        return self.raw.get("prim_path", "/World/Sensors/fixed_d435_01")

    def pose(self) -> dict:
        return self.raw.get("pose", {}) or {}


@dataclass
class CameraMount:
    default_mode: str
    modes: dict

    def get(self, mode: str | None = None) -> MountMode:
        name = mode or self.default_mode
        if name not in self.modes:
            raise KeyError(
                f"camera mount mode '{name}' not found. Available: {sorted(self.modes)}"
            )
        return MountMode(name=name, raw=self.modes[name])


def load_camera_mount(config_path: str) -> CameraMount:
    cfg = load_yaml(config_path)
    return CameraMount(
        default_mode=cfg.get("default_mode", CAMERA_MODE_EE),
        modes=cfg.get("modes", {}) or {},
    )


def look_at_to_T_world_camera(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Build T_world_camera (camera->world) for a camera at ``eye`` looking at ``target``.

    Uses the camera-optical convention: +Z forward (toward target), +X right,
    +Y down. Returns a 4x4 SE(3) matrix mapping camera-frame points to world.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    z = target - eye  # forward (+Z optical)
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("look_at: eye and target coincide")
    z /= n
    x = np.cross(z, up)  # right (+X)
    nx = np.linalg.norm(x)
    if nx < 1e-9:  # up parallel to view dir; pick an alternate up
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)  # down (+Y), right-handed with x,z
    R = np.column_stack([x, y, z])  # columns are camera axes in world
    return make_T(R=R, t=eye)
