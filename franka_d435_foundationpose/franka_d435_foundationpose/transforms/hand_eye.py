"""Load the hand-eye extrinsic ``T_ee_camera`` from ``configs/hand_eye.yaml``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from .frame_conventions import BASE_FRAME, CAMERA_FRAME, EE_FRAME
from .se3 import make_T, validate_T


@dataclass
class HandEye:
    """Hand-eye calibration result: camera -> ee extrinsic plus frame names."""

    T_ee_camera: np.ndarray
    camera_frame: str = CAMERA_FRAME
    ee_frame: str = EE_FRAME
    base_frame: str = BASE_FRAME
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.T_ee_camera = np.asarray(self.T_ee_camera, dtype=np.float64)
        validate_T(self.T_ee_camera)


def load_hand_eye(config_path: str) -> HandEye:
    """Load ``configs/hand_eye.yaml`` and build a :class:`HandEye`.

    The YAML must provide ``T_ee_camera`` either as:
      * ``translation_m`` + ``quaternion_xyzw``, or
      * a 4x4 ``matrix`` (list of lists).
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"hand_eye config not found: {config_path}\n"
            "Expected a YAML like configs/hand_eye.yaml with a T_ee_camera entry."
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    te = cfg.get("T_ee_camera", {})
    if "matrix" in te and te["matrix"] is not None:
        T = np.asarray(te["matrix"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_ee_camera.matrix must be 4x4, got {T.shape}")
    else:
        trans = np.asarray(te.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        quat = np.asarray(te.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        T = make_T(R=quat, t=trans)

    return HandEye(
        T_ee_camera=T,
        camera_frame=cfg.get("camera_frame", CAMERA_FRAME),
        ee_frame=cfg.get("ee_frame", EE_FRAME),
        base_frame=cfg.get("base_frame", BASE_FRAME),
        notes=cfg.get("notes", []) or [],
    )
