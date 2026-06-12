"""Unified observation/action schemas shared between IsaacLab and the policy server.

UNITS AND CONVENTIONS (read carefully, everything downstream depends on this):
  * positions / lengths : meters
  * rotations           : radians (for delta rot vectors), quaternion order is XYZW
                          unless a field name ends with `_wxyz`.
  * joint angles        : radians
  * gripper_width       : meters (physical opening). gripper *action* is normalized [-1, 1]
                          where -1 = close, +1 = open.
  * timestamps          : float seconds (time.time()).

These dataclasses are intentionally dependency-light (stdlib only) so they can be
imported from BOTH the IsaacLab conda env and the isolated OpenPI venv without
pulling heavy deps.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RobotState:
    joint_positions: list[float] = dataclasses.field(default_factory=list)  # rad
    joint_velocities: list[float] = dataclasses.field(default_factory=list)  # rad/s
    ee_position: list[float] = dataclasses.field(default_factory=list)  # [x,y,z] m
    ee_quat: list[float] = dataclasses.field(default_factory=list)  # [x,y,z,w] XYZW
    gripper_width: float = 0.0  # m


@dataclasses.dataclass
class ImageRef:
    """An image reference. Exactly one of `path` / `base64` is normally set.

    mode == "none"   -> both None (state-only observation)
    mode == "path"   -> `path` points to a png/jpg on a shared filesystem
    mode == "base64" -> `base64` holds a small jpeg/png (debug only)
    """

    mode: str = "none"  # none | path | base64
    path: Optional[str] = None
    base64: Optional[str] = None
    shape: Optional[list[int]] = None  # [H, W, C]


@dataclasses.dataclass
class ObjectPose:
    """Reserved for FoundationPose 6D pose integration."""

    name: str = ""
    position: list[float] = dataclasses.field(default_factory=list)  # world m
    quat: list[float] = dataclasses.field(default_factory=list)  # XYZW
    confidence: float = 0.0
    pose_in_camera: Optional[list[float]] = None  # 4x4 flattened, optional
    mesh_path: Optional[str] = None
    mask_path: Optional[str] = None


@dataclasses.dataclass
class Metadata:
    env_name: str = ""
    episode_id: int = 0
    step_id: int = 0


@dataclasses.dataclass
class Observation:
    timestamp: float = 0.0
    task_instruction: str = ""
    robot: RobotState = dataclasses.field(default_factory=RobotState)
    images: dict[str, ImageRef] = dataclasses.field(default_factory=dict)  # e.g. {"front_rgb":..,"wrist_rgb":..}
    objects: list[ObjectPose] = dataclasses.field(default_factory=list)
    metadata: Metadata = dataclasses.field(default_factory=Metadata)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Observation":
        d = dict(d or {})
        robot = RobotState(**(d.get("robot") or {}))
        images = {k: ImageRef(**v) for k, v in (d.get("images") or {}).items()}
        objects = [ObjectPose(**o) for o in (d.get("objects") or [])]
        meta = Metadata(**(d.get("metadata") or {}))
        return Observation(
            timestamp=float(d.get("timestamp", 0.0)),
            task_instruction=str(d.get("task_instruction", "")),
            robot=robot,
            images=images,
            objects=objects,
            metadata=meta,
        )


# --------------------------------------------------------------------------- #
# Action
# --------------------------------------------------------------------------- #
VALID_ACTION_TYPES = ("delta_ee_pose", "joint_position", "joint_delta")


@dataclasses.dataclass
class Action:
    action_type: str = "delta_ee_pose"  # see VALID_ACTION_TYPES
    delta_ee_position: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])  # m
    delta_ee_rot: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])  # rad axis-angle (3) or quat(4)
    gripper: float = 0.0  # normalized [-1,1], -1 close / +1 open
    # For joint_position / joint_delta action types:
    joint_targets: Optional[list[float]] = None  # rad
    # Optional multi-step chunk: list of full action vectors (model-native layout)
    chunk: Optional[list[list[float]]] = None
    raw_model_output: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Action":
        d = dict(d or {})
        return Action(
            action_type=d.get("action_type", "delta_ee_pose"),
            delta_ee_position=list(d.get("delta_ee_position", [0.0, 0.0, 0.0])),
            delta_ee_rot=list(d.get("delta_ee_rot", [0.0, 0.0, 0.0])),
            gripper=float(d.get("gripper", 0.0)),
            joint_targets=d.get("joint_targets"),
            chunk=d.get("chunk"),
            raw_model_output=d.get("raw_model_output"),
        )

    def as_7d(self) -> list[float]:
        """Return the canonical 7D vector [dx,dy,dz, rx,ry,rz, gripper]."""
        rot = list(self.delta_ee_rot)[:3]
        while len(rot) < 3:
            rot.append(0.0)
        return list(self.delta_ee_position)[:3] + rot + [self.gripper]


def validate_action_dict(d: dict[str, Any]) -> tuple[bool, str]:
    """Lightweight structural validation; returns (ok, message)."""
    if not isinstance(d, dict):
        return False, "action is not a dict"
    at = d.get("action_type", "delta_ee_pose")
    if at not in VALID_ACTION_TYPES:
        return False, f"unknown action_type '{at}'"
    if at == "delta_ee_pose":
        if len(d.get("delta_ee_position", [])) != 3:
            return False, "delta_ee_position must have length 3"
    else:
        if d.get("joint_targets") is None:
            return False, f"action_type '{at}' requires joint_targets"
    return True, "ok"
