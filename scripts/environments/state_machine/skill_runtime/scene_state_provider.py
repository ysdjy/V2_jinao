"""Centralized scene state reads for skill execution.

Frame contract used by this package:
    * Control frame: Franka TCP.
    * TCP transform: panda_hand local +Z offset of 0.1034 m.
    * Quaternion order: (w, x, y, z), matching Isaac Lab math utilities.
    * IK action pose order: [x, y, z, qw, qx, qy, qz, gripper].

The IK action is configured with the same body_offset as the ee_frame sensor, so
the observed TCP, commanded TCP, debug visuals, and error metrics describe the
same physical frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PoseState:
    pos_w: torch.Tensor
    quat_w: torch.Tensor

    def as_pose_tensor(self) -> torch.Tensor:
        return torch.cat((self.pos_w, self.quat_w), dim=-1)


@dataclass
class ObjectState:
    name: str
    pose: PoseState
    lin_vel_w: torch.Tensor | None = None
    ang_vel_w: torch.Tensor | None = None
    links: dict[str, PoseState] = field(default_factory=dict)
    joint_pos: dict[str, float] = field(default_factory=dict)
    joint_vel: dict[str, float] = field(default_factory=dict)


@dataclass
class RobotState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    tcp_pose: PoseState
    gripper_width: float


@dataclass
class SceneState:
    env_id: int
    env_origin_w: torch.Tensor
    robot: RobotState
    objects: dict[str, ObjectState]
    sim_time: float


class SceneStateProvider:
    """Reads all runtime state needed by skills from the Isaac Lab scene."""

    def __init__(self, env, env_id: int = 0):
        self.env = env
        self.scene = env.unwrapped.scene
        self.env_id = env_id
        self.device = self.scene.device
        self._finger_joint_ids = self._find_joints("panda_finger_.*")
        self._sim_time = 0.0

    def set_sim_time(self, sim_time: float):
        self._sim_time = sim_time

    def get_state(self) -> SceneState:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]
        env_origin_w = self.scene.env_origins[self.env_id]
        tcp_pos_w = ee_frame.data.target_pos_w[self.env_id, 0].clone()
        tcp_quat_w = ee_frame.data.target_quat_w[self.env_id, 0].clone()
        gripper_width = self._read_gripper_width(robot)

        objects: dict[str, ObjectState] = {}
        for name in ("cube_1", "cube_2", "cube_3"):
            if name in self.scene.keys():
                objects[name] = self._rigid_object_state(name)
        if "knife" in self.scene.keys():
            objects["knife"] = self._articulation_state("knife", link_names=("base", "joint_0", "HandleProxy"))
        if "cabinet" in self.scene.keys():
            objects["cabinet"] = self._articulation_state(
                "cabinet", link_names=("link_1", "drawer", "handle", "HandleProxy")
            )

        return SceneState(
            env_id=self.env_id,
            env_origin_w=env_origin_w.clone(),
            robot=RobotState(
                joint_pos=robot.data.joint_pos[self.env_id].clone(),
                joint_vel=robot.data.joint_vel[self.env_id].clone(),
                tcp_pose=PoseState(tcp_pos_w, tcp_quat_w),
                gripper_width=gripper_width,
            ),
            objects=objects,
            sim_time=self._sim_time,
        )

    def make_action(self, tcp_pose_w: PoseState | torch.Tensor, gripper_command: float) -> torch.Tensor:
        """Build an absolute IK action for the configured TCP frame.

        Isaac Lab manager actions use environment-local position with world-frame
        quaternion. The returned tensor has shape (num_envs, 8).
        """

        if isinstance(tcp_pose_w, PoseState):
            pos_w = tcp_pose_w.pos_w
            quat_w = tcp_pose_w.quat_w
        else:
            pos_w = tcp_pose_w[:3]
            quat_w = tcp_pose_w[3:7]
        pos_env = pos_w - self.scene.env_origins[self.env_id]
        action = torch.zeros((self.env.unwrapped.num_envs, 8), device=self.device)
        action[:, :3] = pos_env
        action[:, 3:7] = quat_w
        action[:, 7] = gripper_command
        return action

    def hold_action(self, state: SceneState, gripper_command: float) -> torch.Tensor:
        if gripper_command not in (-1.0, 1.0):
            raise ValueError("gripper_command must be -1.0 or 1.0")
        return self.make_action(state.robot.tcp_pose, gripper_command)

    def reset_scene_deterministic(self):
        raise RuntimeError("Scene reset moved to SceneLayoutManager.reset_layout()")

    def _find_joints(self, pattern: str) -> torch.Tensor:
        robot = self.scene["robot"]
        try:
            joint_ids, _ = robot.find_joints(pattern)
            return torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        except Exception:
            names = getattr(robot.data, "joint_names", [])
            ids = [idx for idx, name in enumerate(names) if "panda_finger" in name]
            return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _read_gripper_width(self, robot) -> float:
        if self._finger_joint_ids.numel() == 0:
            return 0.0
        values = robot.data.joint_pos[self.env_id, self._finger_joint_ids]
        return float(values.sum().detach().cpu())

    def _rigid_object_state(self, name: str) -> ObjectState:
        asset = self.scene[name]
        return ObjectState(
            name=name,
            pose=PoseState(asset.data.root_pos_w[self.env_id].clone(), asset.data.root_quat_w[self.env_id].clone()),
            lin_vel_w=getattr(asset.data, "root_lin_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
            ang_vel_w=getattr(asset.data, "root_ang_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
        )

    def _articulation_state(self, name: str, link_names: tuple[str, ...]) -> ObjectState:
        asset = self.scene[name]
        links = {}
        for link_name in link_names:
            pose = self._link_pose(asset, link_name)
            if pose is not None:
                links[link_name] = pose
        joint_pos: dict[str, float] = {}
        joint_vel: dict[str, float] = {}
        for joint_name, joint_id in self._joint_id_map(asset).items():
            joint_pos[joint_name] = float(asset.data.joint_pos[self.env_id, joint_id].detach().cpu())
            joint_vel[joint_name] = float(asset.data.joint_vel[self.env_id, joint_id].detach().cpu())
        return ObjectState(
            name=name,
            pose=PoseState(asset.data.root_pos_w[self.env_id].clone(), asset.data.root_quat_w[self.env_id].clone()),
            lin_vel_w=getattr(asset.data, "root_lin_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
            ang_vel_w=getattr(asset.data, "root_ang_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
            links=links,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def _link_pose(self, asset: Any, link_name: str) -> PoseState | None:
        names = getattr(asset.data, "body_names", [])
        candidates = [idx for idx, name in enumerate(names) if link_name in name]
        if not candidates:
            try:
                ids, _ = asset.find_bodies(link_name)
                candidates = list(ids)
            except Exception:
                return None
        idx = int(candidates[0])
        if not hasattr(asset.data, "body_pos_w") or idx >= asset.data.body_pos_w.shape[1]:
            return None
        return PoseState(asset.data.body_pos_w[self.env_id, idx].clone(), asset.data.body_quat_w[self.env_id, idx].clone())

    def _joint_id_map(self, asset: Any) -> dict[str, int]:
        names = getattr(asset.data, "joint_names", [])
        return {name: idx for idx, name in enumerate(names)}

    def _zero_articulation_joints(self, asset: Any):
        if not hasattr(asset, "write_joint_state_to_sim") or not hasattr(asset.data, "joint_pos"):
            return
        joint_pos = torch.zeros_like(asset.data.joint_pos[:1])
        joint_vel = torch.zeros_like(asset.data.joint_vel[:1])
        try:
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=torch.tensor([self.env_id], device=self.device))
        except Exception:
            return
