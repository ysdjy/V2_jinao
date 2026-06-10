"""Target registration and affordance computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_joint_pos_env_cfg

from .scene_state_provider import PoseState, SceneState
from .skill_types import FailureReason


@dataclass(frozen=True)
class TargetConfig:
    name: str
    scene_key: str
    display_name: str
    geometry_type: str
    size: tuple[float, float, float]
    local_grasp_pos: tuple[float, float, float]
    local_approach_dir: tuple[float, float, float]
    pre_grasp_clearance: float
    gripper_open_command: float
    gripper_close_command: float
    lift_distance: float
    min_gripper_width: float


@dataclass
class GraspPlan:
    target_name: str
    target_pose: PoseState
    grasp_pose: PoseState
    pre_grasp_pose: PoseState
    approach_dir_w: torch.Tensor
    lift_distance: float
    gripper_open_command: float
    gripper_close_command: float
    min_gripper_width: float
    valid: bool = True
    failure_reason: FailureReason = FailureReason.NONE
    message: str = ""


class TargetRegistry:
    """Keeps static target affordances separate from real-time object state."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        cube_size = (0.0406, 0.0406, 0.0406)
        self.targets: dict[str, TargetConfig] = {
            "cube_1": self._cube_cfg("cube_1", "Blue Cube (cube_1)", cube_size),
            "cube_2": self._cube_cfg("cube_2", "Red Cube (cube_2)", cube_size),
            "cube_3": self._cube_cfg("cube_3", "Green Cube (cube_3)", cube_size),
            "knife": TargetConfig(
                name="knife",
                scene_key="knife",
                display_name="Knife (knife)",
                geometry_type="knife_handle",
                size=stack_joint_pos_env_cfg.KNIFE_HANDLE_PROXY_SIZE,
                local_grasp_pos=stack_joint_pos_env_cfg.KNIFE_HANDLE_PROXY_OFFSET,
                local_approach_dir=(0.0, 0.0, 1.0),
                pre_grasp_clearance=0.120,
                gripper_open_command=1.0,
                gripper_close_command=-1.0,
                lift_distance=0.120,
                min_gripper_width=0.002,
            ),
        }

    def display_targets(self) -> list[tuple[str, str]]:
        return [(key, cfg.display_name) for key, cfg in self.targets.items()]

    def compute_grasp_plan(self, target_name: str, state: SceneState) -> GraspPlan:
        cfg = self.targets.get(target_name)
        if cfg is None:
            return self._invalid(target_name, FailureReason.REQUEST_INVALID, f"unknown target: {target_name}", state)
        if cfg.scene_key not in state.objects:
            return self._invalid(target_name, FailureReason.TARGET_LOST, f"target not found: {cfg.scene_key}", state)
        if cfg.geometry_type == "cube":
            return self._cube_grasp_plan(cfg, state)
        if cfg.geometry_type == "knife_handle":
            return self._knife_grasp_plan(cfg, state)
        return self._invalid(target_name, FailureReason.REQUEST_INVALID, "unsupported target type", state)

    def _cube_cfg(self, scene_key: str, display_name: str, size: tuple[float, float, float]) -> TargetConfig:
        return TargetConfig(
            name=scene_key,
            scene_key=scene_key,
            display_name=display_name,
            geometry_type="cube",
            size=size,
            # TCP grasp point in the selected face normal direction. This is
            # relative to the live cube pose, not a fixed world height.
            local_grasp_pos=(0.0, 0.0, 0.040),
            local_approach_dir=(0.0, 0.0, 1.0),
            pre_grasp_clearance=0.120,
            gripper_open_command=1.0,
            gripper_close_command=-1.0,
            lift_distance=0.120,
            min_gripper_width=0.0,
        )

    def _cube_grasp_plan(self, cfg: TargetConfig, state: SceneState) -> GraspPlan:
        obj = state.objects[cfg.scene_key]
        pos = obj.pose.pos_w
        quat = math_utils.normalize(obj.pose.quat_w.unsqueeze(0))[0]
        rot = math_utils.matrix_from_quat(quat.unsqueeze(0))[0]
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        axes = [rot[:, 0], -rot[:, 0], rot[:, 1], -rot[:, 1], rot[:, 2], -rot[:, 2]]
        normal = max(axes, key=lambda axis: float(torch.dot(axis, world_up)))
        if float(torch.dot(normal, world_up)) < 0.35:
            return self._invalid(cfg.name, FailureReason.TARGET_UNSAFE, "no upward graspable cube face", state)

        grasp_pos = pos + normal * cfg.local_grasp_pos[2]
        pre_grasp_pos = grasp_pos + normal * cfg.pre_grasp_clearance
        lateral_hint = self._best_lateral_hint(rot, normal)
        tcp_quat = self._tcp_quat_from_approach(normal, lateral_hint)
        return GraspPlan(
            target_name=cfg.name,
            target_pose=obj.pose,
            grasp_pose=PoseState(grasp_pos, tcp_quat),
            pre_grasp_pose=PoseState(pre_grasp_pos, tcp_quat),
            approach_dir_w=normal,
            lift_distance=cfg.lift_distance,
            gripper_open_command=cfg.gripper_open_command,
            gripper_close_command=cfg.gripper_close_command,
            min_gripper_width=cfg.min_gripper_width,
        )

    def _knife_grasp_plan(self, cfg: TargetConfig, state: SceneState) -> GraspPlan:
        obj = state.objects[cfg.scene_key]
        link_pose = obj.links.get(stack_joint_pos_env_cfg.KNIFE_BODY_LINK, obj.pose)
        handle_offset = torch.tensor(cfg.local_grasp_pos, dtype=torch.float32, device=self.device)
        handle_pos = link_pose.pos_w + math_utils.quat_apply(link_pose.quat_w.unsqueeze(0), handle_offset.unsqueeze(0))[0]
        handle_axis_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        handle_axis_w = math_utils.quat_apply(link_pose.quat_w.unsqueeze(0), handle_axis_local.unsqueeze(0))[0]
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        if float(handle_pos[2]) < 0.015:
            return self._invalid(cfg.name, FailureReason.TARGET_UNSAFE, "knife handle is too close to ground", state)
        tcp_quat = self._tcp_quat_from_approach(world_up, handle_axis_w)
        grasp_pose = PoseState(handle_pos + world_up * 0.012, tcp_quat)
        pre_grasp_pose = PoseState(grasp_pose.pos_w + world_up * cfg.pre_grasp_clearance, tcp_quat)
        return GraspPlan(
            target_name=cfg.name,
            target_pose=PoseState(handle_pos, link_pose.quat_w),
            grasp_pose=grasp_pose,
            pre_grasp_pose=pre_grasp_pose,
            approach_dir_w=world_up,
            lift_distance=cfg.lift_distance,
            gripper_open_command=cfg.gripper_open_command,
            gripper_close_command=cfg.gripper_close_command,
            min_gripper_width=cfg.min_gripper_width,
        )

    def _best_lateral_hint(self, rot: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        candidates = [rot[:, 0], rot[:, 1], rot[:, 2]]
        best = min(candidates, key=lambda axis: abs(float(torch.dot(axis, normal))))
        return best

    def _tcp_quat_from_approach(self, approach_dir_w: torch.Tensor, lateral_hint_w: torch.Tensor) -> torch.Tensor:
        approach = math_utils.normalize(approach_dir_w.unsqueeze(0))[0]
        z_axis = -approach
        x_axis = lateral_hint_w - torch.dot(lateral_hint_w, z_axis) * z_axis
        if torch.linalg.norm(x_axis) < 1.0e-5:
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
        x_axis = math_utils.normalize(x_axis.unsqueeze(0))[0]
        y_axis = torch.linalg.cross(z_axis, x_axis)
        y_axis = math_utils.normalize(y_axis.unsqueeze(0))[0]
        x_axis = torch.linalg.cross(y_axis, z_axis)
        rot = torch.stack((x_axis, y_axis, z_axis), dim=-1).unsqueeze(0)
        return math_utils.quat_unique(math_utils.quat_from_matrix(rot))[0]

    def _invalid(self, target_name: str, reason: FailureReason, message: str, state: SceneState) -> GraspPlan:
        tcp = state.robot.tcp_pose
        return GraspPlan(
            target_name=target_name,
            target_pose=tcp,
            grasp_pose=tcp,
            pre_grasp_pose=tcp,
            approach_dir_w=torch.tensor([0.0, 0.0, 1.0], device=self.device),
            lift_distance=0.0,
            gripper_open_command=1.0,
            gripper_close_command=-1.0,
            min_gripper_width=0.0,
            valid=False,
            failure_reason=reason,
            message=message,
        )
