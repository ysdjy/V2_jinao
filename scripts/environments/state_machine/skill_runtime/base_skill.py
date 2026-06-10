"""Base helpers for skill state machines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils

from .scene_state_provider import PoseState, SceneState
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason


@dataclass
class SkillCommand:
    tcp_pose_w: PoseState
    gripper_command: float
    status: ExecutionStatus


@dataclass
class PoseError:
    position: float
    orientation: float


class BaseSkill:
    status = ExecutionStatus.IDLE
    failure_reason = FailureReason.NONE

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        raise NotImplementedError

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

    def result(self, state: SceneState) -> SkillResult:
        raise NotImplementedError


def pose_error(current: PoseState, desired: PoseState) -> PoseError:
    pos_error = float(torch.linalg.norm(current.pos_w - desired.pos_w).detach().cpu())
    current_quat = math_utils.normalize(current.quat_w.unsqueeze(0))[0]
    desired_quat = math_utils.normalize(desired.quat_w.unsqueeze(0))[0]
    dot = torch.abs(torch.dot(current_quat, desired_quat)).clamp(max=1.0)
    ori_error = float((2.0 * torch.acos(dot)).detach().cpu())
    return PoseError(pos_error, ori_error)


def step_pose(current: PoseState, desired: PoseState, max_pos_step: float, max_ori_step: float) -> PoseState:
    delta = desired.pos_w - current.pos_w
    dist = torch.linalg.norm(delta)
    if float(dist) > max_pos_step:
        pos = current.pos_w + delta / dist * max_pos_step
    else:
        pos = desired.pos_w
    current_quat = math_utils.normalize(current.quat_w.unsqueeze(0))[0]
    desired_quat = math_utils.normalize(desired.quat_w.unsqueeze(0))[0]
    if float(torch.dot(current_quat, desired_quat).detach().cpu()) < 0.0:
        desired_quat = -desired_quat
    angle = math_utils.quat_error_magnitude(current_quat.unsqueeze(0), desired_quat.unsqueeze(0))[0]
    if float(angle) > max_ori_step and float(angle) > 1.0e-6:
        tau = max_ori_step / max(float(angle), 1.0e-6)
        quat = math_utils.quat_slerp(current_quat, desired_quat, tau)
    else:
        quat = desired_quat
    return PoseState(pos, math_utils.normalize(quat.unsqueeze(0))[0])


def pose_tensor(pose: PoseState | None) -> torch.Tensor | None:
    if pose is None:
        return None
    return torch.cat((pose.pos_w, pose.quat_w), dim=-1)


def finite_pose(pose: PoseState) -> bool:
    return bool(torch.isfinite(pose.pos_w).all() and torch.isfinite(pose.quat_w).all())


def is_pose_reached(error: PoseError, pos_threshold: float, ori_threshold: float) -> bool:
    return error.position <= pos_threshold and error.orientation <= ori_threshold


def clamp_workspace(pose: PoseState, lower: torch.Tensor, upper: torch.Tensor) -> PoseState:
    pos = torch.minimum(torch.maximum(pose.pos_w, lower), upper)
    return PoseState(pos, pose.quat_w)


def format_pose(pose: PoseState) -> list[float]:
    return [round(float(x), 5) for x in pose.as_pose_tensor().detach().cpu().tolist()]


def safe_acos_dot(q1: torch.Tensor, q2: torch.Tensor) -> float:
    dot = float(torch.abs(torch.dot(q1, q2)).clamp(max=1.0).detach().cpu())
    return 2.0 * math.acos(dot)
