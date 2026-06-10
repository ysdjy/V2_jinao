"""Runtime owner for one active skill request."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from isaaclab.utils import math as math_utils

from .base_skill import SkillCommand
from .drawer_skill import DrawerSkill
from .grasp_skill import GraspSkill
from .place_skill import PlaceSkill
from .scene_state_provider import PoseState, SceneState
from .skill_request import SkillRequest
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason, SkillType
from .target_registry import TargetRegistry


@dataclass
class LatchedCommand:
    tcp_pose: PoseState
    gripper_command: float


@dataclass
class HeldObjectContext:
    object_name: str
    object_to_tcp_pos: torch.Tensor
    object_to_tcp_quat: torch.Tensor


class SkillExecutor:
    def __init__(self, registry: TargetRegistry, log_path: str | os.PathLike = "logs/skill_tests/grasp_results.jsonl"):
        self.registry = registry
        self.active_skill = None
        self.active_request: SkillRequest | None = None
        self.status = ExecutionStatus.IDLE
        self.status_text = ExecutionStatus.IDLE.value
        self.last_result: SkillResult | None = None
        self.latched_command: LatchedCommand | None = None
        self.held_object: HeldObjectContext | None = None
        self.paused = False
        self.pause_started_at: float | None = None
        self.log_path = Path(log_path)

    @property
    def current_state_name(self) -> str:
        if self.status_text == "holding":
            return "HOLDING"
        if self.active_skill is None:
            return "IDLE"
        return getattr(self.active_skill, "current_state", self.status.value)

    @property
    def runtime_status(self) -> str:
        return self.status_text

    def start(self, request: SkillRequest, state: SceneState) -> SkillResult | None:
        if request.skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER) and self.held_object is not None:
            return self._make_immediate_failure(
                request,
                FailureReason.RELEASE_HELD_OBJECT_FIRST,
                "Place or release the held object before operating the drawer.",
            )

        if self.active_skill is not None:
            self._pause_active_skill(state)

        if request.skill_type == SkillType.PLACE:
            if self.held_object is None:
                return self._make_immediate_failure(
                    request,
                    FailureReason.REQUEST_INVALID,
                    "NO_HELD_OBJECT",
                )
            request.source_object = self.held_object.object_name

        self.paused = False
        self.pause_started_at = None
        self.active_request = request
        if request.skill_type == SkillType.GRASP:
            self.active_skill = GraspSkill(request, self.registry)
            self.active_skill.start(state)
            self.status = ExecutionStatus.RUNNING
            self.status_text = self.status.value
            return None
        if request.skill_type == SkillType.PLACE:
            self.active_skill = PlaceSkill(request, held_object=self.held_object)
            self.active_skill.start(state)
            self.status = self.active_skill.status
            self.status_text = self.status.value
            return None
        elif request.skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            self.active_skill = DrawerSkill(request)
            self.active_skill.start(state)
            self.status = self.active_skill.status
            self.status_text = self.status.value
            return None
        else:
            self.active_skill = None
        self.status = ExecutionStatus.NOT_IMPLEMENTED
        self.status_text = self.status.value
        result = SkillResult(
            request_id=request.request_id,
            skill_type=request.skill_type,
            target_name=request.source_object or request.destination_object,
            success=False,
            final_status=ExecutionStatus.NOT_IMPLEMENTED,
            failure_reason=FailureReason.NOT_IMPLEMENTED.value,
            elapsed_time=0.0,
        )
        self.last_result = result
        self._append_result(result)
        print("NOT_IMPLEMENTED: this skill will be implemented in a later stage")
        return result

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.paused:
            return self._latched_or_safe_command(state)
        if self.active_skill is None:
            return self._latched_or_safe_command(state)
        command = self.active_skill.step(state, dt)
        self._latch(command)
        self.status = command.status
        self.status_text = self.status.value
        if self.status == ExecutionStatus.SUCCEEDED:
            self._handle_skill_success(state)
        elif self.status == ExecutionStatus.FAILED:
            self._handle_skill_failure(state)
        elif self.status == ExecutionStatus.STOPPED:
            self._handle_skill_stopped(state)
        return command

    def pause(self, state: SceneState) -> SkillCommand:
        self.paused = True
        self.pause_started_at = state.sim_time
        command = SkillCommand(state.robot.tcp_pose, self._current_gripper_command(), self.status)
        self._latch(command)
        self.status_text = "paused"
        return command

    def resume(self, state: SceneState) -> bool:
        if not self.paused:
            return False
        if self.active_skill is None:
            self.paused = False
            self.pause_started_at = None
            self.status_text = "holding" if self.held_object is not None else self.status.value
            return False
        paused_duration = state.sim_time - (self.pause_started_at or state.sim_time)
        runtime = getattr(self.active_skill, "runtime", None)
        if runtime is not None:
            runtime.start_time += paused_duration
            runtime.state_start_time += paused_duration
        self.paused = False
        self.pause_started_at = None
        self.status = ExecutionStatus.RUNNING
        self.status_text = self.status.value
        return True

    def stop(self, state: SceneState) -> SkillCommand:
        return self.pause(state)

    def reset(self):
        self.active_skill = None
        self.active_request = None
        self.held_object = None
        self.latched_command = None
        self.paused = False
        self.pause_started_at = None
        self.status = ExecutionStatus.IDLE
        self.status_text = self.status.value

    def _append_result(self, result: SkillResult):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(result.to_json() + "\n")

    def _latch(self, command: SkillCommand):
        self.latched_command = LatchedCommand(
            tcp_pose=PoseState(
                command.tcp_pose_w.pos_w.clone(),
                command.tcp_pose_w.quat_w.clone(),
            ),
            gripper_command=float(command.gripper_command),
        )

    def _latched_or_safe_command(self, state: SceneState) -> SkillCommand:
        if self.latched_command is not None:
            return SkillCommand(
                self.latched_command.tcp_pose,
                self.latched_command.gripper_command,
                self.status,
            )
        return SkillCommand(state.robot.tcp_pose, 1.0, self.status)

    def _current_gripper_command(self) -> float:
        if self.latched_command is not None:
            return self.latched_command.gripper_command
        return 1.0

    def _pause_active_skill(self, state: SceneState) -> None:
        command = SkillCommand(state.robot.tcp_pose, self._current_gripper_command(), self.status)
        self._latch(command)
        self.paused = True
        self.pause_started_at = state.sim_time
        self.status_text = "paused"

    def _handle_skill_success(self, state: SceneState) -> None:
        if self.active_skill is None or self.active_request is None:
            return
        result = self.active_skill.result(state)
        self.last_result = result
        self._append_result(result)
        if self.active_request.skill_type == SkillType.GRASP and self.active_request.source_object in state.objects:
            self._save_held_object_context(self.active_request.source_object, state)
            self.status_text = "holding"
        elif self.active_request.skill_type == SkillType.PLACE:
            self.held_object = None
            self.status_text = self.status.value
        else:
            self.status_text = self.status.value
        self.active_skill = None
        self.active_request = None

    def _handle_skill_failure(self, state: SceneState) -> None:
        if self.active_skill is None:
            return
        self.last_result = self.active_skill.result(state)
        self._append_result(self.last_result)
        self.status_text = self.status.value
        self.active_skill = None
        self.active_request = None

    def _handle_skill_stopped(self, state: SceneState) -> None:
        if self.active_skill is None:
            return
        self.last_result = self.active_skill.result(state)
        self._append_result(self.last_result)
        self.status_text = self.status.value
        self.active_skill = None
        self.active_request = None

    def _save_held_object_context(self, object_name: str, state: SceneState) -> None:
        object_state = state.objects[object_name]
        tcp_state = state.robot.tcp_pose
        object_to_tcp_pos, object_to_tcp_quat = math_utils.subtract_frame_transforms(
            object_state.pose.pos_w.unsqueeze(0),
            object_state.pose.quat_w.unsqueeze(0),
            tcp_state.pos_w.unsqueeze(0),
            tcp_state.quat_w.unsqueeze(0),
        )
        self.held_object = HeldObjectContext(
            object_name=object_name,
            object_to_tcp_pos=object_to_tcp_pos[0].clone(),
            object_to_tcp_quat=object_to_tcp_quat[0].clone(),
        )

    def _make_immediate_failure(
        self,
        request: SkillRequest,
        reason: FailureReason,
        message: str,
    ) -> SkillResult:
        self.status = ExecutionStatus.FAILED
        self.status_text = self.status.value
        result = SkillResult(
            request_id=request.request_id,
            skill_type=request.skill_type,
            target_name=request.source_object or request.destination_object,
            success=False,
            final_status=ExecutionStatus.FAILED,
            failure_reason=f"{reason.value}: {message}",
            elapsed_time=0.0,
        )
        self.last_result = result
        self._append_result(result)
        return result
