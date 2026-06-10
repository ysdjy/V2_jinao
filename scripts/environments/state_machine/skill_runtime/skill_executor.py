"""Runtime owner for one active skill request."""

from __future__ import annotations

import os
from pathlib import Path

from .base_skill import SkillCommand
from .drawer_skill import DrawerSkill
from .grasp_skill import GraspSkill
from .place_skill import PlaceSkill
from .scene_state_provider import SceneState
from .skill_request import SkillRequest
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason, SkillType
from .target_registry import TargetRegistry


class SkillExecutor:
    def __init__(self, registry: TargetRegistry, log_path: str | os.PathLike = "logs/skill_tests/grasp_results.jsonl"):
        self.registry = registry
        self.active_skill = None
        self.active_request: SkillRequest | None = None
        self.status = ExecutionStatus.IDLE
        self.last_result: SkillResult | None = None
        self.log_path = Path(log_path)

    @property
    def current_state_name(self) -> str:
        if self.active_skill is None:
            return "IDLE"
        return getattr(self.active_skill, "current_state", self.status.value)

    def start(self, request: SkillRequest, state: SceneState) -> SkillResult | None:
        if self.status == ExecutionStatus.RUNNING:
            return None
        self.active_request = request
        if request.skill_type == SkillType.GRASP:
            self.active_skill = GraspSkill(request, self.registry)
            self.active_skill.start(state)
            self.status = ExecutionStatus.RUNNING
            return None
        if request.skill_type == SkillType.PLACE:
            self.active_skill = PlaceSkill(request)
            self.active_skill.start(state)
            self.status = self.active_skill.status
            return None
        elif request.skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            self.active_skill = DrawerSkill(request)
        else:
            self.active_skill = None
        self.status = ExecutionStatus.NOT_IMPLEMENTED
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
        if self.active_skill is None:
            return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)
        command = self.active_skill.step(state, dt)
        self.status = command.status
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            self.last_result = self.active_skill.result(state)
            self._append_result(self.last_result)
            self.active_skill = None
            self.active_request = None
        return command

    def stop(self, state: SceneState) -> SkillCommand:
        if self.active_skill is None:
            self.status = ExecutionStatus.STOPPED
            return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)
        command = self.active_skill.cancel(state)
        self.status = command.status
        self.last_result = self.active_skill.result(state)
        self._append_result(self.last_result)
        self.active_skill = None
        self.active_request = None
        return command

    def reset(self):
        self.active_skill = None
        self.active_request = None
        self.status = ExecutionStatus.IDLE

    def _append_result(self, result: SkillResult):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(result.to_json() + "\n")
