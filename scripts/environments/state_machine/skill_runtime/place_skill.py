"""Phase-two place skill placeholder."""

from __future__ import annotations

from .base_skill import SkillCommand
from .scene_state_provider import SceneState
from .skill_request import SkillRequest
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason


class PlaceSkill:
    def __init__(self, request: SkillRequest):
        self.request = request
        self.status = ExecutionStatus.NOT_IMPLEMENTED

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.request.source_object,
            success=False,
            final_status=ExecutionStatus.NOT_IMPLEMENTED,
            failure_reason=FailureReason.NOT_IMPLEMENTED.value,
            elapsed_time=0.0,
        )
