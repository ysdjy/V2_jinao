"""Low-cost place skill for env-local support-surface points."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from isaaclab.utils import math as math_utils

from .base_skill import SkillCommand, finite_pose, format_pose, pose_error, pose_tensor, step_pose
from .scene_state_provider import SceneState
from .scene_state_provider import PoseState
from .skill_request import SkillRequest
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason, SkillType


OBJECT_SUPPORT_OFFSET_Z = {
    "cube_1": 0.0203,
    "cube_2": 0.0203,
    "cube_3": 0.0203,
    "knife": 0.095,
}
PLACE_CLEARANCE = 0.002
PRE_PLACE_HEIGHT = 0.100


@dataclass
class PlacePlan:
    point_name: str
    held_object_name: str
    target_surface_xyz: list[float]
    support_offset_z: float
    target_pose: PoseState
    pre_object_pose: PoseState
    place_tcp_pose: PoseState
    pre_place_tcp_pose: PoseState


@dataclass
class PlaceSkillConfig:
    position_threshold: float = 0.012
    orientation_threshold: float = math.radians(8.0)
    stable_cycles: int = 5
    max_position_step: float = 0.012
    max_orientation_step: float = math.radians(5.0)
    descend_position_step: float = 0.006
    descend_orientation_step: float = math.radians(4.0)
    open_duration: float = 0.45
    state_timeout: float = 10.0


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    plan: PlacePlan | None = None
    filtered_plan: PlacePlan | None = None
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


class PlaceSkill:
    def __init__(self, request: SkillRequest, config: PlaceSkillConfig | None = None):
        self.request = request
        self.cfg = config or PlaceSkillConfig()
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.runtime = _Runtime()

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        self.runtime = _Runtime(
            state="MOVE_TO_PRE_PLACE",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            last_command_pose=state.robot.tcp_pose,
        )
        plan = self._make_plan(state)
        if plan is None:
            return
        self.runtime.plan = plan
        self.runtime.filtered_plan = plan
        self._log_plan(plan)
        self._record_transition(state, "IDLE", "MOVE_TO_PRE_PLACE")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

        plan = self.runtime.plan
        if plan is None:
            self._fail(state, FailureReason.REQUEST_INVALID, "place plan is missing")
            return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

        desired = state.robot.tcp_pose
        command_pose = state.robot.tcp_pose
        gripper = -1.0

        if self.runtime.state == "MOVE_TO_PRE_PLACE":
            desired = plan.pre_place_tcp_pose
            command_pose = self._bounded_command(
                state,
                desired,
                self.cfg.max_position_step,
                self.cfg.max_orientation_step,
            )
            self._advance_when_reached(state, desired, "DESCEND_TO_PLACE")
        elif self.runtime.state == "DESCEND_TO_PLACE":
            desired = plan.place_tcp_pose
            command_pose = self._bounded_command(
                state,
                desired,
                self.cfg.descend_position_step,
                self.cfg.descend_orientation_step,
            )
            self._advance_when_reached(state, desired, "OPEN_GRIPPER")
        elif self.runtime.state == "OPEN_GRIPPER":
            desired = plan.place_tcp_pose
            command_pose = desired
            gripper = 1.0
            if self._state_elapsed(state) >= self.cfg.open_duration:
                self._succeed(state)

        error = pose_error(state.robot.tcp_pose, desired)
        self.runtime.final_error_pos = error.position
        self.runtime.final_error_ori = error.orientation
        self.runtime.last_command_pose = command_pose
        return SkillCommand(command_pose, gripper, self.status)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

    def result(self, state: SceneState) -> SkillResult:
        obj_pose = None
        if self.request.source_object in state.objects:
            obj_pose = state.objects[self.request.source_object or ""].pose
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.request.source_object,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=self.failure_reason.value or None,
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
            final_tcp_pose=pose_tensor(state.robot.tcp_pose),
            final_object_pose=pose_tensor(obj_pose),
            position_error=self.runtime.final_error_pos,
            orientation_error=self.runtime.final_error_ori,
            gripper_width=state.robot.gripper_width,
            state_history=self.runtime.history,
        )

    def _make_plan(self, state: SceneState) -> PlacePlan | None:
        held_name = self.request.source_object
        if not held_name:
            self._fail(state, FailureReason.REQUEST_INVALID, "place request missing source_object")
            return None
        held = state.objects.get(held_name)
        if held is None:
            self._fail(state, FailureReason.TARGET_LOST, f"held object not found: {held_name}")
            return None
        if held_name not in OBJECT_SUPPORT_OFFSET_Z:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unsupported place object: {held_name}")
            return None

        surface_xyz = self._parse_surface_xyz(state)
        if surface_xyz is None:
            return None

        device = state.env_origin_w.device
        support_offset_z = OBJECT_SUPPORT_OFFSET_Z[held_name]
        object_target_local = torch.tensor(
            [
                surface_xyz[0],
                surface_xyz[1],
                surface_xyz[2] + support_offset_z + PLACE_CLEARANCE,
            ],
            dtype=torch.float32,
            device=device,
        )
        object_target_pos_w = state.env_origin_w + object_target_local
        object_target_quat_w = math_utils.normalize(held.pose.quat_w.unsqueeze(0))[0]
        pre_object_pos_w = object_target_pos_w + torch.tensor([0.0, 0.0, PRE_PLACE_HEIGHT], dtype=torch.float32, device=device)

        object_to_tcp_pos, object_to_tcp_quat = math_utils.subtract_frame_transforms(
            held.pose.pos_w.unsqueeze(0),
            object_target_quat_w.unsqueeze(0),
            state.robot.tcp_pose.pos_w.unsqueeze(0),
            state.robot.tcp_pose.quat_w.unsqueeze(0),
        )
        place_tcp_pos, place_tcp_quat = math_utils.combine_frame_transforms(
            object_target_pos_w.unsqueeze(0),
            object_target_quat_w.unsqueeze(0),
            object_to_tcp_pos,
            object_to_tcp_quat,
        )
        pre_tcp_pos, pre_tcp_quat = math_utils.combine_frame_transforms(
            pre_object_pos_w.unsqueeze(0),
            object_target_quat_w.unsqueeze(0),
            object_to_tcp_pos,
            object_to_tcp_quat,
        )
        target_pose = PoseState(object_target_pos_w, object_target_quat_w)
        pre_object_pose = PoseState(pre_object_pos_w, object_target_quat_w)
        place_tcp_pose = PoseState(place_tcp_pos[0], math_utils.normalize(place_tcp_quat)[0])
        pre_place_tcp_pose = PoseState(pre_tcp_pos[0], math_utils.normalize(pre_tcp_quat)[0])
        if not all(finite_pose(pose) for pose in (target_pose, pre_object_pose, place_tcp_pose, pre_place_tcp_pose)):
            self._fail(state, FailureReason.REQUEST_INVALID, "computed place target contains non-finite values")
            return None
        return PlacePlan(
            point_name=self.request.destination_object or "point",
            held_object_name=held_name,
            target_surface_xyz=surface_xyz,
            support_offset_z=support_offset_z,
            target_pose=target_pose,
            pre_object_pose=pre_object_pose,
            place_tcp_pose=place_tcp_pose,
            pre_place_tcp_pose=pre_place_tcp_pose,
        )

    def _parse_surface_xyz(self, state: SceneState) -> list[float] | None:
        value = self.request.parameters.get("target_surface_xyz")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            self._fail(state, FailureReason.REQUEST_INVALID, "target_surface_xyz must contain exactly three values")
            return None
        try:
            xyz = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            self._fail(state, FailureReason.REQUEST_INVALID, "target_surface_xyz values must be numeric")
            return None
        if not all(math.isfinite(v) for v in xyz):
            self._fail(state, FailureReason.REQUEST_INVALID, "target_surface_xyz values must be finite")
            return None
        return xyz

    def _bounded_command(
        self,
        state: SceneState,
        desired: PoseState,
        max_position_step: float,
        max_rotation_step: float,
    ) -> PoseState:
        command_from = self.runtime.last_command_pose or state.robot.tcp_pose
        return step_pose(command_from, desired, max_position_step, max_rotation_step)

    def _advance_when_reached(self, state: SceneState, desired: PoseState, next_state: str) -> bool:
        error = pose_error(state.robot.tcp_pose, desired)
        if error.position <= self.cfg.position_threshold and error.orientation <= self.cfg.orientation_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self.cfg.state_timeout:
            self._fail(
                state,
                FailureReason.POSITION_TIMEOUT,
                f"place pose error p={error.position:.4f}, orientation_error_deg={math.degrees(error.orientation):.2f}",
            )
        return False

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state: SceneState, new_state: str):
        old_state = self.runtime.state
        if old_state == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record_transition(state, old_state, new_state)

    def _record_transition(self, state: SceneState, old_state: str, new_state: str):
        plan = self.runtime.plan
        command_pose = self.runtime.last_command_pose or state.robot.tcp_pose
        error = pose_error(state.robot.tcp_pose, command_pose)
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": SkillType.PLACE.value,
            "target": self.request.source_object,
            "from": old_state,
            "to": new_state,
            "object_target_pose": None if plan is None else format_pose(plan.target_pose),
            "tcp_pose": format_pose(state.robot.tcp_pose),
            "pre_place_tcp_pose": None if plan is None else format_pose(plan.pre_place_tcp_pose),
            "place_tcp_pose": None if plan is None else format_pose(plan.place_tcp_pose),
            "position_error": round(error.position, 5),
            "orientation_error_deg": round(math.degrees(error.orientation), 3),
            "gripper_width": round(state.robot.gripper_width, 5),
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[PlaceSkill] transition {record}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")
        print(f"[PlaceSkill] failure request={self.request.request_id} reason={reason.value} {message}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(f"[PlaceSkill] success request={self.request.request_id} target={self.request.source_object}", flush=True)

    def _log_plan(self, plan: PlacePlan):
        record = {
            "point_name": plan.point_name,
            "target_surface_xyz": plan.target_surface_xyz,
            "support_offset_z": plan.support_offset_z,
            "place_clearance": PLACE_CLEARANCE,
            "resolved_object_target_pos_w": self._tensor_list(plan.target_pose.pos_w),
            "resolved_tcp_target_pos_w": self._tensor_list(plan.place_tcp_pose.pos_w),
        }
        print(f"[PlaceSkill] target {record}", flush=True)

    def _tensor_list(self, tensor: torch.Tensor) -> list[float]:
        return [round(float(v), 5) for v in tensor.detach().cpu().tolist()]
