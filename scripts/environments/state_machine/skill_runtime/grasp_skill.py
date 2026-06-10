"""Generic phase-one grasp skill."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .base_skill import SkillCommand, finite_pose, format_pose, is_pose_reached, pose_error, pose_tensor, step_pose
from .scene_state_provider import PoseState, SceneState
from .skill_request import SkillRequest
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason, GraspState, SkillType
from .target_registry import GraspPlan, TargetRegistry


@dataclass
class GraspSkillConfig:
    position_threshold: float = 0.015
    orientation_threshold: float = 0.35
    stable_cycles: int = 3
    max_position_step: float = 0.020
    max_orientation_step: float = 0.16
    target_filter_alpha: float = 0.35
    max_target_jump: float = 0.120
    close_duration: float = 0.75
    verify_duration: float = 0.30
    state_timeout: float = 10.0
    lift_timeout: float = 5.0
    object_near_tcp_distance: float = 0.120
    min_lift_height: float = 0.050
    relative_motion_tolerance: float = 0.035
    workspace_lower: tuple[float, float, float] = (0.20, -0.55, 0.03)
    workspace_upper: tuple[float, float, float] = (0.95, 0.55, 0.70)


@dataclass
class _Runtime:
    state: GraspState = GraspState.IDLE
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    filtered_plan: GraspPlan | None = None
    locked_grasp_pose: PoseState | None = None
    locked_lift_pose: PoseState | None = None
    locked_object_pose: PoseState | None = None
    locked_tcp_object_offset: torch.Tensor | None = None
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    history: list[dict] = field(default_factory=list)


class GraspSkill:
    """Grasps a registered object using live target pose until gripper close."""

    def __init__(self, request: SkillRequest, registry: TargetRegistry, config: GraspSkillConfig | None = None):
        self.request = request
        self.registry = registry
        self.cfg = config or GraspSkillConfig()
        self.runtime = _Runtime()
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE

    @property
    def current_state(self) -> str:
        return self.runtime.state.value

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        now = state.sim_time
        self.runtime = _Runtime(
            state=GraspState.ACQUIRE_TARGET,
            start_time=now,
            state_start_time=now,
            last_command_pose=state.robot.tcp_pose,
        )
        self._record_transition(state, GraspState.IDLE, GraspState.ACQUIRE_TARGET)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return SkillCommand(state.robot.tcp_pose, self._close_command(), self.status)

        plan = self._update_plan(state) if self.runtime.state in {
            GraspState.ACQUIRE_TARGET,
            GraspState.MOVE_TO_PRE_GRASP,
            GraspState.ALIGN_GRASP,
            GraspState.APPROACH_GRASP,
        } else self.runtime.filtered_plan

        if plan is None or not plan.valid:
            self._fail(state, plan.failure_reason if plan else FailureReason.TARGET_LOST, plan.message if plan else "")
            return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

        if self.runtime.state == GraspState.ACQUIRE_TARGET:
            self._transition(state, GraspState.MOVE_TO_PRE_GRASP)

        command_pose = state.robot.tcp_pose
        gripper = plan.gripper_open_command
        desired = plan.pre_grasp_pose

        if self.runtime.state == GraspState.MOVE_TO_PRE_GRASP:
            # First translate above the object while preserving the current TCP
            # orientation. The next state performs the stricter orientation
            # alignment at that safer height.
            desired = PoseState(plan.pre_grasp_pose.pos_w, state.robot.tcp_pose.quat_w)
            command_pose = self._bounded_command(state, desired)
            self._advance_position_when_reached(state, desired, GraspState.APPROACH_GRASP)
        elif self.runtime.state == GraspState.ALIGN_GRASP:
            desired = plan.pre_grasp_pose
            if is_pose_reached(pose_error(state.robot.tcp_pose, desired), self.cfg.position_threshold, self.cfg.orientation_threshold):
                self._transition(state, GraspState.APPROACH_GRASP)
                desired = PoseState(plan.grasp_pose.pos_w, state.robot.tcp_pose.quat_w)
                command_pose = self._bounded_command(state, desired)
            else:
                command_pose = self._bounded_command(state, desired)
        elif self.runtime.state == GraspState.APPROACH_GRASP:
            desired = PoseState(plan.grasp_pose.pos_w, state.robot.tcp_pose.quat_w)
            command_pose = self._bounded_command(state, desired)
            if self._advance_position_when_reached(state, desired, GraspState.CLOSE_GRIPPER):
                self.runtime.locked_grasp_pose = desired
                self.runtime.locked_object_pose = plan.target_pose
        elif self.runtime.state == GraspState.CLOSE_GRIPPER:
            gripper = plan.gripper_close_command
            command_pose = self.runtime.locked_grasp_pose or plan.grasp_pose
            if self._state_elapsed(state) >= self.cfg.close_duration:
                if state.robot.gripper_width <= plan.min_gripper_width:
                    self._fail(state, FailureReason.GRASP_VERIFICATION_FAILED, "gripper closed empty")
                else:
                    obj_pose = self._current_object_pose(state)
                    if obj_pose is None:
                        self._fail(state, FailureReason.TARGET_LOST, "target disappeared during close")
                    else:
                        self.runtime.locked_object_pose = obj_pose
                        self.runtime.locked_tcp_object_offset = obj_pose.pos_w - state.robot.tcp_pose.pos_w
                        lift_pos = state.robot.tcp_pose.pos_w + torch.tensor(
                            [0.0, 0.0, plan.lift_distance], device=state.robot.tcp_pose.pos_w.device
                        )
                        self.runtime.locked_lift_pose = PoseState(lift_pos, state.robot.tcp_pose.quat_w)
                        self._transition(state, GraspState.VERIFY_GRASP)
        elif self.runtime.state == GraspState.VERIFY_GRASP:
            gripper = plan.gripper_close_command
            command_pose = self.runtime.locked_grasp_pose or state.robot.tcp_pose
            if self._state_elapsed(state) >= self.cfg.verify_duration:
                if self._verify_pre_lift(state, plan):
                    self._transition(state, GraspState.LIFT)
                else:
                    self._fail(state, FailureReason.GRASP_VERIFICATION_FAILED, "pre-lift verification failed")
        elif self.runtime.state == GraspState.LIFT:
            gripper = plan.gripper_close_command
            desired = self.runtime.locked_lift_pose or state.robot.tcp_pose
            command_pose = self._bounded_command(state, desired)
            self._advance_when_reached(state, desired, GraspState.VERIFY_LIFT, timeout=self.cfg.lift_timeout)
        elif self.runtime.state == GraspState.VERIFY_LIFT:
            gripper = plan.gripper_close_command
            command_pose = self.runtime.locked_lift_pose or state.robot.tcp_pose
            if self._verify_lift(state, plan):
                self._transition(state, GraspState.HOLD)
                self._succeed(state)
            elif self._state_elapsed(state) > self.cfg.verify_duration + 1.0:
                self._fail(state, FailureReason.OBJECT_DROPPED, "object did not remain stable after lift")
        elif self.runtime.state == GraspState.HOLD:
            gripper = plan.gripper_close_command
            command_pose = self.runtime.locked_lift_pose or state.robot.tcp_pose

        self.runtime.last_command_pose = command_pose
        self.runtime.final_error_pos = pose_error(state.robot.tcp_pose, desired).position
        self.runtime.final_error_ori = pose_error(state.robot.tcp_pose, desired).orientation
        return SkillCommand(command_pose, gripper, self.status)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, GraspState.CANCELLED)
        return SkillCommand(state.robot.tcp_pose, state.robot.gripper_width, self.status)

    def result(self, state: SceneState) -> SkillResult:
        obj_pose = self._current_object_pose(state)
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=SkillType.GRASP,
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

    def _update_plan(self, state: SceneState) -> GraspPlan:
        plan = self.registry.compute_grasp_plan(self.request.source_object or "", state)
        if not plan.valid:
            return plan
        previous = self.runtime.filtered_plan
        if previous is None:
            self.runtime.filtered_plan = plan
            return plan
        jump = torch.linalg.norm(plan.grasp_pose.pos_w - previous.grasp_pose.pos_w)
        if float(jump) > self.cfg.max_target_jump:
            plan.valid = False
            plan.failure_reason = FailureReason.TARGET_LOST
            plan.message = f"target jump exceeded limit: {float(jump):.3f} m"
            return plan
        alpha = self.cfg.target_filter_alpha
        grasp_pos = previous.grasp_pose.pos_w * (1.0 - alpha) + plan.grasp_pose.pos_w * alpha
        pre_pos = previous.pre_grasp_pose.pos_w * (1.0 - alpha) + plan.pre_grasp_pose.pos_w * alpha
        filtered = GraspPlan(
            target_name=plan.target_name,
            target_pose=plan.target_pose,
            grasp_pose=PoseState(grasp_pos, plan.grasp_pose.quat_w),
            pre_grasp_pose=PoseState(pre_pos, plan.pre_grasp_pose.quat_w),
            approach_dir_w=plan.approach_dir_w,
            lift_distance=plan.lift_distance,
            gripper_open_command=plan.gripper_open_command,
            gripper_close_command=plan.gripper_close_command,
            min_gripper_width=plan.min_gripper_width,
        )
        self.runtime.filtered_plan = filtered
        return filtered

    def _bounded_command(self, state: SceneState, desired: PoseState) -> PoseState:
        command_from = self.runtime.last_command_pose or state.robot.tcp_pose
        lower = torch.tensor(self.cfg.workspace_lower, dtype=torch.float32, device=desired.pos_w.device)
        upper = torch.tensor(self.cfg.workspace_upper, dtype=torch.float32, device=desired.pos_w.device)
        bounded = PoseState(torch.minimum(torch.maximum(desired.pos_w, lower), upper), desired.quat_w)
        if not finite_pose(bounded):
            self._fail(state, FailureReason.IK_UNREACHABLE, "non-finite target pose")
            return state.robot.tcp_pose
        return step_pose(command_from, bounded, self.cfg.max_position_step, self.cfg.max_orientation_step)

    def _advance_when_reached(
        self, state: SceneState, desired: PoseState, next_state: GraspState, timeout: float | None = None
    ) -> bool:
        error = pose_error(state.robot.tcp_pose, desired)
        if is_pose_reached(error, self.cfg.position_threshold, self.cfg.orientation_threshold):
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        allowed = self.cfg.state_timeout if timeout is None else timeout
        if self._state_elapsed(state) > allowed:
            reason = FailureReason.POSITION_TIMEOUT if error.position > self.cfg.position_threshold else FailureReason.ORIENTATION_TIMEOUT
            self._fail(state, reason, f"pose error p={error.position:.4f}, q={error.orientation:.4f}")
        return False

    def _advance_position_when_reached(self, state: SceneState, desired: PoseState, next_state: GraspState) -> bool:
        error = pose_error(state.robot.tcp_pose, desired)
        if error.position <= self.cfg.position_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self.cfg.state_timeout:
            self._fail(state, FailureReason.POSITION_TIMEOUT, f"position error p={error.position:.4f}")
        return False

    def _verify_pre_lift(self, state: SceneState, plan: GraspPlan) -> bool:
        obj_pose = self._current_object_pose(state)
        if obj_pose is None:
            return False
        near_tcp = torch.linalg.norm(obj_pose.pos_w - state.robot.tcp_pose.pos_w) <= self.cfg.object_near_tcp_distance
        return bool(near_tcp and state.robot.gripper_width > plan.min_gripper_width)

    def _verify_lift(self, state: SceneState, plan: GraspPlan) -> bool:
        obj_pose = self._current_object_pose(state)
        locked_obj = self.runtime.locked_object_pose
        locked_offset = self.runtime.locked_tcp_object_offset
        if obj_pose is None or locked_obj is None or locked_offset is None:
            return False
        height_gain = float((obj_pose.pos_w[2] - locked_obj.pos_w[2]).detach().cpu())
        relative_offset = obj_pose.pos_w - state.robot.tcp_pose.pos_w
        relative_error = float(torch.linalg.norm(relative_offset - locked_offset).detach().cpu())
        stable = height_gain >= self.cfg.min_lift_height and relative_error <= self.cfg.relative_motion_tolerance
        if stable and state.robot.gripper_width > plan.min_gripper_width:
            self.runtime.stable_count += 1
        else:
            self.runtime.stable_count = 0
        return self.runtime.stable_count >= self.cfg.stable_cycles

    def _current_object_pose(self, state: SceneState) -> PoseState | None:
        if self.request.source_object not in state.objects:
            return None
        return state.objects[self.request.source_object].pose

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _close_command(self) -> float:
        plan = self.runtime.filtered_plan
        return -1.0 if plan is None else plan.gripper_close_command

    def _transition(self, state: SceneState, new_state: GraspState):
        old_state = self.runtime.state
        if old_state == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record_transition(state, old_state, new_state)

    def _record_transition(self, state: SceneState, old_state: GraspState, new_state: GraspState):
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": SkillType.GRASP.value,
            "target": self.request.source_object,
            "from": old_state.value,
            "to": new_state.value,
            "tcp_pose": format_pose(state.robot.tcp_pose),
            "gripper_width": round(state.robot.gripper_width, 5),
        }
        self.runtime.history.append(record)
        print(f"[SkillRuntime] transition {record}")

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self._transition(state, GraspState.FAILED)
        print(f"[SkillRuntime] failure request={self.request.request_id} reason={reason.value} {message}")

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, GraspState.SUCCEEDED)
        print(f"[SkillRuntime] success request={self.request.request_id} target={self.request.source_object}")
