"""Bottom-drawer open/close skill using physical handle contact."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from isaaclab.utils import math as math_utils

from .base_skill import SkillCommand, finite_pose, format_pose, pose_error, pose_tensor, step_pose
from .scene_state_provider import PoseState, SceneState
from .skill_request import SkillRequest
from .skill_result import SkillResult
from .skill_types import ExecutionStatus, FailureReason, SkillType


CABINET_SCALE = 0.62

BOTTOM_DRAWER_LINK = "link_1"
BOTTOM_DRAWER_JOINT = "joint_0"

BOTTOM_HANDLE_LOCAL_POS_UNSCALED = (
    0.11946,
    0.01491,
    1.06183,
)
BOTTOM_HANDLE_LOCAL_POS = tuple(value * CABINET_SCALE for value in BOTTOM_HANDLE_LOCAL_POS_UNSCALED)
BOTTOM_HANDLE_LOCAL_QUAT = (
    1.0,
    0.0,
    0.0,
    0.0,
)


@dataclass
class DrawerPlan:
    handle_pose: PoseState
    pre_handle_pose: PoseState
    grasp_pose: PoseState
    pull_pose: PoseState
    push_contact_pose: PoseState
    push_target_pose: PoseState
    retreat_pose: PoseState
    outward: torch.Tensor
    initial_joint_pos: float


@dataclass
class DrawerSkillConfig:
    pre_distance: float = 0.10
    pull_distance: float = 0.22
    retreat_distance: float = 0.08
    push_margin: float = 0.04
    push_contact_offset: float = 0.005

    move_position_step: float = 0.010
    move_orientation_step: float = math.radians(4.0)
    contact_position_step: float = 0.004
    pull_position_step: float = 0.005
    push_position_step: float = 0.005

    position_threshold: float = 0.010
    orientation_threshold: float = math.radians(7.0)
    stable_cycles: int = 5

    settle_duration: float = 0.20
    close_gripper_duration: float = 0.50
    release_duration: float = 0.40
    state_timeout: float = 12.0

    open_joint_threshold: float = 0.15
    closed_joint_threshold: float = 0.03
    min_handle_grasp_width: float = 0.008


@dataclass
class DrawerRuntime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    plan: DrawerPlan | None = None
    filtered_plan: DrawerPlan | None = None
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    initial_joint_pos: float = 0.0
    max_joint_displacement: float = 0.0
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


class DrawerSkill:
    def __init__(self, request: SkillRequest, config: DrawerSkillConfig | None = None):
        self.request = request
        self.cfg = config or DrawerSkillConfig()
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.runtime = DrawerRuntime()

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        initial_state = "ACQUIRE_HANDLE" if self.request.skill_type == SkillType.OPEN_DRAWER else "ACQUIRE_PUSH_TARGET"
        self.runtime = DrawerRuntime(
            state=initial_state,
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            last_command_pose=state.robot.tcp_pose,
        )
        self._record_transition(state, "IDLE", initial_state)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return SkillCommand(self.runtime.last_command_pose or state.robot.tcp_pose, 1.0, self.status)

        if self.request.skill_type == SkillType.OPEN_DRAWER:
            return self._step_open(state)
        if self.request.skill_type == SkillType.CLOSE_DRAWER:
            return self._step_close(state)

        self._fail(state, FailureReason.REQUEST_INVALID, f"unsupported drawer skill: {self.request.skill_type}")
        return SkillCommand(state.robot.tcp_pose, 1.0, self.status)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return SkillCommand(state.robot.tcp_pose, 1.0, self.status)

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.request.destination_object,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=self.failure_reason.value or None,
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
            final_tcp_pose=pose_tensor(state.robot.tcp_pose),
            position_error=self.runtime.final_error_pos,
            orientation_error=self.runtime.final_error_ori,
            gripper_width=state.robot.gripper_width,
            state_history=self.runtime.history,
        )

    def _step_open(self, state: SceneState) -> SkillCommand:
        gripper = 1.0
        desired = state.robot.tcp_pose
        command_pose = state.robot.tcp_pose

        if self.runtime.state == "ACQUIRE_HANDLE":
            plan = self._make_plan(state)
            if plan is None:
                return SkillCommand(state.robot.tcp_pose, 1.0, self.status)
            self.runtime.plan = plan
            self.runtime.filtered_plan = plan
            self.runtime.initial_joint_pos = plan.initial_joint_pos
            self._log_plan(state, plan)
            if abs(plan.initial_joint_pos) >= self.cfg.open_joint_threshold:
                self._succeed(state)
                return SkillCommand(state.robot.tcp_pose, 1.0, self.status)
            self._transition(state, "MOVE_TO_PRE_GRASP")

        plan = self.runtime.plan
        if plan is None:
            self._fail(state, FailureReason.TARGET_LOST, "drawer plan missing")
            return SkillCommand(state.robot.tcp_pose, 1.0, self.status)

        if self.runtime.state == "MOVE_TO_PRE_GRASP":
            desired = plan.pre_handle_pose
            command_pose = self._bounded_command(state, desired, self.cfg.move_position_step, self.cfg.move_orientation_step)
            self._advance_when_reached(state, desired, "MOVE_TO_HANDLE", FailureReason.POSITION_TIMEOUT)
        elif self.runtime.state == "MOVE_TO_HANDLE":
            desired = plan.grasp_pose
            command_pose = self._bounded_command(state, desired, self.cfg.contact_position_step, self.cfg.move_orientation_step)
            self._advance_when_reached(state, desired, "SETTLE_AT_HANDLE", FailureReason.POSITION_TIMEOUT)
        elif self.runtime.state == "SETTLE_AT_HANDLE":
            desired = plan.grasp_pose
            command_pose = desired
            if self._state_elapsed(state) >= self.cfg.settle_duration:
                self._transition(state, "CLOSE_GRIPPER")
        elif self.runtime.state == "CLOSE_GRIPPER":
            desired = plan.grasp_pose
            command_pose = desired
            gripper = -1.0
            if self._state_elapsed(state) >= self.cfg.close_gripper_duration:
                self._transition(state, "VERIFY_HANDLE_GRASP")
        elif self.runtime.state == "VERIFY_HANDLE_GRASP":
            desired = plan.grasp_pose
            command_pose = desired
            gripper = -1.0
            if state.robot.gripper_width < self.cfg.min_handle_grasp_width:
                self._fail(
                    state,
                    FailureReason.HANDLE_GRASP_FAILED,
                    f"gripper closed empty width={state.robot.gripper_width:.5f}",
                )
            else:
                self._lock_pull_pose(state, plan)
                self._transition(state, "PULL_OPEN")
        elif self.runtime.state == "PULL_OPEN":
            desired = plan.pull_pose
            command_pose = self._bounded_command(state, desired, self.cfg.pull_position_step, self.cfg.move_orientation_step)
            gripper = -1.0
            self._advance_when_reached(state, desired, "VERIFY_OPEN", FailureReason.DRAWER_OPEN_TIMEOUT)
        elif self.runtime.state == "VERIFY_OPEN":
            desired = plan.pull_pose
            command_pose = desired
            gripper = -1.0
            if self._joint_displacement(state) >= self.cfg.open_joint_threshold:
                self._transition(state, "OPEN_GRIPPER")
            elif self._state_elapsed(state) > self.cfg.state_timeout:
                self._fail(state, FailureReason.DRAWER_OPEN_TIMEOUT, "drawer joint did not move enough")
        elif self.runtime.state == "OPEN_GRIPPER":
            desired = plan.pull_pose
            command_pose = desired
            if self._state_elapsed(state) >= self.cfg.release_duration:
                self._lock_retreat_pose(state, plan)
                self._transition(state, "RETREAT")
        elif self.runtime.state == "RETREAT":
            desired = plan.retreat_pose
            command_pose = self._bounded_command(state, desired, self.cfg.move_position_step, self.cfg.move_orientation_step)
            if self._advance_when_reached(state, desired, "SUCCEEDED", FailureReason.POSITION_TIMEOUT):
                self._succeed(state)

        return self._finish_step(state, desired, command_pose, gripper)

    def _step_close(self, state: SceneState) -> SkillCommand:
        gripper = 1.0
        desired = state.robot.tcp_pose
        command_pose = state.robot.tcp_pose

        if self.runtime.state == "ACQUIRE_PUSH_TARGET":
            plan = self._make_plan(state)
            if plan is None:
                return SkillCommand(state.robot.tcp_pose, 1.0, self.status)
            self.runtime.plan = plan
            self.runtime.filtered_plan = plan
            self.runtime.initial_joint_pos = plan.initial_joint_pos
            self._log_plan(state, plan)
            if abs(plan.initial_joint_pos) <= self.cfg.closed_joint_threshold:
                self._succeed(state)
                return SkillCommand(state.robot.tcp_pose, 1.0, self.status)
            self._transition(state, "MOVE_TO_PRE_PUSH")

        plan = self.runtime.plan
        if plan is None:
            self._fail(state, FailureReason.TARGET_LOST, "drawer plan missing")
            return SkillCommand(state.robot.tcp_pose, 1.0, self.status)

        if self.runtime.state == "MOVE_TO_PRE_PUSH":
            desired = plan.pre_handle_pose
            command_pose = self._bounded_command(state, desired, self.cfg.move_position_step, self.cfg.move_orientation_step)
            self._advance_when_reached(state, desired, "MOVE_TO_PUSH_CONTACT", FailureReason.POSITION_TIMEOUT)
        elif self.runtime.state == "MOVE_TO_PUSH_CONTACT":
            desired = plan.push_contact_pose
            command_pose = self._bounded_command(state, desired, self.cfg.contact_position_step, self.cfg.move_orientation_step)
            if self._advance_when_reached(state, desired, "PUSH_CLOSED", FailureReason.POSITION_TIMEOUT):
                self._lock_push_target(state, plan)
        elif self.runtime.state == "PUSH_CLOSED":
            desired = plan.push_target_pose
            command_pose = self._bounded_command(state, desired, self.cfg.push_position_step, self.cfg.move_orientation_step)
            self._advance_when_reached(state, desired, "VERIFY_CLOSED", FailureReason.DRAWER_CLOSE_TIMEOUT)
        elif self.runtime.state == "VERIFY_CLOSED":
            desired = plan.push_target_pose
            command_pose = desired
            joint_pos = self._drawer_joint_pos(state)
            if joint_pos is not None and abs(joint_pos) <= self.cfg.closed_joint_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= self.cfg.stable_cycles:
                    self._lock_retreat_pose(state, plan)
                    self._transition(state, "RETREAT")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.state_timeout:
                self._fail(state, FailureReason.DRAWER_CLOSE_TIMEOUT, "drawer joint did not close")
        elif self.runtime.state == "RETREAT":
            desired = plan.retreat_pose
            command_pose = self._bounded_command(state, desired, self.cfg.move_position_step, self.cfg.move_orientation_step)
            if self._advance_when_reached(state, desired, "SUCCEEDED", FailureReason.POSITION_TIMEOUT):
                self._succeed(state)

        return self._finish_step(state, desired, command_pose, gripper)

    def _make_plan(self, state: SceneState) -> DrawerPlan | None:
        if self.request.destination_object not in (None, "bottom_drawer"):
            self._fail(state, FailureReason.REQUEST_INVALID, "only bottom_drawer is supported")
            return None
        if self.request.parameters.get("drawer_link", BOTTOM_DRAWER_LINK) != BOTTOM_DRAWER_LINK:
            self._fail(state, FailureReason.REQUEST_INVALID, "only link_1 is supported")
            return None
        if self.request.parameters.get("drawer_joint", BOTTOM_DRAWER_JOINT) != BOTTOM_DRAWER_JOINT:
            self._fail(state, FailureReason.REQUEST_INVALID, "only joint_0 is supported")
            return None
        cabinet = state.objects.get("cabinet")
        if cabinet is None or BOTTOM_DRAWER_LINK not in cabinet.links:
            self._fail(state, FailureReason.TARGET_LOST, "cabinet link_1 pose missing")
            return None
        if BOTTOM_DRAWER_JOINT not in cabinet.joint_pos:
            self._fail(state, FailureReason.DRAWER_JOINT_MISSING, "joint_0 missing")
            return None

        drawer_link_pose = cabinet.links[BOTTOM_DRAWER_LINK]
        handle_pose = self._handle_pose(drawer_link_pose)
        if handle_pose is None:
            return None
        outward = handle_pose.pos_w - cabinet.pose.pos_w
        outward = outward.clone()
        outward[2] = 0.0
        outward_norm = torch.linalg.norm(outward)
        if float(outward_norm.detach().cpu()) < 0.05:
            self._fail(state, FailureReason.DRAWER_GEOMETRY_INVALID, "handle outward direction too small")
            return None
        outward = outward / outward_norm
        drawer_quat = self._drawer_grasp_quat(outward)
        initial_joint_pos = float(cabinet.joint_pos[BOTTOM_DRAWER_JOINT])

        pre_pos = handle_pose.pos_w + outward * self.cfg.pre_distance
        grasp_pose = PoseState(handle_pose.pos_w, drawer_quat)
        pre_handle_pose = PoseState(pre_pos, drawer_quat)
        pull_pose = PoseState(handle_pose.pos_w + outward * self.cfg.pull_distance, drawer_quat)
        push_contact_pose = PoseState(handle_pose.pos_w + outward * self.cfg.push_contact_offset, drawer_quat)
        push_target_pose = PoseState(
            handle_pose.pos_w - outward * (abs(initial_joint_pos) + self.cfg.push_margin),
            drawer_quat,
        )
        retreat_pose = PoseState(handle_pose.pos_w + outward * self.cfg.retreat_distance, drawer_quat)
        plan = DrawerPlan(
            handle_pose=PoseState(handle_pose.pos_w, handle_pose.quat_w),
            pre_handle_pose=pre_handle_pose,
            grasp_pose=grasp_pose,
            pull_pose=pull_pose,
            push_contact_pose=push_contact_pose,
            push_target_pose=push_target_pose,
            retreat_pose=retreat_pose,
            outward=outward,
            initial_joint_pos=initial_joint_pos,
        )
        if not all(
            finite_pose(pose)
            for pose in (
                plan.handle_pose,
                plan.pre_handle_pose,
                plan.grasp_pose,
                plan.pull_pose,
                plan.push_contact_pose,
                plan.push_target_pose,
                plan.retreat_pose,
            )
        ):
            self._fail(state, FailureReason.DRAWER_GEOMETRY_INVALID, "computed drawer pose is non-finite")
            return None
        return plan

    def _handle_pose(self, drawer_link_pose: PoseState) -> PoseState | None:
        device = drawer_link_pose.pos_w.device
        handle_local_pos = torch.tensor(BOTTOM_HANDLE_LOCAL_POS, dtype=torch.float32, device=device).unsqueeze(0)
        handle_local_quat = torch.tensor(BOTTOM_HANDLE_LOCAL_QUAT, dtype=torch.float32, device=device).unsqueeze(0)
        handle_pos_w, handle_quat_w = math_utils.combine_frame_transforms(
            drawer_link_pose.pos_w.unsqueeze(0),
            drawer_link_pose.quat_w.unsqueeze(0),
            handle_local_pos,
            handle_local_quat,
        )
        return PoseState(handle_pos_w[0], math_utils.normalize(handle_quat_w)[0])

    def _drawer_grasp_quat(self, outward: torch.Tensor) -> torch.Tensor:
        device = outward.device
        tool_z_w = -outward
        tool_y_w = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        tool_x_w = torch.linalg.cross(tool_y_w, tool_z_w, dim=0)
        tool_x_w = tool_x_w / torch.linalg.norm(tool_x_w).clamp_min(1.0e-6)
        tool_y_w = torch.linalg.cross(tool_z_w, tool_x_w, dim=0)
        tool_y_w = tool_y_w / torch.linalg.norm(tool_y_w).clamp_min(1.0e-6)
        rotation_matrix = torch.stack([tool_x_w, tool_y_w, tool_z_w], dim=-1)
        return math_utils.quat_from_matrix(rotation_matrix.unsqueeze(0))[0]

    def _bounded_command(
        self,
        state: SceneState,
        desired: PoseState,
        max_position_step: float,
        max_rotation_step: float,
    ) -> PoseState:
        command_from = self.runtime.last_command_pose or state.robot.tcp_pose
        return step_pose(command_from, desired, max_position_step, max_rotation_step)

    def _advance_when_reached(
        self,
        state: SceneState,
        desired: PoseState,
        next_state: str,
        timeout_reason: FailureReason,
    ) -> bool:
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
                timeout_reason,
                f"drawer pose error p={error.position:.4f}, orientation_error_deg={math.degrees(error.orientation):.2f}",
            )
        return False

    def _lock_pull_pose(self, state: SceneState, plan: DrawerPlan) -> None:
        plan.grasp_pose = PoseState(state.robot.tcp_pose.pos_w.clone(), state.robot.tcp_pose.quat_w.clone())
        plan.pull_pose = PoseState(
            state.robot.tcp_pose.pos_w + plan.outward * self.cfg.pull_distance,
            state.robot.tcp_pose.quat_w.clone(),
        )

    def _lock_push_target(self, state: SceneState, plan: DrawerPlan) -> None:
        joint_pos = self._drawer_joint_pos(state)
        push_distance = (abs(joint_pos) if joint_pos is not None else abs(plan.initial_joint_pos)) + self.cfg.push_margin
        plan.push_contact_pose = PoseState(state.robot.tcp_pose.pos_w.clone(), state.robot.tcp_pose.quat_w.clone())
        plan.push_target_pose = PoseState(
            state.robot.tcp_pose.pos_w - plan.outward * push_distance,
            state.robot.tcp_pose.quat_w.clone(),
        )

    def _lock_retreat_pose(self, state: SceneState, plan: DrawerPlan) -> None:
        plan.retreat_pose = PoseState(
            state.robot.tcp_pose.pos_w + plan.outward * self.cfg.retreat_distance,
            state.robot.tcp_pose.quat_w.clone(),
        )

    def _finish_step(
        self,
        state: SceneState,
        desired: PoseState,
        command_pose: PoseState,
        gripper: float,
    ) -> SkillCommand:
        error = pose_error(state.robot.tcp_pose, desired)
        self.runtime.final_error_pos = error.position
        self.runtime.final_error_ori = error.orientation
        self.runtime.last_command_pose = command_pose
        self._update_joint_displacement(state)
        return SkillCommand(command_pose, gripper, self.status)

    def _drawer_joint_pos(self, state: SceneState) -> float | None:
        cabinet = state.objects.get("cabinet")
        if cabinet is None:
            return None
        return cabinet.joint_pos.get(BOTTOM_DRAWER_JOINT)

    def _joint_displacement(self, state: SceneState) -> float:
        joint_pos = self._drawer_joint_pos(state)
        if joint_pos is None:
            return 0.0
        displacement = abs(joint_pos - self.runtime.initial_joint_pos)
        self.runtime.max_joint_displacement = max(self.runtime.max_joint_displacement, displacement)
        return displacement

    def _update_joint_displacement(self, state: SceneState) -> None:
        self._joint_displacement(state)

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
        joint_pos = self._drawer_joint_pos(state)
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": self.request.skill_type.value,
            "from": old_state,
            "to": new_state,
            "joint_0": None if joint_pos is None else round(joint_pos, 5),
            "tcp_position_error": round(error.position, 5),
            "tcp_orientation_error_deg": round(math.degrees(error.orientation), 3),
            "gripper_width": round(state.robot.gripper_width, 5),
            "handle_pose": None if plan is None else format_pose(plan.handle_pose),
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[DrawerSkill] transition {record}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")
        print(f"[DrawerSkill] failure request={self.request.request_id} reason={reason.value} {message}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(
            "[DrawerSkill] success "
            f"request={self.request.request_id} "
            f"max_joint_displacement={self.runtime.max_joint_displacement:.5f}",
            flush=True,
        )

    def _log_plan(self, state: SceneState, plan: DrawerPlan):
        cabinet = state.objects["cabinet"]
        drawer_link_pose = cabinet.links[BOTTOM_DRAWER_LINK]
        record = {
            "skill_type": self.request.skill_type.value,
            "drawer_link": BOTTOM_DRAWER_LINK,
            "drawer_joint": BOTTOM_DRAWER_JOINT,
            "drawer_joint_initial": round(plan.initial_joint_pos, 5),
            "cabinet_root_pose": format_pose(cabinet.pose),
            "drawer_link_pose": format_pose(drawer_link_pose),
            "computed_handle_pose": format_pose(plan.handle_pose),
            "outward": [round(float(v), 5) for v in plan.outward.detach().cpu().tolist()],
            "grasp_quaternion": [round(float(v), 5) for v in plan.grasp_pose.quat_w.detach().cpu().tolist()],
        }
        print(f"[DrawerSkill] plan {record}", flush=True)
