"""General IK-based CLOSE-DRAWER skill (no learned policy, no joint-target cheating).

Mirror of OpenDrawerIKSkill: reach the (currently open) drawer's handle, grip it, and push along the
drawer's CLOSING direction (toward the cabinet body) until the drawer joint returns near closed.
Handle pose + orientation are read live every step; geometry is derived from the live handle-vs-cabinet
vector, so it works for any drawer / cabinet. ``drawer_joint_target`` stays None;
``set_cabinet_joint_target`` is never called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from runtime.base_skill import SkillCommand, PoseState, pose_error, pose_tensor, step_pose
from runtime.drawer_ik_common import grasp_quat_from_open_dir, open_direction_world
from runtime.drawer_obs_adapter import SelectedDrawerObsAdapter
from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason


@dataclass
class CloseDrawerIKConfig:
    pre_grasp_clearance: float = 0.0  # no OUTWARD back-off: would drag a gripped/free drawer further open
    push_lead: float = 0.12  # how far toward the cabinet to aim while pushing (seat it fully closed)
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    reach_pos_threshold: float = 0.02
    reach_stable_cycles: int = 6
    close_duration: float = 1.0
    reach_timeout: float = 10.0
    push_timeout: float = 8.0
    close_success_threshold: float = 0.01  # drawer considered closed when joint <= this (less gap)


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    drawer_control_mode: str = "ik_push"
    drawer_joint_name: str = ""
    drawer_joint_target: float | None = None
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


class CloseDrawerIKSkill:
    backend = "ik_pull"

    def __init__(self, request: SkillRequest, env, ik_adapter, config: CloseDrawerIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or CloseDrawerIKConfig()
        self.target_drawer = request.destination_object or "top_drawer"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        cfg = DRAWER_TARGETS.get(self.target_drawer, {})
        self.runtime = _Runtime(drawer_joint_name=cfg.get("joint_name", ""))
        self.obs_adapter: SelectedDrawerObsAdapter | None = None
        self.last_q = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def _handle_pos(self) -> torch.Tensor:
        return self.obs_adapter.selected_handle_pos_w()[self.adapter.env_id]

    def _cabinet_quat(self) -> torch.Tensor:
        return self.env.unwrapped.scene["cabinet"].data.root_quat_w[self.adapter.env_id]

    def _drawer_pos(self) -> float:
        return self.obs_adapter.selected_drawer_joint_pos()

    def _grasp_pose(self, lead: float) -> PoseState:
        """Live target: handle + lead*open_dir. lead>0 = outward (pre-grasp), lead<0 = toward cabinet (push).

        open_dir comes from the cabinet orientation (same for every drawer).
        """
        handle = self._handle_pos()
        open_dir = open_direction_world(self._cabinet_quat())
        quat = grasp_quat_from_open_dir(open_dir, handle.device)
        return PoseState(handle + open_dir * lead, quat)

    def start(self, state: SceneState):
        cfg = DRAWER_TARGETS.get(self.target_drawer)
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if cfg is None:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown target_drawer '{self.target_drawer}'")
            return
        self.obs_adapter = SelectedDrawerObsAdapter(self.env, self.target_drawer, env_id=self.adapter.env_id)
        self.runtime = _Runtime(
            state="MOVE_TO_PRE_GRASP",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            drawer_joint_name=cfg["joint_name"],
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_joint_pos = self._drawer_pos()
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self._record(state, "IDLE", "MOVE_TO_PRE_GRASP")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)

        self.runtime.current_joint_pos = self._drawer_pos()
        gripper = 1.0
        target = None

        if self.runtime.state == "MOVE_TO_PRE_GRASP":
            target = self._grasp_pose(self.cfg.pre_grasp_clearance)
            gripper = 1.0
            self._advance_when_reached(state, target, "APPROACH", self.cfg.reach_timeout)
        elif self.runtime.state == "APPROACH":
            target = self._grasp_pose(0.0)
            gripper = 1.0
            self._advance_when_reached(state, target, "CLOSE_GRIPPER", self.cfg.reach_timeout)
        elif self.runtime.state == "CLOSE_GRIPPER":
            target = self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self.cfg.close_duration:
                self._transition(state, "PUSH")
        elif self.runtime.state == "PUSH":
            target = self._grasp_pose(-self.cfg.push_lead)  # aim toward the cabinet (closing)
            gripper = -1.0
            if self.runtime.current_joint_pos <= self.cfg.close_success_threshold:
                self._succeed(state)
            elif self._state_elapsed(state) > self.cfg.push_timeout:
                self._fail(
                    state,
                    FailureReason.DRAWER_CLOSE_TIMEOUT,
                    f"push did not close {self.target_drawer}: pos={self.runtime.current_joint_pos:.4f}",
                )

        return self._command(state, target, gripper)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return self._hold(state)

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.target_drawer,
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

    def _command(self, state: SceneState, target: PoseState | None, gripper: float) -> SkillCommand:
        if target is None:
            return self._hold(state, gripper)
        cmd = step_pose(state.robot.tcp_pose, target, self.cfg.max_pos_step, self.cfg.max_ori_step)
        ik = self.adapter.solve(cmd)
        err = pose_error(state.robot.tcp_pose, target)
        self.runtime.final_error_pos = err.position
        self.runtime.final_error_ori = err.orientation
        self.runtime.last_command_pose = cmd
        if not ik.success:
            q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
            return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint", joint_target=q,
                                drawer_joint_target=None)
        self.last_q = ik.q_des
        return SkillCommand(cmd, gripper, self.status, control_mode="joint", joint_target=ik.q_des, drawer_joint_target=None)

    def _hold(self, state: SceneState, gripper: float = -1.0) -> SkillCommand:
        q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
        return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint", joint_target=q,
                            drawer_joint_target=None)

    def _advance_when_reached(self, state: SceneState, target: PoseState, next_state: str, timeout: float) -> bool:
        err = pose_error(state.robot.tcp_pose, target)
        if err.position <= self.cfg.reach_pos_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.reach_stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > timeout:
            self._fail(state, FailureReason.POSITION_TIMEOUT, f"{self.runtime.state} reach timeout (err={err.position:.3f})")
        return False

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state: SceneState, new_state: str):
        old = self.runtime.state
        if old == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record(state, old, new_state)

    def _record(self, state: SceneState, old: str, new: str):
        rec = {
            "time": round(state.sim_time, 4),
            "skill": self.request.skill_type.value,
            "backend": self.backend,
            "target_drawer": self.target_drawer,
            "drawer_joint_name": self.runtime.drawer_joint_name,
            "from": old,
            "to": new,
            "drawer_joint_pos": round(self.runtime.current_joint_pos, 5),
            "drawer_joint_target": None,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(rec)
        print(f"[CloseDrawerIKSkill] {rec}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(f"[CloseDrawerIKSkill] success target={self.target_drawer} drawer_pos={self.runtime.current_joint_pos:.4f}", flush=True)
