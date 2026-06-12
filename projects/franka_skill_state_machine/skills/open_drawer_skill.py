"""General IK-based OPEN-DRAWER skill (no learned policy, no joint-target cheating).

Given a target drawer, the skill reaches the handle, grips it, and pulls along the drawer's opening
direction until the drawer joint passes the success threshold. The handle pose AND orientation are
read live every step (they move as the drawer slides and may differ per cabinet), and the pull/grasp
geometry is derived from the live handle-vs-cabinet vector, so the same skill works for the top /
middle / bottom drawer and for differently shaped cabinets. The drawer is opened ONLY by physical
interaction; ``drawer_joint_target`` is always None and ``set_cabinet_joint_target`` is never called.

Output: joint commands (q_des from the encapsulated DLS IK) for the joint-position env.
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
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType


@dataclass
class OpenDrawerIKConfig:
    pre_grasp_clearance: float = 0.12  # back-off along open dir before grasping
    pull_lead: float = 0.08  # how far ahead of the live handle to aim while pulling
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    reach_pos_threshold: float = 0.015
    reach_stable_cycles: int = 6
    close_duration: float = 1.0
    reach_timeout: float = 16.0
    pull_timeout: float = 16.0
    home_joint_threshold: float = 0.12  # rad: arm considered "home" when ||q - q_home|| below this
    home_timeout: float = 6.0


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    drawer_control_mode: str = "ik_pull"
    drawer_joint_name: str = ""
    drawer_joint_target: float | None = None  # always None (physical pull only)
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


class OpenDrawerIKSkill:
    backend = "ik_pull"

    def __init__(self, request: SkillRequest, env, ik_adapter, config: OpenDrawerIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or OpenDrawerIKConfig()
        self.target_drawer = request.destination_object or "top_drawer"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        cfg = DRAWER_TARGETS.get(self.target_drawer, {})
        self.success_threshold = cfg.get("success_threshold", 0.20)
        self.runtime = _Runtime(drawer_joint_name=cfg.get("joint_name", ""))
        self.obs_adapter: SelectedDrawerObsAdapter | None = None
        self.last_q = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    # ---- live scene reads -------------------------------------------------
    def _handle_pos(self) -> torch.Tensor:
        return self.obs_adapter.selected_handle_pos_w()[self.adapter.env_id]

    def _cabinet_quat(self) -> torch.Tensor:
        return self.env.unwrapped.scene["cabinet"].data.root_quat_w[self.adapter.env_id]

    def _drawer_pos(self) -> float:
        return self.obs_adapter.selected_drawer_joint_pos()

    def _grasp_pose(self, lead: float) -> PoseState:
        """Live grasp/pull target: handle + lead*open_dir, with a derived front-grasp orientation.

        open_dir comes from the cabinet orientation (same for every drawer), so the grasp orientation
        is identical for top/middle/bottom.
        """
        handle = self._handle_pos()
        open_dir = open_direction_world(self._cabinet_quat())
        quat = grasp_quat_from_open_dir(open_dir, handle.device)
        return PoseState(handle + open_dir * lead, quat)

    # ---- skill API --------------------------------------------------------
    def start(self, state: SceneState):
        cfg = DRAWER_TARGETS.get(self.target_drawer)
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if cfg is None:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown target_drawer '{self.target_drawer}'")
            return
        self.obs_adapter = SelectedDrawerObsAdapter(self.env, self.target_drawer, env_id=self.adapter.env_id)
        # neutral seed config: the robot's default arm pose (a good IK seed; avoids the joint6 limit
        # solution the DLS IK falls into from an awkward post-place start).
        robot = self.env.unwrapped.scene["robot"]
        self.home_q = robot.data.default_joint_pos[self.adapter.env_id, self.adapter._joint_ids].clone()
        self.runtime = _Runtime(
            state="MOVE_TO_HOME",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            drawer_joint_name=cfg["joint_name"],
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_joint_pos = self._drawer_pos()
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self._record(state, "IDLE", "MOVE_TO_HOME")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)

        self.runtime.current_joint_pos = self._drawer_pos()
        gripper = 1.0
        target = None

        if self.runtime.state == "MOVE_TO_HOME":
            return self._home_step(state)
        if self.runtime.state == "MOVE_TO_PRE_GRASP":
            target = self._grasp_pose(self.cfg.pre_grasp_clearance)
            gripper = 1.0
            if self._advance_when_reached(state, target, "APPROACH", self.cfg.reach_timeout):
                pass
        elif self.runtime.state == "APPROACH":
            target = self._grasp_pose(0.0)
            gripper = 1.0
            self._advance_when_reached(state, target, "CLOSE_GRIPPER", self.cfg.reach_timeout)
        elif self.runtime.state == "CLOSE_GRIPPER":
            target = self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self.cfg.close_duration:
                self._transition(state, "PULL")
        elif self.runtime.state == "PULL":
            target = self._grasp_pose(self.cfg.pull_lead)  # aim a bit beyond the live handle, outward
            gripper = -1.0
            if self.runtime.current_joint_pos >= self.success_threshold:
                self._succeed(state)
            elif self._state_elapsed(state) > self.cfg.pull_timeout:
                self._fail(
                    state,
                    FailureReason.DRAWER_OPEN_TIMEOUT,
                    f"pull did not open {self.target_drawer}: pos={self.runtime.current_joint_pos:.4f} "
                    f"thr={self.success_threshold:.2f}",
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

    # ---- helpers ----------------------------------------------------------
    def _home_step(self, state: SceneState) -> SkillCommand:
        """Drive the arm to the neutral seed config (direct joint target, gripper open)."""
        arm_q = state.robot.joint_pos[self.adapter._joint_ids]
        dist = float(torch.linalg.norm(arm_q - self.home_q))
        if dist <= self.cfg.home_joint_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= 3:
                self._transition(state, "MOVE_TO_PRE_GRASP")
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self.cfg.home_timeout:
            self._transition(state, "MOVE_TO_PRE_GRASP")
        self.last_q = self.home_q.clone()
        return SkillCommand(
            state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
            joint_target=self.home_q.clone(), drawer_joint_target=None,
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
        print(f"[OpenDrawerIKSkill] {rec}", flush=True)

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
        print(f"[OpenDrawerIKSkill] success target={self.target_drawer} drawer_pos={self.runtime.current_joint_pos:.4f}", flush=True)
