"""IK-based OPEN/CLOSE microwave-door skills (revolute door, arc trajectory).

General, state-machine-callable door skills that mirror the open/close DRAWER skills, but follow a
circular ARC about the door hinge instead of a straight pull/push. The state machine only provides
the target door name (e.g. "microwave"); the hinge position/axis, the live handle pose, and the door
angle are all read live from the articulation, so the same skill works for any revolute door.

Geometry (calibrated, microwave_door_config):
  * door = articulation body ``link_0``; link_0's body origin IS the hinge -> hinge_w = link_0 pos_w.
  * hinge axis (world) = R(link_0_quat) @ local_Y  (= world +Z for the microwave: a vertical door).
  * handle (graspable free edge) = combine_frame_transforms(link_0 pos_w, link_0 quat_w, offset).
  * door outward face normal (world) = R(link_0_quat) @ (0,0,-1).

Opening physically rotates the handle about the hinge so the door joint angle increases (validated:
on this microwave joint+ corresponds to rotating the handle about world -Z). The door is opened ONLY
by physical interaction; the door joint target is never commanded.

Output: joint commands (q_des from the encapsulated DLS IK) for the joint-position env.

ASSET CAVEAT: the current microwave asset's door/body collision hulls overlap, so the door physically
opens only ~8-10deg before contact stops it (open_success_angle is set accordingly). The skill logic
is general; raise the threshold once the asset collision is fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import isaaclab.utils.math as math_utils

from runtime.base_skill import SkillCommand, PoseState, pose_error, pose_tensor, step_pose
from runtime.microwave_door_config import DOOR_TARGETS
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType


@dataclass
class DoorIKConfig:
    pre_grasp_clearance: float = 0.08   # back-off along the approach line before grasping
    approach_pitch_deg: float = 55.0    # tilt the forward approach downward (Franka's strongest reach)
    arc_lead_deg: float = 12.0          # how far ahead of the live door angle to aim while sweeping
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    reach_pos_threshold: float = 0.03
    reach_stable_cycles: int = 5
    close_duration: float = 1.0
    reach_timeout: float = 16.0
    sweep_timeout: float = 18.0
    home_joint_threshold: float = 0.12
    home_timeout: float = 6.0


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    door_control_mode: str = "ik_arc"
    door_joint_name: str = ""
    drawer_joint_target: float | None = None  # never set (physical interaction only)
    initial_angle: float = 0.0
    current_angle: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


def _rotate_about_z(vec: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a 3-vector about world +Z by ``angle`` rad (z component unchanged)."""
    c, s = math.cos(angle), math.sin(angle)
    x = c * vec[0] - s * vec[1]
    y = s * vec[0] + c * vec[1]
    return torch.stack((x, y, vec[2]))


def _grasp_quat_vertical_edge(approach: torch.Tensor, device) -> torch.Tensor:
    """TCP quaternion for grasping a VERTICAL handle bar, given the (horizontal) approach direction.

    approach (TCP +Z) = the given direction (gripper moves along it onto the bar);
    bar axis (TCP +X) = world up (the free edge is vertical);
    finger-open axis (TCP +Y) = horizontal, so the fingers straddle the bar and their closing
    contact normal lies in the horizontal plane (it can push the door along the opening tangent).

    We use a FORWARD-ish approach (toward the hinge along the door radial) rather than a face-on +Y
    approach, because the Franka reaches a forward-pointing gripper far more comfortably.
    """
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    z = approach / torch.linalg.norm(approach)
    x = up - torch.dot(up, z) * z  # up component perpendicular to approach -> bar runs vertical
    nx = torch.linalg.norm(x)
    if float(nx) < 1e-6:
        x = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        x = x / nx
    y = torch.linalg.cross(z, x)
    R = torch.stack((x, y, z), dim=1)
    return math_utils.quat_from_matrix(R.unsqueeze(0))[0]


class DoorIKSkill:
    """Shared open/close revolute-door skill; subclasses set the sweep direction + success test."""

    backend = "ik_arc"
    is_open = True

    def __init__(self, request: SkillRequest, env, ik_adapter, config: DoorIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or DoorIKConfig()
        self.target_door = request.destination_object or "microwave"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        dcfg = DOOR_TARGETS.get(self.target_door, {})
        self.asset_name = dcfg.get("asset_name", "microwave")
        self.door_joint_name = dcfg.get("joint_name", "joint_0")
        self.door_link_name = dcfg.get("link_name", "link_0")
        self.handle_offset = torch.tensor(
            dcfg.get("handle_offset", (0.0, 0.0, 0.0)), dtype=torch.float32, device=ik_adapter.env.unwrapped.device
        )
        self.open_success_angle = dcfg.get("open_success_angle", 0.14)
        self.close_success_angle = dcfg.get("close_success_angle", 0.03)
        self.runtime = _Runtime(door_joint_name=self.door_joint_name)
        self.last_q = None
        self._asset = None
        self._door_idx = None
        self._joint_id = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    # ---- live scene reads -------------------------------------------------
    def _bind_asset(self):
        self._asset = self.env.unwrapped.scene[self.asset_name]
        bnames = list(self._asset.data.body_names)
        self._door_idx = next((i for i, n in enumerate(bnames) if n == self.door_link_name), None)
        if self._door_idx is None:
            self._door_idx = next((i for i, n in enumerate(bnames) if self.door_link_name in n), 0)
        jnames = list(self._asset.data.joint_names)
        self._joint_id = jnames.index(self.door_joint_name) if self.door_joint_name in jnames else 0

    def _door_pose(self):
        eid = self.adapter.env_id
        return self._asset.data.body_pos_w[eid], self._asset.data.body_quat_w[eid]

    def _link0(self):
        eid = self.adapter.env_id
        return (
            self._asset.data.body_pos_w[eid, self._door_idx],
            self._asset.data.body_quat_w[eid, self._door_idx],
        )

    def _hinge_pos(self) -> torch.Tensor:
        return self._link0()[0]

    def _door_angle(self) -> float:
        return float(self._asset.data.joint_pos[self.adapter.env_id, self._joint_id])

    def _handle_pos(self) -> torch.Tensor:
        link_pos, link_quat = self._link0()
        handle, _ = math_utils.combine_frame_transforms(
            link_pos.unsqueeze(0), link_quat.unsqueeze(0), self.handle_offset.unsqueeze(0)
        )
        return handle[0]

    def _radial_dir(self) -> torch.Tensor:
        """Horizontal unit vector from the hinge to the handle (outward along the door)."""
        r = (self._handle_pos() - self._hinge_pos()).clone()
        r[2] = 0.0
        n = torch.linalg.norm(r)
        if float(n) < 1e-6:
            return torch.tensor([-1.0, 0.0, 0.0], device=r.device)
        return r / n

    def _approach_dir(self) -> torch.Tensor:
        """Forward+down approach: toward the hinge along the door radial, pitched down for a natural
        Franka wrist."""
        radial = self._radial_dir()
        p = math.radians(self.cfg.approach_pitch_deg)
        approach = -radial * math.cos(p) + torch.tensor([0.0, 0.0, -1.0], device=radial.device) * math.sin(p)
        return approach / torch.linalg.norm(approach)

    def _grasp_pose(self, clearance: float) -> PoseState:
        """Handle target with a forward+down approach grasp of the vertical free edge.

        The gripper approaches along ``_approach_dir`` (toward the hinge, pitched down); ``clearance``
        backs the pre-grasp off along -approach so the gripper comes in from the open / upper side.
        """
        handle = self._handle_pos()
        approach = self._approach_dir()
        quat = _grasp_quat_vertical_edge(approach, handle.device)
        return PoseState(handle - approach * clearance, quat)

    def _arc_target(self) -> PoseState:
        """Aim a lead angle ahead of the live door angle, along the opening (or closing) arc."""
        hinge = self._hinge_pos()
        handle = self._handle_pos()
        approach = self._approach_dir()
        lead = math.radians(self.cfg.arc_lead_deg) * self._sweep_sign()
        target_handle = hinge + _rotate_about_z(handle - hinge, lead)
        quat = _grasp_quat_vertical_edge(approach, handle.device)
        return PoseState(target_handle, quat)

    # ---- subclass hooks ---------------------------------------------------
    def _sweep_sign(self) -> float:
        """Sign of the world-Z rotation that drives the door in the desired direction.

        Validated for the microwave: door OPENS (joint+) when the handle rotates about world -Z, so
        the open sweep uses a negative Z-rotation; closing uses the opposite.
        """
        return -1.0 if self.is_open else +1.0

    def _reached_goal(self) -> bool:
        if self.is_open:
            return self.runtime.current_angle >= self.open_success_angle
        return self.runtime.current_angle <= self.close_success_angle

    # ---- skill API --------------------------------------------------------
    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if self.target_door not in DOOR_TARGETS:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown door '{self.target_door}'")
            return
        self._bind_asset()
        robot = self.env.unwrapped.scene["robot"]
        self.home_q = robot.data.default_joint_pos[self.adapter.env_id, self.adapter._joint_ids].clone()
        self.runtime = _Runtime(
            state="MOVE_TO_HOME",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            door_joint_name=self.door_joint_name,
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_angle = self._door_angle()
        self.runtime.current_angle = self.runtime.initial_angle
        self._record(state, "IDLE", "MOVE_TO_HOME")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)

        self.runtime.current_angle = self._door_angle()
        gripper = 1.0
        target = None

        if self.runtime.state == "MOVE_TO_HOME":
            return self._home_step(state)
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
                self._transition(state, "SWEEP")
        elif self.runtime.state == "SWEEP":
            target = self._arc_target()
            gripper = -1.0
            if self._reached_goal():
                self._succeed(state)
            elif self._state_elapsed(state) > self.cfg.sweep_timeout:
                reason = FailureReason.DOOR_OPEN_TIMEOUT if self.is_open else FailureReason.DOOR_CLOSE_TIMEOUT
                self._fail(
                    state, reason,
                    f"door sweep timeout {self.target_door}: angle={self.runtime.current_angle:.4f}",
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
            target_name=self.target_door,
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
            return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint",
                                joint_target=q, drawer_joint_target=None)
        self.last_q = ik.q_des
        return SkillCommand(cmd, gripper, self.status, control_mode="joint", joint_target=ik.q_des,
                            drawer_joint_target=None)

    def _hold(self, state: SceneState, gripper: float = -1.0) -> SkillCommand:
        q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
        return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint",
                            joint_target=q, drawer_joint_target=None)

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
            "target_door": self.target_door,
            "door_joint_name": self.runtime.door_joint_name,
            "from": old,
            "to": new,
            "door_angle": round(self.runtime.current_angle, 5),
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(rec)
        print(f"[DoorIKSkill] {rec}", flush=True)

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
        print(f"[DoorIKSkill] success {self.request.skill_type.value} target={self.target_door} "
              f"angle={self.runtime.current_angle:.4f}", flush=True)


class OpenDoorIKSkill(DoorIKSkill):
    backend = "ik_arc"
    is_open = True


class CloseDoorIKSkill(DoorIKSkill):
    backend = "ik_arc"
    is_open = False
