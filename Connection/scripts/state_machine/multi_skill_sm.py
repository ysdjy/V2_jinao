# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the cabinet-only Connection scene with a task-space state machine.

Active station:
  * Cabinet: open cabinet 44853's bottom drawer, place the knife, close it.

.. code-block:: bash

    ./isaaclab.sh -p Connection/scripts/state_machine/multi_skill_sm.py --num_envs 1
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Connection cabinet + knife Franka state machine.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized copies of the cabinet scene.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many env steps if > 0.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=900, help="Number of steps to record.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence
import os

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors import FrameTransformer

from connection_tasks.robots.cabinet_44853 import CABINET_BOTTOM_DRAWER_JOINT
import connection_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_ID = "Connection-Multi-Skill-Franka-IK-Abs-v0"

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
TOP_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)


def _quat_tensor(quat: tuple[float, float, float, float], device: torch.device | str, num_envs: int) -> torch.Tensor:
    return torch.tensor(quat, device=device).repeat(num_envs, 1)

def _make_pose(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos, quat], dim=-1)


def _distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


def _offset_pose_local(pos: torch.Tensor, quat: torch.Tensor, local_offset: torch.Tensor) -> torch.Tensor:
    return pos + math_utils.quat_apply(quat, local_offset)


class DoorOpenCloseSm:
    """Task-space state machine for a hinged appliance door."""

    REST = 0
    APPROACH_FRONT = 1
    APPROACH_NEAR = 2
    GRASP = 3
    OPEN = 4
    HOLD_OPEN = 5
    CLOSE = 6
    RELEASE = 7
    RETREAT = 8
    DONE = 9

    STATE_NAMES = {
        REST: "REST",
        APPROACH_FRONT: "APPROACH_FRONT",
        APPROACH_NEAR: "APPROACH_NEAR",
        GRASP: "GRASP",
        OPEN: "OPEN",
        HOLD_OPEN: "HOLD_OPEN",
        CLOSE: "CLOSE",
        RELEASE: "RELEASE",
        RETREAT: "RETREAT",
        DONE: "DONE",
    }

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        open_offset: tuple[float, float, float],
        close_offset: tuple[float, float, float],
        open_threshold: float = 0.35,
        close_threshold: float = 0.05,
        position_threshold: float = 0.06,
    ):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.position_threshold = position_threshold
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.progress = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_des_pos = torch.zeros((num_envs, 3), device=device)
        self.stable_steps = {
            self.APPROACH_FRONT: 6,
            self.APPROACH_NEAR: 6,
            self.GRASP: 10,
            self.OPEN: 4,
            self.HOLD_OPEN: 10,
            self.CLOSE: 4,
            self.RELEASE: 6,
            self.RETREAT: 6,
        }
        self.front_offset = torch.tensor([-0.18, 0.0, 0.04], device=device).repeat(num_envs, 1)
        self.near_offset = torch.tensor([-0.07, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.grasp_offset = torch.tensor([0.018, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.open_offset = torch.tensor(open_offset, device=device).repeat(num_envs, 1)
        self.close_offset = torch.tensor(close_offset, device=device).repeat(num_envs, 1)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = self.REST
        self.progress[env_ids] = 0

    def compute(
        self,
        ee_pose_b: torch.Tensor,
        handle_pose_b: torch.Tensor,
        joint_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        des_pos = ee_pose_b[:, :3].clone()
        des_quat = handle_pose_b[:, 3:].clone()
        gripper = torch.full((self.num_envs,), GRIPPER_OPEN, device=self.device)

        hp = handle_pose_b[:, :3]
        hq = handle_pose_b[:, 3:]
        for i in range(self.num_envs):
            state = int(self.state[i].item())
            if state == self.REST:
                des_pos[i] = ee_pose_b[i, :3]
                des_quat[i] = ee_pose_b[i, 3:]
                self._advance(i, self.APPROACH_FRONT)
            elif state == self.APPROACH_FRONT:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.front_offset[i : i + 1])[0]
                if self._stable(i, self._at_target(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.APPROACH_NEAR)
            elif state == self.APPROACH_NEAR:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.near_offset[i : i + 1])[0]
                if self._stable(i, self._at_target(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.GRASP)
            elif state == self.GRASP:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.grasp_offset[i : i + 1])[0]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at_target(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.OPEN)
            elif state == self.OPEN:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.open_offset[i : i + 1])[0]
                gripper[i] = GRIPPER_CLOSE
                opened = joint_pos is None or abs(float(joint_pos[i].item())) >= self.open_threshold
                if self._stable(i, opened, state):
                    self._advance(i, self.HOLD_OPEN)
            elif state == self.HOLD_OPEN:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.grasp_offset[i : i + 1])[0]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at_target(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.CLOSE)
            elif state == self.CLOSE:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.close_offset[i : i + 1])[0]
                gripper[i] = GRIPPER_CLOSE
                closed = joint_pos is None or abs(float(joint_pos[i].item())) <= self.close_threshold
                if self._stable(i, closed, state):
                    self._advance(i, self.RELEASE)
            elif state == self.RELEASE:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.front_offset[i : i + 1])[0]
                gripper[i] = GRIPPER_OPEN
                if self._stable(i, self._at_target(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.RETREAT)
            elif state == self.RETREAT:
                des_pos[i] = _offset_pose_local(hp[i : i + 1], hq[i : i + 1], self.front_offset[i : i + 1])[0]
                gripper[i] = GRIPPER_OPEN
                if self._stable(i, self._at_target(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.DONE)
            else:
                des_pos[i] = ee_pose_b[i, :3]
                des_quat[i] = ee_pose_b[i, 3:]
                gripper[i] = GRIPPER_OPEN

        self.last_des_pos = des_pos.clone()
        return torch.cat([_make_pose(des_pos, des_quat), gripper.unsqueeze(-1)], dim=-1)

    def _at_target(self, env_id: int, cur_pos: torch.Tensor, des_pos: torch.Tensor) -> bool:
        return bool(_distance(cur_pos[env_id : env_id + 1], des_pos[env_id : env_id + 1])[0] < self.position_threshold)

    def _stable(self, env_id: int, condition: bool, state: int) -> bool:
        self.progress[env_id] = self.progress[env_id] + 1 if condition else 0
        return bool(self.progress[env_id] >= self.stable_steps[state])

    def _advance(self, env_id: int, next_state: int):
        self.state[env_id] = next_state
        self.progress[env_id] = 0


class CabinetKnifeSm:
    """Open bottom drawer, place knife, then close the drawer."""

    REST = 0
    APPROACH_DRAWER_FRONT = 1
    APPROACH_DRAWER_HANDLE = 2
    GRASP_DRAWER = 3
    OPEN_DRAWER = 4
    RELEASE_DRAWER = 5
    APPROACH_KNIFE_ABOVE = 6
    APPROACH_KNIFE = 7
    GRASP_KNIFE = 8
    LIFT_KNIFE = 9
    MOVE_TO_DRAWER = 10
    LOWER_IN_DRAWER = 11
    RELEASE_KNIFE = 12
    RETREAT_FROM_DRAWER = 13
    APPROACH_HANDLE_TO_CLOSE = 14
    GRASP_HANDLE_TO_CLOSE = 15
    CLOSE_DRAWER = 16
    RELEASE_HANDLE = 17
    DONE = 18

    STATE_NAMES = {
        REST: "REST",
        APPROACH_DRAWER_FRONT: "APPROACH_DRAWER_FRONT",
        APPROACH_DRAWER_HANDLE: "APPROACH_DRAWER_HANDLE",
        GRASP_DRAWER: "GRASP_DRAWER",
        OPEN_DRAWER: "OPEN_DRAWER",
        RELEASE_DRAWER: "RELEASE_DRAWER",
        APPROACH_KNIFE_ABOVE: "APPROACH_KNIFE_ABOVE",
        APPROACH_KNIFE: "APPROACH_KNIFE",
        GRASP_KNIFE: "GRASP_KNIFE",
        LIFT_KNIFE: "LIFT_KNIFE",
        MOVE_TO_DRAWER: "MOVE_TO_DRAWER",
        LOWER_IN_DRAWER: "LOWER_IN_DRAWER",
        RELEASE_KNIFE: "RELEASE_KNIFE",
        RETREAT_FROM_DRAWER: "RETREAT_FROM_DRAWER",
        APPROACH_HANDLE_TO_CLOSE: "APPROACH_HANDLE_TO_CLOSE",
        GRASP_HANDLE_TO_CLOSE: "GRASP_HANDLE_TO_CLOSE",
        CLOSE_DRAWER: "CLOSE_DRAWER",
        RELEASE_HANDLE: "RELEASE_HANDLE",
        DONE: "DONE",
    }

    def __init__(self, dt: float, num_envs: int, device: torch.device | str, position_threshold: float = 0.06):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.progress = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_des_pos = torch.zeros((num_envs, 3), device=device)
        self.stable_steps = {
            self.APPROACH_DRAWER_FRONT: 6,
            self.APPROACH_DRAWER_HANDLE: 6,
            self.GRASP_DRAWER: 10,
            self.OPEN_DRAWER: 4,
            self.RELEASE_DRAWER: 6,
            self.APPROACH_KNIFE_ABOVE: 6,
            self.APPROACH_KNIFE: 6,
            self.GRASP_KNIFE: 10,
            self.LIFT_KNIFE: 6,
            self.MOVE_TO_DRAWER: 6,
            self.LOWER_IN_DRAWER: 6,
            self.RELEASE_KNIFE: 6,
            self.RETREAT_FROM_DRAWER: 6,
            self.APPROACH_HANDLE_TO_CLOSE: 6,
            self.GRASP_HANDLE_TO_CLOSE: 10,
            self.CLOSE_DRAWER: 4,
            self.RELEASE_HANDLE: 6,
        }
        self.drawer_front_offset = torch.tensor([-0.14, 0.0, 0.03], device=device).repeat(num_envs, 1)
        self.drawer_open_threshold = 0.18
        self.drawer_close_threshold = 0.025
        self.drawer_near_offset = torch.tensor([-0.055, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_grasp_offset = torch.tensor([0.025, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_open_offset = torch.tensor([-0.220, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_close_offset = torch.tensor([0.140, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.knife_above_offset = torch.tensor([0.0, 0.0, 0.16], device=device).repeat(num_envs, 1)
        self.knife_grasp_offset = torch.tensor([0.0, 0.0, 0.035], device=device).repeat(num_envs, 1)
        self.drawer_drop_offset = torch.tensor([0.16, 0.0, -0.060], device=device).repeat(num_envs, 1)
        self.drawer_drop_above_offset = torch.tensor([0.12, 0.0, 0.105], device=device).repeat(num_envs, 1)
        self.retreat_offset = torch.tensor([-0.14, 0.0, 0.12], device=device).repeat(num_envs, 1)
        self.knife_quat = _quat_tensor(TOP_DOWN_QUAT, device, num_envs)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = self.REST
        self.progress[env_ids] = 0

    def compute(
        self,
        ee_pose_b: torch.Tensor,
        drawer_pose_b: torch.Tensor,
        knife_pose_b: torch.Tensor,
        drawer_joint_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        des_pos = ee_pose_b[:, :3].clone()
        des_quat = drawer_pose_b[:, 3:].clone()
        gripper = torch.full((self.num_envs,), GRIPPER_OPEN, device=self.device)

        drawer_pos = drawer_pose_b[:, :3]
        drawer_handle_quat = drawer_pose_b[:, 3:]
        knife_pos = knife_pose_b[:, :3]
        for i in range(self.num_envs):
            state = int(self.state[i].item())
            if state == self.REST:
                des_pos[i] = ee_pose_b[i, :3]
                des_quat[i] = ee_pose_b[i, 3:]
                self._advance(i, self.APPROACH_DRAWER_FRONT)
            elif state == self.APPROACH_DRAWER_FRONT:
                des_pos[i] = drawer_pos[i] + self.drawer_front_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.APPROACH_DRAWER_HANDLE)
            elif state == self.APPROACH_DRAWER_HANDLE:
                des_pos[i] = drawer_pos[i] + self.drawer_near_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.GRASP_DRAWER)
            elif state == self.GRASP_DRAWER:
                des_pos[i] = drawer_pos[i] + self.drawer_grasp_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.OPEN_DRAWER)
            elif state == self.OPEN_DRAWER:
                des_pos[i] = drawer_pos[i] + self.drawer_open_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_CLOSE
                opened = drawer_joint_pos is None or abs(float(drawer_joint_pos[i].item())) >= self.drawer_open_threshold
                if self._stable(i, opened, state):
                    self._advance(i, self.RELEASE_DRAWER)
            elif state == self.RELEASE_DRAWER:
                des_pos[i] = drawer_pos[i] + self.drawer_front_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_OPEN
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.APPROACH_KNIFE_ABOVE)
            elif state == self.APPROACH_KNIFE_ABOVE:
                des_pos[i] = knife_pos[i] + self.knife_above_offset[i]
                des_quat[i] = self.knife_quat[i]
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.APPROACH_KNIFE)
            elif state == self.APPROACH_KNIFE:
                des_pos[i] = knife_pos[i] + self.knife_grasp_offset[i]
                des_quat[i] = self.knife_quat[i]
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.GRASP_KNIFE)
            elif state == self.GRASP_KNIFE:
                des_pos[i] = knife_pos[i] + self.knife_grasp_offset[i]
                des_quat[i] = self.knife_quat[i]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.LIFT_KNIFE)
            elif state == self.LIFT_KNIFE:
                des_pos[i] = knife_pos[i] + self.knife_above_offset[i]
                des_quat[i] = self.knife_quat[i]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.MOVE_TO_DRAWER)
            elif state == self.MOVE_TO_DRAWER:
                des_pos[i] = drawer_pos[i] + self.drawer_drop_above_offset[i]
                des_quat[i] = self.knife_quat[i]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.LOWER_IN_DRAWER)
            elif state == self.LOWER_IN_DRAWER:
                des_pos[i] = drawer_pos[i] + self.drawer_drop_offset[i]
                des_quat[i] = self.knife_quat[i]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.RELEASE_KNIFE)
            elif state == self.RELEASE_KNIFE:
                des_pos[i] = drawer_pos[i] + self.drawer_drop_offset[i]
                des_quat[i] = self.knife_quat[i]
                gripper[i] = GRIPPER_OPEN
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.RETREAT_FROM_DRAWER)
            elif state == self.RETREAT_FROM_DRAWER:
                des_pos[i] = drawer_pos[i] + self.retreat_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_OPEN
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.APPROACH_HANDLE_TO_CLOSE)
            elif state == self.APPROACH_HANDLE_TO_CLOSE:
                des_pos[i] = drawer_pos[i] + self.drawer_front_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.GRASP_HANDLE_TO_CLOSE)
            elif state == self.GRASP_HANDLE_TO_CLOSE:
                des_pos[i] = drawer_pos[i] + self.drawer_grasp_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_CLOSE
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.CLOSE_DRAWER)
            elif state == self.CLOSE_DRAWER:
                des_pos[i] = drawer_pos[i] + self.drawer_close_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_CLOSE
                closed = drawer_joint_pos is None or abs(float(drawer_joint_pos[i].item())) <= self.drawer_close_threshold
                if self._stable(i, closed, state):
                    self._advance(i, self.RELEASE_HANDLE)
            elif state == self.RELEASE_HANDLE:
                des_pos[i] = drawer_pos[i] + self.drawer_front_offset[i]
                des_quat[i] = drawer_handle_quat[i]
                gripper[i] = GRIPPER_OPEN
                if self._stable(i, self._at(i, ee_pose_b[:, :3], des_pos), state):
                    self._advance(i, self.DONE)
            else:
                des_pos[i] = ee_pose_b[i, :3]
                des_quat[i] = ee_pose_b[i, 3:]
                gripper[i] = GRIPPER_OPEN

        self.last_des_pos = des_pos.clone()
        return torch.cat([_make_pose(des_pos, des_quat), gripper.unsqueeze(-1)], dim=-1)

    def _at(self, env_id: int, cur_pos: torch.Tensor, des_pos: torch.Tensor) -> bool:
        return bool(_distance(cur_pos[env_id : env_id + 1], des_pos[env_id : env_id + 1])[0] < self.position_threshold)

    def _stable(self, env_id: int, condition: bool, state: int) -> bool:
        self.progress[env_id] = self.progress[env_id] + 1 if condition else 0
        return bool(self.progress[env_id] >= self.stable_steps[state])

    def _advance(self, env_id: int, next_state: int):
        self.state[env_id] = next_state
        self.progress[env_id] = 0


def _target_pose_in_robot_frame(env, frame_name: str, robot_name: str) -> torch.Tensor:
    frame: FrameTransformer = env.unwrapped.scene[frame_name]
    pos_w = frame.data.target_pos_w[..., 0, :].clone()
    quat_w = frame.data.target_quat_w[..., 0, :].clone()
    robot = env.unwrapped.scene[robot_name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        pos_w,
        quat_w,
    )
    return torch.cat([pos_b, quat_b], dim=-1)


def _pack_actions(env, cabinet_cmd: torch.Tensor) -> torch.Tensor:
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    idx = 0
    for name, dim in zip(env.unwrapped.action_manager.active_terms, env.unwrapped.action_manager.action_term_dim):
        if "arm_action" in name:
            actions[:, idx : idx + dim] = cabinet_cmd[:, :dim]
        elif "gripper_action" in name:
            actions[:, idx] = cabinet_cmd[:, -1]
        idx += dim
    return actions


def _state_names(sm, states: torch.Tensor) -> list[str]:
    return [sm.STATE_NAMES.get(int(s), str(int(s))) for s in states.detach().cpu().tolist()]


def main():
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "videos", "multi_skill")
        )
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[multi_skill_sm] recording video to {video_dir}", flush=True)

    env.reset()

    cabinet = env.unwrapped.scene["cabinet"]
    cabinet_joint_idx = int(cabinet.find_joints(CABINET_BOTTOM_DRAWER_JOINT)[0][0])

    dt = env_cfg.sim.dt * env_cfg.decimation
    cabinet_sm = CabinetKnifeSm(dt, env.unwrapped.num_envs, env.unwrapped.device)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    actions[:, 7] = GRIPPER_OPEN

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            cabinet_ee_pose = _target_pose_in_robot_frame(env, "cabinet_ee_frame", "cabinet_robot")
            cabinet_handle_pose = _target_pose_in_robot_frame(env, "cabinet_frame", "cabinet_robot")
            knife_pose = _target_pose_in_robot_frame(env, "knife_frame", "cabinet_robot")

            cabinet_cmd = cabinet_sm.compute(
                cabinet_ee_pose,
                cabinet_handle_pose,
                knife_pose,
                cabinet.data.joint_pos[:, cabinet_joint_idx],
            )
            actions = _pack_actions(env, cabinet_cmd)

            if step_count % 120 == 0:
                drawer = cabinet.data.joint_pos[:, cabinet_joint_idx].cpu().tolist()
                cabinet_dist = _distance(cabinet_ee_pose[:, :3], cabinet_handle_pose[:, :3]).cpu().tolist()
                cabinet_err = _distance(cabinet_ee_pose[:, :3], cabinet_sm.last_des_pos).cpu().tolist()
                print(
                    "[multi_skill_sm] "
                    f"step={step_count:4d} "
                    f"cabinet={_state_names(cabinet_sm, cabinet_sm.state)} drawer={[round(v, 3) for v in drawer]} "
                    f"ee_h={[round(v, 3) for v in cabinet_dist]} err={[round(v, 3) for v in cabinet_err]}",
                    flush=True,
                )

            if dones.any():
                env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                cabinet_sm.reset_idx(env_ids)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                print(f"[multi_skill_sm] reached max_steps={args_cli.max_steps}", flush=True)
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
