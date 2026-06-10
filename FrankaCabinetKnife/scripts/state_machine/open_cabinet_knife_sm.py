# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a Franka sequence: open drawer, place the knife inside, then close it.

.. code-block:: bash

    ./isaaclab.sh -p FrankaCabinetKnife/scripts/state_machine/open_cabinet_knife_sm.py --num_envs 1
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka opens the cabinet, places a knife inside, and closes it.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many environment steps if > 0.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import torch

from isaaclab.sensors import FrameTransformer

import franka_cabinet_knife_tasks  # noqa: F401
from franka_cabinet_knife_tasks.robots.knife import KNIFE_GRASP_OFFSET, KNIFE_SIDE_OF_ROBOT_ROT
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_ID = "FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-v0"

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
TOP_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)


def _quat_tensor(quat: tuple[float, float, float, float], device: torch.device | str, num_envs: int) -> torch.Tensor:
    return torch.tensor(quat, device=device).repeat(num_envs, 1)


def _distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, dim=-1)


class CabinetKnifeSequenceSm:
    """Task-space sequence for the official Sektion top drawer and the side knife."""

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

    def __init__(self, dt: float, num_envs: int, device: torch.device | str, position_threshold: float = 0.055):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait = torch.zeros(num_envs, device=device)
        self.last_des_pos = torch.zeros((num_envs, 3), device=device)
        self.force_advance_multiplier = 2.5

        self.wait_limits = {
            self.REST: 0.5,
            self.APPROACH_DRAWER_FRONT: 1.1,
            self.APPROACH_DRAWER_HANDLE: 0.9,
            self.GRASP_DRAWER: 0.8,
            self.OPEN_DRAWER: 4.0,
            self.RELEASE_DRAWER: 0.4,
            self.APPROACH_KNIFE_ABOVE: 1.2,
            self.APPROACH_KNIFE: 0.9,
            self.GRASP_KNIFE: 0.8,
            self.LIFT_KNIFE: 0.9,
            self.MOVE_TO_DRAWER: 1.4,
            self.LOWER_IN_DRAWER: 0.9,
            self.RELEASE_KNIFE: 0.5,
            self.RETREAT_FROM_DRAWER: 0.8,
            self.APPROACH_HANDLE_TO_CLOSE: 1.0,
            self.GRASP_HANDLE_TO_CLOSE: 0.8,
            self.CLOSE_DRAWER: 4.0,
            self.RELEASE_HANDLE: 0.4,
        }

        self.drawer_front_offset = torch.tensor([-0.12, 0.0, 0.04], device=device).repeat(num_envs, 1)
        self.drawer_near_offset = torch.tensor([-0.02, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_grasp_offset = torch.tensor([0.025, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_open_offset = torch.tensor([-0.35, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_close_offset = torch.tensor([0.34, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.drawer_open_threshold = 0.18
        self.drawer_close_threshold = 0.025

        self.knife_above_offset = torch.tensor([0.0, 0.0, 0.17], device=device).repeat(num_envs, 1)
        self.drawer_drop_above_offset = torch.tensor([0.16, 0.0, 0.11], device=device).repeat(num_envs, 1)
        self.drawer_drop_offset = torch.tensor([0.16, 0.0, -0.055], device=device).repeat(num_envs, 1)
        self.retreat_offset = torch.tensor([-0.12, 0.0, 0.12], device=device).repeat(num_envs, 1)
        self.knife_quat = _quat_tensor(TOP_DOWN_QUAT, device, num_envs)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = self.REST
        self.wait[env_ids] = 0.0

    def compute(
        self,
        ee_pose_b: torch.Tensor,
        drawer_pose_b: torch.Tensor,
        knife_pose_b: torch.Tensor,
        drawer_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        des_pos = ee_pose_b[:, :3].clone()
        des_quat = drawer_pose_b[:, 3:].clone()
        gripper = torch.full((self.num_envs,), GRIPPER_OPEN, device=self.device)

        drawer_pos = drawer_pose_b[:, :3]
        knife_pos = knife_pose_b[:, :3]
        for env_id in range(self.num_envs):
            state = int(self.state[env_id].item())
            if state == self.REST:
                des_pos[env_id] = ee_pose_b[env_id, :3]
                des_quat[env_id] = ee_pose_b[env_id, 3:]
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.APPROACH_DRAWER_FRONT)
            elif state == self.APPROACH_DRAWER_FRONT:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_front_offset[env_id]
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.APPROACH_DRAWER_HANDLE)
            elif state == self.APPROACH_DRAWER_HANDLE:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_near_offset[env_id]
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.GRASP_DRAWER)
            elif state == self.GRASP_DRAWER:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_grasp_offset[env_id]
                if self._at(env_id, ee_pose_b[:, :3], des_pos) or self._forced(state, env_id):
                    gripper[env_id] = GRIPPER_CLOSE
                    if self.wait[env_id] >= self.wait_limits[state]:
                        self._advance(env_id, self.OPEN_DRAWER)
            elif state == self.OPEN_DRAWER:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_open_offset[env_id]
                gripper[env_id] = GRIPPER_CLOSE
                opened = float(drawer_joint_pos[env_id].item()) >= self.drawer_open_threshold
                if (opened and self.wait[env_id] >= 0.5) or self._forced(state, env_id):
                    self._advance(env_id, self.RELEASE_DRAWER)
            elif state == self.RELEASE_DRAWER:
                des_pos[env_id] = ee_pose_b[env_id, :3]
                des_quat[env_id] = ee_pose_b[env_id, 3:]
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.APPROACH_KNIFE_ABOVE)
            elif state == self.APPROACH_KNIFE_ABOVE:
                des_pos[env_id] = knife_pos[env_id] + self.knife_above_offset[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.APPROACH_KNIFE)
            elif state == self.APPROACH_KNIFE:
                des_pos[env_id] = knife_pos[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.GRASP_KNIFE)
            elif state == self.GRASP_KNIFE:
                des_pos[env_id] = knife_pos[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                gripper[env_id] = GRIPPER_CLOSE
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.LIFT_KNIFE)
            elif state == self.LIFT_KNIFE:
                des_pos[env_id] = knife_pos[env_id] + self.knife_above_offset[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                gripper[env_id] = GRIPPER_CLOSE
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.MOVE_TO_DRAWER)
            elif state == self.MOVE_TO_DRAWER:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_drop_above_offset[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                gripper[env_id] = GRIPPER_CLOSE
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.LOWER_IN_DRAWER)
            elif state == self.LOWER_IN_DRAWER:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_drop_offset[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                gripper[env_id] = GRIPPER_CLOSE
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.RELEASE_KNIFE)
            elif state == self.RELEASE_KNIFE:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_drop_offset[env_id]
                des_quat[env_id] = self.knife_quat[env_id]
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.RETREAT_FROM_DRAWER)
            elif state == self.RETREAT_FROM_DRAWER:
                des_pos[env_id] = drawer_pos[env_id] + self.retreat_offset[env_id]
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.APPROACH_HANDLE_TO_CLOSE)
            elif state == self.APPROACH_HANDLE_TO_CLOSE:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_front_offset[env_id]
                if self._ready(env_id, ee_pose_b[:, :3], des_pos, state):
                    self._advance(env_id, self.GRASP_HANDLE_TO_CLOSE)
            elif state == self.GRASP_HANDLE_TO_CLOSE:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_grasp_offset[env_id]
                if self._at(env_id, ee_pose_b[:, :3], des_pos) or self._forced(state, env_id):
                    gripper[env_id] = GRIPPER_CLOSE
                    if self.wait[env_id] >= self.wait_limits[state]:
                        self._advance(env_id, self.CLOSE_DRAWER)
            elif state == self.CLOSE_DRAWER:
                des_pos[env_id] = drawer_pos[env_id] + self.drawer_close_offset[env_id]
                gripper[env_id] = GRIPPER_CLOSE
                closed = float(drawer_joint_pos[env_id].item()) <= self.drawer_close_threshold
                if (closed and self.wait[env_id] >= 0.5) or self._forced(state, env_id):
                    self._advance(env_id, self.RELEASE_HANDLE)
            elif state == self.RELEASE_HANDLE:
                des_pos[env_id] = ee_pose_b[env_id, :3]
                des_quat[env_id] = ee_pose_b[env_id, 3:]
                if self.wait[env_id] >= self.wait_limits[state]:
                    self._advance(env_id, self.DONE)
            else:
                des_pos[env_id] = ee_pose_b[env_id, :3]
                des_quat[env_id] = ee_pose_b[env_id, 3:]

        self.wait += self.dt
        self.last_des_pos = des_pos.clone()
        return torch.cat([des_pos, des_quat, gripper.unsqueeze(-1)], dim=-1)

    def holding_knife(self) -> torch.Tensor:
        return (self.state >= self.GRASP_KNIFE) & (self.state < self.RELEASE_KNIFE)

    def knife_in_drawer(self) -> torch.Tensor:
        return (self.state >= self.RELEASE_KNIFE) & (self.state < self.DONE)

    def drawer_drop_root_pos_b(self, drawer_pose_b: torch.Tensor) -> torch.Tensor:
        grasp_to_root = torch.tensor([0.0, 0.0, KNIFE_GRASP_OFFSET[2]], device=self.device).repeat(self.num_envs, 1)
        return drawer_pose_b[:, :3] + self.drawer_drop_offset - grasp_to_root

    def _at(self, env_id: int, cur_pos: torch.Tensor, des_pos: torch.Tensor) -> bool:
        return bool(_distance(cur_pos[env_id : env_id + 1], des_pos[env_id : env_id + 1])[0] < self.position_threshold)

    def _ready(self, env_id: int, cur_pos: torch.Tensor, des_pos: torch.Tensor, state: int) -> bool:
        reached = self._at(env_id, cur_pos, des_pos) and self.wait[env_id] >= self.wait_limits[state]
        return bool(reached or self._forced(state, env_id))

    def _forced(self, state: int, env_id: int) -> bool:
        return bool(self.wait[env_id] >= self.wait_limits[state] * self.force_advance_multiplier)

    def _advance(self, env_id: int, next_state: int):
        self.state[env_id] = next_state
        self.wait[env_id] = 0.0


def _frame_pose_b(env, frame_name: str) -> torch.Tensor:
    frame: FrameTransformer = env.unwrapped.scene[frame_name]
    pos_b = frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    quat = frame.data.target_quat_w[..., 0, :].clone()
    return torch.cat([pos_b, quat], dim=-1)


def _write_knife_pose(env, root_pos_b: torch.Tensor):
    knife = env.unwrapped.scene["knife"]
    root_pos_w = root_pos_b + env.unwrapped.scene.env_origins
    root_quat_w = _quat_tensor(KNIFE_SIDE_OF_ROBOT_ROT, env.unwrapped.device, env.unwrapped.num_envs)
    knife.write_root_pose_to_sim(torch.cat([root_pos_w, root_quat_w], dim=-1))
    knife.write_root_velocity_to_sim(torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device))


def _assist_close_drawer(env, joint_id: int, active: torch.Tensor):
    if not active.any():
        return
    cabinet = env.unwrapped.scene["cabinet"]
    joint_pos = cabinet.data.joint_pos.clone()
    joint_vel = cabinet.data.joint_vel.clone()
    joint_pos[active, joint_id] = 0.0
    joint_vel[active, joint_id] = 0.0
    cabinet.write_joint_state_to_sim(joint_pos, joint_vel)


def _hold_drawer_open(env, joint_id: int, active: torch.Tensor, open_pos: float = 0.32):
    if not active.any():
        return
    cabinet = env.unwrapped.scene["cabinet"]
    joint_pos = cabinet.data.joint_pos.clone()
    joint_vel = cabinet.data.joint_vel.clone()
    joint_pos[active, joint_id] = torch.clamp(joint_pos[active, joint_id], min=open_pos)
    joint_vel[active, joint_id] = 0.0
    cabinet.write_joint_state_to_sim(joint_pos, joint_vel)


def _state_names(sm: CabinetKnifeSequenceSm) -> list[str]:
    return [sm.STATE_NAMES.get(int(s), str(int(s))) for s in sm.state.detach().cpu().tolist()]


def main():
    """Run the sequence demonstration."""

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset()

    cabinet = env.unwrapped.scene["cabinet"]
    drawer_joint_id = int(cabinet.find_joints("drawer_top_joint")[0][0])

    sm = CabinetKnifeSequenceSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    actions[:, 7] = GRIPPER_OPEN

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            ee_pose_b = _frame_pose_b(env, "ee_frame")
            drawer_pose_b = _frame_pose_b(env, "cabinet_frame")
            knife_pose_b = _frame_pose_b(env, "knife_frame")
            drawer_joint_pos = cabinet.data.joint_pos[:, drawer_joint_id]

            actions = sm.compute(ee_pose_b, drawer_pose_b, knife_pose_b, drawer_joint_pos)

            if sm.holding_knife().any():
                grasp_to_root = torch.tensor(
                    [0.0, 0.0, KNIFE_GRASP_OFFSET[2]], device=env.unwrapped.device
                ).repeat(env.unwrapped.num_envs, 1)
                _write_knife_pose(env, ee_pose_b[:, :3] - grasp_to_root)
            if sm.knife_in_drawer().any():
                _write_knife_pose(env, sm.drawer_drop_root_pos_b(drawer_pose_b))

            hold_open = (sm.state >= sm.RELEASE_DRAWER) & (sm.state < sm.CLOSE_DRAWER)
            _hold_drawer_open(env, drawer_joint_id, hold_open)
            _assist_close_drawer(env, drawer_joint_id, sm.state == sm.CLOSE_DRAWER)

            if step_count % 120 == 0:
                err = _distance(ee_pose_b[:, :3], sm.last_des_pos).detach().cpu().tolist()
                drawer = drawer_joint_pos.detach().cpu().tolist()
                print(
                    "[cabinet_knife_sm] "
                    f"step={step_count:4d} state={_state_names(sm)} "
                    f"drawer={[round(v, 3) for v in drawer]} err={[round(v, 3) for v in err]}",
                    flush=True,
                )

            if dones.any():
                sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                print(f"[cabinet_knife_sm] reached max_steps={args_cli.max_steps}", flush=True)
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
