# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the KitchenV0 fridge-only task with a warp-based state machine.

States: REST -> APPROACH_INFRONT_HANDLE -> APPROACH_HANDLE -> GRASP_HANDLE ->
OPEN_DOOR -> RELEASE_HANDLE.

Unlike a drawer (linear pull), a fridge door rotates about a hinge (``joint_0``).
The OPEN_DOOR state therefore commands a target that is the *current* handle
position offset by a fixed world-frame vector toward the robot, while keeping the
grasp orientation. As the door swings, the handle moves and the robot keeps
pulling, which rotates the door open.

.. code-block:: bash

    ./isaaclab.sh -p KitchenV0/scripts/state_machine/open_fridge_sm.py --num_envs 1
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka open-fridge state machine (KitchenV0).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--target_angle",
    type=int,
    choices=(15, 45),
    default=45,
    help="Physical door angle at which the state machine releases the handle. This does not drive the door joint.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="If > 0, stop after this many environment steps and print a diagnostic summary. "
    "If 0 (default), run indefinitely.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# enabling cameras is required for off-screen video recording
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import os
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sensors import FrameTransformer

# Importing the project package registers the Kitchen-* gym tasks.
import kitchen_tasks  # noqa: F401
from kitchen_tasks.robots.fridge import FRIDGE_DOOR_JOINT
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_ID = "Kitchen-Fridge-Franka-IK-Abs-v0"

# Fixed grasp orientation (robot base frame), wxyz. Gripper approach axis (hand +z)
# points horizontally toward the door (+x world), fingers close along world y to
# span the vertical handle. This is Ry(+90deg). Used for all manipulation states.
GRASP_QUAT_WXYZ = (0.7071, 0.0, 0.7071, 0.0)

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class OpenFridgeSmState:
    """States for the fridge-door opening state machine."""

    REST = wp.constant(0)
    APPROACH_INFRONT_HANDLE = wp.constant(1)
    APPROACH_HANDLE = wp.constant(2)
    GRASP_HANDLE = wp.constant(3)
    OPEN_DOOR = wp.constant(4)
    RELEASE_HANDLE = wp.constant(5)


class OpenFridgeSmWaitTime:
    """Additional wait times (in s) for states before switching."""

    REST = wp.constant(0.5)
    APPROACH_INFRONT_HANDLE = wp.constant(1.25)
    APPROACH_HANDLE = wp.constant(1.0)
    GRASP_HANDLE = wp.constant(1.5)
    OPEN_DOOR = wp.constant(6.5)
    RELEASE_HANDLE = wp.constant(0.2)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    handle_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    grasp_quat: wp.array(dtype=wp.quat),
    infront_offset: wp.array(dtype=wp.vec3),
    near_offset: wp.array(dtype=wp.vec3),
    grasp_offset: wp.array(dtype=wp.vec3),
    open_offset: wp.array(dtype=wp.vec3),
    position_threshold: float,
    approach_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # handle position and the fixed grasp orientation we command throughout
    hp = wp.transform_get_translation(handle_pose[tid])
    gq = grasp_quat[tid]
    # decide next state
    if state == OpenFridgeSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= OpenFridgeSmWaitTime.REST:
            sm_state[tid] = OpenFridgeSmState.APPROACH_INFRONT_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenFridgeSmState.APPROACH_INFRONT_HANDLE:
        des_ee_pose[tid] = wp.transform(hp + infront_offset[tid], gq)
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            approach_threshold,
        ):
            if sm_wait_time[tid] >= OpenFridgeSmWaitTime.APPROACH_INFRONT_HANDLE:
                sm_state[tid] = OpenFridgeSmState.APPROACH_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenFridgeSmState.APPROACH_HANDLE:
        # stay in front of the handle; never command the handle center directly
        des_ee_pose[tid] = wp.transform(hp + near_offset[tid], gq)
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            approach_threshold,
        ):
            if sm_wait_time[tid] >= OpenFridgeSmWaitTime.APPROACH_HANDLE:
                sm_state[tid] = OpenFridgeSmState.GRASP_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenFridgeSmState.GRASP_HANDLE:
        # move to the grasp pose first; only close the gripper once the EE is there.
        des_t = hp + grasp_offset[tid]
        des_ee_pose[tid] = wp.transform(des_t, gq)
        at_grasp = distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            des_t,
            position_threshold,
        )
        if at_grasp:
            gripper_state[tid] = GripperState.CLOSE
            if sm_wait_time[tid] >= OpenFridgeSmWaitTime.GRASP_HANDLE:
                sm_state[tid] = OpenFridgeSmState.OPEN_DOOR
                sm_wait_time[tid] = 0.0
        else:
            gripper_state[tid] = GripperState.OPEN
            sm_wait_time[tid] = 0.0
    elif state == OpenFridgeSmState.OPEN_DOOR:
        # target = current handle position + fixed world pull, keep fixed grasp orientation.
        des_ee_pose[tid] = wp.transform(hp + open_offset[tid], gq)
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= OpenFridgeSmWaitTime.OPEN_DOOR:
            sm_state[tid] = OpenFridgeSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenFridgeSmState.RELEASE_HANDLE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= OpenFridgeSmWaitTime.RELEASE_HANDLE:
            sm_state[tid] = OpenFridgeSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class OpenFridgeSm:
    """A task-space state machine to open a refrigerator door."""

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str = "cpu",
        position_threshold=0.03,
        approach_threshold=0.08,
    ):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.approach_threshold = approach_threshold
        # state machine buffers
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # base grasp orientation. During OPEN we rotate it about world-z by the door
        # angle so the gripper follows the (vertical) handle as the door swings.
        self.forward_quat_wxyz = torch.tensor([list(GRASP_QUAT_WXYZ)], device=self.device).repeat(
            self.num_envs, 1
        )
        # sign of the world-z rotation that follows the door opening (flip if needed)
        self.open_follow_sign = -1.0
        # current per-env grasp orientation, stored as (x, y, z, w) for warp
        gx, gy, gz, gw = GRASP_QUAT_WXYZ[1], GRASP_QUAT_WXYZ[2], GRASP_QUAT_WXYZ[3], GRASP_QUAT_WXYZ[0]
        self.grasp_quat = torch.tensor([[gx, gy, gz, gw]], device=self.device).repeat(self.num_envs, 1)

        # world-frame approach offsets (all relative to the *current* handle position).
        # Stage 1: far in front + slightly above -> clear the door plane entirely.
        self.infront_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.infront_offset[:, 0] = -0.16
        self.infront_offset[:, 2] = 0.03
        # Stage 2: close to the handle but still in front of the door surface.
        self.near_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.near_offset[:, 0] = -0.08
        # Stage 3: minimal offset for finger wrap (not a push into the door).
        self.grasp_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasp_offset[:, 0] = 0.01

        # Pull applied during OPEN_DOOR. The base direction points toward the robot
        # (-x) and along the initial hinge arc (-y); it is rotated about world-z by
        # the door angle each step (see compute) so it always follows the arc
        # tangent instead of fighting it as the door swings.
        self.open_offset_base = torch.tensor([[-0.30, -0.08, 0.0]], device=self.device).repeat(
            self.num_envs, 1
        )
        self.open_offset = self.open_offset_base.clone()

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.grasp_quat_wp = wp.from_torch(self.grasp_quat.contiguous(), wp.quat)
        self.infront_offset_wp = wp.from_torch(self.infront_offset, wp.vec3)
        self.near_offset_wp = wp.from_torch(self.near_offset, wp.vec3)
        self.grasp_offset_wp = wp.from_torch(self.grasp_offset, wp.vec3)
        self.open_offset_wp = wp.from_torch(self.open_offset, wp.vec3)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor, door_angle: torch.Tensor):
        # Update the per-env grasp orientation: in the OPEN_DOOR state, rotate the
        # base (forward) orientation about world-z by the door angle so the gripper
        # tracks the rotating handle and does not slip off.
        theta = self.open_follow_sign * door_angle  # (num_envs,)
        half = 0.5 * theta
        rz_wxyz = torch.zeros((self.num_envs, 4), device=self.device)
        rz_wxyz[:, 0] = torch.cos(half)
        rz_wxyz[:, 3] = torch.sin(half)
        open_q_wxyz = math_utils.quat_mul(rz_wxyz, self.forward_quat_wxyz)
        in_open = (self.sm_state == 4).unsqueeze(-1)  # 4 == OPEN_DOOR
        cur_wxyz = torch.where(in_open, open_q_wxyz, self.forward_quat_wxyz)
        # write into grasp_quat (x, y, z, w) in place so the warp view sees it
        self.grasp_quat[:, 0] = cur_wxyz[:, 1]
        self.grasp_quat[:, 1] = cur_wxyz[:, 2]
        self.grasp_quat[:, 2] = cur_wxyz[:, 3]
        self.grasp_quat[:, 3] = cur_wxyz[:, 0]

        # rotate the pull offset about world-z by the same door angle so it follows
        # the arc tangent as the door swings open
        ca = torch.cos(theta)
        sa = torch.sin(theta)
        bx = self.open_offset_base[:, 0]
        by = self.open_offset_base[:, 1]
        self.open_offset[:, 0] = bx * ca - by * sa
        self.open_offset[:, 1] = bx * sa + by * ca

        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        handle_pose = handle_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        handle_pose_wp = wp.from_torch(handle_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                handle_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.grasp_quat_wp,
                self.infront_offset_wp,
                self.near_offset_wp,
                self.grasp_offset_wp,
                self.open_offset_wp,
                self.position_threshold,
                self.approach_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=render_mode)

    # wrap for video recording
    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "videos", "open_fridge")
        video_dir = os.path.abspath(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[open_fridge_sm] recording video to {video_dir}", flush=True)

    env.reset()

    # locate the fridge door joint index for diagnostics
    fridge = env.unwrapped.scene["fridge"]
    door_joint_idx = int(fridge.find_joints(FRIDGE_DOOR_JOINT)[0][0])

    # create action buffers (position + quaternion + gripper)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # create state machine
    open_sm = OpenFridgeSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    target_angle_rad = 3.141592653589793 * float(args_cli.target_angle) / 180.0

    max_door_angle = torch.zeros((env.unwrapped.num_envs,), device=env.unwrapped.device)

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # -- end-effector frame
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            # -- handle frame
            fridge_frame_tf: FrameTransformer = env.unwrapped.scene["fridge_frame"]
            handle_position = fridge_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            handle_orientation = fridge_frame_tf.data.target_quat_w[..., 0, :].clone()

            # current door angle (signed), used to make the gripper follow the door
            door_angle = fridge.data.joint_pos[:, door_joint_idx]

            # advance state machine
            actions = open_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([handle_position, handle_orientation], dim=-1),
                door_angle,
            )
            reached_target = (open_sm.sm_state == 4) & (door_angle.abs() >= target_angle_rad)
            if reached_target.any():
                open_sm.sm_state[reached_target] = 5  # RELEASE_HANDLE
                open_sm.sm_wait_time[reached_target] = 0.0

            max_door_angle = torch.maximum(max_door_angle, door_angle.abs())

            # periodic progress log
            if step_count % 90 == 0:
                st = open_sm.sm_state.detach().to("cpu").tolist()
                cur_deg = (door_angle.abs() * 180.0 / 3.14159265).detach().to("cpu").tolist()
                grip = actions[:, -1].detach().to("cpu").tolist()
                dist0 = (tcp_rest_position[0] - handle_position[0]).norm().item()
                print(
                    f"[open_fridge_sm] step={step_count:4d} states={st} door_deg={[round(d,1) for d in cur_deg]} "
                    f"gripper={grip} ee_handle_dist={round(dist0,3)}",
                    flush=True,
                )

            # reset state machine
            if dones.any():
                open_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                max_door_angle[dones.nonzero(as_tuple=False).squeeze(-1)] = 0.0

            # optional bounded run for verification / calibration
            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                states = open_sm.sm_state.detach().to("cpu").tolist()
                ang = (max_door_angle * 180.0 / 3.14159265).detach().to("cpu").tolist()
                hp = handle_position[0].detach().to("cpu").tolist()
                ep = tcp_rest_position[0].detach().to("cpu").tolist()
                hq = handle_orientation[0].detach().to("cpu").tolist()
                eq = tcp_rest_orientation[0].detach().to("cpu").tolist()
                print(f"[open_fridge_sm] reached max_steps={args_cli.max_steps}", flush=True)
                print(f"[open_fridge_sm] per-env SM state    = {states}", flush=True)
                print("[open_fridge_sm] (states: 0=REST 1=APPROACH_INFRONT 2=APPROACH 3=GRASP 4=OPEN 5=RELEASE)", flush=True)
                print(f"[open_fridge_sm] per-env max door deg = {[round(a,1) for a in ang]}", flush=True)
                print(f"[open_fridge_sm] env0 handle pos(env) = {[round(v,3) for v in hp]}", flush=True)
                print(f"[open_fridge_sm] env0 ee     pos(env) = {[round(v,3) for v in ep]}", flush=True)
                print(f"[open_fridge_sm] env0 handle quat(wxyz)= {[round(v,3) for v in hq]}", flush=True)
                print(f"[open_fridge_sm] env0 ee     quat(wxyz)= {[round(v,3) for v in eq]}", flush=True)
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
