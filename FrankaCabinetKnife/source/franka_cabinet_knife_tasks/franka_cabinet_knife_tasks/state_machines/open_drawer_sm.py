# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Warp state machine copied from the official Franka cabinet-opening example."""

from collections.abc import Sequence

import torch
import warp as wp


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class OpenDrawerSmState:
    """States for the cabinet drawer opening state machine."""

    REST = wp.constant(0)
    APPROACH_INFRONT_HANDLE = wp.constant(1)
    APPROACH_HANDLE = wp.constant(2)
    GRASP_HANDLE = wp.constant(3)
    OPEN_DRAWER = wp.constant(4)
    RELEASE_HANDLE = wp.constant(5)


class OpenDrawerSmWaitTime:
    """Additional wait times in seconds before switching states."""

    REST = wp.constant(0.5)
    APPROACH_INFRONT_HANDLE = wp.constant(1.25)
    APPROACH_HANDLE = wp.constant(1.0)
    GRASP_HANDLE = wp.constant(1.0)
    OPEN_DRAWER = wp.constant(3.0)
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
    handle_approach_offset: wp.array(dtype=wp.transform),
    handle_grasp_offset: wp.array(dtype=wp.transform),
    drawer_opening_rate: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == OpenDrawerSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.REST:
            sm_state[tid] = OpenDrawerSmState.APPROACH_INFRONT_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.APPROACH_INFRONT_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_approach_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= OpenDrawerSmWaitTime.APPROACH_INFRONT_HANDLE:
                sm_state[tid] = OpenDrawerSmState.APPROACH_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.APPROACH_HANDLE:
        des_ee_pose[tid] = handle_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= OpenDrawerSmWaitTime.APPROACH_HANDLE:
                sm_state[tid] = OpenDrawerSmState.GRASP_HANDLE
                sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.GRASP_HANDLE:
        des_ee_pose[tid] = wp.transform_multiply(handle_grasp_offset[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.GRASP_HANDLE:
            sm_state[tid] = OpenDrawerSmState.OPEN_DRAWER
            sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.OPEN_DRAWER:
        des_ee_pose[tid] = wp.transform_multiply(drawer_opening_rate[tid], handle_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.OPEN_DRAWER:
            sm_state[tid] = OpenDrawerSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0
    elif state == OpenDrawerSmState.RELEASE_HANDLE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= OpenDrawerSmWaitTime.RELEASE_HANDLE:
            sm_state[tid] = OpenDrawerSmState.RELEASE_HANDLE
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class OpenDrawerSm:
    """Task-space finite-state machine for opening the cabinet drawer."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        self.handle_approach_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_approach_offset[:, 0] = -0.1
        self.handle_approach_offset[:, -1] = 1.0

        self.handle_grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.handle_grasp_offset[:, 0] = 0.025
        self.handle_grasp_offset[:, -1] = 1.0

        self.drawer_opening_rate = torch.zeros((self.num_envs, 7), device=self.device)
        self.drawer_opening_rate[:, 0] = -0.015
        self.drawer_opening_rate[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.handle_approach_offset_wp = wp.from_torch(self.handle_approach_offset, wp.transform)
        self.handle_grasp_offset_wp = wp.from_torch(self.handle_grasp_offset, wp.transform)
        self.drawer_opening_rate_wp = wp.from_torch(self.drawer_opening_rate, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset selected state-machine instances."""

        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor) -> torch.Tensor:
        """Compute desired end-effector pose and gripper command."""

        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        handle_pose = handle_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        handle_pose_wp = wp.from_torch(handle_pose.contiguous(), wp.transform)

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
                self.handle_approach_offset_wp,
                self.handle_grasp_offset_wp,
                self.drawer_opening_rate_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

