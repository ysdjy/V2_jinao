# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run the Franka cube stack environment with a simple pick-place state machine.

The state machine picks ``cube_2`` and places it on top of ``cube_1``. It is meant
for visual calibration of the modified ground-level Franka stack scene with the
cabinet and knife assets loaded.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/stack_cube_sm.py --num_envs 1

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick and place state machine for the Franka stack environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--disable_collision_debug_vis",
    action="store_true",
    default=False,
    help="Disable PhysX collision-shape debug visualization in the viewport.",
)
parser.add_argument(
    "--grasp_offset_z",
    type=float,
    default=0.100,
    help="TCP local-z offset from the source cube pose used for grasping.",
)
parser.add_argument(
    "--approach_offset_z",
    type=float,
    default=0.120,
    help="Additional local-z offset above the grasp pose for source-cube approach.",
)
parser.add_argument(
    "--carry_offset_z",
    type=float,
    default=0.170,
    help="Additional local-z offset above the grasp/place pose while carrying the cube.",
)
parser.add_argument(
    "--place_clearance_z",
    type=float,
    default=0.006,
    help="Additional local-z clearance above the stacked pose before opening the gripper.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import carb
import gymnasium as gym
import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
import isaaclab.utils.math as math_utils
import omni.usd
from pxr import UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

wp.config.kernel_cache_dir = "/tmp/warp_cache"
wp.init()


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickPlaceSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    APPROACH_ABOVE_TARGET = wp.constant(5)
    PLACE_OBJECT = wp.constant(6)
    OPEN_GRIPPER = wp.constant(7)
    RETREAT = wp.constant(8)


class PickPlaceSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.4)
    APPROACH_OBJECT = wp.constant(0.4)
    GRASP_OBJECT = wp.constant(0.4)
    LIFT_OBJECT = wp.constant(0.3)
    APPROACH_ABOVE_TARGET = wp.constant(0.3)
    PLACE_OBJECT = wp.constant(0.4)
    OPEN_GRIPPER = wp.constant(0.4)
    RETREAT = wp.constant(1.0)


class PickPlaceSmMaxTime:
    APPROACH_ABOVE_OBJECT = wp.constant(2.0)
    APPROACH_OBJECT = wp.constant(2.0)
    LIFT_OBJECT = wp.constant(2.0)
    APPROACH_ABOVE_TARGET = wp.constant(2.0)
    PLACE_OBJECT = wp.constant(2.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    above_object_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    lift_pose: wp.array(dtype=wp.transform),
    pre_place_pose: wp.array(dtype=wp.transform),
    place_pose: wp.array(dtype=wp.transform),
    retreat_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickPlaceSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.REST:
            sm_state[tid] = PickPlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = above_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickPlaceSmMaxTime.APPROACH_ABOVE_OBJECT:
            if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PickPlaceSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickPlaceSmMaxTime.APPROACH_OBJECT:
            if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickPlaceSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickPlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.LIFT_OBJECT:
        des_ee_pose[tid] = lift_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickPlaceSmMaxTime.LIFT_OBJECT:
            if sm_wait_time[tid] >= PickPlaceSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickPlaceSmState.APPROACH_ABOVE_TARGET
                sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.APPROACH_ABOVE_TARGET:
        des_ee_pose[tid] = pre_place_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickPlaceSmMaxTime.APPROACH_ABOVE_TARGET:
            if sm_wait_time[tid] >= PickPlaceSmWaitTime.APPROACH_ABOVE_TARGET:
                sm_state[tid] = PickPlaceSmState.PLACE_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.PLACE_OBJECT:
        des_ee_pose[tid] = place_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickPlaceSmMaxTime.PLACE_OBJECT:
            if sm_wait_time[tid] >= PickPlaceSmWaitTime.PLACE_OBJECT:
                sm_state[tid] = PickPlaceSmState.OPEN_GRIPPER
                sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.OPEN_GRIPPER:
        des_ee_pose[tid] = place_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.OPEN_GRIPPER:
            sm_state[tid] = PickPlaceSmState.RETREAT
            sm_wait_time[tid] = 0.0

    elif state == PickPlaceSmState.RETREAT:
        des_ee_pose[tid] = retreat_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickPlaceSmWaitTime.RETREAT:
            sm_state[tid] = PickPlaceSmState.RETREAT
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndPlaceSm:
    """Task-space pick-place state machine for cube stacking."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.035):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)

    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self,
        ee_pose: torch.Tensor,
        above_object_pose: torch.Tensor,
        object_pose: torch.Tensor,
        lift_pose: torch.Tensor,
        pre_place_pose: torch.Tensor,
        place_pose: torch.Tensor,
        retreat_pose: torch.Tensor,
    ) -> torch.Tensor:
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        above_object_pose = above_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        lift_pose = lift_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        pre_place_pose = pre_place_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        place_pose = place_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        retreat_pose = retreat_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(above_object_pose.contiguous(), wp.transform),
                wp.from_torch(object_pose.contiguous(), wp.transform),
                wp.from_torch(lift_pose.contiguous(), wp.transform),
                wp.from_torch(pre_place_pose.contiguous(), wp.transform),
                wp.from_torch(place_pose.contiguous(), wp.transform),
                wp.from_torch(retreat_pose.contiguous(), wp.transform),
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

    @property
    def state(self) -> torch.Tensor:
        return self.sm_state


class PoseAxesVisualizer:
    """Draw pose axes with USD curves without loading external marker assets."""

    def __init__(self, prim_prefix: str, axis_length: float, width: float):
        self.axis_length = axis_length
        self.stage = omni.usd.get_context().get_stage()
        self.curves = []
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.35, 1.0)]
        for axis_name, color in zip(("x", "y", "z"), colors):
            curve = UsdGeom.BasisCurves.Define(self.stage, f"{prim_prefix}_{axis_name}")
            curve.CreateTypeAttr("linear")
            curve.CreateBasisAttr("bezier")
            curve.CreateDisplayColorAttr([color])
            curve.CreateWidthsAttr([width])
            self.curves.append(curve)

    def visualize(self, positions: torch.Tensor, orientations: torch.Tensor):
        num_poses = positions.shape[0]
        local_axes = torch.tensor(
            [[self.axis_length, 0.0, 0.0], [0.0, self.axis_length, 0.0], [0.0, 0.0, self.axis_length]],
            device=positions.device,
            dtype=positions.dtype,
        )
        for axis_id, curve in enumerate(self.curves):
            axis = local_axes[axis_id].repeat(num_poses, 1)
            rotated_axis = math_utils.quat_apply(orientations, axis)
            starts = positions.detach().cpu()
            ends = (positions + rotated_axis).detach().cpu()
            points = []
            for start, end in zip(starts.tolist(), ends.tolist()):
                points.extend([start, end])
            curve.GetPointsAttr().Set(points)
            curve.GetCurveVertexCountsAttr().Set([2] * num_poses)


def enable_collision_debug_visualization():
    """Enable viewport-only PhysX collider overlays without editing simulation prims."""
    settings = carb.settings.get_settings()
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 2)
    settings.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)


def main():
    env_cfg: StackEnvCfg = parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.viewer.eye = (2.0, -2.0, 1.4)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.15)

    env = gym.make("Isaac-Stack-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    if not args_cli.headless and not args_cli.disable_collision_debug_vis:
        enable_collision_debug_visualization()

    current_visualizer = None
    active_target_visualizer = None
    phase_target_visualizers = {}
    if not args_cli.headless:
        current_visualizer = PoseAxesVisualizer("/Visuals/StackSm/current_tcp_pose", axis_length=0.08, width=0.006)
        active_target_visualizer = PoseAxesVisualizer("/Visuals/StackSm/active_target_pose", axis_length=0.14, width=0.012)
        phase_target_visualizers = {
            "approach_above_object": PoseAxesVisualizer(
                "/Visuals/StackSm/targets/p01_approach_above_object", axis_length=0.09, width=0.004
            ),
            "grasp_object": PoseAxesVisualizer("/Visuals/StackSm/targets/p02_grasp_object", axis_length=0.09, width=0.004),
            "lift_object": PoseAxesVisualizer("/Visuals/StackSm/targets/p03_lift_object", axis_length=0.09, width=0.004),
            "approach_above_target": PoseAxesVisualizer(
                "/Visuals/StackSm/targets/p04_approach_above_target", axis_length=0.09, width=0.004
            ),
            "place_object": PoseAxesVisualizer("/Visuals/StackSm/targets/p05_place_object", axis_length=0.09, width=0.004),
            "retreat": PoseAxesVisualizer("/Visuals/StackSm/targets/p06_retreat", axis_length=0.09, width=0.004),
        }

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    down_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    down_orientation[:, 1] = 1.0

    stack_sm = PickAndPlaceSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    state_names = {
        0: "REST",
        1: "APPROACH_ABOVE_OBJECT",
        2: "APPROACH_OBJECT",
        3: "GRASP_OBJECT",
        4: "LIFT_OBJECT",
        5: "APPROACH_ABOVE_TARGET",
        6: "PLACE_OBJECT",
        7: "OPEN_GRIPPER",
        8: "RETREAT",
    }
    previous_command_state = torch.full((env.unwrapped.num_envs,), -1, dtype=torch.int32, device=env.unwrapped.device)
    cube_size = 0.0406
    grasp_tcp_offset_z = args_cli.grasp_offset_z
    above_object_offset_z = args_cli.approach_offset_z
    carry_offset_z = args_cli.carry_offset_z
    place_clearance_z = args_cli.place_clearance_z

    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            source_cube_data: RigidObjectData = env.unwrapped.scene["cube_2"].data
            target_cube_data: RigidObjectData = env.unwrapped.scene["cube_1"].data
            source_position = source_cube_data.root_pos_w - env.unwrapped.scene.env_origins
            target_position = target_cube_data.root_pos_w - env.unwrapped.scene.env_origins
            source_yaw_orientation = math_utils.yaw_quat(source_cube_data.root_quat_w)
            target_yaw_orientation = math_utils.yaw_quat(target_cube_data.root_quat_w)
            source_orientation = math_utils.quat_mul(source_yaw_orientation, down_orientation)
            target_orientation = math_utils.quat_mul(target_yaw_orientation, down_orientation)

            source_grasp_offset = torch.zeros_like(source_position)
            source_grasp_offset[:, 2] = grasp_tcp_offset_z
            source_above_offset = torch.zeros_like(source_position)
            source_above_offset[:, 2] = grasp_tcp_offset_z + above_object_offset_z
            source_lift_offset = torch.zeros_like(source_position)
            source_lift_offset[:, 2] = grasp_tcp_offset_z + carry_offset_z
            target_place_offset = torch.zeros_like(target_position)
            target_place_offset[:, 2] = cube_size + grasp_tcp_offset_z + place_clearance_z
            target_above_offset = torch.zeros_like(target_position)
            target_above_offset[:, 2] = cube_size + grasp_tcp_offset_z + carry_offset_z

            above_object_position = source_position + source_above_offset
            grasp_position = source_position + source_grasp_offset
            lift_position = source_position + source_lift_offset
            pre_place_position = target_position + target_above_offset
            place_position = target_position + target_place_offset
            retreat_position = pre_place_position.clone()

            command_state = stack_sm.state.clone()
            actions = stack_sm.compute(
                torch.cat([tcp_position, tcp_orientation], dim=-1),
                torch.cat([above_object_position, source_orientation], dim=-1),
                torch.cat([grasp_position, source_orientation], dim=-1),
                torch.cat([lift_position, source_orientation], dim=-1),
                torch.cat([pre_place_position, target_orientation], dim=-1),
                torch.cat([place_position, target_orientation], dim=-1),
                torch.cat([retreat_position, target_orientation], dim=-1),
            )

            if current_visualizer is not None and active_target_visualizer is not None:
                current_visualizer.visualize(tcp_position + env.unwrapped.scene.env_origins, tcp_orientation)
                active_target_visualizer.visualize(actions[:, 0:3] + env.unwrapped.scene.env_origins, actions[:, 3:7])
                phase_target_visualizers["approach_above_object"].visualize(
                    above_object_position + env.unwrapped.scene.env_origins, source_orientation
                )
                phase_target_visualizers["grasp_object"].visualize(
                    grasp_position + env.unwrapped.scene.env_origins, source_orientation
                )
                phase_target_visualizers["lift_object"].visualize(
                    lift_position + env.unwrapped.scene.env_origins, source_orientation
                )
                phase_target_visualizers["approach_above_target"].visualize(
                    pre_place_position + env.unwrapped.scene.env_origins, target_orientation
                )
                phase_target_visualizers["place_object"].visualize(
                    place_position + env.unwrapped.scene.env_origins, target_orientation
                )
                phase_target_visualizers["retreat"].visualize(
                    retreat_position + env.unwrapped.scene.env_origins, target_orientation
                )

            changed = command_state != previous_command_state
            if changed.any():
                env0_state = int(command_state[0].item())
                env0_target = actions[0, :7].detach().cpu().tolist()
                print(f"[StackSm] state={state_names[env0_state]} target_pose={env0_target}")
                print(
                    "[StackSmTargets] "
                    f"above={above_object_position[0].detach().cpu().tolist()} "
                    f"grasp={grasp_position[0].detach().cpu().tolist()} "
                    f"lift={lift_position[0].detach().cpu().tolist()} "
                    f"pre_place={pre_place_position[0].detach().cpu().tolist()} "
                    f"place={place_position[0].detach().cpu().tolist()} "
                    f"source_quat={source_orientation[0].detach().cpu().tolist()} "
                    f"target_quat={target_orientation[0].detach().cpu().tolist()}"
                )
                previous_command_state[:] = command_state

            if dones.any():
                stack_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                previous_command_state[dones.nonzero(as_tuple=False).squeeze(-1)] = -1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
