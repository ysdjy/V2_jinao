# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""KitchenV0 sequential state machine in one environment.

V0 scope: validate that the full task order runs in a single scene/episode. The
appliance joints and knife pose are driven directly at the asset level. The next
version should replace each direct asset command with Franka IK grasp/motion
states, following the official Franka state-machine examples.

Sequence:
    rest -> open fridge -> open microwave -> open bottom drawer -> place knife
    -> close drawer -> close microwave -> done

Example:
    ./isaaclab.sh -p KitchenV0/scripts/kitchen_sequence_sm.py --num_envs 1 --fridge_angle 45
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from enum import IntEnum

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="KitchenV0 sequential state-machine scaffold.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--fridge_angle", type=int, choices=(15, 45), default=45, help="Fridge target opening angle.")
parser.add_argument("--max_steps", type=int, default=0, help="If > 0, stop after this many environment steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import kitchen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from kitchen_tasks.robots.fridge import FRIDGE_OPEN_15_DEG, FRIDGE_OPEN_45_DEG, FRIDGE_DOOR_JOINT
from kitchen_tasks.robots.microwave import MICROWAVE_DOOR_JOINT, MICROWAVE_OPEN_45_DEG
from kitchen_tasks.tasks.kitchen_scene.kitchen_scene_env_cfg import (
    CABINET_BOTTOM_DRAWER_JOINT,
    CABINET_BOTTOM_DRAWER_OPEN_POS,
)

TASK_ID = "Kitchen-V0-Franka-IK-Abs-Play-v0"


class KitchenSequenceState(IntEnum):
    REST = 0
    OPEN_FRIDGE = 1
    OPEN_MICROWAVE = 2
    OPEN_BOTTOM_DRAWER = 3
    PLACE_KNIFE_IN_DRAWER = 4
    CLOSE_BOTTOM_DRAWER = 5
    CLOSE_MICROWAVE = 6
    DONE = 7


STATE_DURATIONS = {
    KitchenSequenceState.REST: 0.5,
    KitchenSequenceState.OPEN_FRIDGE: 1.5,
    KitchenSequenceState.OPEN_MICROWAVE: 1.5,
    KitchenSequenceState.OPEN_BOTTOM_DRAWER: 1.5,
    KitchenSequenceState.PLACE_KNIFE_IN_DRAWER: 0.8,
    KitchenSequenceState.CLOSE_BOTTOM_DRAWER: 1.5,
    KitchenSequenceState.CLOSE_MICROWAVE: 1.5,
    KitchenSequenceState.DONE: 9999.0,
}


def _joint_id(asset, joint_name: str) -> int:
    return int(asset.find_joints(joint_name)[0][0])


def _set_single_joint(asset, joint_id: int, value: torch.Tensor | float) -> None:
    joint_pos = asset.data.joint_pos.clone()
    joint_vel = asset.data.joint_vel.clone()
    if isinstance(value, torch.Tensor):
        joint_pos[:, joint_id] = value
    else:
        joint_pos[:, joint_id] = float(value)
    joint_vel[:, joint_id] = 0.0
    asset.write_joint_state_to_sim(joint_pos, joint_vel)
    asset.set_joint_position_target(joint_pos)


def _lerp(start: float, end: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    return start + (end - start) * alpha


def _place_knife_in_drawer(knife, env_origins: torch.Tensor) -> None:
    root_pose = knife.data.root_pose_w.clone()
    root_vel = knife.data.root_vel_w.clone()
    # Approximate bottom-drawer interior in each env frame. This is a v0 asset-level
    # placeholder; the robot-controlled version should compute this from drawer frames.
    drawer_pos_env = torch.tensor([-0.42, 0.0, 0.36], device=root_pose.device).repeat(root_pose.shape[0], 1)
    root_pose[:, :3] = drawer_pos_env + env_origins
    root_pose[:, 3:7] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=root_pose.device)
    root_vel[:] = 0.0
    knife.write_root_pose_to_sim(root_pose)
    knife.write_root_velocity_to_sim(root_vel)


def main() -> None:
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset()

    scene = env.unwrapped.scene
    fridge = scene["fridge"]
    microwave = scene["microwave"]
    cabinet = scene["cabinet"]
    knife = scene["knife"]

    fridge_joint_id = _joint_id(fridge, FRIDGE_DOOR_JOINT)
    microwave_joint_id = _joint_id(microwave, MICROWAVE_DOOR_JOINT)
    drawer_joint_id = _joint_id(cabinet, CABINET_BOTTOM_DRAWER_JOINT)

    fridge_closed = float(fridge.data.joint_pos[0, fridge_joint_id])
    microwave_closed = float(microwave.data.joint_pos[0, microwave_joint_id])
    drawer_closed = float(cabinet.data.joint_pos[0, drawer_joint_id])

    fridge_open = fridge_closed + (FRIDGE_OPEN_15_DEG if args_cli.fridge_angle == 15 else FRIDGE_OPEN_45_DEG)
    # PartNet 7320 imports with a non-zero closed joint offset. Drive relative to
    # the reset pose instead of assuming closed == 0.
    microwave_open = microwave_closed - MICROWAVE_OPEN_45_DEG
    drawer_open = drawer_closed + CABINET_BOTTOM_DRAWER_OPEN_POS

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    state = KitchenSequenceState.REST
    state_elapsed = 0.0
    step_count = 0
    dt = env_cfg.sim.dt * env_cfg.decimation
    printed_state = None

    while simulation_app.is_running():
        duration = STATE_DURATIONS[state]
        alpha = state_elapsed / duration if duration > 0.0 else 1.0

        if printed_state != state:
            print(f"[kitchen_sequence_sm] state={state.name}", flush=True)
            printed_state = state

        with torch.inference_mode():
            if state == KitchenSequenceState.OPEN_FRIDGE:
                _set_single_joint(fridge, fridge_joint_id, _lerp(fridge_closed, fridge_open, alpha))
            elif state == KitchenSequenceState.OPEN_MICROWAVE:
                _set_single_joint(microwave, microwave_joint_id, _lerp(microwave_closed, microwave_open, alpha))
            elif state == KitchenSequenceState.OPEN_BOTTOM_DRAWER:
                _set_single_joint(cabinet, drawer_joint_id, _lerp(drawer_closed, drawer_open, alpha))
            elif state == KitchenSequenceState.PLACE_KNIFE_IN_DRAWER:
                _place_knife_in_drawer(knife, scene.env_origins)
            elif state == KitchenSequenceState.CLOSE_BOTTOM_DRAWER:
                _set_single_joint(
                    cabinet,
                    drawer_joint_id,
                    _lerp(drawer_open, drawer_closed, alpha),
                )
            elif state == KitchenSequenceState.CLOSE_MICROWAVE:
                _set_single_joint(microwave, microwave_joint_id, _lerp(microwave_open, microwave_closed, alpha))

            env.step(actions)

        state_elapsed += dt
        step_count += 1
        if state_elapsed >= duration and state != KitchenSequenceState.DONE:
            state = KitchenSequenceState(state + 1)
            state_elapsed = 0.0

        if step_count % 60 == 0:
            fridge_deg = float(fridge.data.joint_pos[0, fridge_joint_id] * 180.0 / torch.pi)
            microwave_deg = float(microwave.data.joint_pos[0, microwave_joint_id] * 180.0 / torch.pi)
            drawer_m = float(cabinet.data.joint_pos[0, drawer_joint_id])
            print(
                f"[kitchen_sequence_sm] step={step_count} fridge={fridge_deg:.1f}deg "
                f"microwave={microwave_deg:.1f}deg bottom_drawer={drawer_m:.3f}m",
                flush=True,
            )

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break
        if state == KitchenSequenceState.DONE and state_elapsed >= 0.5:
            fridge_deg = float(fridge.data.joint_pos[0, fridge_joint_id] * 180.0 / torch.pi)
            microwave_deg = float(microwave.data.joint_pos[0, microwave_joint_id] * 180.0 / torch.pi)
            drawer_m = float(cabinet.data.joint_pos[0, drawer_joint_id])
            print(
                f"[kitchen_sequence_sm] final fridge={fridge_deg:.1f}deg "
                f"microwave={microwave_deg:.1f}deg bottom_drawer={drawer_m:.3f}m",
                flush=True,
            )
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
