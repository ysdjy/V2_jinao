# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Probe the Franka reachable workspace for a fixed (forward) grasp orientation.

Commands each env a different STATIC desired EE pose (a grid in x,z at y=0 with
the forward grasp orientation), lets the IK settle, and reports the achieved
position error. Used to place the fridge handle in a well-reachable spot.

Run:
    ./isaaclab.sh -p Connection/tools/probe_reach.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import itertools  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import connection_tasks  # noqa: F401, E402
from isaaclab.sensors import FrameTransformer  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

TASK_ID = "Connection-Open-Fridge-Franka-IK-Abs-v0"

# ready-point joint configs for the HANDLE-FRAME grasp orientation (the door
# handle's own orientation, which the SM will track so the gripper follows the
# door as it swings). This orientation lets the gripper rotate with the door.
GRASP_WXYZ = (-0.5, 0.5, -0.5, -0.5)
targets = [
    (0.20, 0.10, 0.55),
    (0.23, 0.10, 0.55),
    (0.20, 0.12, 0.58),
]
num_envs = len(targets)


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=num_envs, use_fabric=True)
    # move the fridge far away so it does not block the reach probe
    env_cfg.scene.fridge.init_state.pos = (5.0, 5.0, 0.5)
    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset()

    dev = env.unwrapped.device
    actions = torch.zeros(env.unwrapped.action_space.shape, device=dev)
    tgt = torch.tensor(targets, device=dev, dtype=torch.float32)
    actions[:, 0:3] = tgt
    actions[:, 3] = GRASP_WXYZ[0]
    actions[:, 4] = GRASP_WXYZ[1]
    actions[:, 5] = GRASP_WXYZ[2]
    actions[:, 6] = GRASP_WXYZ[3]
    actions[:, 7] = 1.0  # gripper open

    for _ in range(400):
        with torch.inference_mode():
            env.step(actions)

    ee: FrameTransformer = env.unwrapped.scene["ee_frame"]
    achieved = ee.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    robot = env.unwrapped.scene["robot"]
    arm_q = robot.data.joint_pos[:, :7].clone()

    lines = ["=== REACH PROBE (forward grasp orientation) ==="]
    for i, t in enumerate(targets):
        a = achieved[i].detach().to("cpu").tolist()
        err = ((achieved[i] - tgt[i]).norm()).item()
        q = arm_q[i].detach().to("cpu").tolist()
        lines.append(
            f"target={tuple(round(v,3) for v in t)}  achieved={tuple(round(v,3) for v in a)}  err={round(err,3)}"
        )
        lines.append(f"   arm_joint_pos = {[round(v,4) for v in q]}")
    with open("/tmp/reach_probe.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
