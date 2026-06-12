"""Reference integration: drive a Franka in ANY IsaacLab joint-position env with GELLO.

This is both a usage example and a smoke test for the ``gello_isaac_teleop`` module. It is
scene-agnostic — point ``--task`` at any IsaacLab manager-based env that has a Franka ``robot``
with a joint-position arm action.

GUI (see the robot follow GELLO):
    sg dialout -c 'DISPLAY=:1 ./isaaclab.sh -p \
      projects/gello_franka_teleop/gello_isaac_teleop/examples/run_gello_teleop_demo.py \
      --task Isaac-Stack-Cube-Franka-JointPolicy-v0'

Headless smoke (bounded):
    sg dialout -c './isaaclab.sh -p \
      projects/gello_franka_teleop/gello_isaac_teleop/examples/run_gello_teleop_demo.py \
      --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --headless --max_steps 200'
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GELLO -> IsaacLab Franka teleop demo (module reference).")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-JointPolicy-v0",
                    help="Any IsaacLab Franka joint-position env.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0, help="0 = run until window closed / Ctrl+C.")
parser.add_argument("--print_every", type=int, default=60)
parser.add_argument("--gello_config", type=str,
                    default="projects/gello_franka_teleop/configs/gello_franka.yaml")
parser.add_argument("--no_gripper", action="store_true", help="Disable gripper control (keep open).")
parser.add_argument("--max_joint_vel", type=float, default=2.5, help="Following speed cap (rad/s).")
parser.add_argument("--smoothing_tau", type=float, default=0.08, help="Smoothing time constant (s).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# make the module importable (projects/gello_franka_teleop on sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gello_isaac_teleop import GelloFrankaTeleop, GelloTeleopConfig  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    teleop = GelloFrankaTeleop(env, GelloTeleopConfig(
        gello_config=args_cli.gello_config,
        enable_gripper=not args_cli.no_gripper,
        max_joint_vel=args_cli.max_joint_vel,
        smoothing_tau=args_cli.smoothing_tau,
    ))
    teleop.start()
    print("[demo] GELLO teleop started. Move the leader arm to drive the Franka.", flush=True)

    step = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                action = teleop.step()          # reads GELLO, returns env action (joint + gripper)
                env.step(action)
            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                t = teleop.telemetry()
                print(f"[demo] hz={t['real_loop_hz']} read_hz={t.get('read_hz',0):.0f} "
                      f"q_cmd={t.get('q_cmd')} franka={t.get('franka_q')} "
                      f"gripper={t.get('gripper_cmd')}({t.get('gripper_state')}) "
                      f"raw={t.get('gripper_raw')}", flush=True)
            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
    except KeyboardInterrupt:
        print("\n[demo] Ctrl+C — exiting.", flush=True)
    finally:
        teleop.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
