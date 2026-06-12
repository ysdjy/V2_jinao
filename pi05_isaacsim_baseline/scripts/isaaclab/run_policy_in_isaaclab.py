"""IsaacLab rollout driven by the HTTP policy server (mock or pi0.5).

Closed loop:  IsaacLab env  ->  Observation  ->  policy server /infer
              ->  safety filter  ->  IsaacLab action  ->  env.step  ->  record.

Run with the IsaacLab launcher:
  ./isaaclab.sh -p pi05_isaacsim_baseline/scripts/isaaclab/run_policy_in_isaaclab.py \
      --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --headless \
      --num_rollouts 1 --max_steps 100 --policy_port 8008 --image_mode none

If the policy server is unreachable, the client returns safe zero actions, so the
rollout still completes and proves the IsaacLab action pipeline works.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a policy-server-driven rollout in IsaacLab.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--policy_host", type=str, default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=8008)
parser.add_argument("--policy_timeout", type=float, default=5.0)
parser.add_argument("--image_mode", choices=["none", "path", "base64"], default="none")
parser.add_argument("--task_instruction", type=str, default="Stack the cubes with the Franka robot.")
parser.add_argument("--env_kind", choices=["auto", "ik_rel", "ik_abs", "joint"], default="auto")
parser.add_argument("--action_source", choices=["policy", "zero"], default="policy",
                    help="zero = bypass server, send safe zero actions (pure env smoke test)")
parser.add_argument("--out_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------- #
import json
import os
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "adapters", "policy_server"))
sys.path.insert(0, os.path.join(_PROJ, "adapters", "action_adapters"))

import isaac_obs_utils as obs_utils  # noqa: E402
from client import PolicyClient  # noqa: E402
from safety_filter import SafetyFilter  # noqa: E402


def main():
    out_dir = args_cli.out_dir or os.path.join(_PROJ, "data", "processed", "rollouts")
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(_PROJ, "data", "processed", "images")
    logs_dir = os.path.join(_PROJ, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    env_kind = obs_utils.detect_env_kind(args_cli.task) if args_cli.env_kind == "auto" else args_cli.env_kind
    print(f"[run_policy] task={args_cli.task} env_kind={env_kind} action_source={args_cli.action_source}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # keep observation terms separate (not concatenated) so we can introspect if needed
    try:
        env_cfg.observations.policy.concatenate_terms = False
    except Exception:
        pass
    env = gym.make(args_cli.task, cfg=env_cfg)

    client = PolicyClient(args_cli.policy_host, args_cli.policy_port, timeout=args_cli.policy_timeout)
    server_ok = False
    if args_cli.action_source == "policy":
        server_ok = client.wait_until_healthy(retries=10, delay=1.0)
        print(f"[run_policy] policy server healthy: {server_ok} ({client.base})")

    sf = SafetyFilter(os.path.join(_PROJ, "configs", "safety_limits.yaml"))
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    ts = time.strftime("%Y%m%d_%H%M%S")
    traj_path = os.path.join(out_dir, f"rollout_{ts}.jsonl")
    traj_file = open(traj_path, "w")

    latencies, ep_lengths, successes = [], [], []
    num_timeouts = num_policy_errors = num_safety_clips = 0

    for ep in range(args_cli.num_rollouts):
        env.reset()
        sf.reset()
        ep_len = 0
        success = False
        for step in range(args_cli.max_steps):
            obs_dict = obs_utils.build_observation(
                env, args_cli.task_instruction, ep, step,
                image_mode=args_cli.image_mode, image_dir=img_dir,
            )
            if args_cli.action_source == "zero" or not server_ok:
                action = sf.safe_zero_action()
                latency = 0.0
            else:
                action, latency = client.infer(obs_dict)
                latencies.append(latency)
                if action.get("_fallback"):
                    if sf.note_policy_error():
                        print("[run_policy] too many policy errors -> ending episode")
                        break
                else:
                    sf.note_policy_ok()

            # safety filter (delta or joint)
            if env_kind in ("ik_rel", "ik_abs"):
                action = sf.filter_delta_ee(action, ee_position=obs_dict["robot"]["ee_position"])
            elif env_kind == "joint" and action.get("joint_targets") is not None:
                jt = sf.filter_joint(action["joint_targets"], current=obs_dict["robot"]["joint_positions"])
                action["joint_targets"] = jt

            act_t = obs_utils.action_to_env_tensor(action, env_kind, obs_dict, num_envs, device)
            step_out = env.step(act_t)
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated[0].item() or truncated[0].item())

            # try to read success
            try:
                success = bool(terminated[0].item())
            except Exception:
                pass

            rec = {
                "episode": ep,
                "step": step,
                "timestamp": obs_dict["timestamp"],
                "observation_summary": {
                    "joint_positions": [round(v, 4) for v in obs_dict["robot"]["joint_positions"]],
                    "ee_position": [round(v, 4) for v in obs_dict["robot"]["ee_position"]],
                    "gripper_width": round(obs_dict["robot"]["gripper_width"], 4),
                },
                "action": action.get("delta_ee_position", []) + action.get("delta_ee_rot", []) + [action.get("gripper", 0.0)]
                if env_kind != "joint" else (action.get("joint_targets", []) + [action.get("gripper", 0.0)]),
                "policy_latency_ms": round(latency, 3),
                "done": done,
                "success": success,
            }
            traj_file.write(json.dumps(rec) + "\n")
            ep_len += 1
            if done:
                break

        ep_lengths.append(ep_len)
        successes.append(success)
        print(f"[run_policy] episode {ep}: length={ep_len} success={success}")

    traj_file.close()
    num_timeouts = client.num_timeouts
    num_policy_errors = client.num_errors
    num_safety_clips = sf.num_clips

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    summary = {
        "task_name": args_cli.task,
        "policy_backend": "mock-or-real(server)" if args_cli.action_source == "policy" else "zero",
        "checkpoint": None,
        "num_rollouts": args_cli.num_rollouts,
        "success_rate": (sum(1 for s in successes if s) / len(successes)) if successes else 0.0,
        "avg_episode_length": (sum(ep_lengths) / len(ep_lengths)) if ep_lengths else 0.0,
        "avg_policy_latency_ms": round(avg_lat, 3),
        "avg_control_frequency_hz": round(1000.0 / avg_lat, 2) if avg_lat > 0 else None,
        "avg_final_ee_error": None,
        "num_timeouts": num_timeouts,
        "num_policy_errors": num_policy_errors,
        "num_safety_clips": num_safety_clips,
        "trajectory_file": traj_path,
        "server_healthy": server_ok,
        "notes": "rollout completed",
    }
    summ_path = os.path.join(logs_dir, f"eval_policy_{ts}.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_policy] summary -> {summ_path}")
    print(json.dumps(summary, indent=2))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
