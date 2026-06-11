# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the selected-drawer custom RL env (Stage 3). No training.

Checks: env creates, reset samples a selected drawer, observation is the expected 31-d, selected
joint / handle are read correctly, rewards compute without error, and random actions run without
crashing. Writes per-reset records to logs/skill_tests/custom_drawer_selected_smoke.jsonl.

    ./isaaclab.sh -p scripts/environments/state_machine/debug_custom_drawer_env.py \
        --num_envs 1 --resets 10 --steps 100 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Custom selected-drawer env smoke test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resets", type=int, default=10)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import json
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack.config.franka import custom_drawer_mdp as cd_mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.custom_drawer_config import FUNCTIONAL_DRAWERS

TASK_ID = "Isaac-Open-CustomDrawer-Selected-Franka-v0"
LOG_PATH = Path("logs/skill_tests/custom_drawer_selected_smoke.jsonl")


def _round(seq, n=4):
    return [round(float(v), n) for v in seq]


def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    u = env.unwrapped
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=args_cli.seed)
    policy_obs = obs["policy"]
    print(f"[smoke] observation['policy'] shape = {tuple(policy_obs.shape)} (expect [N,31])", flush=True)
    print(f"[smoke] action_space = {env.action_space}", flush=True)
    assert policy_obs.shape[-1] == 31, f"expected 31-d obs, got {policy_obs.shape[-1]}"

    distribution = {d: 0 for d in FUNCTIONAL_DRAWERS}
    for r in range(args_cli.resets):
        env.reset()
        rows = u._sel_frame_row.tolist()
        jids = u._sel_joint_id.tolist()
        for row in rows:
            distribution[FUNCTIONAL_DRAWERS[int(row)]] += 1
        sel_jp = cd_mdp._selected_drawer_joint_pos(u)
        handle = cd_mdp._selected_handle_pos(u)
        tcp = cd_mdp._tcp_pos(u)
        rel = cd_mdp.selected_rel_ee_drawer_distance(u)
        rec = {
            "reset": r,
            "selected_drawer": [FUNCTIONAL_DRAWERS[int(x)] for x in rows],
            "selected_frame_row": [int(x) for x in rows],
            "selected_joint_id": [int(x) for x in jids],
            "selected_joint_pos": _round(sel_jp.tolist()),
            "selected_handle_pose": _round(handle[0].tolist()),
            "tcp_pose": _round(tcp[0].tolist()),
            "rel_ee_drawer_distance": _round(rel[0].tolist()),
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(f"[smoke] reset {r}: {rec}", flush=True)

    # random actions: confirm rewards compute and nothing crashes
    action_dim = env.action_space.shape[-1]
    print(f"[smoke] running {args_cli.steps} random steps (action_dim={action_dim}) ...", flush=True)
    rew_min, rew_max = float("inf"), float("-inf")
    for i in range(args_cli.steps):
        action = (torch.rand((u.num_envs, action_dim), device=u.device) * 2.0 - 1.0)
        obs, rew, term, trunc, info = env.step(action)
        rew_min = min(rew_min, float(rew.min()))
        rew_max = max(rew_max, float(rew.max()))
        if not torch.isfinite(rew).all():
            raise RuntimeError(f"non-finite reward at step {i}")
    print(f"[smoke] random steps OK. reward range=[{rew_min:.4f},{rew_max:.4f}]", flush=True)
    print(f"[smoke] selected_drawer distribution over {args_cli.resets} resets x {u.num_envs} envs: {distribution}", flush=True)
    print(f"[smoke] PASS. log -> {LOG_PATH}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
