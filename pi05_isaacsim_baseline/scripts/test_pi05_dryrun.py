"""pi0.5 / pi0 dry-run inference (no IsaacSim).

Tries, in order:
  1. real OpenPI policy (config + checkpoint, or a default env policy)
  2. mock policy fallback

Builds a random/example observation, runs inference, prints action shape/dtype
and timing. Writes logs/pi05_dryrun.txt.

Usage (inside OpenPI venv for the real path):
  .venv_openpi/bin/python scripts/test_pi05_dryrun.py --backend openpi --env-default aloha_sim
  python scripts/test_pi05_dryrun.py --backend mock     # always works
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "adapters", "policy_server"))


def _example_obs():
    import random

    return {
        "timestamp": time.time(),
        "task_instruction": "Stack the cubes with the Franka robot.",
        "robot": {
            "joint_positions": [random.uniform(-1, 1) for _ in range(7)],
            "joint_velocities": [0.0] * 7,
            "ee_position": [0.45, 0.0, 0.30],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_width": 0.04,
        },
        "images": {},
        "objects": [{"name": "cube", "position": [0.5, 0.1, 0.05], "quat": [0, 0, 0, 1], "confidence": 1.0}],
        "metadata": {"env_name": "dryrun", "episode_id": 0, "step_id": 0},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["openpi", "mock"], default="mock")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--env-default", default=None)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    out = os.path.join(_PROJ, "logs", "pi05_dryrun.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    result = {"backend_requested": args.backend, "backend_used": None, "ok": False}
    policy = None
    if args.backend == "openpi":
        try:
            from openpi_policy import OpenPIPolicy

            policy = OpenPIPolicy(config_name=args.config, ckpt_dir=args.ckpt, env_default=args.env_default)
            result["backend_used"] = "openpi"
        except Exception as e:  # noqa
            result["openpi_error"] = f"{type(e).__name__}: {e}"
            print(f"[dryrun] OpenPI backend failed ({e}); falling back to mock")

    if policy is None:
        from mock_policy import MockPolicy

        policy = MockPolicy()
        result["backend_used"] = "mock"

    obs = _example_obs()
    # warm-up
    action = policy.infer(obs)
    times = []
    for _ in range(args.iters):
        t0 = time.time()
        action = policy.infer(obs)
        times.append((time.time() - t0) * 1000.0)

    avg = sum(times) / len(times)
    result["ok"] = True
    result["avg_latency_ms"] = round(avg, 3)
    result["min_latency_ms"] = round(min(times), 3)
    result["action_keys"] = list(action.keys())
    result["delta_ee_position"] = action.get("delta_ee_position")
    result["delta_ee_rot"] = action.get("delta_ee_rot")
    result["gripper"] = action.get("gripper")
    raw = action.get("raw_model_output")
    result["raw_action_len"] = len(raw) if raw else None
    chunk = action.get("chunk")
    result["chunk_shape"] = [len(chunk), len(chunk[0])] if chunk else None

    print(json.dumps(result, indent=2))
    with open(out, "w") as f:
        f.write(json.dumps(result, indent=2))
    print(f"[dryrun] wrote {out}")


if __name__ == "__main__":
    main()
