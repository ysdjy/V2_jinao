"""Minimal example: query a running policy server (mock or pi0.5) from python.

Start a server first, e.g.:
  PYBIN=python3 bash scripts/start_mock_server.sh 8008

Then:
  python examples/example_infer_client.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters", "policy_server"))

from client import PolicyClient  # noqa: E402


def main():
    client = PolicyClient(host="127.0.0.1", port=8008, timeout=5.0)
    if not client.wait_until_healthy(retries=5, delay=0.5):
        print("server not reachable on :8008 — start it with scripts/start_mock_server.sh")
        return
    print("health:", client.health())

    observation = {
        "timestamp": 0.0,
        "task_instruction": "Stack the cubes with the Franka robot.",
        "robot": {
            "joint_positions": [0.0, -0.2, 0.0, -2.5, 0.0, 2.4, 0.7, 0.04, 0.04],
            "joint_velocities": [0.0] * 9,
            "ee_position": [0.45, 0.0, 0.30],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_width": 0.08,
        },
        "images": {},  # image_mode=none
        "objects": [{"name": "cube_1", "position": [0.5, 0.1, 0.05],
                     "quat": [0, 0, 0, 1], "confidence": 1.0}],
        "metadata": {"env_name": "example", "episode_id": 0, "step_id": 0},
    }

    action, latency_ms = client.infer(observation)
    print(f"latency: {latency_ms:.2f} ms")
    print("action:", action)


if __name__ == "__main__":
    main()
