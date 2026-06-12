"""Mock policy backend.

Produces deterministic, *safe*, small actions from an observation so the whole
IsaacLab <-> policy-server <-> action pipeline can be exercised without any model
weights. The action is intentionally tiny (well within the safety limits) and
gently nudges the end effector toward the first detected object if one is present,
otherwise it emits a slow descending motion + periodic gripper toggle.

Pure stdlib + (optional) numpy. No model, no GPU, no network.
"""

from __future__ import annotations

import math
from typing import Any


class MockPolicy:
    backend_name = "mock"

    def __init__(self, action_horizon: int = 1, seed: int = 0):
        self.action_horizon = max(1, int(action_horizon))
        self._step = 0

    # ------------------------------------------------------------------ #
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """obs: Observation.to_dict(). Returns Action.to_dict()."""
        self._step += 1
        robot = obs.get("robot", {}) or {}
        ee = list(robot.get("ee_position", [0.0, 0.0, 0.0]))[:3]
        while len(ee) < 3:
            ee.append(0.0)

        objects = obs.get("objects", []) or []
        if objects and objects[0].get("position"):
            tgt = list(objects[0]["position"])[:3]
            while len(tgt) < 3:
                tgt.append(0.0)
            # proportional step toward target, clipped small
            dpos = [_clip((tgt[i] - ee[i]) * 0.1, -0.02, 0.02) for i in range(3)]
        else:
            # gentle descending probe
            dpos = [0.0, 0.0, -0.005]

        # small oscillating yaw, periodic gripper toggle
        drot = [0.0, 0.0, 0.01 * math.sin(self._step * 0.2)]
        gripper = 1.0 if (self._step // 20) % 2 == 0 else -1.0

        def _vec():
            return [round(v, 6) for v in dpos] + [round(v, 6) for v in drot] + [gripper]

        chunk = [_vec() for _ in range(self.action_horizon)] if self.action_horizon > 1 else None

        return {
            "action_type": "delta_ee_pose",
            "delta_ee_position": [round(v, 6) for v in dpos],
            "delta_ee_rot": [round(v, 6) for v in drot],
            "gripper": gripper,
            "chunk": chunk,
            "raw_model_output": _vec(),
        }


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
