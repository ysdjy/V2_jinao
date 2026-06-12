"""Safety filter: clamp/scrub policy actions before they touch the simulator.

Pure-python (numpy optional). Reads limits from configs/safety_limits.yaml but
also works with sane built-in defaults if yaml/file are unavailable.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

_DEFAULTS = {
    "delta_position_max_norm": 0.03,
    "delta_position_clip": 0.03,
    "delta_rotation_max_norm": 0.15,
    "gripper_clip": [-1.0, 1.0],
    "joint_min": [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8],
    "joint_max": [2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8],
    "joint_delta_clip": 0.10,
    "workspace_min": [0.10, -0.55, 0.0],
    "workspace_max": [0.95, 0.55, 0.9],
    "max_consecutive_policy_errors": 5,
    "nan_replacement": 0.0,
}


def _load_yaml(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        try:
            import yaml

            with open(path) as f:
                return {**_DEFAULTS, **(yaml.safe_load(f) or {})}
        except Exception:
            pass
    return dict(_DEFAULTS)


def _finite(x: float, repl: float) -> float:
    return x if (x is not None and math.isfinite(x)) else repl


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _l2(v) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in v))


class SafetyFilter:
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = _load_yaml(config_path)
        self.consecutive_errors = 0
        self.num_clips = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        self.consecutive_errors = 0

    def note_policy_error(self) -> bool:
        """Call when a policy error/timeout/fallback occurred.
        Returns True if an episode reset should be triggered."""
        self.consecutive_errors += 1
        return self.consecutive_errors >= int(self.cfg["max_consecutive_policy_errors"])

    def note_policy_ok(self):
        self.consecutive_errors = 0

    # ------------------------------------------------------------------ #
    def filter_delta_ee(self, action: dict[str, Any], ee_position: Optional[list[float]] = None) -> dict[str, Any]:
        repl = float(self.cfg["nan_replacement"])
        dpos = [_finite(float(v), repl) for v in action.get("delta_ee_position", [0, 0, 0])[:3]]
        drot = [_finite(float(v), repl) for v in action.get("delta_ee_rot", [0, 0, 0])[:3]]
        grip = _finite(float(action.get("gripper", 0.0)), repl)

        # per-axis position clip
        c = float(self.cfg["delta_position_clip"])
        clipped = [_clip(v, -c, c) for v in dpos]
        if clipped != dpos:
            self.num_clips += 1
        dpos = clipped
        # L2 position norm scale-down
        n = _l2(dpos)
        nmax = float(self.cfg["delta_position_max_norm"])
        if n > nmax > 0:
            dpos = [v * nmax / n for v in dpos]
            self.num_clips += 1

        # rotation L2 clip
        rn = _l2(drot)
        rmax = float(self.cfg["delta_rotation_max_norm"])
        if rn > rmax > 0:
            drot = [v * rmax / rn for v in drot]
            self.num_clips += 1

        # gripper clip
        glo, ghi = self.cfg["gripper_clip"]
        gclip = _clip(grip, float(glo), float(ghi))
        if gclip != grip:
            self.num_clips += 1
        grip = gclip

        # optional workspace clamp on the *resulting* ee position
        if ee_position is not None and len(ee_position) >= 3:
            wmin = self.cfg["workspace_min"]
            wmax = self.cfg["workspace_max"]
            for i in range(3):
                target = float(ee_position[i]) + dpos[i]
                clamped = _clip(target, float(wmin[i]), float(wmax[i]))
                if clamped != target:
                    dpos[i] = clamped - float(ee_position[i])
                    self.num_clips += 1

        out = dict(action)
        out["delta_ee_position"] = dpos
        out["delta_ee_rot"] = drot
        out["gripper"] = grip
        return out

    def filter_joint(self, joint_targets: list[float], current: Optional[list[float]] = None,
                     is_delta: bool = False) -> list[float]:
        repl = float(self.cfg["nan_replacement"])
        jt = [_finite(float(v), repl) for v in joint_targets]
        if is_delta:
            d = float(self.cfg["joint_delta_clip"])
            jt = [_clip(v, -d, d) for v in jt]
            if current is not None:
                jt = [float(current[i]) + jt[i] for i in range(min(len(jt), len(current)))]
        jmin, jmax = self.cfg["joint_min"], self.cfg["joint_max"]
        out = []
        for i, v in enumerate(jt):
            lo = float(jmin[i]) if i < len(jmin) else -6.28
            hi = float(jmax[i]) if i < len(jmax) else 6.28
            cv = _clip(v, lo, hi)
            if cv != v:
                self.num_clips += 1
            out.append(cv)
        return out

    def safe_zero_action(self) -> dict[str, Any]:
        return {
            "action_type": "delta_ee_pose",
            "delta_ee_position": [0.0, 0.0, 0.0],
            "delta_ee_rot": [0.0, 0.0, 0.0],
            "gripper": 0.0,
        }
