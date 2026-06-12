"""Template success metric for a custom task.

Two ways to define success:
  1. Use the env's own `terminations.success` term (preferred when it exists) --
     the rollout runner already reads `terminated`.
  2. Implement a geometric check here (e.g. object within tolerance of a goal).

Edit `check_success` for your task.
"""

from __future__ import annotations

from typing import Any


def check_success(obs: dict[str, Any], goal: dict[str, Any] | None = None, pos_tol: float = 0.03) -> bool:
    """Example: success if the first object is within pos_tol of goal['position']."""
    if not goal:
        return False
    objs = obs.get("objects", [])
    if not objs:
        return False
    p = objs[0].get("position")
    g = goal.get("position")
    if not p or not g:
        return False
    d = sum((float(p[i]) - float(g[i])) ** 2 for i in range(3)) ** 0.5
    return d <= pos_tol


def final_ee_error(obs: dict[str, Any], goal_ee: list[float]) -> float:
    ee = obs.get("robot", {}).get("ee_position", [0, 0, 0])
    return sum((float(ee[i]) - float(goal_ee[i])) ** 2 for i in range(3)) ** 0.5
