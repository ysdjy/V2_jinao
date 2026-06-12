"""Teleop safety/smoothing controller — framework-agnostic (numpy only).

Turns a raw GELLO arm pose + gripper angle into a safe Franka joint command, with behaviour
that is **independent of the sim loop rate** (this is the fix for "slow / low-frequency"
following): both the low-pass and the rate limit are expressed in physical units (seconds /
rad·s⁻¹) and scaled by the per-step ``dt``, so the arm keeps up whether the GUI runs at 4 Hz
or 60 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GripperMapCfg:
    """Binary gripper mapping from the raw GELLO gripper angle (degrees) with hysteresis."""

    enabled: bool = True
    open_raw: float = 211.0       # raw angle (deg) when GELLO gripper is fully OPEN
    close_raw: float = 169.0      # raw angle (deg) when fully CLOSED
    deadband: float = 3.0         # hysteresis band (deg) around the threshold
    invert: bool = False

    @property
    def threshold(self) -> float:
        return 0.5 * (self.open_raw + self.close_raw)


@dataclass
class ControllerCfg:
    """Smoothing + safety gains. Defaults tuned for responsive teleop at any sim rate."""

    smoothing_tau: float = 0.08   # low-pass time constant (s); alpha_eff = clamp(dt/tau, 0, 1)
    max_joint_vel: float = 2.5    # rate limit (rad/s); per-step cap = max_joint_vel * dt
    abs_step_cap: float = 0.6     # hard per-step cap (rad) so a dt spike can't cause a jump
    jump_warn: float = 1.5        # warn if a single raw GELLO sample jumps more than this (rad)
    gripper: GripperMapCfg = None  # set in __post_init__

    def __post_init__(self):
        if self.gripper is None:
            self.gripper = GripperMapCfg()


class TeleopController:
    """Holds ``q_cmd`` and advances it toward the GELLO target each step.

    Usage:
        ctrl = TeleopController(num_arm=7, q_lower=lo, q_upper=hi, cfg=ControllerCfg())
        ctrl.reset(current_franka_q)            # no startup jump: start at the robot's pose
        q_cmd, gripper_cmd = ctrl.step(q_gello, gripper_deg, dt)
    """

    def __init__(self, num_arm: int, q_lower: np.ndarray, q_upper: np.ndarray, cfg: ControllerCfg | None = None):
        self.num_arm = int(num_arm)
        self.cfg = cfg or ControllerCfg()
        self.q_lower = np.asarray(q_lower, dtype=float).reshape(-1)
        self.q_upper = np.asarray(q_upper, dtype=float).reshape(-1)
        self.q_cmd = np.zeros(self.num_arm, dtype=float)
        self.q_target = np.zeros(self.num_arm, dtype=float)
        self._last_gello = None
        self._gripper_state = "open"  # commanded state: open -> +1.0, close -> -1.0
        self._last_gripper_cmd = 1.0

    def reset(self, current_franka_q: np.ndarray):
        """Seed q_cmd/q_target at the robot's current pose so there is no startup jump."""
        q = np.asarray(current_franka_q, dtype=float).reshape(-1)[: self.num_arm]
        self.q_cmd = q.copy()
        self.q_target = q.copy()
        self._last_gello = None

    def update_gripper(self, gripper_deg: float | None) -> float:
        """Binary gripper command (1.0 open / -1.0 close) with hysteresis. None -> hold last."""
        g = self.cfg.gripper
        if not g.enabled or gripper_deg is None:
            return 1.0 if not g.enabled else self._last_gripper_cmd
        thr, db = g.threshold, g.deadband
        if not g.invert:
            if self._gripper_state == "open" and gripper_deg < thr - db:
                self._gripper_state = "close"
            elif self._gripper_state == "close" and gripper_deg > thr + db:
                self._gripper_state = "open"
        else:
            if self._gripper_state == "open" and gripper_deg > thr + db:
                self._gripper_state = "close"
            elif self._gripper_state == "close" and gripper_deg < thr - db:
                self._gripper_state = "open"
        self._last_gripper_cmd = 1.0 if self._gripper_state == "open" else -1.0
        return self._last_gripper_cmd

    def step(self, q_gello: np.ndarray | None, gripper_deg: float | None, dt: float):
        """Advance q_cmd toward q_gello by one step of length ``dt`` (s). Returns (q_cmd, gripper_cmd).

        If ``q_gello`` is None (read failed), q_cmd is held. Behaviour is dt-scaled so the
        following speed is the same at any loop rate.
        """
        dt = max(1e-3, float(dt))
        if q_gello is not None:
            q_g = np.asarray(q_gello, dtype=float).reshape(-1)[: self.num_arm]
            if not np.any(np.isnan(q_g)):
                if self._last_gello is not None:
                    jump = float(np.max(np.abs(q_g - self._last_gello)))
                    if jump > self.cfg.jump_warn:
                        print(f"[teleop-ctrl] large GELLO jump {jump:.2f} rad (dropped packet?)", flush=True)
                self._last_gello = q_g
                self.q_target = np.clip(q_g, self.q_lower, self.q_upper)  # joint-limit clip

        # frame-rate-independent low-pass toward the target
        alpha = min(1.0, dt / max(1e-4, self.cfg.smoothing_tau))
        q_filtered = (1.0 - alpha) * self.q_cmd + alpha * self.q_target
        # rate limit by physical joint velocity, with an absolute safety cap
        max_delta = min(self.cfg.abs_step_cap, self.cfg.max_joint_vel * dt)
        delta = np.clip(q_filtered - self.q_cmd, -max_delta, max_delta)
        self.q_cmd = self.q_cmd + delta

        gripper_cmd = self.update_gripper(gripper_deg)
        return self.q_cmd.copy(), gripper_cmd
