"""GelloFrankaTeleop — one-call facade: connect a GELLO leader arm to an IsaacLab Franka.

Scene-agnostic: works with any IsaacLab manager-based env that has a Franka ``robot`` with
``panda_joint.*`` joints. Typical use inside a host sim loop:

    from gello_isaac_teleop import GelloFrankaTeleop, GelloTeleopConfig

    teleop = GelloFrankaTeleop(env, GelloTeleopConfig(
        gello_config="projects/gello_franka_teleop/configs/gello_franka.yaml"))
    teleop.start()                       # opens device, seeds q_cmd at current Franka q (no jump)
    while running:
        action = teleop.step(dt)         # reads GELLO, returns the env action (joint targets + gripper)
        env.step(action)
    teleop.stop()

For envs whose action space is NOT joint-position (e.g. IK-relative), call
``teleop.compute_joint_targets(dt)`` and map ``(q_cmd, gripper_cmd)`` to your own action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .controller import ControllerCfg, GripperMapCfg, TeleopController
from .isaac_binding import IsaacFrankaBinding
from .reader import GelloReader, ThreadedGelloReader


@dataclass
class GelloTeleopConfig:
    gello_config: str = "projects/gello_franka_teleop/configs/gello_franka.yaml"
    override_port: str | None = None
    read_hz: float = 60.0                 # background serial read rate (decoupled from sim)
    # smoothing / rate (frame-rate independent — the fix for slow/low-Hz following)
    smoothing_tau: float = 0.08
    max_joint_vel: float = 2.5            # rad/s
    abs_step_cap: float = 0.6
    start_tolerance: float = 0.8          # warn if startup |q_gello - q_franka| exceeds this
    # gripper (ON by default — the fix for "gripper not controllable")
    enable_gripper: bool = True
    gripper_open_raw: float = 211.0
    gripper_close_raw: float = 169.0
    gripper_deadband: float = 3.0
    gripper_invert: bool = False
    # env binding overrides (rarely needed)
    robot_key: str = "robot"
    arm_joint_pattern: str = "panda_joint.*"
    arm_action_term: str | None = None
    gripper_action_term: str | None = None


class GelloFrankaTeleop:
    def __init__(self, env, cfg: GelloTeleopConfig | None = None):
        self.cfg = cfg or GelloTeleopConfig()
        self.bind = IsaacFrankaBinding(
            env,
            robot_key=self.cfg.robot_key,
            arm_joint_pattern=self.cfg.arm_joint_pattern,
            arm_action_term=self.cfg.arm_action_term,
            gripper_action_term=self.cfg.gripper_action_term,
        )
        self._reader: ThreadedGelloReader | None = None
        self._ctrl: TeleopController | None = None
        self._last_loop_t = time.perf_counter()
        self._loop_hz = 0.0
        self._last = {"q_gello": None, "gripper_raw": None, "read_hz": 0.0, "read_ms": 0.0, "ok": False}

    # ---- lifecycle ----
    def start(self):
        base = GelloReader(self.cfg.gello_config, use_gripper=self.cfg.enable_gripper,
                           override_port=self.cfg.override_port)
        if base.num_arm != self.bind.num_arm:
            print(f"[gello-isaac][warn] GELLO has {base.num_arm} arm joints but Franka arm has "
                  f"{self.bind.num_arm}; using min.", flush=True)
        lower, upper, src = self.bind.read_arm_limits()
        ctrl_cfg = ControllerCfg(
            smoothing_tau=self.cfg.smoothing_tau,
            max_joint_vel=self.cfg.max_joint_vel,
            abs_step_cap=self.cfg.abs_step_cap,
            gripper=GripperMapCfg(
                enabled=self.cfg.enable_gripper,
                open_raw=self.cfg.gripper_open_raw,
                close_raw=self.cfg.gripper_close_raw,
                deadband=self.cfg.gripper_deadband,
                invert=self.cfg.gripper_invert,
            ),
        )
        self._ctrl = TeleopController(self.bind.num_arm, lower.cpu().numpy(), upper.cpu().numpy(), ctrl_cfg)
        franka_q = self.bind.read_arm_q().cpu().numpy()
        self._ctrl.reset(franka_q)  # no startup jump
        self._reader = ThreadedGelloReader(base, hz=self.cfg.read_hz).start()

        # startup-gap report (read once via the underlying reader)
        q0, _g0, ok0 = base.read()
        if ok0 and q0 is not None:
            gap = float(np.max(np.abs(q0[: self.bind.num_arm] - franka_q[: self.bind.num_arm])))
            print(f"[gello-isaac] joint limits from {src}; startup gap = {gap:.3f} rad "
                  f"(ramps in, no jump). gripper={'on' if self.cfg.enable_gripper else 'off'}", flush=True)
            if gap > self.cfg.start_tolerance:
                print(f"[gello-isaac][warn] large startup gap ({gap:.3f} > {self.cfg.start_tolerance}); "
                      "hold GELLO near the robot pose for a gentler start.", flush=True)
        return self

    def stop(self):
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

    # ---- per-step ----
    def compute_joint_targets(self, dt: float | None = None):
        """Read latest GELLO sample and advance q_cmd. Returns (q_cmd[np], gripper_cmd[float]).

        ``dt`` is the time step (s) used for the frame-rate-independent smoothing/rate-limit.
        If None (recommended), the real wall-clock time since the last call is used, so the
        following speed stays consistent whether the sim loop runs at 4 Hz or 60 Hz.
        """
        assert self._reader is not None and self._ctrl is not None, "call start() first"
        now = time.perf_counter()
        loop_dt = now - self._last_loop_t
        self._last_loop_t = now
        if loop_dt > 0:
            inst = 1.0 / loop_dt
            self._loop_hz = inst if self._loop_hz == 0.0 else 0.9 * self._loop_hz + 0.1 * inst
        step_dt = loop_dt if dt is None else dt
        q_gello, gripper_raw, ok, read_ms, read_hz = self._reader.get_latest()
        q_cmd, gripper_cmd = self._ctrl.step(q_gello, gripper_raw, step_dt)
        self._last = {"q_gello": q_gello, "gripper_raw": gripper_raw, "read_hz": read_hz,
                      "read_ms": read_ms, "ok": ok, "gripper_cmd": gripper_cmd}
        return q_cmd, gripper_cmd

    def step(self, dt: float | None = None):
        """Convenience for joint-position envs: returns a ready-to-use env action tensor."""
        q_cmd, gripper_cmd = self.compute_joint_targets(dt)
        return self.bind.make_action(q_cmd, gripper_cmd)

    def reseat(self):
        """Re-seed q_cmd at the current Franka pose (call after an env.reset())."""
        if self._ctrl is not None:
            self._ctrl.reset(self.bind.read_arm_q().cpu().numpy())

    def telemetry(self) -> dict:
        """Latest diagnostics for logging/printing."""
        ctrl = self._ctrl
        out = dict(self._last)
        out["real_loop_hz"] = round(self._loop_hz, 1)
        if ctrl is not None:
            out["q_cmd"] = np.round(ctrl.q_cmd, 3).tolist()
            out["q_target"] = np.round(ctrl.q_target, 3).tolist()
            out["gripper_state"] = ctrl._gripper_state
            out["franka_q"] = np.round(self.bind.read_arm_q().cpu().numpy(), 3).tolist()
        return out
