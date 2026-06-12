"""GELLO device reader — framework-agnostic (numpy + gello only, no IsaacLab).

Reads a GELLO leader arm over its Dynamixel serial bus and returns arm joint angles
(radians) plus the raw gripper angle (degrees). A background-thread variant decouples the
(blocking) serial read from the caller's loop so the consumer never stalls on I/O.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("gello_isaac_teleop.reader needs pyyaml: pip install pyyaml") from exc


class GelloReader:
    """Synchronous GELLO reader built on gello's own ``DynamixelRobot``.

    :meth:`read` does a SINGLE serial read and returns ``(arm_q[num_arm], gripper_deg|None, ok)``.
    The gripper is the RAW servo angle in degrees (matching ``gello_get_offset.py`` output,
    e.g. open~211 / close~169), NOT gello's normalized [0,1] — so degree-based thresholds work.
    """

    def __init__(self, config_path: str | Path, *, use_gripper: bool = True, override_port: str | None = None):
        from gello.robots.dynamixel import DynamixelRobot  # lazy: only needed at device init

        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"GELLO config not found: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        port = override_port or cfg.get("port")
        if not port or port == "None":
            raise RuntimeError(
                "GELLO config has no 'port'. Run detect_gello_port.sh and fill configs/gello_franka.yaml."
            )
        self.port = port
        joint_ids = list(cfg["joint_ids"])
        joint_offsets = list(cfg["joint_offsets"])
        joint_signs = list(cfg["joint_signs"])
        self.num_arm = len(joint_ids)
        baudrate = int(cfg.get("baudrate", 57600))

        gcfg = cfg.get("gripper") or {}
        gripper_config = None
        if use_gripper and gcfg.get("enabled", False):
            gripper_config = (int(gcfg["id"]), float(gcfg["open_value"]), float(gcfg["close_value"]))

        start = np.array(cfg.get("start_joints", [0.0] * self.num_arm), dtype=float)
        if start.shape[0] != self.num_arm:
            raise ValueError(f"start_joints must have {self.num_arm} entries, got {start.shape[0]}")
        start_full = np.concatenate([start, [0.0]]) if gripper_config is not None else start

        def _build(gc, st):
            robot = DynamixelRobot(
                joint_ids=joint_ids,
                joint_offsets=joint_offsets,
                joint_signs=joint_signs,
                real=True,
                port=port,
                baudrate=baudrate,
                gripper_config=gc,
                start_joints=st,
            )
            # gello silently falls back to a fake (all-zeros) driver if the port can't open
            # (usually a dialout permission problem). Reject it loudly instead of teleoping garbage.
            if getattr(robot._driver, "_is_fake", False):
                raise PermissionError(
                    "GELLO serial port did not open (gello fell back to its fake driver). "
                    "Check dialout permission, e.g. run via: sg dialout -c \"<command>\"."
                )
            return robot

        try:
            self._robot = _build(gripper_config, start_full)
            self.gripper_enabled = gripper_config is not None
        except PermissionError:
            raise
        except Exception as exc:  # gripper servo may be missing -> degrade to arm-only
            print(f"[gello-reader] gripper init failed ({exc}); reading {self.num_arm} arm joints only.", flush=True)
            self._robot = _build(None, start)
            self.gripper_enabled = False

    def read(self):
        """Return ``(arm_q: np.ndarray[num_arm], gripper_deg: float | None, ok: bool)``."""
        try:
            raw = np.asarray(self._robot._driver.get_joints(), dtype=float)  # radians; len num_arm (+1)
        except Exception:  # noqa: BLE001 - keep teleop alive across a transient read error
            return None, None, False
        n = self.num_arm
        if raw.ndim != 1 or raw.shape[0] < n:
            return None, None, False
        offsets = np.asarray(self._robot._joint_offsets, dtype=float)
        signs = np.asarray(self._robot._joint_signs, dtype=float)
        arm_q = (raw[:n] - offsets[:n]) * signs[:n]
        if np.any(np.isnan(arm_q)):
            return None, None, False
        gripper_deg = float(np.rad2deg(raw[n])) if (self.gripper_enabled and raw.shape[0] > n) else None
        return arm_q, gripper_deg, True


class ThreadedGelloReader:
    """Runs a :class:`GelloReader` in a daemon thread; consumers grab the latest sample (non-blocking).

    Serial reads release the GIL, so the read thread genuinely overlaps with the caller's
    sim/render work — this is what keeps teleop responsive even when the sim loop is slow.
    """

    def __init__(self, reader: GelloReader, hz: float = 60.0):
        self._reader = reader
        self._period = 1.0 / max(1.0, float(hz))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="gello-reader", daemon=True)
        self._latest_q = None
        self._latest_gripper = None
        self._last_ok = False
        self._read_ms = 0.0
        self._read_hz = 0.0
        self.port = reader.port
        self.num_arm = reader.num_arm
        self.gripper_enabled = reader.gripper_enabled

    def start(self) -> "ThreadedGelloReader":
        self._thread.start()
        return self

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            q, gripper, ok = self._reader.read()
            t1 = time.perf_counter()
            with self._lock:
                if ok and q is not None:
                    self._latest_q = q
                    self._latest_gripper = gripper
                self._last_ok = ok
                self._read_ms = (t1 - t0) * 1000.0
                dt = t1 - last
                last = t1
                if dt > 0:
                    inst = 1.0 / dt
                    self._read_hz = inst if self._read_hz == 0.0 else 0.9 * self._read_hz + 0.1 * inst
            sleep_t = self._period - (t1 - t0)
            if sleep_t > 0:
                self._stop.wait(sleep_t)

    def get_latest(self):
        """Return ``(arm_q|None, gripper_deg|None, last_ok, read_ms, read_hz)`` (non-blocking)."""
        with self._lock:
            q = None if self._latest_q is None else self._latest_q.copy()
            return q, self._latest_gripper, self._last_ok, self._read_ms, self._read_hz

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
