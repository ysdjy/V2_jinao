"""STUB: convert a policy action into a future Franka FCI command.

This does NOT control hardware. It defines the interface and applies the SAME
safety filter used in sim so sim->real action handling stays consistent.

Real implementation TODO (do NOT mix this into a training/conda env -- run it in
a dedicated real-time control process; see docs/real_franka_d435_migration.md):
  * connect via libfranka / franka_ros2 FCI (requires the FR3/Panda FCI setup)
  * map delta_ee_pose -> Cartesian impedance / motion generator target
  * enforce collision thresholds, joint/velocity limits, watchdog
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "action_adapters"))
from safety_filter import SafetyFilter  # noqa: E402


class FrankaFCIActionStub:
    def __init__(self, safety_config: str | None = None):
        self.sf = SafetyFilter(safety_config)
        self.connected = False

    def connect(self, robot_ip: str = "172.16.0.2"):
        # TODO: libfranka connection. Intentionally not implemented.
        raise NotImplementedError("Real FCI connection not implemented in the stub.")

    def to_fci_command(self, action: dict[str, Any], ee_position=None) -> dict[str, Any]:
        """Returns a dict describing the command we WOULD send (after safety)."""
        safe = self.sf.filter_delta_ee(action, ee_position=ee_position)
        return {
            "type": "cartesian_delta",
            "delta_position_m": safe["delta_ee_position"],
            "delta_rotation_rad": safe["delta_ee_rot"],
            "gripper_normalized": safe["gripper"],
            "safety_clips": self.sf.num_clips,
            "_note": "STUB - not sent to hardware",
        }
