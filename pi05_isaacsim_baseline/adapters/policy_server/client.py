"""Policy-server client used by the IsaacLab side (stdlib only).

On timeout / error it returns a SAFE fallback action (zero motion, hold gripper)
so the IsaacLab rollout never crashes because of the network or the server.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Optional

logger = logging.getLogger("policy_client")


SAFE_ACTION = {
    "action_type": "delta_ee_pose",
    "delta_ee_position": [0.0, 0.0, 0.0],
    "delta_ee_rot": [0.0, 0.0, 0.0],
    "gripper": 0.0,
    "chunk": None,
    "raw_model_output": None,
    "_fallback": True,
}


class PolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8008, timeout: float = 5.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.num_timeouts = 0
        self.num_errors = 0

    def health(self) -> Optional[dict[str, Any]]:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa
            logger.warning("health check failed: %s", e)
            return None

    def wait_until_healthy(self, retries: int = 30, delay: float = 1.0) -> bool:
        for _ in range(retries):
            if self.health() is not None:
                return True
            time.sleep(delay)
        return False

    def infer(self, observation: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Returns (action_dict, latency_ms). Falls back to SAFE_ACTION on failure."""
        body = json.dumps(observation).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/infer", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            latency_ms = (time.time() - t0) * 1000.0
            action = resp.get("action", dict(SAFE_ACTION))
            ok, msg = _basic_shape_check(action)
            if not ok:
                logger.warning("action failed shape check (%s); using safe fallback", msg)
                self.num_errors += 1
                return dict(SAFE_ACTION), latency_ms
            return action, latency_ms
        except TimeoutError:
            self.num_timeouts += 1
            logger.warning("policy server timed out; using safe fallback")
            return dict(SAFE_ACTION), (time.time() - t0) * 1000.0
        except Exception as e:  # noqa
            self.num_errors += 1
            logger.warning("policy inference error (%s); using safe fallback", e)
            return dict(SAFE_ACTION), (time.time() - t0) * 1000.0


def _basic_shape_check(action: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(action, dict):
        return False, "not a dict"
    at = action.get("action_type", "delta_ee_pose")
    if at == "delta_ee_pose":
        if len(action.get("delta_ee_position", [])) != 3:
            return False, "delta_ee_position != 3"
    elif action.get("joint_targets") is None:
        return False, "missing joint_targets"
    return True, "ok"
