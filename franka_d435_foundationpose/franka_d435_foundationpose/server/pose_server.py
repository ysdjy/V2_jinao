"""ZMQ pose server — runs in the `foundationpose` env only.

Receives an encoded RGB-D request from PoseClient, runs the
FoundationPoseEstimator (real or mock), and replies with the pose dict.

This module imports FoundationPoseEstimator, which may import FoundationPose;
DO NOT run it inside env_isaaclab. Start it with:
    conda run -n foundationpose python .../scripts/start_pose_server.py
"""

from __future__ import annotations

import json

import numpy as np

from ..camera.frame_types import RGBDFrame
from ..foundationpose.foundationpose_wrapper import FoundationPoseEstimator
from ..utils.logging_utils import get_logger
from .pose_client import _decode_array

logger = get_logger("pose_server")


class PoseServer:
    """REP-socket server wrapping a FoundationPoseEstimator."""

    def __init__(self, config_path: str, endpoint: str = "tcp://0.0.0.0:5599", force_mock: bool = False):
        try:
            import zmq
        except Exception as e:  # pragma: no cover
            raise ImportError("pyzmq is required for PoseServer: pip install pyzmq") from e
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(endpoint)
        self.endpoint = endpoint
        self.estimator = FoundationPoseEstimator(config_path, force_mock=force_mock)
        logger.info("PoseServer bound at %s (backend=%s)", endpoint, self.estimator.backend)

    def _frame_from_request(self, req: dict) -> RGBDFrame:
        rgb = _decode_array(req["rgb"]).astype(np.uint8)
        depth = _decode_array(req["depth"]).astype(np.float32)
        K = np.asarray(req["K"], dtype=np.float64)
        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            K=K,
            timestamp=float(req.get("timestamp", 0.0)),
            camera_frame=req.get("camera_frame", "d435_color_optical_frame"),
        )

    def handle(self, req: dict) -> dict:
        mode = req.get("mode", "estimate")
        if mode == "ping":
            return {"ok": True, "backend": self.estimator.backend, "mode": "ping"}

        frame = self._frame_from_request(req)
        mesh_path = req.get("mesh_path")
        try:
            if mode == "track":
                prev = np.asarray(req["previous_pose"], dtype=np.float64)
                result = self.estimator.track(frame, prev)
            else:
                mask = None
                if "mask" in req:
                    mask = _decode_array(req["mask"]).astype(bool)
                result = self.estimator.estimate(frame, mesh_path, mask)
        except Exception as e:  # pragma: no cover - report errors to client
            logger.exception("pose computation failed")
            return {"success": False, "error": str(e), "mode": mode}

        return {
            "success": bool(result.success),
            "mode": result.mode,
            "score": result.score,
            "backend": self.estimator.backend,
            "T_camera_object": result.T_camera_object.tolist(),
            "debug": result.debug,
        }

    def serve_forever(self):
        logger.info("PoseServer listening... (Ctrl-C to stop)")
        while True:
            msg = self._sock.recv_string()
            try:
                req = json.loads(msg)
                resp = self.handle(req)
            except Exception as e:  # pragma: no cover
                resp = {"success": False, "error": f"bad request: {e}"}
            self._sock.send_string(json.dumps(resp))

    def close(self):
        self._sock.close(0)
