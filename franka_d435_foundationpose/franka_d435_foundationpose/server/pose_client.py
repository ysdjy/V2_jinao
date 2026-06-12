"""Light ZMQ client for the pose server. Safe to import on the IsaacLab side.

Only depends on numpy + pyzmq. It serializes an RGBDFrame (+ object name and
optional mask) to the server and returns the decoded pose dict. It does NOT
import FoundationPose.

Wire format: a single JSON message (msgpack-free for readability/debuggability).
Images are base64-encoded PNG (rgb, mask) and base64 float32 buffer (depth).
"""

from __future__ import annotations

import base64
import json
import zlib

import numpy as np

from ..camera.frame_types import RGBDFrame


def _encode_array(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    raw = zlib.compress(arr.tobytes())
    return {
        "b64": base64.b64encode(raw).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "compressed": True,
    }


def _decode_array(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["b64"])
    if d.get("compressed"):
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def encode_request(
    frame: RGBDFrame,
    object_name: str,
    mesh_path: str,
    mask: np.ndarray | None = None,
    mode: str = "estimate",
    previous_pose: np.ndarray | None = None,
) -> dict:
    req = {
        "mode": mode,
        "object_name": object_name,
        "mesh_path": mesh_path,
        "camera_frame": frame.camera_frame,
        "timestamp": frame.timestamp,
        "rgb": _encode_array(frame.rgb),
        "depth": _encode_array(frame.depth.astype(np.float32)),
        "K": frame.K.tolist(),
    }
    if mask is not None:
        req["mask"] = _encode_array(np.asarray(mask).astype(np.uint8))
    if previous_pose is not None:
        req["previous_pose"] = np.asarray(previous_pose).tolist()
    return req


class PoseClient:
    """Synchronous request/reply ZMQ client (REQ socket)."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5599", timeout_ms: int = 60000):
        try:
            import zmq
        except Exception as e:  # pragma: no cover
            raise ImportError("pyzmq is required for PoseClient: pip install pyzmq") from e
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)
        self.endpoint = endpoint

    def estimate(self, frame, object_name, mesh_path, mask=None) -> dict:
        return self._send(encode_request(frame, object_name, mesh_path, mask, "estimate"))

    def track(self, frame, object_name, mesh_path, previous_pose) -> dict:
        return self._send(
            encode_request(
                frame, object_name, mesh_path, mode="track", previous_pose=previous_pose
            )
        )

    def ping(self) -> dict:
        return self._send({"mode": "ping"})

    def _send(self, req: dict) -> dict:
        self._sock.send_string(json.dumps(req))
        reply = self._sock.recv_string()
        resp = json.loads(reply)
        if "T_camera_object" in resp and resp["T_camera_object"] is not None:
            resp["T_camera_object"] = np.asarray(resp["T_camera_object"], dtype=np.float64)
        return resp

    def close(self):
        self._sock.close(0)
