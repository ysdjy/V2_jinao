"""A mock FoundationPose estimator for end-to-end plumbing tests.

It returns a deterministic ``T_camera_object`` placing the object ~0.5 m in
front of the camera (along +Z of the optical frame). When a mask is provided it
nudges the XY translation toward the mask centroid via the pinhole model so the
overlay lands roughly on the object — purely cosmetic, NOT a real estimate.

This lets the full pipeline (capture -> transforms -> save -> visualize) run
without FoundationPose, weights, CUDA, or a mesh.
"""

from __future__ import annotations

import numpy as np

from ..camera.frame_types import RGBDFrame
from .foundationpose_wrapper import PoseResult


class MockFoundationPoseEstimator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.default_z = float(self.cfg.get("mock_object_depth_m", 0.5))
        self._last_pose = None

    # ------------------------------------------------------------------ #
    def _pose_from_frame(self, frame: RGBDFrame, mask=None) -> np.ndarray:
        """Build a plausible T_camera_object from the frame + optional mask."""
        z = self.default_z
        cx, cy = frame.cx, frame.cy
        fx, fy = frame.fx, frame.fy

        # Default: straight ahead at the principal point.
        u, v = cx, cy
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
            if mask.any():
                ys, xs = np.nonzero(mask)
                u, v = float(xs.mean()), float(ys.mean())
                # If we have valid depth under the mask, use its median.
                d = frame.depth[ys, xs]
                d = d[(d > 0) & np.isfinite(d)]
                if d.size:
                    z = float(np.median(d))

        # Back-project the (u, v, z) pixel to the camera optical frame.
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [x, y, z]
        return T

    # ------------------------------------------------------------------ #
    def estimate(self, frame: RGBDFrame, mesh_path: str, mask) -> PoseResult:
        T = self._pose_from_frame(frame, mask)
        self._last_pose = T
        return PoseResult(
            T_camera_object=T,
            score=None,
            success=True,
            mode="estimate",
            debug={
                "backend": "mock",
                "note": "MOCK pose — not a real estimate",
                "mesh_path": mesh_path,
            },
        )

    def track(self, frame: RGBDFrame, previous_pose) -> PoseResult:
        T = np.asarray(previous_pose, dtype=np.float64).reshape(4, 4)
        self._last_pose = T
        return PoseResult(
            T_camera_object=T,
            score=None,
            success=True,
            mode="track",
            debug={"backend": "mock", "note": "MOCK track — returns previous pose"},
        )
