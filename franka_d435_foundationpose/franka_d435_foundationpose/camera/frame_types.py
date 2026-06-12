"""Unified RGB-D frame container shared by sim, RealSense and saved providers.

This is the single interface contract between the camera side and the
FoundationPose side. Keep it light (numpy only) so it imports cleanly inside
``env_isaaclab``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..utils import depth_utils, image_io


@dataclass
class RGBDFrame:
    """A single RGB-D frame with intrinsics and provenance metadata.

    Attributes
    ----------
    rgb : (H, W, 3) uint8, RGB order (NOT BGR).
    depth : (H, W) float32, metric depth in **meters**, aligned to color.
    K : (3, 3) float64 camera intrinsics for the color stream.
    timestamp : float seconds.
    camera_frame : name of the optical frame, e.g. "d435_color_optical_frame".
    depth_aligned_to_color : whether depth is aligned to the color image.
    metadata : free-form dict (serial number, sim/real, exposure, ...).
    """

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    timestamp: float = 0.0
    camera_frame: str = "d435_color_optical_frame"
    depth_aligned_to_color: bool = True
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # construction / normalization
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self.rgb = np.asarray(self.rgb)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.timestamp = float(self.timestamp)
        if self.metadata is None:
            self.metadata = {}

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self, raise_on_error: bool = True) -> bool:
        """Check shapes, dtypes, depth units and intrinsics validity."""
        msgs = []
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            msgs.append(f"rgb must be (H, W, 3), got {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            msgs.append(f"rgb must be uint8, got {self.rgb.dtype}")
        if self.depth.ndim != 2:
            msgs.append(f"depth must be (H, W), got {self.depth.shape}")
        if self.depth.dtype != np.float32:
            msgs.append(f"depth must be float32 (meters), got {self.depth.dtype}")
        if self.rgb.ndim == 3 and self.depth.ndim == 2:
            if self.rgb.shape[:2] != self.depth.shape:
                msgs.append(
                    f"rgb {self.rgb.shape[:2]} and depth {self.depth.shape} "
                    "must have the same H x W"
                )
        if self.K.shape != (3, 3):
            msgs.append(f"K must be (3, 3), got {self.K.shape}")
        else:
            if not (self.K[0, 0] > 0 and self.K[1, 1] > 0):
                msgs.append("K must have positive fx, fy")
            if not np.isclose(self.K[2, 2], 1.0):
                msgs.append(f"K[2,2] must be 1, got {self.K[2, 2]}")
        # Heuristic depth-unit sanity check: metric indoor depth rarely > 100 m.
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size and float(finite.max()) > 100.0:
            msgs.append(
                f"max depth {float(finite.max()):.1f} looks like millimeters, "
                "not meters; convert with depth_utils.to_meters(depth, 0.001)"
            )
        if msgs:
            if raise_on_error:
                raise ValueError("Invalid RGBDFrame: " + "; ".join(msgs))
            return False
        return True

    # ------------------------------------------------------------------ #
    # intrinsics dict (shared json schema)
    # ------------------------------------------------------------------ #
    def intrinsics_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "K": self.K.tolist(),
            "depth_unit": "meter",
            "camera_frame": self.camera_frame,
        }

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save_sample(self, output_dir: str) -> str:
        """Save rgb.png, depth.npy, depth_vis.png, camera_intrinsics.json, metadata.json.

        Returns the output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.validate(raise_on_error=False)

        image_io.save_rgb(os.path.join(output_dir, "rgb.png"), self.rgb)
        np.save(os.path.join(output_dir, "depth.npy"), self.depth.astype(np.float32))
        vis = depth_utils.colorize_depth(self.depth)
        image_io.save_bgr(os.path.join(output_dir, "depth_vis.png"), vis)
        image_io.save_json(
            os.path.join(output_dir, "camera_intrinsics.json"), self.intrinsics_dict()
        )
        meta = dict(self.metadata)
        meta.update(
            {
                "timestamp": self.timestamp,
                "camera_frame": self.camera_frame,
                "depth_aligned_to_color": bool(self.depth_aligned_to_color),
                "depth_unit": "meter",
            }
        )
        image_io.save_json(os.path.join(output_dir, "metadata.json"), meta)
        return output_dir

    @classmethod
    def load_sample(cls, input_dir: str) -> "RGBDFrame":
        """Reconstruct an RGBDFrame from a saved sample directory."""
        rgb_path = os.path.join(input_dir, "rgb.png")
        depth_npy = os.path.join(input_dir, "depth.npy")
        depth_png = os.path.join(input_dir, "depth.png")
        intr_path = os.path.join(input_dir, "camera_intrinsics.json")
        meta_path = os.path.join(input_dir, "metadata.json")

        if not os.path.isfile(rgb_path):
            raise FileNotFoundError(f"missing rgb.png in {input_dir}")
        rgb = image_io.load_rgb(rgb_path)

        if os.path.isfile(depth_npy):
            depth = np.load(depth_npy).astype(np.float32)
        elif os.path.isfile(depth_png):
            # depth.png is assumed uint16 millimeters when no .npy is present.
            import cv2

            raw = cv2.imread(depth_png, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"could not read {depth_png}")
            depth = depth_utils.to_meters(raw, 0.001)
        else:
            raise FileNotFoundError(
                f"missing depth.npy (or depth.png) in {input_dir}"
            )

        if not os.path.isfile(intr_path):
            raise FileNotFoundError(f"missing camera_intrinsics.json in {input_dir}")
        with open(intr_path, "r") as f:
            intr = json.load(f)
        K = np.asarray(intr["K"], dtype=np.float64)
        camera_frame = intr.get("camera_frame", "d435_color_optical_frame")

        metadata = {}
        timestamp = 0.0
        depth_aligned = True
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            timestamp = float(metadata.get("timestamp", 0.0))
            depth_aligned = bool(metadata.get("depth_aligned_to_color", True))

        return cls(
            rgb=rgb,
            depth=depth,
            K=K,
            timestamp=timestamp,
            camera_frame=camera_frame,
            depth_aligned_to_color=depth_aligned,
            metadata=metadata,
        )
