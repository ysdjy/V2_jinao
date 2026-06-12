"""Depth helpers: unit conversion and colorized visualization.

IMPORTANT: throughout the project depth is stored in **meters** as float32.
RealSense raw depth is uint16 in millimeters; convert with ``depth_scale``.
"""

from __future__ import annotations

import numpy as np


def to_meters(depth, depth_scale: float = 1.0) -> np.ndarray:
    """Convert a raw depth array to float32 meters.

    ``depth_scale`` is meters-per-unit. RealSense reports ~0.001 (mm -> m).
    If the input is already float meters, pass ``depth_scale=1.0``.
    """
    depth = np.asarray(depth)
    return (depth.astype(np.float32)) * float(depth_scale)


def sanitize_depth(depth, max_depth_m: float = 10.0) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip absurd far values to 0 (invalid)."""
    depth = np.asarray(depth, dtype=np.float32).copy()
    bad = ~np.isfinite(depth)
    depth[bad] = 0.0
    depth[depth > max_depth_m] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def colorize_depth(depth, min_m: float | None = None, max_m: float | None = None):
    """Return an H x W x 3 uint8 BGR visualization of a metric depth map.

    Uses OpenCV's JET colormap when available; falls back to a grayscale
    ramp if cv2 is not importable. Zero/invalid depth is rendered black.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > 0
    if min_m is None:
        min_m = float(depth[valid].min()) if valid.any() else 0.0
    if max_m is None:
        max_m = float(depth[valid].max()) if valid.any() else 1.0
    if max_m <= min_m:
        max_m = min_m + 1e-3

    norm = np.clip((depth - min_m) / (max_m - min_m), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)

    try:
        import cv2

        vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        vis[~valid] = 0
        return vis
    except Exception:
        gray = np.stack([norm_u8] * 3, axis=-1)
        gray[~valid] = 0
        return gray
