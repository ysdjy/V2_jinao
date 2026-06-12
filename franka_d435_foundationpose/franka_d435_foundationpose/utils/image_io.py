"""Image / array IO helpers.

All RGB images in this project are stored as **RGB-ordered uint8** PNGs.
OpenCV works in BGR, so we convert explicitly at the IO boundary.
"""

from __future__ import annotations

import json

import numpy as np


def _require_cv2():
    try:
        import cv2

        return cv2
    except Exception as e:  # pragma: no cover - clear guidance on missing dep
        raise ImportError(
            "opencv-python is required for image IO. Install with:\n"
            "    pip install opencv-python"
        ) from e


def save_rgb(path: str, rgb: np.ndarray) -> None:
    """Save an RGB (H, W, 3) uint8 array as a PNG."""
    cv2 = _require_cv2()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"save_rgb expects (H, W, 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def load_rgb(path: str) -> np.ndarray:
    """Load a PNG/JPG as an RGB (H, W, 3) uint8 array."""
    cv2 = _require_cv2()
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_bgr(path: str, bgr: np.ndarray) -> None:
    """Save an already-BGR array (e.g. a colorized depth vis) directly."""
    cv2 = _require_cv2()
    cv2.imwrite(path, np.asarray(bgr))


def save_mask(path: str, mask: np.ndarray) -> None:
    """Save a boolean / 0-255 mask as a single-channel PNG (0 or 255)."""
    cv2 = _require_cv2()
    mask = np.asarray(mask)
    if mask.dtype == bool:
        out = (mask.astype(np.uint8)) * 255
    elif mask.max() <= 1:
        out = (mask.astype(np.uint8)) * 255
    else:
        out = mask.astype(np.uint8)
    cv2.imwrite(path, out)


def load_mask(path: str) -> np.ndarray:
    """Load a mask PNG as a boolean (H, W) array (>0 -> True)."""
    cv2 = _require_cv2()
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return m > 0


def save_json(path: str, obj: dict) -> None:
    """Write a dict as pretty JSON (numpy arrays converted to lists)."""
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
