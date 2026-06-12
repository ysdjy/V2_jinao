"""STUB: RealSense D435 -> unified Observation images.

Defines how a real D435 stream would populate the same observation schema used in
sim (front_rgb / wrist_rgb). No hardware access here.

Real implementation TODO:
  * pyrealsense2 pipeline (color + depth), align depth to color
  * resize to the training resolution (match configs image_resize, e.g. 256x256)
  * write to image_mode=path (recommended) under data/processed/images/
"""

from __future__ import annotations

from typing import Any


class D435ObservationStub:
    def __init__(self, width: int = 640, height: int = 480, resize=(256, 256)):
        self.width, self.height, self.resize = width, height, resize
        self.pipeline = None

    def start(self):
        # TODO: import pyrealsense2; configure + start pipeline
        raise NotImplementedError("D435 capture not implemented in the stub.")

    def get_image_refs(self, step_id: int, image_dir: str) -> dict[str, Any]:
        """Return {'front_rgb': {mode:path,path:...}, ...} like the sim exporter."""
        raise NotImplementedError(
            "Implement: grab frame -> resize -> save png -> return ImageRef dicts. "
            "Keep the SAME logical keys (front_rgb/wrist_rgb) and resolution as training."
        )
