"""Initial-mask provider for FoundationPose's first ``estimate`` call.

FoundationPose (model-based) needs an initial binary mask of the target object
in the first frame to seed pose estimation. After that, ``track`` propagates.

Three ways to obtain a mask are supported here:
  1. load_from_file  : read an existing mask PNG (from assets/masks or a sample).
  2. from_bbox       : build a rectangular mask from an [x0, y0, x1, y1] box
                       (manual placeholder when no segmentation exists yet).
  3. from_sim_segmentation : turn an IsaacLab semantic/instance segmentation map
                       into a binary mask for a known object id/label.

SAM / SAM2 / Grounded-SAM are intentionally NOT required. See README: a future
mask source can be SAM2 (point/box prompt) or Grounded-SAM (text prompt) to
auto-generate this initial mask; FoundationPose then handles tracking.
"""

from __future__ import annotations

import os

import numpy as np

from ..utils import image_io


class MaskProvider:
    """Produce a boolean (H, W) mask via file / bbox / sim-segmentation."""

    @staticmethod
    def load_from_file(path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"mask file not found: {path}\n"
                "Provide an initial binary mask (white = object) or use "
                "MaskProvider.from_bbox(...) as a placeholder."
            )
        return image_io.load_mask(path)

    @staticmethod
    def from_bbox(height: int, width: int, bbox) -> np.ndarray:
        """Rectangular mask from ``bbox = [x0, y0, x1, y1]`` (pixel coords)."""
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0, x1 = sorted((max(0, x0), min(width, x1)))
        y0, y1 = sorted((max(0, y0), min(height, y1)))
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def from_sim_segmentation(seg: np.ndarray, target_ids) -> np.ndarray:
        """Binary mask from an IsaacLab segmentation map.

        ``seg`` is an (H, W) integer id map; ``target_ids`` is an int or a
        collection of ints identifying the target object.
        """
        seg = np.asarray(seg)
        if seg.ndim == 3:
            seg = seg[..., 0]
        if np.isscalar(target_ids):
            target_ids = [int(target_ids)]
        mask = np.isin(seg, list(target_ids))
        return mask.astype(bool)

    @staticmethod
    def center_box(height: int, width: int, frac: float = 0.4) -> np.ndarray:
        """Convenience placeholder: a centered box covering ``frac`` of the image."""
        bw, bh = int(width * frac), int(height * frac)
        x0 = (width - bw) // 2
        y0 = (height - bh) // 2
        return MaskProvider.from_bbox(height, width, [x0, y0, x0 + bw, y0 + bh])
