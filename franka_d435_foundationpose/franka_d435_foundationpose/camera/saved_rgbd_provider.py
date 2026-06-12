"""Provider that loads RGB-D frames from a saved sample directory on disk.

Useful for testing the FoundationPose wrapper without IsaacLab or a real D435.
"""

from __future__ import annotations

import os

from .frame_types import RGBDFrame


class SavedRGBDProvider:
    """Reads a sample directory written by :meth:`RGBDFrame.save_sample`.

    A sample dir contains: rgb.png, depth.npy (or depth.png), depth_vis.png,
    camera_intrinsics.json, metadata.json, and optionally mask.png /
    transforms.json (read by the higher-level scripts, not here).
    """

    def __init__(self, sample_dir: str):
        if not os.path.isdir(sample_dir):
            raise NotADirectoryError(f"sample directory not found: {sample_dir}")
        self.sample_dir = sample_dir

    def get_frame(self) -> RGBDFrame:
        """Load and return the RGB-D frame from the sample directory."""
        frame = RGBDFrame.load_sample(self.sample_dir)
        frame.validate(raise_on_error=False)
        return frame

    @property
    def mask_path(self) -> str | None:
        p = os.path.join(self.sample_dir, "mask.png")
        return p if os.path.isfile(p) else None

    @property
    def transforms_path(self) -> str | None:
        p = os.path.join(self.sample_dir, "transforms.json")
        return p if os.path.isfile(p) else None

    def close(self):  # symmetry with the live providers
        pass
