"""Adapter from an IsaacLab ``Camera`` sensor to the unified :class:`RGBDFrame`.

Runs inside ``env_isaaclab``. It does NOT import isaaclab at module import time
so that unit tests and the file-exchange tooling can import it anywhere; the
IsaacLab objects are passed in by the caller (the demo script).

It MUST NOT import FoundationPose.
"""

from __future__ import annotations

import time

import numpy as np

from .frame_types import RGBDFrame

# Data-type keys an IsaacLab Camera may expose for depth, in priority order.
_DEPTH_KEYS = ("distance_to_image_plane", "depth")


def _to_numpy(x):
    """Convert a torch tensor (or array) on any device to a numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_rgbd_frame(
    camera,
    cam_index: int = 0,
    camera_frame: str = "d435_color_optical_frame",
    metadata: dict | None = None,
) -> RGBDFrame:
    """Build an :class:`RGBDFrame` from an IsaacLab ``Camera`` sensor.

    Parameters
    ----------
    camera : an ``isaaclab.sensors.Camera`` instance (already updated).
    cam_index : index of the camera in the (num_cameras) batch dimension.
    camera_frame : optical frame name to stamp on the produced frame.
    metadata : extra metadata to merge into the frame.

    Notes
    -----
    * RGB: ``camera.data.output["rgb"]`` is (N, H, W, 3) uint8 (may include an
      alpha channel as RGBA depending on the IsaacLab version; we take 3 chans).
    * Depth: ``distance_to_image_plane`` is metric depth (meters) along the
      optical axis (NOT range), which is exactly what FoundationPose expects.
    * Intrinsics: ``camera.data.intrinsic_matrices`` is (N, 3, 3).
    """
    output = camera.data.output

    # ----- RGB -----
    if "rgb" not in output:
        raise KeyError(
            "Camera output has no 'rgb'. Configure CameraCfg(data_types=[...]) "
            "with 'rgb' and launch with --enable_cameras."
        )
    rgb = _to_numpy(output["rgb"])[cam_index]
    if rgb.shape[-1] == 4:  # RGBA -> RGB
        rgb = rgb[..., :3]
    rgb = np.ascontiguousarray(rgb).astype(np.uint8)

    # ----- depth -----
    depth = None
    for key in _DEPTH_KEYS:
        if key in output:
            depth = _to_numpy(output[key])[cam_index]
            break
    if depth is None:
        raise KeyError(
            "Camera output has no depth channel. Add 'distance_to_image_plane' "
            f"(or 'depth') to CameraCfg.data_types. Available: {list(output.keys())}"
        )
    depth = np.ascontiguousarray(depth).astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    # IsaacLab encodes invalid/too-far depth as +inf; FoundationPose wants 0.
    depth[~np.isfinite(depth)] = 0.0

    # ----- intrinsics -----
    K = _to_numpy(camera.data.intrinsic_matrices)[cam_index].astype(np.float64)

    meta = {"source": "isaaclab_camera", "cam_index": int(cam_index)}
    if metadata:
        meta.update(metadata)

    frame = RGBDFrame(
        rgb=rgb,
        depth=depth,
        K=K,
        timestamp=time.time(),
        camera_frame=camera_frame,
        depth_aligned_to_color=True,  # sim RGB and depth share the same sensor
        metadata=meta,
    )
    frame.validate(raise_on_error=False)
    return frame


class IsaacLabCameraProvider:
    """Thin stateful wrapper around an IsaacLab ``Camera`` sensor.

    The demo script constructs the IsaacLab scene/camera and hands the camera
    sensor to this provider; ``get_frame()`` then returns an RGBDFrame.
    """

    def __init__(
        self,
        camera,
        cam_index: int = 0,
        camera_frame: str = "d435_color_optical_frame",
    ):
        self.camera = camera
        self.cam_index = cam_index
        self.camera_frame = camera_frame

    def get_frame(self, metadata: dict | None = None) -> RGBDFrame:
        return extract_rgbd_frame(
            self.camera,
            cam_index=self.cam_index,
            camera_frame=self.camera_frame,
            metadata=metadata,
        )

    def close(self):
        pass
