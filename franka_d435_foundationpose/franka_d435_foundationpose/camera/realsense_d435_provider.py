"""Intel RealSense D435 RGB-D provider (perception only — no robot control).

Outputs color + aligned-depth-to-color + intrinsics as an :class:`RGBDFrame`.

Safety: this module never sends any motion command. It only reads the camera.
"""

from __future__ import annotations

import time

import numpy as np

from .frame_types import RGBDFrame

_INSTALL_HINT = (
    "pyrealsense2 is not available. Install the RealSense SDK Python bindings:\n"
    "    pip install pyrealsense2\n"
    "and make sure the D435 is connected (check with `rs-enumerate-devices`).\n"
    "Alternatively run in ROS2 mode (see realsense_d435_provider docstring) or\n"
    "use the SavedRGBDProvider on previously captured samples."
)


class RealSenseD435Provider:
    """Capture color + aligned depth from a D435 using pyrealsense2.

    Parameters
    ----------
    width, height : color/depth stream resolution.
    fps : stream frame rate.
    camera_frame : optical frame name stamped onto produced frames.
    align_to_color : if True, depth is aligned to the color image (recommended
        for FoundationPose, which expects depth registered to RGB).

    ROS2 mode (not implemented here, documented for later):
        Instead of pyrealsense2 you may subscribe to the realsense2_camera node:
          * /camera/color/image_raw                 -> rgb
          * /camera/aligned_depth_to_color/image_raw -> depth (uint16 mm)
          * /camera/color/camera_info                -> K
        Convert depth to meters (x 0.001) and wrap in an RGBDFrame identically.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        camera_frame: str = "d435_color_optical_frame",
        align_to_color: bool = True,
        warmup_frames: int = 30,
    ):
        try:
            import pyrealsense2 as rs
        except Exception as e:  # pragma: no cover - hardware/SDK dependent
            raise ImportError(_INSTALL_HINT) from e

        self._rs = rs
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_frame = camera_frame
        self.align_to_color = align_to_color

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._profile = self._pipeline.start(config)

        # Depth scale: raw z16 units -> meters (usually 0.001 for the D435).
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        self._align = rs.align(rs.stream.color) if align_to_color else None

        dev = self._profile.get_device()
        self._serial = dev.get_info(rs.camera_info.serial_number)

        # Discard the first frames so auto-exposure and the depth laser settle;
        # the very first frames are typically dark / mostly-invalid depth.
        for _ in range(max(0, int(warmup_frames))):
            try:
                self._pipeline.wait_for_frames(2000)
            except Exception:
                break

    def get_frame(self, timeout_ms: int = 5000) -> RGBDFrame:
        """Grab one synchronized color+depth frame as an RGBDFrame (depth in m)."""
        rs = self._rs
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete frame set")

        bgr = np.asanyarray(color_frame.get_data())
        rgb = bgr[:, :, ::-1].copy()  # BGR -> RGB, contiguous
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        depth_m = depth_raw.astype(np.float32) * self._depth_scale

        # Intrinsics of the (aligned) color stream.
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        frame = RGBDFrame(
            rgb=rgb,
            depth=depth_m,
            K=K,
            timestamp=time.time(),
            camera_frame=self.camera_frame,
            depth_aligned_to_color=bool(self.align_to_color),
            metadata={
                "source": "realsense_d435",
                "serial_number": self._serial,
                "depth_scale": self._depth_scale,
                "fps": self.fps,
            },
        )
        frame.validate(raise_on_error=False)
        return frame

    def close(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
