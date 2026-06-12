"""Visualize a saved pose_result.json over its RGB-D sample.

Reads a sample directory (rgb.png + camera_intrinsics.json + pose_result.json),
draws the object frame axes onto the RGB image, and writes/optionally shows
pose_overlay.png. Useful to sanity-check a pose without rerunning estimation.

    python franka_d435_foundationpose/scripts/visualize_pose_result.py \
        --input_dir franka_d435_foundationpose/outputs/rgbd_samples/sample_000001
"""

import argparse
import os
import sys

import numpy as np

import _bootstrap  # noqa: F401

from franka_d435_foundationpose.camera.frame_types import RGBDFrame
from franka_d435_foundationpose.utils import image_io
from franka_d435_foundationpose.utils.config import resolve_cli_path
from franka_d435_foundationpose.utils.logging_utils import get_logger

# Reuse the overlay drawing from the saved-RGBD runner.
from run_foundationpose_on_saved_rgbd import draw_pose_overlay  # noqa: E402

logger = get_logger("visualize_pose")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="sample dir with pose_result.json")
    parser.add_argument("--pose_json", default=None, help="override pose_result.json path")
    parser.add_argument("--axis_len", type=float, default=0.05, help="axis length in meters")
    parser.add_argument("--show", action="store_true", help="display the overlay window")
    parser.add_argument("--output", default=None, help="output overlay path")
    args = parser.parse_args()

    input_dir = resolve_cli_path(args.input_dir)
    frame = RGBDFrame.load_sample(input_dir)

    pose_json = args.pose_json or os.path.join(input_dir, "pose_result.json")
    if not os.path.isfile(pose_json):
        logger.error("pose_result.json not found: %s (run estimation first)", pose_json)
        return 1
    pose = image_io.load_json(pose_json)
    T_camera_object = np.asarray(pose["T_camera_object"], dtype=np.float64)

    overlay = draw_pose_overlay(frame.rgb, frame.K, T_camera_object, axis_len=args.axis_len)
    out_path = args.output or os.path.join(input_dir, "pose_overlay.png")
    image_io.save_rgb(out_path, overlay)
    logger.info("wrote %s", out_path)

    if "T_base_object" in pose:
        logger.info(
            "T_base_object (object in %s):\n%s",
            pose.get("base_frame", "base"),
            np.array2string(np.asarray(pose["T_base_object"]), precision=4),
        )

    if args.show:
        try:
            import cv2

            cv2.imshow("pose_overlay", overlay[:, :, ::-1])
            logger.info("press any key in the window to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:  # pragma: no cover
            logger.warning("could not display window: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
