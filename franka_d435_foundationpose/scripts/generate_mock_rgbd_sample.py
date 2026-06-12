"""Generate a fully synthetic RGB-D sample WITHOUT IsaacLab or a real D435.

Produces a sample directory identical in layout to the IsaacLab capture demo, so
the FoundationPose wrapper, mask handling and coordinate transforms can be
exercised entirely offline.

Writes:
    rgb.png  depth.npy  depth_vis.png  mask.png
    camera_intrinsics.json  transforms.json  metadata.json  object_config.yaml

The synthetic scene is a centered "object" (a colored square at ~0.6 m) on a
flat background (~1.2 m). The mask is the object square.

Run:
    cd /home1/banghai/Documents/IsaacLab
    python franka_d435_foundationpose/scripts/generate_mock_rgbd_sample.py
"""

import argparse
import os
import sys

import numpy as np

import _bootstrap  # noqa: F401

from franka_d435_foundationpose.camera.frame_types import RGBDFrame
from franka_d435_foundationpose.transforms.hand_eye import load_hand_eye
from franka_d435_foundationpose.transforms.se3 import compose_T, make_T
from franka_d435_foundationpose.utils import image_io
from franka_d435_foundationpose.utils.config import default_config_path, project_path, resolve_cli_path
from franka_d435_foundationpose.utils.logging_utils import get_logger

logger = get_logger("gen_mock_sample")


def build_synthetic_frame(width, height, fx, fy, cx, cy, obj_depth, bg_depth, box_frac):
    """Return (RGBDFrame, mask) for a centered object square on a background."""
    rgb = np.full((height, width, 3), 70, dtype=np.uint8)  # gray background
    depth = np.full((height, width), float(bg_depth), dtype=np.float32)

    bw, bh = int(width * box_frac), int(height * box_frac)
    x0, y0 = (width - bw) // 2, (height - bh) // 2
    x1, y1 = x0 + bw, y0 + bh

    # Object: a colored square that is closer than the background.
    rgb[y0:y1, x0:x1] = (200, 60, 60)  # RGB order
    depth[y0:y1, x0:x1] = float(obj_depth)

    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    frame = RGBDFrame(
        rgb=rgb,
        depth=depth,
        K=K,
        timestamp=0.0,
        camera_frame="d435_color_optical_frame",
        depth_aligned_to_color=True,
        metadata={
            "source": "generate_mock_rgbd_sample",
            "synthetic": True,
            "object_box_px": [x0, y0, x1, y1],
        },
    )
    return frame, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=None,
                        help="defaults to <project>/outputs/rgbd_samples/sample_mock_000001")
    parser.add_argument("--object_name", default="cube", help="object name hint for downstream")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--object_depth", type=float, default=0.6, help="meters")
    parser.add_argument("--bg_depth", type=float, default=1.2, help="meters")
    parser.add_argument("--box_frac", type=float, default=0.25, help="object square size fraction")
    parser.add_argument("--hand_eye", default=None, help="hand_eye.yaml for T_ee_camera")
    parser.add_argument("--write_asset_mask", action="store_true",
                        help="also write assets/masks/<object>_mask.png")
    args = parser.parse_args()

    output_dir = (
        resolve_cli_path(args.output_dir)
        if args.output_dir
        else project_path("outputs", "rgbd_samples", "sample_mock_000001")
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. Synthetic frame + mask.
    frame, mask = build_synthetic_frame(
        args.width, args.height, args.fx, args.fy, args.cx, args.cy,
        args.object_depth, args.bg_depth, args.box_frac,
    )
    frame.validate()
    frame.save_sample(output_dir)
    image_io.save_mask(os.path.join(output_dir, "mask.png"), mask)
    logger.info("wrote rgb/depth/depth_vis/intrinsics/metadata/mask -> %s", output_dir)

    # 2. Transforms: T_base_ee (placeholder FK), T_ee_camera (hand-eye), T_base_camera.
    hand_eye_path = args.hand_eye or default_config_path("hand_eye.yaml")
    hand_eye = load_hand_eye(hand_eye_path)
    T_ee_camera = hand_eye.T_ee_camera
    # A plausible end-effector pose in the base frame (identity rotation).
    T_base_ee = make_T(t=[0.4, 0.0, 0.5])
    T_base_camera = compose_T(T_base_ee, T_ee_camera)
    image_io.save_json(
        os.path.join(output_dir, "transforms.json"),
        {
            "convention": "T_target_source means transforming a point from the "
            "source frame to the target frame",
            "base_frame": hand_eye.base_frame,
            "ee_frame": hand_eye.ee_frame,
            "camera_frame": hand_eye.camera_frame,
            "T_base_ee": T_base_ee.tolist(),
            "T_ee_camera": T_ee_camera.tolist(),
            "T_base_camera": T_base_camera.tolist(),
            "timestamp": frame.timestamp,
        },
    )
    with open(os.path.join(output_dir, "object_config.yaml"), "w") as f:
        f.write(f"object_name: {args.object_name}\n")
    logger.info("wrote transforms.json + object_config.yaml")

    # 3. Optionally publish the mask as the object's default asset mask.
    if args.write_asset_mask:
        asset_mask = project_path("assets", "masks", f"{args.object_name}_mask.png")
        os.makedirs(os.path.dirname(asset_mask), exist_ok=True)
        image_io.save_mask(asset_mask, mask)
        logger.info("wrote asset default mask -> %s", asset_mask)

    logger.info("done. Sample ready at: %s", output_dir)
    logger.info(
        "next: python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py "
        "--input_dir %s --object_name %s --mock",
        os.path.relpath(output_dir, project_path()), args.object_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
