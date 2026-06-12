"""Live RealSense D435 capture + optional FoundationPose estimation.

Perception only. This script never controls the Franka and sends no motion
commands. It captures one RGB-D frame (color + aligned depth + intrinsics),
saves a sample, and — if a mesh and mask are available — runs FoundationPose to
get the object pose in the camera frame. If a hand_eye.yaml and a current
end-effector pose file are provided, it also reports the pose in the base frame.

Can run in the `foundationpose` env (for real estimation) or any env with
pyrealsense2 (capture-only). See README.

The D435 is always read as **color + aligned-depth-to-color + intrinsics** (the
depth stream is registered to the color image), which is what FoundationPose
expects. Depth is converted to meters.

Examples
--------
Capture only (no pose estimation, no FoundationPose needed):
    python franka_d435_foundationpose/scripts/run_realsense_d435_live.py \
        --save_only \
        --output_dir franka_d435_foundationpose/outputs/realsense_samples/sample_000001

Capture + pose (object mesh + bbox mask), and base-frame pose:
    conda run -n foundationpose python franka_d435_foundationpose/scripts/run_realsense_d435_live.py \
        --object_name target_object --mask_bbox 280 200 360 280 \
        --hand_eye franka_d435_foundationpose/configs/hand_eye.yaml \
        --ee_pose_json current_ee_pose.json
"""

import argparse
import os
import sys

import numpy as np

import _bootstrap  # noqa: F401

from franka_d435_foundationpose.foundationpose.mask_provider import MaskProvider
from franka_d435_foundationpose.transforms.frame_conventions import base_object_from_chain
from franka_d435_foundationpose.transforms.hand_eye import load_hand_eye
from franka_d435_foundationpose.transforms.se3 import make_T
from franka_d435_foundationpose.utils import image_io
from franka_d435_foundationpose.utils.config import default_config_path, resolve_cli_path
from franka_d435_foundationpose.utils.logging_utils import get_logger

logger = get_logger("realsense_live")


def load_ee_pose_json(path):
    """Load T_base_ee from a json. Accepts either a 4x4 'T_base_ee' matrix or
    {'position':[x,y,z], 'quaternion_xyzw':[x,y,z,w]}."""
    data = image_io.load_json(path)
    if "T_base_ee" in data:
        return np.asarray(data["T_base_ee"], dtype=np.float64)
    pos = np.asarray(data["position"], dtype=np.float64)
    quat = np.asarray(data["quaternion_xyzw"], dtype=np.float64)
    return make_T(R=quat, t=pos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, help="where to save the sample")
    parser.add_argument("--object_name", default=None, help="object key for pose estimation")
    parser.add_argument("--config", default=None, help="foundationpose.yaml path")
    parser.add_argument("--object_config", default=None, help="object_assets.yaml path")
    parser.add_argument("--mask_path", default=None, help="mask PNG for estimation")
    parser.add_argument("--mask_bbox", type=int, nargs=4, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"), help="rectangular mask")
    parser.add_argument("--hand_eye", default=None, help="hand_eye.yaml for base-frame pose")
    parser.add_argument("--ee_pose_json", default=None, help="current end-effector pose json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mock", action="store_true", help="force mock estimator")
    parser.add_argument("--save_only", action="store_true",
                        help="only capture + save one RGB-D frame; skip pose estimation")
    args = parser.parse_args()

    # --- pre-flight: pyrealsense2 must be importable before we touch the camera ---
    try:
        import pyrealsense2  # noqa: F401
    except Exception:
        logger.error(
            "pyrealsense2 is not available. Install the RealSense SDK Python "
            "bindings and connect the D435:\n"
            "    pip install pyrealsense2\n"
            "    rs-enumerate-devices   # verify the camera is detected\n"
            "Alternatively run the offline generator "
            "(scripts/generate_mock_rgbd_sample.py) to test without hardware."
        )
        return 1

    output_dir = resolve_cli_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- capture one frame (color + aligned depth to color + intrinsics) ---
    from franka_d435_foundationpose.camera.realsense_d435_provider import (
        RealSenseD435Provider,
    )

    with RealSenseD435Provider(
        args.width, args.height, args.fps, align_to_color=True
    ) as cam:
        frame = cam.get_frame()
    frame.save_sample(output_dir)
    logger.info("captured + saved RGB-D sample (aligned depth->color) -> %s", output_dir)

    if args.save_only:
        logger.info("--save_only set; skipping pose estimation. Done.")
        return 0

    if not args.object_name:
        logger.info("no --object_name; capture-only mode. Done.")
        logger.info("Provide --object_name (+ mesh & mask) to also estimate a pose, "
                    "or pass --save_only to make capture-only explicit.")
        return 0

    # --- mask ---
    if args.mask_path:
        mask = MaskProvider.load_from_file(args.mask_path)
    elif args.mask_bbox:
        mask = MaskProvider.from_bbox(frame.height, frame.width, args.mask_bbox)
    else:
        logger.warning(
            "no mask provided (--mask_path or --mask_bbox). FoundationPose needs "
            "an initial mask. Saved RGB-D only; rerun with a mask to estimate."
        )
        return 0

    # --- estimator ---
    from franka_d435_foundationpose.foundationpose.foundationpose_wrapper import (
        FoundationPoseEstimator,
    )
    from franka_d435_foundationpose.foundationpose.mesh_loader import get_object_asset

    fp_config = args.config or default_config_path("foundationpose.yaml")
    obj_config = args.object_config or default_config_path("object_assets.yaml")
    asset = get_object_asset(obj_config, args.object_name)

    estimator = FoundationPoseEstimator(fp_config, force_mock=args.mock)
    result = estimator.estimate(frame, asset.mesh_path, mask)
    logger.info("backend=%s  T_camera_object=\n%s",
                estimator.backend, np.array2string(result.T_camera_object, precision=4))

    extra = {"object_name": args.object_name, "camera_frame": frame.camera_frame}

    # --- optional base-frame pose ---
    if args.hand_eye and args.ee_pose_json:
        hand_eye = load_hand_eye(args.hand_eye)
        T_base_ee = load_ee_pose_json(args.ee_pose_json)
        T_base_object = base_object_from_chain(
            T_base_ee, hand_eye.T_ee_camera, result.T_camera_object, validate=False
        )
        extra["T_base_object"] = T_base_object.tolist()
        extra["base_frame"] = hand_eye.base_frame
        logger.info("T_base_object=\n%s", np.array2string(T_base_object, precision=4))
    else:
        logger.info("provide --hand_eye and --ee_pose_json to also get base-frame pose")

    result.save_json(os.path.join(output_dir, "pose_result.json"), extra=extra)
    logger.info("wrote pose_result.json")
    if estimator.is_mock:
        logger.warning("MOCK result — not a real pose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
