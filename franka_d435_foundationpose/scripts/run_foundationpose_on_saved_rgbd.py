"""Run FoundationPose (or the mock) on a saved RGB-D sample directory.

Mode-1 (file exchange) entry point — the most stable path. Reads a sample dir,
looks up the object's mesh from configs/object_assets.yaml, runs estimate, and
writes pose_result.json + pose_overlay.png. If transforms.json is present it
also computes T_base_object.

Examples
--------
Mock (any env):
    python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \
        --input_dir franka_d435_foundationpose/outputs/rgbd_samples/sample_000001 \
        --object_name target_object --mock

Real (isolated env):
    conda run -n foundationpose python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \
        --input_dir franka_d435_foundationpose/outputs/rgbd_samples/sample_000001 \
        --object_name target_object
"""

import argparse
import os
import sys

import numpy as np

import _bootstrap  # noqa: F401

from franka_d435_foundationpose.camera.saved_rgbd_provider import SavedRGBDProvider
from franka_d435_foundationpose.foundationpose.foundationpose_wrapper import (
    FoundationPoseEstimator,
)
from franka_d435_foundationpose.foundationpose.mask_provider import MaskProvider
from franka_d435_foundationpose.foundationpose.mesh_loader import get_object_asset
from franka_d435_foundationpose.transforms.frame_conventions import (
    BASE_FRAME,
    base_object_from_chain,
)
from franka_d435_foundationpose.transforms.se3 import compose_T
from franka_d435_foundationpose.utils import image_io
from franka_d435_foundationpose.utils.config import default_config_path, resolve_cli_path
from franka_d435_foundationpose.utils.logging_utils import get_logger

logger = get_logger("run_fp_saved")


def draw_pose_overlay(rgb, K, T_camera_object, axis_len=0.05):
    """Draw the object frame axes onto the RGB image (RGB in -> BGR-safe out).

    Returns an RGB uint8 image with X/Y/Z axes (red/green/blue) drawn from the
    object origin. Works for both real and mock poses.
    """
    import cv2

    img = np.ascontiguousarray(rgb[:, :, ::-1]).copy()  # RGB -> BGR for cv2
    origin = np.array([0, 0, 0, 1.0])
    axes = np.array(
        [
            [0, 0, 0, 1.0],
            [axis_len, 0, 0, 1.0],
            [0, axis_len, 0, 1.0],
            [0, 0, axis_len, 1.0],
        ]
    ).T
    cam_pts = (T_camera_object @ axes)[:3, :]  # (3, 4) in camera frame
    # Project; guard against points behind the camera.
    pts2d = []
    for i in range(cam_pts.shape[1]):
        x, y, z = cam_pts[:, i]
        if z <= 1e-6:
            pts2d.append(None)
            continue
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        pts2d.append((int(round(u)), int(round(v))))

    o = pts2d[0]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X red, Y green, Z blue (BGR)
    if o is not None:
        for i, c in zip(range(1, 4), colors):
            if pts2d[i] is not None:
                cv2.line(img, o, pts2d[i], c, 2)
        cv2.circle(img, o, 4, (255, 255, 255), -1)
    return np.ascontiguousarray(img[:, :, ::-1])  # back to RGB


def resolve_mask(input_dir, frame, asset, args):
    """Pick a mask: explicit > sample mask.png > object default > placeholder box."""
    if args.mask_path:
        logger.info("using mask from --mask_path: %s", args.mask_path)
        return MaskProvider.load_from_file(args.mask_path), "arg"
    sample_mask = os.path.join(input_dir, "mask.png")
    if os.path.isfile(sample_mask):
        logger.info("using sample mask: %s", sample_mask)
        return MaskProvider.load_from_file(sample_mask), "sample"
    if asset.default_mask_path and os.path.isfile(asset.default_mask_path):
        logger.info("using object default mask: %s", asset.default_mask_path)
        return MaskProvider.load_from_file(asset.default_mask_path), "object_default"
    logger.warning(
        "no mask found; falling back to a centered placeholder box. Provide a "
        "real mask (mask.png / --mask_path / object default) for correct results."
    )
    return MaskProvider.center_box(frame.height, frame.width, frac=0.4), "placeholder_box"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="saved RGB-D sample directory")
    parser.add_argument("--object_name", required=True, help="key in object_assets.yaml")
    parser.add_argument("--config", default=None, help="path to foundationpose.yaml")
    parser.add_argument("--object_config", default=None, help="path to object_assets.yaml")
    parser.add_argument("--mask_path", default=None, help="explicit mask PNG override")
    parser.add_argument("--mock", action="store_true", help="force the mock estimator")
    parser.add_argument("--output_dir", default=None, help="defaults to input_dir")
    args = parser.parse_args()

    input_dir = resolve_cli_path(args.input_dir)
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)

    fp_config = args.config or default_config_path("foundationpose.yaml")
    obj_config = args.object_config or default_config_path("object_assets.yaml")

    # 1. Load the sample.
    provider = SavedRGBDProvider(input_dir)
    frame = provider.get_frame()
    logger.info("loaded frame %dx%d from %s", frame.width, frame.height, input_dir)

    # 2. Resolve the object asset + mesh path.
    asset = get_object_asset(obj_config, args.object_name)
    mesh_path = asset.mesh_path
    if not args.mock and (not mesh_path or not os.path.isfile(mesh_path)):
        logger.warning(
            "mesh for '%s' not found (%s). The real backend needs a mesh; the "
            "mock does not. Continuing — will mock if backend unavailable.",
            args.object_name, mesh_path,
        )

    # 3. Mask.
    mask, mask_src = resolve_mask(input_dir, frame, asset, args)

    # 4. Estimator (real or mock).
    estimator = FoundationPoseEstimator(fp_config, force_mock=args.mock)
    logger.info("estimator backend: %s", estimator.backend)

    result = estimator.estimate(frame, mesh_path, mask)
    logger.info("estimate success=%s mode=%s", result.success, result.mode)
    logger.info("T_camera_object=\n%s", np.array2string(result.T_camera_object, precision=4))

    # 5. Optional: compose T_base_object from transforms.json.
    extra = {
        "object_name": args.object_name,
        "camera_mode": None,
        "camera_frame": frame.camera_frame,
        "base_frame": BASE_FRAME,
        "timestamp": frame.timestamp,
        "mask_source": mask_src,
        "backend": estimator.backend,
    }
    # Three transform cases, in priority order:
    #   1. end-effector camera : T_base_ee + T_ee_camera -> T_base_object
    #   2. fixed scene camera  : T_world_camera          -> T_world_object
    #   3. none                : only T_camera_object
    transforms_path = provider.transforms_path
    if transforms_path:
        tf = image_io.load_json(transforms_path)
        extra["camera_mode"] = tf.get("camera_mode")
        T_base_ee = tf.get("T_base_ee")
        T_ee_camera = tf.get("T_ee_camera")
        T_world_camera = tf.get("T_world_camera")
        if T_base_ee is not None and T_ee_camera is not None:
            T_base_object = base_object_from_chain(
                np.asarray(T_base_ee, dtype=np.float64),
                np.asarray(T_ee_camera, dtype=np.float64),
                result.T_camera_object,
                validate=False,
            )
            extra["T_base_object"] = T_base_object.tolist()
            extra["base_frame"] = tf.get("base_frame", BASE_FRAME)
            logger.info("[end-effector] T_base_object=\n%s",
                        np.array2string(T_base_object, precision=4))
        elif T_world_camera is not None:
            T_world_object = compose_T(
                np.asarray(T_world_camera, dtype=np.float64), result.T_camera_object
            )
            extra["T_world_object"] = T_world_object.tolist()
            extra["world_frame"] = tf.get("world_frame", "world")
            logger.info("[fixed scene] T_world_object=\n%s",
                        np.array2string(T_world_object, precision=4))
        else:
            logger.info("transforms.json present but no usable external transform "
                        "(need T_base_ee+T_ee_camera or T_world_camera); "
                        "reporting T_camera_object only")
    else:
        logger.info("no transforms.json; reporting T_camera_object only")

    # 6. Save outputs.
    pose_json = os.path.join(output_dir, "pose_result.json")
    result.save_json(pose_json, extra=extra)
    logger.info("wrote %s", pose_json)

    try:
        overlay = draw_pose_overlay(frame.rgb, frame.K, result.T_camera_object)
        overlay_path = os.path.join(output_dir, "pose_overlay.png")
        image_io.save_rgb(overlay_path, overlay)
        logger.info("wrote %s", overlay_path)
    except Exception as e:  # pragma: no cover - overlay is best-effort
        logger.warning("failed to render overlay: %s", e)

    if estimator.is_mock:
        logger.warning(
            "RESULT IS FROM THE MOCK ESTIMATOR — not a real pose. Configure "
            "FoundationPose for real estimates (see envs/README.md)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
