"""IsaacLab demo: Franka + end-effector D435-like camera -> save RGB-D sample.

Runs inside env_isaaclab via isaaclab.sh. It launches the simulator, attaches a
D435-like RGB-D camera to the Franka hand, looks at a target cube, reads
RGB/depth/intrinsics + the end-effector pose, computes the transform chain, and
writes a sample directory compatible with run_foundationpose_on_saved_rgbd.py.

Does NOT import FoundationPose.

Run:
    cd /home1/banghai/Documents/IsaacLab
    ./isaaclab.sh -p franka_d435_foundationpose/scripts/run_isaaclab_franka_d435_demo.py --enable_cameras
"""

import argparse
import os

import _bootstrap  # noqa: F401

# ---------------------------------------------------------------------------- #
# 1. argparse + launch the simulator BEFORE importing any isaaclab modules.
#
# IsaacLab/IsaacSim is only importable inside the Isaac Sim runtime, which is set
# up by isaaclab.sh. If this script is run under plain python (e.g. `python
# run_isaaclab_franka_d435_demo.py`), `pxr` / `isaaclab` are missing and we fail
# with a clear message instead of an obscure ModuleNotFoundError.
# ---------------------------------------------------------------------------- #
try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as e:
    raise SystemExit(
        "[run_isaaclab_franka_d435_demo] IsaacLab/IsaacSim runtime not found "
        f"({e}).\nThis demo must be launched via isaaclab.sh, NOT plain python:\n"
        "    cd /home1/banghai/Documents/IsaacLab\n"
        "    ./isaaclab.sh -p franka_d435_foundationpose/scripts/"
        "run_isaaclab_franka_d435_demo.py --enable_cameras\n"
        "Run it on a machine with a GPU and Isaac Sim (GUI or headless) available."
    )

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_warmup_steps", type=int, default=24, help="sim steps before capture")
parser.add_argument("--num_samples", type=int, default=1, help="number of samples to save")
parser.add_argument("--sample_stride", type=int, default=12, help="sim steps between samples")
parser.add_argument("--output_root", default=None, help="defaults to <project>/outputs/rgbd_samples")
parser.add_argument("--hand_eye", default=None, help="path to hand_eye.yaml")
parser.add_argument("--object_name", default="target_object", help="object name for object_config.yaml hint")
parser.add_argument("--save_mask", action="store_true", help="try to save a segmentation mask")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Segmentation requires cameras + the annotator; enabling cameras is mandatory.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------- #
# 2. Now it is safe to import isaaclab + our isaaclab-side helpers.
# ---------------------------------------------------------------------------- #
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from franka_d435_foundationpose.camera.isaaclab_camera_provider import (  # noqa: E402
    IsaacLabCameraProvider,
)
from franka_d435_foundationpose.isaaclab.franka_d435_scene import build_scene_cfg  # noqa: E402
from franka_d435_foundationpose.transforms.hand_eye import load_hand_eye  # noqa: E402
from franka_d435_foundationpose.transforms.se3 import compose_T, invert_T, make_T  # noqa: E402
from franka_d435_foundationpose.utils import image_io  # noqa: E402
from franka_d435_foundationpose.utils.config import (  # noqa: E402
    default_config_path,
    project_path,
    resolve_cli_path,
)
from franka_d435_foundationpose.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("isaaclab_demo")

EE_BODY = "panda_hand"
BASE_BODY = "panda_link0"


def pose_w_to_T(pose7):
    """IsaacLab world pose [px,py,pz, qw,qx,qy,qz] -> 4x4 T_world_body."""
    p = np.asarray(pose7[:3], dtype=np.float64)
    qw, qx, qy, qz = [float(v) for v in pose7[3:7]]
    return make_T(R=np.array([qx, qy, qz, qw]), t=p)  # se3 uses xyzw


def main():
    output_root = (
        resolve_cli_path(args_cli.output_root)
        if args_cli.output_root
        else project_path("outputs", "rgbd_samples")
    )
    os.makedirs(output_root, exist_ok=True)

    hand_eye_path = args_cli.hand_eye or default_config_path("hand_eye.yaml")
    hand_eye = load_hand_eye(hand_eye_path)
    T_ee_camera = hand_eye.T_ee_camera
    logger.info("loaded hand-eye T_ee_camera from %s", hand_eye_path)

    # --- build scene ---
    scene_cfg = build_scene_cfg(num_envs=1, hand_eye_yaml=hand_eye_path)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.5, 0.0, 0.3])

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    logger.info("scene ready; bodies=%s", scene["robot"].data.body_names)

    robot = scene["robot"]
    camera = scene["camera"]
    provider = IsaacLabCameraProvider(camera, cam_index=0, camera_frame=hand_eye.camera_frame)

    ee_idx = robot.data.body_names.index(EE_BODY)
    base_idx = robot.data.body_names.index(BASE_BODY)

    dt = sim.get_physics_dt()
    saved = 0
    step = 0
    next_capture = args_cli.num_warmup_steps

    while simulation_app.is_running() and saved < args_cli.num_samples:
        sim.step()
        scene.update(dt)
        step += 1
        if step < next_capture:
            continue

        # --- read RGB-D ---
        frame = provider.get_frame(metadata={"sim_step": step, "engine": "isaaclab"})

        # --- read transforms: T_base_ee = inv(T_world_base) @ T_world_ee ---
        ee_pose_w = robot.data.body_pose_w[0, ee_idx].detach().cpu().numpy()
        base_pose_w = robot.data.body_pose_w[0, base_idx].detach().cpu().numpy()
        T_world_ee = pose_w_to_T(ee_pose_w)
        T_world_base = pose_w_to_T(base_pose_w)
        T_base_ee = compose_T(invert_T(T_world_base), T_world_ee)
        T_base_camera = compose_T(T_base_ee, T_ee_camera)

        # --- save sample ---
        sample_dir = os.path.join(output_root, f"sample_{saved + 1:06d}")
        frame.save_sample(sample_dir)

        image_io.save_json(
            os.path.join(sample_dir, "transforms.json"),
            {
                "convention": "T_target_source means transforming a point from "
                "the source frame to the target frame",
                "base_frame": BASE_BODY,
                "ee_frame": hand_eye.ee_frame,
                "camera_frame": hand_eye.camera_frame,
                "T_base_ee": T_base_ee.tolist(),
                "T_ee_camera": T_ee_camera.tolist(),
                "T_base_camera": T_base_camera.tolist(),
                "timestamp": frame.timestamp,
            },
        )
        # object_config.yaml hint for downstream lookups.
        with open(os.path.join(sample_dir, "object_config.yaml"), "w") as f:
            f.write(f"object_name: {args_cli.object_name}\n")

        # --- best-effort mask from segmentation ---
        if args_cli.save_mask:
            _try_save_mask(camera, sample_dir, args_cli.object_name)
        else:
            logger.info(
                "mask not saved (run with --save_mask to attempt sim segmentation). "
                "Downstream will fall back to a placeholder mask if none is provided."
            )

        logger.info("saved sample -> %s", sample_dir)
        saved += 1
        next_capture = step + args_cli.sample_stride

    logger.info("done; saved %d sample(s) under %s", saved, output_root)
    simulation_app.close()


def _try_save_mask(camera, sample_dir, object_name):
    """Best-effort: derive a binary mask for object_name from semantic seg output."""
    try:
        output = camera.data.output
        seg_key = None
        for k in ("semantic_segmentation", "instance_segmentation_fast", "instance_segmentation"):
            if k in output:
                seg_key = k
                break
        if seg_key is None:
            logger.warning(
                "no segmentation channel in camera output; add 'semantic_segmentation' "
                "to CameraCfg.data_types to enable mask export. Skipping mask."
            )
            return
        seg = output[seg_key]
        seg = seg.detach().cpu().numpy()[0] if hasattr(seg, "detach") else np.asarray(seg)[0]
        if seg.ndim == 3:
            seg = seg[..., 0]
        # Without robust id->label mapping we save a foreground mask (non-background).
        ids, counts = np.unique(seg, return_counts=True)
        bg = ids[np.argmax(counts)]  # most frequent id assumed to be background
        mask = seg != bg
        image_io.save_mask(os.path.join(sample_dir, "mask.png"), mask)
        logger.info("saved best-effort mask (foreground = non-background id %s)", bg)
    except Exception as e:  # pragma: no cover
        logger.warning("mask export failed (%s); continuing without a mask.", e)


if __name__ == "__main__":
    main()
