"""PRIMARY demo: saved scene USD + Franka + end-effector D435-like camera.

Loads a pre-saved scene USD (read as data; SceneLayoutModule Python code is NOT
imported), finds the Franka and its end-effector link, mounts a D435-like RGB-D
camera on the end-effector (default) or as a fixed scene camera (fallback/debug),
captures one RGB-D frame, and saves a FoundationPose-ready sample.

Does NOT import FoundationPose.

Run (default = end-effector camera):
    cd /home1/banghai/Documents/IsaacLab
    ./isaaclab.sh -p franka_d435_foundationpose/scripts/run_saved_scene_ee_d435_demo.py \
        --enable_cameras \
        --scene_usd SceneLayoutModule/saved_scenes/scene_v0_20260611_191056.usd \
        --object_name coffee_machine \
        --object_prim_path /coffeemachine \
        --allow_mock_mask

Fallback/debug (fixed environment camera):
    ... add  --camera_mode fixed_scene
"""

import argparse
import os

import _bootstrap  # noqa: F401

# ---------------------------------------------------------------------------- #
# 1. argparse + launch the simulator BEFORE importing any isaaclab modules.
# ---------------------------------------------------------------------------- #
try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as e:
    raise SystemExit(
        "[run_saved_scene_ee_d435_demo] IsaacLab/IsaacSim runtime not found "
        f"({e}).\nLaunch via isaaclab.sh, NOT plain python:\n"
        "    cd /home1/banghai/Documents/IsaacLab\n"
        "    ./isaaclab.sh -p franka_d435_foundationpose/scripts/"
        "run_saved_scene_ee_d435_demo.py --enable_cameras --scene_usd <usd>\n"
        "Requires a GPU + Isaac Sim (GUI or headless)."
    )

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene_usd", required=True, help="path to the saved scene USD")
parser.add_argument("--camera_mode", default=None,
                    choices=["end_effector", "fixed_scene"],
                    help="override camera mount mode (default: from camera_mount.yaml)")
parser.add_argument("--object_name", default="coffee_machine", help="object name hint")
parser.add_argument("--object_prim_path", default=None, help="target object prim path")
parser.add_argument("--allow_mock_mask", action="store_true",
                    help="if no real mask, write a center-rectangle mock mask")
parser.add_argument("--output_dir", default=None,
                    help="defaults to <project>/outputs/ee_d435_samples/sample_000001")
parser.add_argument("--camera_mount", default=None, help="path to camera_mount.yaml")
parser.add_argument("--num_warmup_steps", type=int, default=16, help="sim steps before capture")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------- #
# 2. Safe to import isaaclab + our isaaclab-side helpers now.
# ---------------------------------------------------------------------------- #
import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from franka_d435_foundationpose.camera.isaaclab_camera_provider import (  # noqa: E402
    extract_rgbd_frame,
)
from franka_d435_foundationpose.camera.mount_config import (  # noqa: E402
    CAMERA_MODE_EE,
    load_camera_mount,
)
from franka_d435_foundationpose.foundationpose.mask_provider import MaskProvider  # noqa: E402
from franka_d435_foundationpose.isaaclab.attach_d435_to_franka import (  # noqa: E402
    build_camera_cfg_for_mount,
)
from franka_d435_foundationpose.isaaclab import saved_scene_loader as ssl  # noqa: E402
from franka_d435_foundationpose.transforms.se3 import compose_T, invert_T  # noqa: E402
from franka_d435_foundationpose.utils import image_io  # noqa: E402
from franka_d435_foundationpose.utils.config import (  # noqa: E402
    default_config_path,
    project_path,
    resolve_cli_path,
)
from franka_d435_foundationpose.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("saved_scene_ee_demo")

BASE_LINK_NAMES = ("panda_link0", "base", "base_link")


def read_link_poses(robot_path, ee_link_path, stage, sim, dt):
    """Return (T_world_base, T_world_ee) trying the Articulation wrapper first,
    falling back to pxr USD xforms. Returns (None, None, reason) on total failure."""
    ee_name = ee_link_path.rsplit("/", 1)[-1]

    # --- Backend A: IsaacLab Articulation (physics-accurate) ---
    try:
        from isaaclab.assets import Articulation, ArticulationCfg

        robot = Articulation(ArticulationCfg(prim_path=robot_path, spawn=None, actuators={}))
        sim.reset()
        for _ in range(args_cli.num_warmup_steps):
            sim.step()
        robot.update(dt)
        names = list(robot.body_names)
        pose_w = robot.data.body_pose_w[0].detach().cpu().numpy()  # (num_bodies, 7) pos+quat wxyz

        ee_idx = names.index(ee_name) if ee_name in names else None
        base_idx = next((names.index(b) for b in BASE_LINK_NAMES if b in names), 0)
        if ee_idx is None:
            logger.warning("EE link '%s' not in articulation bodies %s; using pxr fallback",
                           ee_name, names)
        else:
            T_world_ee = _pose7_wxyz_to_T(pose_w[ee_idx])
            T_world_base = _pose7_wxyz_to_T(pose_w[base_idx])
            logger.info("link poses via IsaacLab Articulation (base=%s, ee=%s)",
                        names[base_idx], ee_name)
            return T_world_base, T_world_ee, robot
    except Exception as e:  # pragma: no cover - depends on runtime
        logger.warning("Articulation pose read failed (%s); using pxr fallback", e)
        try:
            sim.reset()
            for _ in range(args_cli.num_warmup_steps):
                sim.step()
        except Exception:
            pass

    # --- Backend B: pxr USD xforms ---
    base_path = _find_base_path(stage, robot_path)
    pos_b, quat_b = ssl.get_world_pose_pxr(stage, base_path)
    pos_e, quat_e = ssl.get_world_pose_pxr(stage, ee_link_path)
    T_world_base = ssl.world_pose_to_T(pos_b, quat_b)
    T_world_ee = ssl.world_pose_to_T(pos_e, quat_e)
    logger.info("link poses via pxr USD xforms (base=%s)", base_path)
    return T_world_base, T_world_ee, None


def _pose7_wxyz_to_T(pose7):
    from franka_d435_foundationpose.transforms.se3 import make_T

    p = np.asarray(pose7[:3], dtype=np.float64)
    qw, qx, qy, qz = [float(v) for v in pose7[3:7]]
    return make_T(R=np.array([qx, qy, qz, qw]), t=p)


def _find_base_path(stage, robot_path):
    for b in BASE_LINK_NAMES:
        cand = f"{robot_path}/{b}"
        prim = stage.GetPrimAtPath(cand)
        if prim and prim.IsValid():
            return cand
    return robot_path  # fall back to the root


def make_mask(frame, camera, allow_mock):
    """Return (mask, mask_source). Real segmentation if available, else mock box."""
    output = camera.data.output
    for key in ("semantic_segmentation", "instance_segmentation_fast", "instance_segmentation"):
        if key in output:
            try:
                seg = output[key]
                seg = seg.detach().cpu().numpy()[0] if hasattr(seg, "detach") else np.asarray(seg)[0]
                if seg.ndim == 3:
                    seg = seg[..., 0]
                ids, counts = np.unique(seg, return_counts=True)
                bg = ids[np.argmax(counts)]
                mask = seg != bg
                if mask.any():
                    return mask.astype(bool), "sim_segmentation_foreground"
            except Exception as e:  # pragma: no cover
                logger.warning("segmentation mask failed (%s)", e)
            break
    if allow_mock:
        mask = MaskProvider.center_box(frame.height, frame.width, frac=0.35)
        return mask, "mock_center_rectangle"
    return None, "none"


def main():
    scene_usd = resolve_cli_path(args_cli.scene_usd)
    mount_path = args_cli.camera_mount or default_config_path("camera_mount.yaml")
    output_dir = (
        resolve_cli_path(args_cli.output_dir)
        if args_cli.output_dir
        else project_path("outputs", "ee_d435_samples", "sample_000001")
    )
    os.makedirs(output_dir, exist_ok=True)

    mount = load_camera_mount(mount_path)
    mode = mount.get(args_cli.camera_mode)  # None -> default_mode (end_effector)
    logger.info("camera mount mode: %s (default=%s)", mode.name, mount.default_mode)

    # --- open the saved scene + simulation context ---
    stage = ssl.open_saved_scene(scene_usd)
    logger.info("opened scene USD: %s", scene_usd)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # --- discover prims ---
    ee_cands = mode.parent_link_candidates if mode.name == CAMERA_MODE_EE else None
    discovered = ssl.discover_prims(stage, ee_candidates=ee_cands)
    if args_cli.object_prim_path:
        logger.info("target object prim (user): %s", args_cli.object_prim_path)

    # --- build + attach the camera ---
    if mode.name == CAMERA_MODE_EE:
        robot_path = ssl.find_franka_root(stage, discovered)
        ee_link_path = ssl.find_ee_link(stage, robot_path, mode.parent_link_candidates or
                                        ssl.DEFAULT_EE_CANDIDATES)
        cam_prim_path = f"{ee_link_path}/{mode.prim_path_suffix}"
        logger.info("attaching EE camera at %s", cam_prim_path)
    else:
        robot_path = None
        ee_link_path = None
        cam_prim_path = mode.prim_path
        logger.info("placing fixed scene camera at %s", cam_prim_path)

    camera_cfg, extrinsic = build_camera_cfg_for_mount(mode, cam_prim_path)

    from isaaclab.sensors import Camera

    camera = Camera(camera_cfg)

    # --- step + capture ---
    dt = sim_cfg.dt
    if mode.name == CAMERA_MODE_EE:
        T_world_base, T_world_ee, _robot = read_link_poses(robot_path, ee_link_path, stage, sim, dt)
    else:
        sim.reset()
        for _ in range(args_cli.num_warmup_steps):
            sim.step()
        T_world_base = T_world_ee = None

    camera.update(dt)
    frame = extract_rgbd_frame(
        camera, cam_index=0, camera_frame=mode.camera_frame,
        metadata={"engine": "isaaclab", "scene_usd": os.path.basename(scene_usd),
                  "camera_mode": mode.name},
    )

    # --- transforms ---
    transforms = {
        "convention": "T_target_source means transforming a point from the source "
        "frame to the target frame",
        "base_frame": "panda_link0",
        "ee_frame": "franka_hand",
        "camera_frame": mode.camera_frame,
        "timestamp": frame.timestamp,
    }
    if mode.name == CAMERA_MODE_EE:
        T_ee_camera = extrinsic  # camera -> ee
        T_base_ee = compose_T(invert_T(T_world_base), T_world_ee)
        T_base_camera = compose_T(T_base_ee, T_ee_camera)
        transforms.update({
            "camera_mode": "end_effector_d435",
            "T_base_ee": T_base_ee.tolist(),
            "T_ee_camera": T_ee_camera.tolist(),
            "T_base_camera": T_base_camera.tolist(),
        })
        logger.info("T_base_ee=\n%s", np.array2string(T_base_ee, precision=4))
    else:
        T_world_camera = extrinsic  # camera -> world
        transforms.update({
            "camera_mode": "fixed_scene_d435",
            "T_world_camera": T_world_camera.tolist(),
        })
        logger.info("T_world_camera=\n%s", np.array2string(T_world_camera, precision=4))

    # --- mask ---
    mask, mask_source = make_mask(frame, camera, args_cli.allow_mock_mask)

    # --- save sample ---
    frame.metadata["mask_source"] = mask_source
    frame.metadata["object_name"] = args_cli.object_name
    frame.metadata["object_prim_path"] = args_cli.object_prim_path
    frame.save_sample(output_dir)
    image_io.save_json(os.path.join(output_dir, "transforms.json"), transforms)
    with open(os.path.join(output_dir, "object_config.yaml"), "w") as f:
        f.write(f"object_name: {args_cli.object_name}\n")
        if args_cli.object_prim_path:
            f.write(f"object_prim_path: {args_cli.object_prim_path}\n")
    if mask is not None:
        image_io.save_mask(os.path.join(output_dir, "mask.png"), mask)
        logger.info("saved mask (source=%s)", mask_source)
    else:
        logger.warning("no mask saved (mask_source=none). Pass --allow_mock_mask or "
                       "provide a mask for FoundationPose estimate.")

    logger.info("saved sample -> %s", output_dir)
    logger.info("next: python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py "
                "--input_dir %s --object_name %s --mock",
                os.path.relpath(output_dir, project_path()), args_cli.object_name)
    simulation_app.close()


if __name__ == "__main__":
    main()
