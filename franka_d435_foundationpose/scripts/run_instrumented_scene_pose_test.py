"""Load an instrumented scene, capture from the EE D435, estimate + validate pose.

Pipeline (all outputs written to disk; Isaac Sim swallows stdout on exit):
  1. load the instrumented scene USD (camera already baked in)
  2. find the D435 camera prim + the target cube
  3. capture rgb/depth/K (+ mask) from the camera
  4. read T_base_ee (robot FK via USD), build the EE transform chain
  5. run mock (or real) FoundationPose -> T_camera_object -> T_base_object_pred
  6. read the cube's ground-truth pose from the sim -> T_base_object_gt
  7. compute errors + write a pose validation report

Does not re-create the camera unless --ensure_sensor is passed.

Run:
    cd /home1/banghai/Documents/IsaacLab
    conda run -n env_isaaclab python franka_d435_foundationpose/scripts/run_instrumented_scene_pose_test.py \
        --enable_cameras --load_latest_instrumented_scene \
        --object_name cube_0 --allow_mock_mask --mock_foundationpose
"""

import argparse
import os
import time

import _bootstrap  # noqa: F401

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as e:
    raise SystemExit(
        f"[run_instrumented_scene_pose_test] IsaacLab runtime not found ({e}).\n"
        "Run with env_isaaclab python (has Isaac Sim) and --enable_cameras."
    )

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene_usd", default=None, help="explicit instrumented scene USD")
parser.add_argument("--load_latest_instrumented_scene", action="store_true",
                    help="auto-find latest *_with_ee_d435.usd")
parser.add_argument("--object_name", default="cube_0", help="target object label")
parser.add_argument("--object_prim_path", default=None, help="explicit target cube prim")
parser.add_argument("--camera_mount", default=None, help="camera_mount.yaml path")
parser.add_argument("--allow_mock_mask", action="store_true",
                    help="use a center-rectangle mock mask if no real mask")
parser.add_argument("--mock_foundationpose", action="store_true",
                    help="force the mock FoundationPose estimator")
parser.add_argument("--ensure_sensor", action="store_true",
                    help="spawn the camera if it is missing from the scene")
parser.add_argument("--output_dir", default=None, help="defaults to a timestamped dir")
parser.add_argument("--num_warmup_steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from franka_d435_foundationpose.camera.isaaclab_camera_provider import extract_rgbd_frame  # noqa: E402
from franka_d435_foundationpose.camera.mount_config import CAMERA_MODE_EE, load_camera_mount  # noqa: E402
from franka_d435_foundationpose.foundationpose.foundationpose_wrapper import FoundationPoseEstimator  # noqa: E402
from franka_d435_foundationpose.foundationpose.mask_provider import MaskProvider  # noqa: E402
from franka_d435_foundationpose.foundationpose.mesh_loader import get_object_asset  # noqa: E402
from franka_d435_foundationpose.isaaclab import saved_scene_loader as ssl  # noqa: E402
from franka_d435_foundationpose.transforms.se3 import compose_T, invert_T  # noqa: E402
from franka_d435_foundationpose.utils import image_io  # noqa: E402
from franka_d435_foundationpose.utils.config import default_config_path, project_path, resolve_cli_path  # noqa: E402
from franka_d435_foundationpose.utils.logging_utils import get_logger  # noqa: E402
from franka_d435_foundationpose.validation.pose_metrics import build_validation, write_validation_report_md  # noqa: E402

from run_foundationpose_on_saved_rgbd import draw_pose_overlay  # noqa: E402

logger = get_logger("instrumented_pose_test")
BASE_LINK_NAMES = ("panda_link0", "base", "base_link")


def find_camera_prim(stage, discovered, mode, ee_link_path):
    for cpath in discovered.get("cameras", []):
        prim = stage.GetPrimAtPath(cpath)
        a = prim.GetAttribute("fp:camera_frame")
        if a and a.Get() == mode.camera_frame:
            return cpath
    cand = f"{ee_link_path}/{mode.prim_path_suffix}" if ee_link_path else mode.prim_path
    prim = stage.GetPrimAtPath(cand)
    return cand if prim and prim.IsValid() else None


def find_base_path(stage, robot_path):
    for b in BASE_LINK_NAMES:
        p = f"{robot_path}/{b}"
        pr = stage.GetPrimAtPath(p)
        if pr and pr.IsValid():
            return p
    return robot_path


def _parse_seg_key(k):
    """Parse an idToLabels key: an RGBA tuple string '(r, g, b, a)' or an int."""
    s = str(k).strip()
    if s.startswith("("):
        return tuple(int(x) for x in s.strip("()").split(","))
    return int(s)


def _mask_from_seg(seg, keys):
    """Build a boolean mask from an RGBA (H,W,4) or integer (H,W) seg + keys."""
    seg = np.asarray(seg)
    if seg.ndim == 3:  # RGBA-coded segmentation
        m = np.zeros(seg.shape[:2], dtype=bool)
        for k in keys:
            rgba = np.array(_parse_seg_key(k))
            n = min(len(rgba), seg.shape[2])
            m |= np.all(seg[..., :n] == rgba[:n], axis=-1)
        return m
    ids = [_parse_seg_key(k) for k in keys]  # integer-coded
    return np.isin(seg, ids)


def make_mask(frame, camera, target_label, target_prim, allow_mock):
    """Real segmentation mask for the target (by prim path or class), else mock box.

    The sim segmentation is RGBA-coded; idToLabels maps an RGBA-tuple-string to a
    class (semantic) or prim path (instance). Classes may be merged comma-lists
    (e.g. 'cube_0,cube_1'), so we match the target as an exact token / path.
    """
    output = camera.data.output
    info = camera.data.info[0] if getattr(camera.data, "info", None) else {}
    target_leaf = target_prim.rsplit("/", 1)[-1] if target_prim else None
    for key, kind in (
        ("instance_segmentation_fast", "prim"),
        ("instance_segmentation", "prim"),
        ("semantic_segmentation", "label"),
    ):
        if key not in output:
            continue
        try:
            seg = output[key]
            seg = seg.detach().cpu().numpy()[0] if hasattr(seg, "detach") else np.asarray(seg)[0]
            id2l = (info.get(key) or {}).get("idToLabels", {})
            want = []
            for k, v in id2l.items():
                label = v.get("class") if isinstance(v, dict) else str(v)
                if not label:
                    continue
                if kind == "prim":
                    hit = (target_prim and target_prim in str(label)) or \
                          (target_leaf and target_leaf in str(label))
                else:
                    tokens = [t.strip() for t in str(label).split(",")]
                    hit = target_label in tokens
                if hit:
                    want.append(k)
            if want:
                mask = _mask_from_seg(seg, want)
                if mask.any():
                    src = "sim_instance_segmentation" if kind == "prim" else "sim_semantic_segmentation"
                    return mask.astype(bool), src
        except Exception as e:  # pragma: no cover
            logger.warning("%s mask failed (%s)", key, e)
    if allow_mock:
        return MaskProvider.center_box(frame.height, frame.width, frac=0.35), "mock_center_rectangle"
    return None, "none"


def main():
    saved_dir = ssl.scene_layout_saved_dir()
    if args_cli.scene_usd:
        scene_usd = resolve_cli_path(args_cli.scene_usd)
    elif args_cli.load_latest_instrumented_scene:
        scene_usd = ssl.find_latest_instrumented_scene(saved_dir)
    else:
        raise SystemExit("provide --scene_usd or --load_latest_instrumented_scene")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_cli_path(args_cli.output_dir) if args_cli.output_dir else \
        project_path("outputs", "instrumented_scene_pose_tests", ts)
    os.makedirs(out_dir, exist_ok=True)
    log_lines = [f"scene_usd: {scene_usd}", f"output_dir: {out_dir}"]

    mount = load_camera_mount(args_cli.camera_mount or default_config_path("camera_mount.yaml"))
    mode = mount.get(CAMERA_MODE_EE)

    # --- open scene + sim ---
    stage = ssl.open_saved_scene(scene_usd)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    discovered = ssl.discover_prims(stage, ee_candidates=mode.parent_link_candidates)
    robot_path = ssl.find_franka_root(stage, discovered)
    ee_link_path = ssl.find_ee_link(stage, robot_path, mode.parent_link_candidates)
    base_path = find_base_path(stage, robot_path)
    target_prim = ssl.select_target_cube(stage, args_cli.object_name, args_cli.object_prim_path, discovered)
    log_lines += [f"robot: {robot_path}", f"ee_link: {ee_link_path}", f"target_prim: {target_prim}"]

    # --- camera: reuse baked prim (spawn=None) unless --ensure_sensor ---
    cam_prim = find_camera_prim(stage, discovered, mode, ee_link_path)
    from franka_d435_foundationpose.isaaclab.attach_d435_to_franka import build_camera_cfg_for_mount

    cam_path = cam_prim or f"{ee_link_path}/{mode.prim_path_suffix}"
    camera_cfg, _extr = build_camera_cfg_for_mount(mode, cam_path)
    if cam_prim and not args_cli.ensure_sensor:
        camera_cfg.spawn = None  # reuse the camera already baked into the scene
        log_lines.append(f"camera: reusing baked prim {cam_prim}")
    else:
        log_lines.append(f"camera: spawning at {cam_path} (ensure_sensor={args_cli.ensure_sensor})")

    from isaaclab.sensors import Camera

    camera = Camera(camera_cfg)

    # --- step + capture ---
    sim.reset()
    dt = sim_cfg.dt
    for _ in range(args_cli.num_warmup_steps):
        sim.step()
    camera.update(dt)
    frame = extract_rgbd_frame(camera, 0, mode.camera_frame,
                               metadata={"engine": "isaaclab", "scene_usd": os.path.basename(scene_usd),
                                         "camera_mode": "end_effector"})

    # --- transforms ---
    # Read PHYSICS-accurate world poses of the hand + base (after the scene
    # settles under sim.reset). The USD-authored xform is the pre-physics pose;
    # the camera is rigidly attached to the physics hand, so we must use the
    # settled pose or the chain is inconsistent with the rendered image.
    from franka_d435_foundationpose.transforms.se3 import make_T

    def _xfv_pose(prim_path):
        from isaaclab.sim.views import XformPrimView

        view = XformPrimView(prim_path)
        pos, quat = view.get_world_poses()  # (1,3), (1,4) wxyz
        pos = pos[0].detach().cpu().numpy()
        w, x, y, z = [float(v) for v in quat[0].detach().cpu().numpy()]
        return make_T(R=np.array([x, y, z, w]), t=pos)

    T_ee_camera = mode.T_ee_camera()
    try:
        T_world_ee = _xfv_pose(ee_link_path)
        T_world_base = _xfv_pose(base_path)
        pose_source = "xform_prim_view_physics"
    except Exception as e:  # pragma: no cover - fallback to authored USD xform
        logger.warning("XformPrimView failed (%s); using USD authored xform", e)
        pos_e, q_e = ssl.get_world_pose_pxr(stage, ee_link_path)
        pos_b, q_b = ssl.get_world_pose_pxr(stage, base_path)
        T_world_ee = ssl.world_pose_to_T(pos_e, q_e)
        T_world_base = ssl.world_pose_to_T(pos_b, q_b)
        pose_source = "usd_xform_fallback"
    T_base_ee = compose_T(invert_T(T_world_base), T_world_ee)
    T_base_camera = compose_T(T_base_ee, T_ee_camera)
    log_lines.append(f"camera_pose_source: {pose_source}")

    # --- mask ---
    mask, mask_source = make_mask(frame, camera, args_cli.object_name, target_prim, args_cli.allow_mock_mask)
    frame.metadata["mask_source"] = mask_source
    frame.metadata["object_name"] = args_cli.object_name
    frame.metadata["object_prim_path"] = target_prim

    # --- save sample ---
    frame.save_sample(out_dir)
    image_io.save_json(os.path.join(out_dir, "transforms.json"), {
        "convention": "T_target_source means transforming a point from the source frame to the target frame",
        "camera_mode": "end_effector_d435",
        "base_frame": base_path.rsplit("/", 1)[-1],
        "ee_frame": "franka_hand",
        "camera_frame": mode.camera_frame,
        "T_base_ee": T_base_ee.tolist(),
        "T_ee_camera": T_ee_camera.tolist(),
        "T_base_camera": T_base_camera.tolist(),
        "timestamp": frame.timestamp,
    })
    if mask is not None:
        image_io.save_mask(os.path.join(out_dir, "mask.png"), mask)
    log_lines.append(f"mask_source: {mask_source}")

    # --- ground-truth cube pose from sim ---
    gt = ssl.get_object_gt_pose(stage, target_prim, robot_base_path=base_path)
    image_io.save_json(os.path.join(out_dir, "gt_pose.json"), {
        "convention": "T_target_source",
        "object_name": args_cli.object_name,
        "object_prim_path": target_prim,
        "base_frame": gt.get("base_frame"),
        "T_world_object": gt["T_world_object"].tolist(),
        "T_base_object": gt["T_base_object"].tolist(),
        "source": "sim_usd_xform",
    })
    T_base_object_gt = gt["T_base_object"]

    # --- FoundationPose (mock or real) ---
    asset = get_object_asset(default_config_path("object_assets.yaml"), args_cli.object_name)
    estimator = FoundationPoseEstimator(default_config_path("foundationpose.yaml"),
                                        force_mock=args_cli.mock_foundationpose)
    est_mask = mask if mask is not None else MaskProvider.center_box(frame.height, frame.width)
    result = estimator.estimate(frame, asset.mesh_path, est_mask)
    # T_base_camera == T_base_ee @ T_ee_camera, so this equals the full chain.
    T_base_object_pred = compose_T(T_base_camera, result.T_camera_object)
    result.save_json(os.path.join(out_dir, "pose_result.json"), extra={
        "object_name": args_cli.object_name,
        "camera_mode": "end_effector_d435",
        "camera_frame": mode.camera_frame,
        "base_frame": gt.get("base_frame"),
        "mask_source": mask_source,
        "backend": estimator.backend,
        "T_base_object": T_base_object_pred.tolist(),
    })
    try:
        overlay = draw_pose_overlay(frame.rgb, frame.K, result.T_camera_object)
        image_io.save_rgb(os.path.join(out_dir, "pose_overlay.png"), overlay)
    except Exception as e:  # pragma: no cover
        log_lines.append(f"overlay failed: {e}")

    # --- validation ---
    val = build_validation(T_base_object_pred, T_base_object_gt, args_cli.object_name,
                           frame="base", symmetry="cube", backend=estimator.backend)
    image_io.save_json(os.path.join(out_dir, "pose_validation_report.json"), val)
    write_validation_report_md(val, os.path.join(out_dir, "pose_validation_report.md"))

    log_lines += [
        f"backend: {estimator.backend}",
        f"translation_error_m: {val['translation_error_m']:.4f}",
        f"rotation_error_deg: {val['rotation_error_deg']:.2f}",
        f"rotation_error_deg_symmetric: {val['rotation_error_deg_symmetric']:.2f}",
        f"warnings: {val['warnings']}",
    ]
    # Persist a plain-text run log (stdout is unreliable under Isaac Sim).
    with open(os.path.join(out_dir, "run_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
