"""Inject a D435-like RGB-D camera into the latest SceneLayoutModule scene.

Loads the latest saved scene USD (read as data; SceneLayoutModule Python is NOT
imported), authors a D435-like camera on the Franka end-effector (default) or a
fixed scene camera (fallback), adds semantic labels to the cubes, and SAVES A
NEW instrumented scene (USD + JSON manifest + markdown report) back into
SceneLayoutModule/saved_scenes/ WITHOUT modifying the original scene.

Run:
    cd /home1/banghai/Documents/IsaacLab
    conda run -n env_isaaclab python franka_d435_foundationpose/scripts/instrument_latest_scene_with_sensors.py \
        --load_latest_saved --camera_mode end_effector --save_to_scene_layout
  (or via ./isaaclab.sh -p ... if your launcher binds env_isaaclab)
"""

import argparse
import os

import _bootstrap  # noqa: F401

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as e:
    raise SystemExit(
        f"[instrument_latest_scene_with_sensors] IsaacLab runtime not found ({e}).\n"
        "Run with env_isaaclab python (it has Isaac Sim), e.g.:\n"
        "    conda run -n env_isaaclab python franka_d435_foundationpose/scripts/"
        "instrument_latest_scene_with_sensors.py --load_latest_saved\n"
    )

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene_usd", default=None, help="explicit source scene USD")
parser.add_argument("--load_latest_saved", action="store_true",
                    help="auto-find the latest (non-instrumented) saved scene")
parser.add_argument("--camera_mode", default=None, choices=["end_effector", "fixed_scene"])
parser.add_argument("--save_to_scene_layout", action="store_true",
                    help="save outputs into SceneLayoutModule/saved_scenes (default)")
parser.add_argument("--camera_mount", default=None, help="camera_mount.yaml path")
parser.add_argument("--no_semantics", action="store_true", help="skip cube semantic labels")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402

from franka_d435_foundationpose.camera.mount_config import (  # noqa: E402
    CAMERA_MODE_EE,
    load_camera_mount,
)
from franka_d435_foundationpose.isaaclab import saved_scene_loader as ssl  # noqa: E402
from franka_d435_foundationpose.utils import image_io  # noqa: E402
from franka_d435_foundationpose.utils.config import (  # noqa: E402
    default_config_path,
    project_path,
    resolve_cli_path,
)
from franka_d435_foundationpose.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("instrument_scene")


def _nonconflicting(path):
    """Return path, or path with a numeric suffix if it already exists."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{stem}_{i}{ext}"):
        i += 1
    return f"{stem}_{i}{ext}"


def main():
    saved_dir = ssl.scene_layout_saved_dir()
    if args_cli.scene_usd:
        source_usd = resolve_cli_path(args_cli.scene_usd)
    elif args_cli.load_latest_saved:
        source_usd = ssl.find_latest_scene(saved_dir)
    else:
        raise SystemExit("provide --scene_usd or --load_latest_saved")
    logger.info("source scene USD: %s", source_usd)

    mount = load_camera_mount(args_cli.camera_mount or default_config_path("camera_mount.yaml"))
    mode = mount.get(args_cli.camera_mode)
    logger.info("camera mode: %s", mode.name)

    # --- open source stage (read as data) ---
    stage = ssl.open_saved_scene(source_usd)

    # --- discovery (+ persist a discovery report) ---
    ee_cands = mode.parent_link_candidates if mode.name == CAMERA_MODE_EE else None
    discovered = ssl.discover_prims(stage, ee_candidates=ee_cands)
    discovery_report = project_path("outputs", "scene_discovery", "latest_scene_discovery_report.md")
    ssl.write_discovery_report(discovered, source_usd, discovery_report)
    logger.info("wrote discovery report -> %s", discovery_report)

    # --- decide camera prim path + mounted link ---
    if mode.name == CAMERA_MODE_EE:
        try:
            robot_path = ssl.find_franka_root(stage, discovered)
            ee_link_path = ssl.find_ee_link(stage, robot_path, mode.parent_link_candidates
                                            or ssl.DEFAULT_EE_CANDIDATES)
            cam_prim_path = f"{ee_link_path}/{mode.prim_path_suffix}"
            mounted_link = ee_link_path.rsplit("/", 1)[-1]
        except Exception as e:
            logger.error("EE link discovery failed (%s). Falling back to fixed_scene. "
                         "Re-run with --camera_mode fixed_scene to silence this.", e)
            mode = mount.get("fixed_scene")
            cam_prim_path = mode.prim_path
            mounted_link = None
    else:
        cam_prim_path = mode.prim_path
        mounted_link = None

    # --- author the camera prim ---
    if mode.name == CAMERA_MODE_EE:
        extrinsic = mode.T_ee_camera()
    else:
        from franka_d435_foundationpose.camera.mount_config import look_at_to_T_world_camera
        p = mode.pose()
        extrinsic = look_at_to_T_world_camera(p["translation_m"], p["look_at_m"])
    ssl.author_usd_camera(
        stage, cam_prim_path, extrinsic, mode.sim_optics,
        {"width": mode.width, "height": mode.height}, mode.camera_frame, mode.name,
    )
    logger.info("authored D435-like camera prim at %s (mode=%s, link=%s)",
                cam_prim_path, mode.name, mounted_link)

    # --- cube discovery + semantic labels ---
    cubes = ssl.find_cubes(stage, discovered)
    labels_added = []
    if not args_cli.no_semantics:
        for i, cube in enumerate(cubes):
            label = f"cube_{i}"
            if ssl.add_semantic_label(stage, cube, label):
                labels_added.append(label)
        logger.info("semantic labels added: %s", labels_added or "(none / unsupported)")
    default_cube_prim = cubes[0] if cubes else None
    default_cube = "cube_0" if cubes else None

    # --- export the instrumented scene (new files, never overwrite) ---
    src_stem = os.path.splitext(os.path.basename(source_usd))[0]
    out_usd = _nonconflicting(os.path.join(saved_dir, f"{src_stem}_with_ee_d435.usd"))
    ssl.export_stage(stage, out_usd)
    logger.info("saved instrumented scene USD -> %s", out_usd)

    out_stem = os.path.splitext(out_usd)[0]
    manifest = ssl.build_instrument_manifest(
        source_scene_usd=source_usd,
        instrumented_scene_usd=out_usd,
        camera_prim_path=cam_prim_path,
        camera_mode=mode.name,
        mounted_link=mounted_link,
        camera_frame=mode.camera_frame,
        resolution={"width": mode.width, "height": mode.height},
        sensor_types={"rgb": True, "depth": True, "segmentation": mode.want_segmentation},
        discovered_cubes=cubes,
        default_target_cube=default_cube,
        default_target_cube_prim=default_cube_prim,
        semantic_labels_added=labels_added,
        extra={"T_ee_camera": extrinsic.tolist() if mode.name == CAMERA_MODE_EE else None,
               "knife_candidates": discovered.get("knife", [])},
    )
    manifest_path = out_stem + ".json"
    image_io.save_json(manifest_path, manifest)
    report_path = out_stem + "_report.md"
    ssl.write_instrument_report_md(manifest, report_path)
    logger.info("saved manifest -> %s", manifest_path)
    logger.info("saved report   -> %s", report_path)

    logger.info("DONE. original scene untouched: %s", source_usd)
    logger.info("next: run_instrumented_scene_pose_test.py --load_latest_instrumented_scene "
                "--object_name %s --allow_mock_mask --mock_foundationpose", default_cube or "cube_0")
    simulation_app.close()


if __name__ == "__main__":
    main()
