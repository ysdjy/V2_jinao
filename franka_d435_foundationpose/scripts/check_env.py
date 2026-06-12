"""Environment diagnostic for both sides of the project.

Run on the IsaacLab side (must NOT fail just because FoundationPose is missing):
    cd /home1/banghai/Documents/IsaacLab
    ./isaaclab.sh -p franka_d435_foundationpose/scripts/check_env.py --side isaaclab

Run on the FoundationPose side:
    conda run -n foundationpose python franka_d435_foundationpose/scripts/check_env.py --side foundationpose
"""

import argparse
import importlib
import os
import sys

import _bootstrap  # noqa: F401  (puts project root on sys.path)

from franka_d435_foundationpose.utils.config import (
    PROJECT_ROOT,
    default_config_path,
    load_yaml,
    resolve_path,
)


def _check_import(name: str):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Check the project environment.")
    parser.add_argument(
        "--side",
        choices=["isaaclab", "foundationpose", "auto"],
        default="auto",
        help="Which side to check. 'auto' infers from available packages.",
    )
    args = parser.parse_args()

    ok = True
    print("=" * 70)
    print("franka_d435_foundationpose :: environment check")
    print("=" * 70)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"Project root      : {PROJECT_ROOT}")

    # Are we near the IsaacLab root?
    isaaclab_root = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
    has_isaaclab_sh = os.path.isfile(os.path.join(isaaclab_root, "isaaclab.sh"))
    print(f"IsaacLab root     : {isaaclab_root}  (isaaclab.sh: {has_isaaclab_sh})")

    # Light deps required on BOTH sides.
    print("\n[light dependencies]")
    for mod in ["numpy", "cv2", "yaml", "scipy"]:
        good, info = _check_import(mod)
        print(f"  {'OK ' if good else 'MISS'}  {mod:12s} {info if good else '-> ' + info}")
        if not good:
            ok = False

    # Optional deps.
    print("\n[optional dependencies]")
    for mod in ["zmq", "pyrealsense2", "trimesh", "torch"]:
        good, info = _check_import(mod)
        print(f"  {'OK ' if good else '...'}  {mod:12s} {info if good else 'not installed (optional)'}")

    # IsaacLab package.
    print("\n[IsaacLab]")
    good, info = _check_import("isaaclab")
    print(f"  {'OK ' if good else '...'}  isaaclab    {info if good else 'not importable (run via isaaclab.sh -p ...)'}")

    # FoundationPose config + repo + weights.
    print("\n[FoundationPose]")
    fp_cfg_path = default_config_path("foundationpose.yaml")
    fp_repo = fp_weights = None
    use_mock = True
    if os.path.isfile(fp_cfg_path):
        cfg = load_yaml(fp_cfg_path)
        fp_repo = resolve_path(cfg.get("foundationpose_repo"))
        fp_weights = resolve_path(cfg.get("weights_dir"))
        use_mock = bool(cfg.get("use_mock_if_unavailable", True))
        print(f"  config            : {fp_cfg_path}")
        print(f"  foundationpose_repo: {fp_repo}  (exists: {os.path.isdir(fp_repo) if fp_repo else False})")
        print(f"  weights_dir        : {fp_weights}  (exists: {os.path.isdir(fp_weights) if fp_weights else False})")
        print(f"  use_mock_if_unavailable: {use_mock}")
    else:
        print(f"  config not found: {fp_cfg_path}")

    repo_ok = bool(fp_repo) and os.path.isdir(fp_repo)
    if not repo_ok:
        print("  -> FoundationPose repo NOT found.")
        print("     git clone https://github.com/NVlabs/FoundationPose " + (fp_repo or "<path>"))
        print("     download weights into <repo>/weights, then build per its README or use Docker.")
        print(f"     Mock fallback is {'ENABLED' if use_mock else 'DISABLED'}.")

    # Can we import FoundationPose itself?
    fp_importable = False
    if repo_ok:
        if fp_repo not in sys.path:
            sys.path.insert(0, fp_repo)
        fp_importable, fp_info = _check_import("estimater")
        print(f"  import estimater   : {'OK' if fp_importable else 'FAILED -> ' + fp_info}")

    # Side-specific verdict.
    side = args.side
    if side == "auto":
        side = "foundationpose" if fp_importable else "isaaclab"
    print("\n" + "=" * 70)
    print(f"Side under evaluation: {side}")
    if side == "isaaclab":
        print("IsaacLab-side check: light deps are what matter here.")
        print("FoundationPose absence is OK — capture works; pose runs via the")
        print("separate `foundationpose` env or the mock estimator.")
    else:
        if not fp_importable:
            print("FoundationPose-side check: FoundationPose is NOT importable.")
            print("Fix the repo/weights/CUDA build (or use Docker) before real estimation.")
            if not use_mock:
                ok = False
        else:
            print("FoundationPose-side check: real estimator available. ")

    print("\n[recommended commands]")
    print("  Capture (IsaacLab):")
    print("    ./isaaclab.sh -p franka_d435_foundationpose/scripts/run_isaaclab_franka_d435_demo.py --enable_cameras")
    print("  Pose on a saved sample (mock):")
    print("    python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \\")
    print("        --input_dir franka_d435_foundationpose/outputs/rgbd_samples/sample_000001 \\")
    print("        --object_name target_object --mock")
    print("  Pose on a saved sample (real, isolated env):")
    print("    conda run -n foundationpose python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \\")
    print("        --input_dir .../sample_000001 --object_name target_object")
    print("=" * 70)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
