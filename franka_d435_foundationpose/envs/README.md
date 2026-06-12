# Environments

This project uses a **two-environment architecture** to keep the heavy
FoundationPose dependencies away from your stable IsaacLab/IsaacSim env.

## A. IsaacLab side — `env_isaaclab` (your existing env)

Responsible for: launching IsaacLab/IsaacSim, the Franka, the D435-like camera,
reading RGB/depth/intrinsics + end-effector pose, and saving RGB-D samples or
sending them to the pose server via the light `PoseClient`.

It must **never** import FoundationPose and must **never** install FoundationPose's
heavy/compiled dependencies.

Install only the light extras (most are already present):

```bash
conda run -n env_isaaclab pip install -r requirements_isaaclab_side.txt
```

## B. FoundationPose side — `foundationpose` (new, isolated env)

Responsible for: loading FoundationPose, the object mesh, running
`estimate`/`track`, and returning the 6D pose.

1. **Clone FoundationPose OUTSIDE this project** (do not copy its source in):

   ```bash
   git clone https://github.com/NVlabs/FoundationPose /home1/banghai/Documents/FoundationPose
   ```

   (or `/home1/banghai/Documents/third_party/FoundationPose`). Update
   `configs/foundationpose.yaml: foundationpose_repo` to match.

2. **Download the weights** into `<repo>/weights` per the FoundationPose README.

3. **Create the env** (starting point — follow the official README for the
   CUDA-compiled extensions, versions depend on your GPU/driver):

   ```bash
   conda env create -f environment_foundationpose.yml
   conda activate foundationpose
   # then build nvdiffrast / pytorch3d / etc. per FoundationPose's README
   ```

4. **Prefer the official Docker image** if the conda build is painful. Set
   `use_docker: true` in `configs/foundationpose.yaml`. Docker is the most
   reliable way to get FoundationPose running.

Until FoundationPose is configured, everything still runs against the
**MockFoundationPoseEstimator** (set `use_mock_if_unavailable: true`, the default),
so you can validate the full data flow first. With
`use_mock_if_unavailable: false`, a missing/broken FoundationPose instead raises a
clear error (repo path, weights, build/Docker hints) rather than silently mocking.

## Isolation rules (do not break these)

1. **Never** `pip install` FoundationPose's heavy/compiled deps (torch build,
   pytorch3d, nvdiffrast, custom CUDA ops) into `env_isaaclab`.
2. **Never** copy the FoundationPose source into this project. It stays in its
   own clone; this project only references it via
   `configs/foundationpose.yaml: foundationpose_repo` and adds it to `sys.path`
   lazily inside the `foundationpose` env.
3. The IsaacLab side talks to the FoundationPose side only through the disk
   sample format (mode 1) or the ZMQ `PoseClient` (mode 2) — never by importing
   FoundationPose.
4. `pyzmq` is optional and only needed for mode 2; it is not required for the
   default file-exchange workflow.

## C. Real D435 side — `realsense` (isolated, verified)

A separate light env for the real RealSense D435 (perception only; never touches
`env_isaaclab`). The pip wheel of `pyrealsense2` bundles the SDK, so no Intel apt
repo is required for the Python capture path:

```bash
conda create -n realsense python=3.10 -y
conda run -n realsense pip install pyrealsense2 numpy opencv-python pyyaml
# verify the device is seen:
conda run -n realsense python -c "import pyrealsense2 as rs; print(len(rs.context().query_devices()), 'device(s)')"
# capture one RGB-D sample (color + aligned depth + intrinsics):
conda run -n realsense python scripts/run_realsense_d435_live.py --save_only \
    --output_dir outputs/realsense_samples/sample_000001
```

Notes:
- The provider discards ~30 warmup frames so auto-exposure / the depth laser
  settle (the first frames are dark with mostly-invalid depth). Tunable via the
  `warmup_frames` arg of `RealSenseD435Provider`.
- `realsense-viewer` / `rs-enumerate-devices` are NOT installed by the pip wheel.
  Only install them if you want the GUI: add Intel's apt repo, then
  `sudo apt install librealsense2-utils librealsense2-dev`.
- udev rules: only needed if you hit permission errors enumerating/streaming as a
  non-root user. If so, install `99-realsense-libusb.rules` from the librealsense
  repo into `/etc/udev/rules.d/` and `sudo udevadm control --reload && udevadm trigger`.
- For real pose estimation, run FoundationPose (`foundationpose` env / Docker) on
  the saved sample dir; D435 capture and FoundationPose stay in separate envs.
