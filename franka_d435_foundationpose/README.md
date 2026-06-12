# franka_d435_foundationpose

Bridge an Intel RealSense **D435** mounted on a **Franka** end-effector with
**FoundationPose** to extract the **6D pose** of a target object relative to the
camera, the end-effector, and the robot base — in **IsaacLab simulation** and on
the **real robot**.

This is a deployment/wrapper project. It does **not** train FoundationPose; it
packages model-based FoundationPose inference (mesh + RGB + depth + intrinsics +
initial mask → 6D pose) behind a clean, environment-isolated interface.

---

## 1. Goal

Given an RGB-D frame from a D435 (real) or a D435-like sim camera, plus the
target object's CAD mesh and an initial mask, produce:

- `T_camera_object` — object pose in the camera optical frame (FoundationPose output)
- `T_base_object`   — object pose in the robot base frame, via the transform chain

so downstream code (grasping, manipulation) can act on the object in base coordinates.

### Primary pipeline: instrument the latest SceneLayoutModule scene, then test

The default workflow takes a scene authored in **SceneLayoutModule**, bakes a
D435-like camera onto the Franka end-effector, saves it back as a new
*instrumented* scene, then captures + estimates + validates a cube's pose.

```bash
cd /home1/banghai/Documents/IsaacLab

# (1) Author/save your scene in SceneLayoutModule (Save Both in the UI):
./isaaclab.sh -p SceneLayoutModule/scene_layout_ui.py --num_envs 1 --load_latest_saved

# (2) Instrument the latest saved scene with the EE D435 (writes a NEW scene;
#     original is never modified; SceneLayoutModule Python is NOT imported):
conda run -n env_isaaclab python franka_d435_foundationpose/scripts/instrument_latest_scene_with_sensors.py \
    --load_latest_saved --camera_mode end_effector --save_to_scene_layout --headless
#  -> SceneLayoutModule/saved_scenes/<stem>_with_ee_d435.usd  (+ .json + _report.md)

# (3) Capture + (mock) FoundationPose + ground-truth + validation:
conda run -n env_isaaclab python franka_d435_foundationpose/scripts/run_instrumented_scene_pose_test.py \
    --enable_cameras --headless --load_latest_instrumented_scene \
    --object_name cube_0 --allow_mock_mask --mock_foundationpose
#  -> outputs/instrumented_scene_pose_tests/<timestamp>/  (sample + pose_result.json
#     + gt_pose.json + pose_validation_report.{json,md})
```

> `./isaaclab.sh -p` works if your launcher binds `env_isaaclab`; on some setups
> it resolves to base Python (no Isaac Sim), so the commands above use
> `conda run -n env_isaaclab python` which is always correct.

The camera is on the end-effector, so the estimate chains to the base frame:
`T_base_object = T_base_ee @ T_ee_camera @ T_camera_object`, and is compared to
the cube's ground-truth pose read from the sim. **Cube symmetry:** translation
error is the reliable metric; rotation error is reported both naively and
minimized over the 24 cube rotations (see the validation report).

**Fallback / debug: fixed environment camera.** Add `--camera_mode fixed_scene`
to the instrument step; the chain becomes
`T_world_object = T_world_camera @ T_camera_object`. Modes live in
`configs/camera_mount.yaml` (`default_mode: end_effector`).

**Single-shot demo (no instrumented scene):**
`run_saved_scene_ee_d435_demo.py --scene_usd <usd> --object_name <name>
--allow_mock_mask` loads a scene and captures in one step (camera spawned on the
fly, not baked into a saved scene).

The saved scene is read **as data only** (USD/JSON) — SceneLayoutModule's Python
code is never imported. Objects are NOT all under `/World/envs/env_0` (e.g. the
coffee machine is at `/coffeemachine`); prim discovery handles this.

## 2. Why NOT install FoundationPose into `env_isaaclab`

FoundationPose pulls in PyTorch, CUDA-compiled extensions (nvdiffrast, custom
ops), pytorch3d, trimesh, OpenCV, scipy, etc. Installing those into your working
IsaacLab/IsaacSim env (`env_isaaclab`) risks breaking a known-good simulation
setup (torch/CUDA version conflicts especially). We keep them apart.

## 3. Two-environment architecture

```
┌─────────────────────────────┐      file exchange or ZMQ      ┌──────────────────────────────┐
│  IsaacLab side               │  ───────────────────────────▶ │  FoundationPose side          │
│  env: env_isaaclab           │   rgb, depth, K, mask, T's     │  env: foundationpose          │
│                              │                                │                               │
│  • launch IsaacLab/IsaacSim  │  ◀───────────────────────────  │  • load FoundationPose + mesh │
│  • Franka + EE D435 camera   │     T_camera_object, score     │  • estimate / track           │
│  • read RGB/depth/K + T_be   │                                │  • return 6D pose             │
│  • save samples / PoseClient │                                │  • PoseServer (ZMQ)           │
│  • NO FoundationPose import   │                                │  • heavy CUDA deps live here  │
└─────────────────────────────┘                                └──────────────────────────────┘
```

- **IsaacLab side** (`env_isaaclab`): light deps only (numpy, opencv, pyyaml,
  scipy, pyzmq). Never imports FoundationPose.
- **FoundationPose side** (`foundationpose`, separate conda env or Docker):
  the heavy estimator. Cloned **outside** this project; this repo only holds the
  wrapper, configs, scripts, and IO contracts.

See `envs/README.md` for setup details.

## 4. Responsibility boundary

| IsaacLab side (`env_isaaclab`)                   | FoundationPose side (`foundationpose`) |
|--------------------------------------------------|----------------------------------------|
| Start sim, load Franka, attach D435-like camera  | Load FoundationPose + object mesh      |
| Read RGB, depth, intrinsics                      | Read RGB-D + K + mask                  |
| Read `T_base_ee`; load `T_ee_camera` (hand-eye)  | Run `estimate` / `track`               |
| Save samples to disk / send via `PoseClient`     | Output `T_camera_object`               |
| Capture from the real D435 (`pyrealsense2`)      | Run `PoseServer`                       |

## 5. Directory layout

```
franka_d435_foundationpose/
├── README.md                     # this file
├── pyproject.toml                # IsaacLab-side (light) package metadata
├── configs/                      # all runtime config (yaml)
│   ├── camera_d435.yaml          # camera/intrinsics reference + sim params
│   ├── foundationpose.yaml       # repo path, weights, mock fallback, iters
│   ├── object_assets.yaml        # object_name -> mesh + default mask
│   └── hand_eye.yaml             # T_ee_camera extrinsic (PLACEHOLDER)
├── envs/                         # environment setup
│   ├── requirements_isaaclab_side.txt
│   ├── environment_foundationpose.yml
│   └── README.md
├── assets/
│   ├── meshes/  (+README)        # YOU put object CAD meshes here (meters!)
│   └── masks/   (+README)        # initial masks
├── outputs/     (+README)        # generated samples + pose results
├── scripts/                      # runnable entry points (all argparse)
│   ├── check_env.py
│   ├── run_isaaclab_franka_d435_demo.py
│   ├── run_foundationpose_on_saved_rgbd.py
│   ├── run_realsense_d435_live.py
│   ├── start_pose_server.py
│   └── visualize_pose_result.py
├── franka_d435_foundationpose/   # the python package
│   ├── camera/                   # RGBDFrame + providers (sim/realsense/saved)
│   ├── foundationpose/           # wrapper, mock, mask & mesh loaders
│   ├── server/                   # ZMQ pose server (FP side) + light client
│   ├── transforms/               # se3, hand_eye, frame_conventions
│   ├── isaaclab/                 # scene + camera attach (import in sim only)
│   └── utils/                    # image_io, depth_utils, config, logging
└── tests/                        # pytest
```

## 6. Collect a D435-like RGB-D sample in IsaacLab

```bash
cd /home1/banghai/Documents/IsaacLab
./isaaclab.sh -p franka_d435_foundationpose/scripts/run_isaaclab_franka_d435_demo.py --enable_cameras
```

`--enable_cameras` is **required** (IsaacLab disables the render products / camera
annotators otherwise). This writes:

```
franka_d435_foundationpose/outputs/rgbd_samples/sample_000001/
├── rgb.png  depth.npy  depth_vis.png
├── camera_intrinsics.json  transforms.json  metadata.json
└── (mask.png if --save_mask and segmentation is available)
```

Useful flags: `--num_samples`, `--sample_stride`, `--save_mask`, `--hand_eye <path>`.

## 7. Estimate a pose on a saved sample (FoundationPose side)

Mock (no FoundationPose needed — validates the whole flow):

```bash
cd /home1/banghai/Documents/IsaacLab
python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \
    --input_dir franka_d435_foundationpose/outputs/rgbd_samples/sample_000001 \
    --object_name target_object --mock
```

Real (isolated env, after FoundationPose is configured):

```bash
conda run -n foundationpose python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \
    --input_dir franka_d435_foundationpose/outputs/rgbd_samples/sample_000001 \
    --object_name target_object
```

Outputs `pose_result.json` (with `T_camera_object`, and `T_base_object` when
`transforms.json` is present) and `pose_overlay.png`.

## 8. Real D435 (perception only — never controls the robot)

```bash
# capture only
python franka_d435_foundationpose/scripts/run_realsense_d435_live.py \
    --output_dir franka_d435_foundationpose/outputs/rgbd_samples/live_001

# capture + pose (mesh + a rectangular initial mask) + base-frame pose
conda run -n foundationpose python franka_d435_foundationpose/scripts/run_realsense_d435_live.py \
    --output_dir .../live_001 --object_name target_object \
    --mask_bbox 280 200 360 280 \
    --hand_eye franka_d435_foundationpose/configs/hand_eye.yaml \
    --ee_pose_json current_ee_pose.json
```

Uses **color + aligned-depth-to-color + intrinsics**. `current_ee_pose.json` is
your robot's current `T_base_ee` (a 4x4 `T_base_ee`, or `position` +
`quaternion_xyzw`). ROS2 mode is documented in
`camera/realsense_d435_provider.py` (subscribe to
`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`,
`/camera/color/camera_info`).

## 9. Object meshes

Put your CAD mesh under `assets/meshes/` (e.g. `target_object.obj`) and register
it in `configs/object_assets.yaml`. **Meshes must be in METERS.** See
`assets/meshes/README.md`.

## 10. Initial mask

FoundationPose's first `estimate` needs an initial binary mask of the object
(white = object). After that, `track` propagates the pose. Sources: a file, a
bounding box, or sim segmentation (`MaskProvider`). See `assets/masks/README.md`.
**Future:** plug in SAM2 / Grounded-SAM to auto-generate the initial mask.

## 11. Coordinate convention — `T_target_source`

Every transform is named `T_<target>_<source>` and maps a point from the source
frame into the target frame:

```
p_target = T_target_source @ p_source
```

- `T_camera_object` : object → camera   `p_camera = T_camera_object @ p_object`  (FoundationPose output)
- `T_ee_camera`     : camera → ee        `p_ee     = T_ee_camera @ p_camera`      (hand-eye extrinsic)
- `T_base_ee`       : ee → base          `p_base   = T_base_ee @ p_ee`            (robot FK)

FoundationPose natively returns `ob_in_cam` = object-in-camera = `T_camera_object`,
which is exactly our convention — no inversion needed (made explicit in
`foundationpose/foundationpose_wrapper.py`). The D435 color **optical** frame is
+X right, +Y down, +Z forward.

## 12. Object pose in the base frame

```
T_base_object = T_base_ee @ T_ee_camera @ T_camera_object
```

Implemented in `transforms/frame_conventions.py::base_object_from_chain`. In the
sim demo, `T_base_ee = inv(T_world_base) @ T_world_ee` from the Franka body poses.

## 13. Hand-eye calibration matters

`configs/hand_eye.yaml` ships with **placeholder** `T_ee_camera` values. On the
real robot you **must** run a hand-eye calibration (e.g. `easy_handeye`, MoveIt
hand-eye, or an AprilTag/Charuco routine) and replace them. Without an accurate
`T_ee_camera`, `T_camera_object` may be fine but `T_base_object` will be wrong.

## 14. MockFoundationPoseEstimator — purpose and limits

When FoundationPose / weights / CUDA are unavailable (and
`use_mock_if_unavailable: true`), the wrapper falls back to a mock that returns a
deterministic pose ~0.5 m in front of the camera (nudged toward the mask
centroid). It exists **only** to validate data flow, transforms, saving, and
visualization. **Mock poses are not real estimates.** Set
`use_mock_if_unavailable: false` to require the real backend.

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `FoundationPose repo not found` | `git clone https://github.com/NVlabs/FoundationPose` to the path in `configs/foundationpose.yaml`; update `foundationpose_repo`. |
| weights missing | download into `<repo>/weights` per FoundationPose README. |
| `pyrealsense2` not available | `pip install pyrealsense2`; check `rs-enumerate-devices`. Or use a saved sample. |
| no mask | provide `mask.png` / `--mask_path` / `--mask_bbox`, or set an object default mask. The runner falls back to a centered placeholder (wrong but non-fatal). |
| depth looks ~1000× off | depth must be **meters** float32. RealSense raw is mm → multiply by `depth_scale` (~0.001). `RGBDFrame.validate()` flags this. |
| intrinsics missing | the sim/real providers always populate `K`; for saved samples ensure `camera_intrinsics.json` exists. |
| sim camera produces no image | you forgot `--enable_cameras`. |
| pose axes point the wrong way | check optical-frame convention (+Z forward) and that your mesh is in meters; verify `T_ee_camera` orientation. |
| `No module named cv2/zmq` in `env_isaaclab` | `conda run -n env_isaaclab pip install -r envs/requirements_isaaclab_side.txt`. |

## 16. Minimal runnable commands

```bash
# 0. (once) install light deps on the IsaacLab side
conda run -n env_isaaclab pip install -r franka_d435_foundationpose/envs/requirements_isaaclab_side.txt

# 1. environment check (must NOT fail if FoundationPose is missing)
cd /home1/banghai/Documents/IsaacLab
./isaaclab.sh -p franka_d435_foundationpose/scripts/check_env.py --side isaaclab

# 2. PRIMARY (default): instrument the latest SceneLayoutModule scene, then test
#    (needs GPU + Isaac Sim). Original scene is never modified.
conda run -n env_isaaclab python franka_d435_foundationpose/scripts/instrument_latest_scene_with_sensors.py \
    --load_latest_saved --camera_mode end_effector --save_to_scene_layout --headless
#    -> SceneLayoutModule/saved_scenes/<stem>_with_ee_d435.usd (+ .json + _report.md)
conda run -n env_isaaclab python franka_d435_foundationpose/scripts/run_instrumented_scene_pose_test.py \
    --enable_cameras --headless --load_latest_instrumented_scene \
    --object_name cube_0 --allow_mock_mask --mock_foundationpose
#    -> outputs/instrumented_scene_pose_tests/<timestamp>/ (sample + pose_result.json
#       + gt_pose.json + pose_validation_report.{json,md})

# 2-alt. NO-HARDWARE: fully synthetic sample (no IsaacLab, no D435)
python franka_d435_foundationpose/scripts/generate_mock_rgbd_sample.py --object_name cube
#    -> outputs/rgbd_samples/sample_mock_000001/

# 3. (optional) run mock FoundationPose standalone on any saved sample dir
python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \
    --input_dir franka_d435_foundationpose/outputs/instrumented_scene_pose_tests/<timestamp> \
    --object_name cube_0 --mock
#    -> pose_result.json (T_camera_object + T_base_object), pose_overlay.png

# 4. (after configuring FoundationPose) real estimation in the isolated env,
#    re-using the sample captured by the instrumented pose test
conda run -n foundationpose python franka_d435_foundationpose/scripts/run_foundationpose_on_saved_rgbd.py \
    --input_dir franka_d435_foundationpose/outputs/instrumented_scene_pose_tests/<timestamp> \
    --object_name cube_0
```

> The earlier self-contained synthetic-scene EE demo
> (`run_isaaclab_franka_d435_demo.py`, spawns its own cube) is still available
> for a quick sim smoke test without a saved scene.

### Real D435 (perception only)

```bash
# capture-only (no FoundationPose); fails clearly if pyrealsense2 is missing
python franka_d435_foundationpose/scripts/run_realsense_d435_live.py \
    --save_only \
    --output_dir franka_d435_foundationpose/outputs/realsense_samples/sample_000001
```
Always uses color + **aligned depth to color** + intrinsics. Never controls the robot.

### Mode 2 (optional): ZMQ inference service

```bash
# FoundationPose side
conda run -n foundationpose python franka_d435_foundationpose/scripts/start_pose_server.py
# IsaacLab side uses franka_d435_foundationpose.server.pose_client.PoseClient
# (no FoundationPose dependency) to send RGB-D and receive the pose.
```

## Tests

```bash
conda run -n env_isaaclab python -m pytest franka_d435_foundationpose/tests -q
```

## Assumptions made

- Project lives at `/home1/banghai/Documents/IsaacLab/franka_d435_foundationpose`;
  FoundationPose is cloned separately at `/home1/banghai/Documents/FoundationPose`.
- EE frame = `panda_hand`, base frame = `panda_link0`, camera optical frame =
  `d435_color_optical_frame`.
- Sim depth uses `distance_to_image_plane` (metric, along optical axis), which is
  what FoundationPose expects.
- `T_ee_camera` in `hand_eye.yaml` is a placeholder until real calibration.
- Sim camera focal length is tuned so `fx ≈ 615 px @ 640×480` to approximate a D435.
- FoundationPose's `register()`/`track_one()` return `ob_in_cam` == `T_camera_object`.
  If your FoundationPose version differs, adjust the wrapper's conversion explicitly.
