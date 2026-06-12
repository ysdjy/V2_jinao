# Sim → Real: Franka + RealSense D435 + FoundationPose

> Status: **interfaces only** (`adapters/real_robot/*_stub.py`). No hardware is
> driven yet. This doc is the migration plan and the safety contract.

## 1. Aligning sim and real observations
The unified Observation schema is the contract. Real must produce the *same*
fields the model was trained on:
| field | sim source | real source |
|-------|-----------|-------------|
| `robot.joint_positions/velocities` | articulation data | libfranka robot state |
| `robot.ee_position` / `ee_quat` | hand body pose | FK from joint state / FCI O_T_EE |
| `robot.gripper_width` | finger joints | Franka gripper width |
| `images.front_rgb` / `wrist_rgb` | sim cameras | D435 color (aligned) |
| `objects[]` | scene GT | FoundationPose 6D |
Keep **logical camera names, resolution, units, quaternion order** identical.

## 2. D435 replacing the IsaacSim camera
`d435_observation_stub.py`:
- `pyrealsense2` pipeline (color + depth), align depth→color.
- Resize color to the training resolution (match `configs ... image_resize`, e.g.
  256×256) and the same logical keys (`front_rgb`, `wrist_rgb`).
- Use `image_mode=path` (write png, pass path) — never base64 at control rate.

## 3. FoundationPose 6D pose into the observation
`foundationpose_object_stub.py`:
- Inputs: RGB-D + object mesh + initial mask.
- Output via `make_object_entry(name, position_world, quat_world_xyzw, confidence,
  pose_in_camera, mesh_path, mask_path)` → append to `observation['objects']`.
- Apply camera→world extrinsics (hand-eye calibration) to get world pose.
- Train with the same `objects[]` content you will have at deploy time, or the
  model won't learn to use it.

## 4. Policy action → Franka FCI
`franka_fci_action_stub.py`:
- Reuses the **same** `safety_filter` as sim (identical clamps).
- `delta_ee_pose` → Cartesian impedance / motion-generator target via libfranka or
  `franka_ros2`. Requires the FCI setup (see project memory: FR3 FCI setup).
- Run the control loop in a **dedicated real-time process**, not the training env.

## 5. Safety requirements (real)
- Hard Cartesian/joint/velocity limits + collision thresholds in the FCI layer
  (the python safety_filter is a *second* line, not the only one).
- Watchdog: policy timeout → hold position (the client already returns zero-delta).
- Workspace box enforced (`configs/safety_limits.yaml`); start with a tight box.
- E-stop reachable; low speed scaling for first runs; human supervision.
- Reset on N consecutive policy errors (already implemented).

## 6. Why NOT mix real control and training in one conda env
- The real-time control stack (libfranka/franka_ros2, RT kernel, low-latency loop)
  has different, often conflicting, dependencies vs. JAX/torch training.
- A training process can stall the GIL / saturate the GPU and **miss FCI deadlines**
  → the robot faults or behaves unsafely.
- Different lifecycles: training is batch/offline; control is hard-real-time.
- Isolation mirrors the sim design: IsaacLab ⇄ HTTP ⇄ OpenPI. For real, run the
  policy server on the training box and a thin RT client on the control box; they
  exchange Observation/Action over the network, exactly like sim.

## Migration checklist
- [ ] hand-eye calibration (camera↔robot)
- [ ] D435 stream → observation (matching keys/resolution)
- [ ] FoundationPose → `objects[]`
- [ ] FK/EE pose from FCI matches sim convention (XYZW, robot frame)
- [ ] safety_filter limits reviewed for hardware
- [ ] FCI hard limits + watchdog + E-stop verified
- [ ] separate RT control process / env
- [ ] low-speed supervised dry run before autonomy
