# Migrating the pipeline to your own IsaacSim task

This pipeline was built task-agnostic. To move from `Isaac-Stack-Cube-Franka-IK-Rel-v0`
to your own task (tabletop grasp / place / insertion / assembly), you touch a small
number of well-marked places.

## 1. What observations to implement
Start from `adapters/observation_adapters/custom_task_observation.py`. Fill:
- `robot_key` — your scene articulation key (usually `"robot"`).
- `ee_body` — the end-effector body name (`panda_hand` for Franka).
- `camera_map` — `{logical_name: scene_sensor_key}` for `front_rgb` / `wrist_rgb`.
- `object_keys` — scene rigid-object keys to expose as `observation.objects[]`
  (or fill these later from FoundationPose; see §8).

The default `scripts/isaaclab/isaac_obs_utils.build_observation` already reads
joint state + EE pose generically, so for many tasks you only adjust camera/object
mapping.

## 2. What actions to implement
Pick the env action space and set `env_kind`:
| your env id contains | env_kind | action layout |
|----------------------|----------|---------------|
| `IK-Rel` | `ik_rel` | `[dx,dy,dz, rx,ry,rz, grip]` (7D) |
| `IK-Abs` | `ik_abs` | `[x,y,z, qw,qx,qy,qz, grip]` (8D, delta integrated) |
| `Joint`  | `joint`  | `[q1..q7, grip]` (8D) |

`detect_env_kind()` infers this from the task name; override with `--env_kind`.
If the policy emits delta-EE but your env needs joints, add an IK step on the
IsaacLab side using the env Jacobian (note in `joint_action_adapter.py`).

## 3. task_instruction
One natural-language sentence describing the goal, e.g.
`"Insert the peg into the hole."` It is sent in every Observation and used as the
VLA prompt (and stored per-frame in the LeRobot dataset).

## 4. Collect demos
```bash
bash scripts/collect_demos.sh --task <YourTaskId> --teleop_device <device> \
    --num_demos 20 --enable_cameras
```
Use `configs/custom_task_template.yaml` to record the task's metadata.

## 5. Convert dataset
First inspect the new HDF5 (field names may differ):
```bash
python adapters/data_conversion/inspect_hdf5.py --input data/raw_hdf5/<file>.hdf5
```
Update the candidate keys in `configs/dataset_mapping_isaaclab_franka.yaml` (or copy
it per-task), then run `hdf5_to_lerobot.py`. The converter reports missing fields
instead of crashing.

## 6. Fine-tune
```bash
bash scripts/train_pi05.sh --repo-id <your_repo_id> --steps 10 --batch-size 1   # smoke
bash scripts/train_pi05.sh --repo-id <your_repo_id> --full                       # real
```
`register_openpi_config.py` registers a `pi05_<name>` config pointing at your
dataset (reuses the libero-style data transform, which matches our converter).

## 7. Evaluate & success metric
- Prefer the env's own `terminations.success` term (the rollout runner reads
  `terminated`).
- For a custom geometric metric, edit `adapters/evaluation/custom_success_metric.py`
  (`check_success`, `final_ee_error`) and wire it into the rollout loop.
- Eval JSON already reports success rate, episode length, latency, control freq,
  timeouts, policy errors, safety clips.

## 8. Add FoundationPose 6D pose to the observation
`observation.objects[]` is already in the schema. To populate it:
- In sim: read object root pose from the scene (see `custom_task_observation.py`).
- With FoundationPose (sim or real): use
  `adapters/real_robot/foundationpose_object_stub.make_object_entry(...)` and append
  to `observation['objects']`. Keep `name` stable for tracking; include
  `pose_in_camera`, `mesh_path`, `mask_path` as available.
The VLA can then condition on object pose if your training data includes it.

## 9. Avoiding sim→real action-space mismatch
- Train and deploy in the **same** action space (`env_kind`). If you collect demos
  in `ik_rel`, serve and evaluate in `ik_rel`.
- Keep units identical (m / rad / XYZW / gripper [-1,1]) across sim, dataset, and
  real (the stubs reuse the same `safety_filter`).
- Match camera resolution and logical names (`front_rgb`/`wrist_rgb`) between sim
  and real so the model sees the same input layout.
- Re-fit normalization stats per dataset (`compute_norm_stats.py`, run by
  `train_pi05.sh`).

## Checklist
- [ ] task id registered & runs headless
- [ ] observation adapter fills robot/cameras/objects
- [ ] env_kind + action adapter correct
- [ ] demos collected, HDF5 inspected, mapping updated
- [ ] dataset converted + validated
- [ ] norm stats computed, smoke train passes
- [ ] eval runs, success metric defined
