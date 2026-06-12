# TELEOP HANDOFF — IsaacSim Franka Teleoperation Data Collection for π0.5/OpenPI Baseline

> **Audience:** the AI responsible for **teleoperation device setup + demonstration
> data collection**. This is NOT a general intro. Read it once and you should know
> exactly what is already done, what the current scene/task is, what you must NOT
> touch, where to put data, and how to hand the data back to the π0.5 pipeline.
>
> **Your scope in one sentence:** drive the user's Franka in their IsaacSim scene
> with their teleop device, record clean HDF5 demonstrations into the project's
> `data/raw_hdf5/`, inspect + replay them, and leave a status file. You do **not**
> touch OpenPI, the policy server, or training.

---

## 0. TL;DR for the teleop AI
1. Task is **`Isaac-Stack-Cube-Franka-IK-Rel-v0`** — do not change it.
2. Record with **`scripts/collect_demos.sh`** → HDF5 lands in **`data/raw_hdf5/`**.
3. Start with **3 test demos, no cameras**. Inspect → replay → only then scale up.
4. Leave **`logs/TELEOP_STATUS.md`** + inspect log + replay report.
5. Don't touch `.venv_openpi`, OpenPI, policy server, training, env_isaaclab, drivers.

---

## 1. Current project state (facts)

| item | value |
|------|-------|
| IsaacLab root | `/home1/banghai/Documents/IsaacLab` |
| Baseline project | `/home1/banghai/Documents/IsaacLab/pi05_isaacsim_baseline` |
| Current task (CONFIRMED by user in GUI) | `Isaac-Stack-Cube-Franka-IK-Rel-v0` |
| OpenPI isolated venv | `/home1/banghai/Documents/IsaacLab/pi05_isaacsim_baseline/.venv_openpi` |
| IsaacLab conda env | `env_isaaclab` (the wrapper scripts auto-`conda activate` it) |
| Raw demo output dir | `pi05_isaacsim_baseline/data/raw_hdf5/` |

What is already **done and verified** (you do NOT need to redo any of this):
- **π0.5 / OpenPI baseline is deployed** in the isolated `.venv_openpi` (jax sees both
  RTX 8000 GPUs, torch+cuda, lerobot). `env_isaaclab` was **not** polluted.
- **Policy server + client** built (`adapters/policy_server/`), HTTP on port 8008,
  backends `mock` and `openpi`.
- **IsaacLab closed-loop rollout verified** with the mock backend. Most recent result:
  - policy server healthy: **True**
  - task: `Isaac-Stack-Cube-Franka-IK-Rel-v0`
  - rollout length: **60** steps
  - avg_policy_latency_ms: **~1.5 ms**
  - server_healthy: **true**
  - success: **false** — expected, because the backend is the **mock** policy, not a
    trained policy. This number is NOT a task success metric.
- **Real π0.5 DROID checkpoint** completed a GPU dry-run (proves the OpenPI inference
  stack works here). BUT the DROID checkpoint's **action space ≠ this Stack task's
  action space**, so it cannot represent task success either. The real task policy
  must come from **fine-tuning on the user's own demonstrations** — which is exactly
  why we need YOU.

The user already opened and confirmed the scene with:
```bash
cd /home1/banghai/Documents/IsaacLab
./isaaclab.sh -p pi05_isaacsim_baseline/scripts/isaaclab/run_policy_in_isaaclab.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --num_rollouts 1 --max_steps 100
```
(That driver also needs the mock server running and accepts `--headless`; see the
project README. It is shown here only to confirm the scene is valid.)

### The one thing that's actually missing
**High-quality teleoperation demonstrations in the user's own scene.** Without demos:
- no HDF5 → LeRobot conversion can be validated,
- no π0.5 fine-tuning smoke test can run,
- no real task-success evaluation is possible.

That gap is your job.

---

## 2. Goal of THIS phase

Not tuning OpenPI. Not tuning the mock policy. The goal is for **you (the teleop AI)**
to close the data-collection link:
1. Stably teleoperate the Franka in the user's IsaacSim/IsaacLab scene.
2. Record HDF5 demos using the project-wrapped recorder (or official `record_demos.py`).
3. Ensure the recording contains what imitation learning needs: observation, action,
   episode structure, timestamps (and images if cameras are enabled).
4. Prioritize a **small number of high-quality successful** demos first.
5. Confirm each batch is replayable.
6. Hand the data to the π0.5 pipeline for HDF5 → LeRobot → training smoke test.

---

## 3. Checklist

### A. Confirm the teleop device controls the CURRENT task
Use **only** `Isaac-Stack-Cube-Franka-IK-Rel-v0`. Do not switch tasks without the
user's explicit request. Confirm:
- [ ] robot moves smoothly
- [ ] end-effector can reach near the target cube
- [ ] gripper opens/closes
- [ ] you can complete at least one valid grasp (or near-grasp)
- [ ] the sim does not crash during operation
- [ ] you did NOT modify OpenPI / policy server / training code

> This is an **IK-Rel** task: the env action is 7D
> `[dx, dy, dz, drx, dry, drz, gripper]` (position delta in meters, rotation delta as
> axis-angle in radians, gripper scalar). Keep that in mind when reasoning about the
> device mapping — but you do not implement the action mapping here; `record_demos.py`
> handles teleop→action internally.

### B. Confirm the recorder is usable
Prefer the project wrapper:
```bash
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/scripts/collect_demos.sh --help 2>/dev/null || \
  sed -n '1,40p' pi05_isaacsim_baseline/scripts/collect_demos.sh   # read its usage header
```
`collect_demos.sh` is a thin wrapper over IsaacLab's official
`scripts/tools/record_demos.py`. It forwards `--task`, `--teleop_device`,
`--num_demos`, `--dataset_file`, and any extra args (e.g. `--enable_cameras`,
`--device cuda:0`) straight through, and it **auto-`conda activate env_isaaclab`**.

Accepted flags (wrapper): `--task`, `--teleop_device`, `--num_demos`,
`--dataset_file`, plus pass-through extras. If you omit `--dataset_file`, it
auto-names `data/raw_hdf5/franka_demo_<timestamp>.hdf5`.

If the wrapper doesn't fit your device, fall back to the official script directly —
but the output HDF5 **must** still go into
`/home1/banghai/Documents/IsaacLab/pi05_isaacsim_baseline/data/raw_hdf5/`:
```bash
cd /home1/banghai/Documents/IsaacLab
source /home1/banghai/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device <real_device> \
    --num_demos 3 \
    --dataset_file pi05_isaacsim_baseline/data/raw_hdf5/franka_test.hdf5
```
> Recorder behavior to know: it exports **successful episodes only**
> (`EXPORT_SUCCEEDED_ONLY`). Press **`R`** to reset/abort the current episode. A demo
> is marked successful when the task's success term holds for ~10 consecutive steps.

### C. Record 3 TEST demos first (no cameras)
Reduce variables — start without cameras:
```bash
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/scripts/collect_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device <real_device> \
    --num_demos 3
```
`<real_device>` is filled by YOU based on the actual hardware and what the local
scripts support — e.g. `keyboard`, `spacemouse`, `gamepad`, or a custom name
registered in the env's `teleop_devices`. Use what the project/IsaacLab actually
supports; do not assume.

### D. Verify the HDF5 was created
```bash
ls -lh /home1/banghai/Documents/IsaacLab/pi05_isaacsim_baseline/data/raw_hdf5/
```
Confirm at least one `.hdf5` file exists and is non-trivial in size.

### E. Inspect HDF5 fields (save the log)
```bash
cd /home1/banghai/Documents/IsaacLab
DEMO=$(ls -t pi05_isaacsim_baseline/data/raw_hdf5/*.hdf5 | head -1)
TS=$(date +%Y%m%d_%H%M%S)
pi05_isaacsim_baseline/.venv_openpi/bin/python \
    pi05_isaacsim_baseline/adapters/data_conversion/inspect_hdf5.py \
    --input "$DEMO" \
    --json-out "pi05_isaacsim_baseline/logs/teleop_hdf5_inspect_${TS}.json" \
    | tee "pi05_isaacsim_baseline/logs/teleop_hdf5_inspect_${TS}.txt"
```
> Note: `inspect_hdf5.py` only needs `h5py`+`numpy`. `.venv_openpi/bin/python` has
> them; so does `env_isaaclab`'s python. Either works — pick one and be consistent.

Check the output for:
- [ ] `actions`  (expect shape `(T, 7)` for this IK-Rel task)
- [ ] `obs` group with robot state
- [ ] joint positions (e.g. `obs/joint_pos`)
- [ ] gripper state
- [ ] end-effector pose (e.g. `obs/eef_pos`, `obs/eef_quat`)
- [ ] episode/demo structure (`data/demo_0`, `data/demo_1`, ...)
- [ ] timestamps / per-episode metadata
- [ ] image datasets — only if you recorded with `--enable_cameras`

If field NAMES differ from the above, that's fine — just note them; the converter's
mapping (`configs/dataset_mapping_isaaclab_franka.yaml`) is candidate-based and can be
extended. Record the actual names in your status file so the π0.5 side can map them.

### F. Replay the demonstrations
```bash
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/scripts/replay_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --dataset_file "$DEMO"
```
Confirm during replay:
- [ ] Franka moves in the correct direction
- [ ] gripper actions are correct
- [ ] no crash mid-episode
- [ ] no severe jitter in the trajectory
- [ ] if the goal was grasp/stack, replay should approach/reach success
Write what you observed into `logs/teleop_replay_report_<timestamp>.md`.

### G. Scale up only after the 3 test demos pass
Staged collection:
- **Batch 1 — 10 high-quality successful demos, no/low cameras** → validates
  HDF5 → LeRobot → training smoke test.
- **Batch 2 — 20 demos WITH cameras** → validates the visual VLA data path.
- **Batch 3 — 50+ high-quality successful demos** → real π0.5 fine-tuning seed set.

Commands:
```bash
# Batch 1 (no cameras)
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/scripts/collect_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device <real_device> \
    --num_demos 10

# Batch 2 (with cameras)
bash pi05_isaacsim_baseline/scripts/collect_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device <real_device> \
    --num_demos 20 \
    --enable_cameras
```
> Tip: keep batches in separate `--dataset_file` names (e.g.
> `data/raw_hdf5/batch1_nocam_<ts>.hdf5`, `data/raw_hdf5/batch2_cam_<ts>.hdf5`) so
> test/no-cam/cam sets never get mixed.

---

## 4. Data quality requirements
**Quality > quantity.** Do not flood the set with failed/jittery/random-motion demos.
A good demonstration:
- smooth robot motion
- clear action intent
- sensible gripper open/close timing
- consistent task steps across demos
- a few failed demos may be kept **for analysis**, but Batch 1 training data should be
  successful demos only
- if an episode fails, it must be distinguishable (filename / metadata / a note)
- never mix test/random-motion demos into the formal training set
- replay-confirm every batch after collecting it

---

## 5. Do NOT do
1. Do not reinstall OpenPI.
2. Do not modify `.venv_openpi`.
3. Do not modify `env_isaaclab` core dependencies.
4. Do not upgrade NVIDIA driver / CUDA / IsaacSim.
5. Do not switch task unless the user explicitly asks.
6. Do not modify the policy server.
7. Do not modify the π0.5 training scripts.
8. Do not save data to scattered paths outside the project — only `data/raw_hdf5/`.
9. Do not mix failed and successful demos without marking them.
10. Do not disable safety or inflate action magnitude just to collect data faster.

---

## 6. Deliverables to leave for the π0.5 pipeline
1. HDF5 files → `pi05_isaacsim_baseline/data/raw_hdf5/*.hdf5`
2. Inspect logs → `pi05_isaacsim_baseline/logs/teleop_hdf5_inspect_*.txt` (+ `.json`)
3. Replay report → `pi05_isaacsim_baseline/logs/teleop_replay_report_*.md`
4. Status file → `pi05_isaacsim_baseline/logs/TELEOP_STATUS.md`

`TELEOP_STATUS.md` must contain:
- task name used
- teleop device used
- number of demos collected
- number of successful demos
- number of failed demos
- whether cameras were enabled
- HDF5 file path(s)
- whether replay was done
- whether replay looked correct
- problems found (incl. actual HDF5 field names if they differ from §3.E)
- next-step recommendation

Suggested `TELEOP_STATUS.md` skeleton:
```markdown
# TELEOP STATUS
- task: Isaac-Stack-Cube-Franka-IK-Rel-v0
- teleop_device: <real_device>
- demos_collected: N
- demos_successful: N
- demos_failed: N
- cameras_enabled: yes/no
- hdf5_files:
  - data/raw_hdf5/<file>.hdf5
- replay_done: yes/no
- replay_ok: yes/no
- hdf5_fields_observed: { actions: (T,7), obs/joint_pos: ..., obs/eef_pos: ..., ... }
- problems:
  - ...
- next_step: hand off to π0.5 pipeline (see §7)
```

---

## 7. Next phase commands (for the π0.5 pipeline, after collection)
Once demos exist and replay is OK, the π0.5 side runs:
```bash
cd /home1/banghai/Documents/IsaacLab
DEMO=$(ls -t pi05_isaacsim_baseline/data/raw_hdf5/*.hdf5 | head -1)

# 1. HDF5 -> (normalized intermediate + LeRobot) dataset
pi05_isaacsim_baseline/.venv_openpi/bin/python \
    pi05_isaacsim_baseline/adapters/data_conversion/hdf5_to_lerobot.py \
    --input "$DEMO" \
    --output pi05_isaacsim_baseline/data/lerobot/franka_stack_cube_pi05 \
    --config pi05_isaacsim_baseline/configs/dataset_mapping_isaaclab_franka.yaml \
    --task_instruction "Stack the cubes with the Franka robot."

# 2. π0.5 fine-tuning smoke test (10 steps)
#    NOTE: train_pi05.sh takes --repo-id (the dataset dir name under data/lerobot/),
#    NOT --dataset. The repo-id must equal the --output basename above.
bash pi05_isaacsim_baseline/scripts/train_pi05.sh \
    --repo-id franka_stack_cube_pi05 \
    --steps 10 --batch-size 1
```
> ⚠️ Interface note: an earlier draft showed `train_pi05.sh --dataset <path> --steps 10`.
> The **actual** flag is `--repo-id <name>` (optionally `--lerobot-root <dir>`), not
> `--dataset`. Use the command above. `hdf5_to_lerobot.py` always writes a normalized
> intermediate even if the LeRobot build step is skipped, so the data is never lost.

That's the entire teleop handoff. Collect clean demos, verify them, leave the status
file, and the π0.5 pipeline takes it from §7.
