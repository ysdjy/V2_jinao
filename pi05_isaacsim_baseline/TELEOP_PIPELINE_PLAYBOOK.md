# TELEOP → TRAINING PIPELINE PLAYBOOK

> **Audience:** the AI (or engineer) that will run the **data-collection → π0.5/OpenPI
> training** flow for this project. Read this one file and you know: what the module is,
> how to validate the whole pipeline *autonomously* (no human, no GUI), how a human
> collects *real* demos, where everything lives, and what is already verified.
>
> **Task is fixed:** `Isaac-Stack-Cube-Franka-IK-Rel-v0` (7-D action `[dx,dy,dz,drx,dry,drz,gripper]`).
> **Do not** modify OpenPI internals, the policy server, or `.venv_openpi`. You *run*
> the training scripts; you don't rewrite them.

---

## 0. The one reality you must know

**An AI agent cannot perform real-time keyboard teleoperation in the GUI.** Completing a
pick-and-stack by watching the viewport and pressing keys is a *human* sensorimotor task.
So the demonstration *actions* come from one of two sources:

| source | who/how | task-successful? | use |
|--------|---------|------------------|-----|
| **SYNTHETIC** | `gen_synthetic_demos.sh` — scripted actions, headless, fully autonomous | ❌ no | **validate the collect→convert→train *plumbing*** (this is what an AI can do alone) |
| **REAL** | human at the keyboard (`collect_demos.sh`), or GELLO arm later | ✅ yes | the actual training data |

> Synthetic demos are **throwaway**: correct *format*, not correct *behavior*. Use them to
> prove the pipeline runs end-to-end, then collect real demos for a model that actually works.

---

## 1. Pipeline at a glance

```
  [data source]                                      this playbook validated ──┐
  synthetic (auto)  ┐                                                          │
  human keyboard    ├─► record HDF5 ─► inspect ─► (replay, GUI) ─► HDF5→LeRobot ─► π0.5 train ─► serve ─► eval
  GELLO (later)     ┘   data/raw_hdf5/   logs/      GUI-only        data/lerobot/   smoke test   :8008   IsaacLab
```

All HDF5 lands in `data/raw_hdf5/`. All LeRobot datasets land in `data/lerobot/<repo_id>/`.

---

## 2. AUTONOMOUS plumbing validation (an AI can run this start-to-finish, headless)

This is the "打通流程" path: prove `collect → HDF5 → LeRobot → train smoke` works with zero
human input, using synthetic data.

```bash
cd /home1/banghai/Documents/IsaacLab

# 1) Generate 3 synthetic demos (headless, scripted actions, ~1 min after sim loads)
bash pi05_isaacsim_baseline/scripts/gen_synthetic_demos.sh \
    --num_demos 3 --episode_len 80 \
    --dataset_file data/raw_hdf5/synthetic_plumbing.hdf5 --headless

# 2) Inspect the HDF5 fields (expect actions (T,7), obs/eef_pos, obs/eef_quat, obs/joint_pos, obs/gripper_pos)
DEMO=pi05_isaacsim_baseline/data/raw_hdf5/synthetic_plumbing.hdf5
TS=$(date +%Y%m%d_%H%M%S)
pi05_isaacsim_baseline/.venv_openpi/bin/python \
    pi05_isaacsim_baseline/adapters/data_conversion/inspect_hdf5.py \
    --input "$DEMO" --json-out pi05_isaacsim_baseline/logs/teleop_hdf5_inspect_${TS}.json \
    | tee pi05_isaacsim_baseline/logs/teleop_hdf5_inspect_${TS}.txt

# 3) Convert HDF5 -> normalized + LeRobot dataset
pi05_isaacsim_baseline/.venv_openpi/bin/python \
    pi05_isaacsim_baseline/adapters/data_conversion/hdf5_to_lerobot.py \
    --input "$DEMO" \
    --output pi05_isaacsim_baseline/data/lerobot/synthetic_plumbing_check \
    --config pi05_isaacsim_baseline/configs/dataset_mapping_isaaclab_franka.yaml \
    --task_instruction "Stack the cubes with the Franka robot." \
    --report-out pi05_isaacsim_baseline/logs/synthetic_convert_report_${TS}.json

# 4) π0.5 fine-tune SMOKE TEST (10 steps). repo-id == the data/lerobot/<dir> basename.
bash pi05_isaacsim_baseline/scripts/train_pi05.sh \
    --repo-id synthetic_plumbing_check --steps 10 --batch-size 1
```

Steps 1–3 are **verified working** (see §6). Step 4 is the training side — run it to close the
loop. If step 4 passes on synthetic data, the plumbing is proven and you only need *real* demos
to get a working policy.

> `--episode_len`/`--num_demos`/`--action_mode {scripted,random}` are tunable. Synthetic demos
> are deliberately small-magnitude (`--max_delta 0.02`) — do not inflate them.

---

## 3. REAL demo collection (human-driven; needed for an actually-working policy)

### 3a. Keyboard (works now, zero hardware)
```bash
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/scripts/collect_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device keyboard --num_demos 3 \
    --dataset_file data/raw_hdf5/real_keyboard_test.hdf5
```
A **human** drives. Click the 3D viewport first so it has focus. Controls:

| move X | move Y | move Z | rot X | rot Y | rot Z | gripper | abort episode |
|--------|--------|--------|-------|-------|-------|---------|---------------|
| W / S  | A / D  | Q / E  | Z / X | T / G | C / V | K (toggle) | R |

Recording starts immediately; only **successful** episodes are saved (success term held ~10
steps); it exits after `--num_demos` successes. Then inspect (step 2 above) and **replay in the
GUI** (§4) to confirm direction/gripper look right.

### 3b. GELLO leader arm (later, for natural high-quality demos)
GELLO outputs *joint angles*; this task wants *EE deltas*. So GELLO needs a custom
`Se3Gello` device (joint → Franka FK → EE-pose delta → 7-D action) registered in the env's
`teleop_devices`. **Deferred** until the pipeline is validated. See [[gello-franka-teleop]]
for the existing GELLO reader to reuse.

---

## 4. Replay (GUI only — visual sanity check)

`replay_demos.py` needs the GUI (`omni.appwindow`); it **cannot run headless**. With a display:
```bash
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/scripts/replay_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --dataset_file data/raw_hdf5/<file>.hdf5
```
Confirm: correct direction, sensible gripper, no crash/jitter. (For *synthetic* data this just
shows the scripted motion — it won't stack anything; that's expected.)

---

## 5. File / script map

| path | what |
|------|------|
| `scripts/gen_synthetic_demos.sh` → `scripts/isaaclab/generate_synthetic_demos.py` | **NEW** autonomous synthetic demo generator (headless, EXPORT_ALL) |
| `scripts/collect_demos.sh` | human teleop recorder (wraps IsaacLab `record_demos.py`, EXPORT_SUCCEEDED_ONLY) |
| `scripts/replay_demos.sh` | GUI replay |
| `adapters/data_conversion/inspect_hdf5.py` | dump HDF5 fields (+ `--json-out`) |
| `adapters/data_conversion/hdf5_to_lerobot.py` | HDF5 → normalized + LeRobot (`--report-out` for a JSON summary) |
| `scripts/train_pi05.sh` | π0.5 fine-tune (`--repo-id <dir under data/lerobot>`, `--steps`, `--batch-size`, `--full`) |
| `scripts/serve_pi05_checkpoint.sh` / `scripts/start_mock_server.sh` | HTTP policy server (:8008) |
| `scripts/eval_pi05_in_isaaclab.sh` / `scripts/isaaclab/run_policy_in_isaaclab.py` | closed-loop eval |
| `data/raw_hdf5/` | all HDF5 demos | `data/lerobot/<repo_id>/` | LeRobot datasets | `logs/` | all logs/reports |
| `configs/dataset_mapping_isaaclab_franka.yaml` | HDF5-field → policy-field mapping (candidate-based, extensible) |

---

## 6. Verified status (2026-06-12)

- ✅ **Synthetic generator** builds a 654 KB HDF5 with 3 episodes, `actions (T,7)`, full
  `obs/{eef_pos,eef_quat,joint_pos,gripper_pos,cube_*,object}` + `states/` + `initial_state/` —
  **structurally identical to real `record_demos.py` output**.
- ✅ **inspect_hdf5.py** parses it; all expected fields present.
- ✅ **hdf5_to_lerobot.py** converts it: mappings all `found`, `missing: []`, 3 episodes /
  240 frames, `lerobot_built: true`, parquet + meta written.
- ✅ **train smoke (step 4)** — PASSED (2026-06-12). π0.5 **LoRA** finetune, single GPU, batch=1,
  10 steps, `train exit: 0`, losses ~0.44/0.24/0.03/0.31, checkpoint saved (8.8 GB) at
  `policies/checkpoints/pi05_isaaclab_20260612_133228/.../9`. Notes: converter run with
  `--synthesize-images 224` (state-only → zero images for the pi0.5/libero layout); the registered
  config is **LoRA low-mem** because full pi0.5 finetune OOMs even sharded on 2×48 GB. See STATUS.md
  "Synthetic Plumbing Validation Summary" for the exact fixes.
- ⚠️ `replay_demos.py` is GUI-only (headless `omni.appwindow` error) — expected tool limitation.
- ℹ️ `data/raw_hdf5/test3_keyboard.hdf5` (96 B) is an **empty** leftover from a human keyboard
  attempt that recorded 0 successful demos — ignore/delete it; it is not a real dataset.

Logs: `logs/gen_synthetic_*.log`, `logs/teleop_hdf5_inspect_*.{txt,json}`,
`logs/synthetic_convert_*.log` + `logs/synthetic_convert_report_*.json`, `logs/TELEOP_STATUS.md`.

---

## 7. Do / Don't

**Do:** run the scripts above; collect real demos when ready; keep synthetic and real data in
separate, clearly-named files; replay-confirm real batches in the GUI.

**Don't:** train on synthetic data for a real policy; modify OpenPI / `.venv_openpi` / the policy
server / `record_demos.py` / `replay_demos.py` (IsaacLab core); mix synthetic into a real training
set; inflate action magnitude.
