# WORKFLOW — single front door for the full sim-data → π0.5 pipeline

> **For another AI / engineer:** this folder is the *operational entry point* for the
> whole flow — **teleop control → data collection → storage → training → inference →
> evaluation**. Run everything from here; one config file, numbered stage scripts.
> The heavy implementations live elsewhere in the project (mapped below) and are NOT
> duplicated here — these scripts are thin wrappers so paths/conventions stay correct.
>
> Task is fixed: **`Isaac-Stack-Cube-Franka-IK-Rel-v0`**, 7-D action
> `[dx,dy,dz,drx,dry,drz,gripper]`. Don't switch tasks without the user.

---

## The flow (5 stages)

```
            pipeline.env  (one config for everything)
                 │
 ┌──────────┐ 1 ┌──────────┐ 2 ┌──────────┐ 3 ┌──────────┐ 4 ┌──────────┐ 5 ┌──────────┐
 │  TELEOP  │──▶│ COLLECT  │──▶│  STORE   │──▶│  TRAIN   │──▶│  SERVE   │──▶│   EVAL   │
 │  device  │   │  HDF5    │   │ LeRobot  │   │  π0.5    │   │ policy   │   │ closed   │
 │ keyboard │   │          │   │ dataset  │   │  LoRA    │   │ server   │   │ loop sim │
 └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
   human/GELLO   1_collect.sh   2_convert.sh   3_train.sh    4_serve.sh    5_eval.sh
                 inspect.sh                                  (:8008)
```

Run in order:
```bash
bash WORKFLOW/1_collect.sh     # teleop → data/raw_hdf5/<name>.hdf5     (human drives)
bash WORKFLOW/inspect.sh       # check the HDF5 fields
bash WORKFLOW/2_convert.sh     # HDF5 → data/lerobot/$REPO_ID
bash WORKFLOW/3_train.sh       # π0.5 LoRA fine-tune → policies/checkpoints/...
bash WORKFLOW/4_serve.sh       # policy server on :8008  (newest ckpt, else mock)
bash WORKFLOW/5_eval.sh        # closed-loop rollout in IsaacLab → logs/eval_policy_*.json
bash WORKFLOW/stop.sh          # stop the server
```
Everything is parameterised in **`pipeline.env`** (task, teleop device, num demos, repo-id,
train steps, port, …). Edit there, not in the scripts.

---

## Stage → real implementation map (where to LOCATE things)

| Stage | WORKFLOW wrapper | Real implementation | Notes |
|------|------------------|---------------------|-------|
| 1 Teleop+collect | `1_collect.sh` | `../scripts/collect_demos.sh` → IsaacLab `scripts/tools/record_demos.py` | EXPORT_SUCCEEDED_ONLY; `R`=abort |
| — Inspect | `inspect.sh` | `../adapters/data_conversion/inspect_hdf5.py` | dumps fields (+ `--json-out`) |
| 2 Store/convert | `2_convert.sh` | `../adapters/data_conversion/hdf5_to_lerobot.py` (+ `validate_lerobot_dataset.py`) | mapping: `../configs/dataset_mapping_isaaclab_franka.yaml` |
| 3 Train | `3_train.sh` | `../scripts/train_pi05.sh` (+ `register_openpi_config.py`) | pi0.5 **LoRA** config |
| 4 Serve | `4_serve.sh` | `../scripts/serve_pi05_checkpoint.sh` / `start_mock_server.sh` | HTTP `:8008`, `/health` `/infer` |
| 5 Eval | `5_eval.sh` | `../scripts/eval_pi05_in_isaaclab.sh` → `../scripts/isaaclab/run_policy_in_isaaclab.py` | obs/action/safety adapters |
| stop | `stop.sh` | `../scripts/stop_server.sh` | |

Deeper architecture, schemas, migration: `../docs/architecture.md`,
`../docs/migration_to_custom_task.md`, `../docs/pi05_training.md`. Teleop specifics +
verified status: `../TELEOP_PIPELINE_PLAYBOOK.md`, `../logs/TELEOP_STATUS.md`.

---

## Data storage (where everything lands)

| What | Location |
|------|----------|
| Raw teleop demos (HDF5) | `../data/raw_hdf5/*.hdf5` |
| Normalized intermediate | `../data/processed/normalized_dataset/<repo_id>/` |
| LeRobot training dataset | `../data/lerobot/<repo_id>/` |
| Trained checkpoints | `../policies/checkpoints/pi05_isaaclab_<ts>/<config>/<exp>/<step>/` |
| Rollout trajectories | `../data/processed/rollouts/*.jsonl` |
| Logs / reports / eval json | `../logs/` |

Keep **real** and **synthetic/throwaway** datasets in different `repo_id`s — never mix.

---

## Environments (two, isolated — do not cross them)

| Env | Used for | How |
|-----|----------|-----|
| `env_isaaclab` (conda) | sim: collect / serve-mock / eval | the wrappers `conda activate` it for `isaaclab.sh` |
| `.venv_openpi` (uv) | OpenPI: convert / train / real-policy serve | `../.venv_openpi/bin/python` |

They talk over the HTTP policy server (`:8008`) only. Never install OpenPI into
`env_isaaclab`, and never edit OpenPI internals / `.venv_openpi`.

---

## Teleop device (the standard sim-data device)

- `TELEOP_DEVICE` in `pipeline.env`. Works today: **keyboard** (no hardware),
  `spacemouse`, `gamepad`.
- **GELLO leader arm** (planned standard): GELLO outputs *joint angles* but this task wants
  *EE deltas*, so it needs a custom `Se3Gello` device (joint → Franka FK → EE-pose delta →
  7-D action) registered in the env's `teleop_devices`. Until that exists, use keyboard.
  See `../TELEOP_PIPELINE_PLAYBOOK.md` §3b.

---

## Important behaviours / gotchas

- **Cameras:** set `ENABLE_CAMERAS=1` (default) to record front/wrist RGB — needed for a
  visual VLA. If you collect *without* cameras (state-only), set `SYNTH_IMAGES=224` in
  `pipeline.env` so stage 2 injects zero images to match the pi0.5 input layout (plumbing only).
- **Training memory:** the registered config is **pi0.5 LoRA** so it fits one 48 GB GPU at
  batch=1. *Full* fine-tune needs ~50 GB and OOMs on 48 GB even sharded — use more GPU memory
  and `--fsdp` with an even batch for that.
- **repo-id must match:** stage 2 output `data/lerobot/<REPO_ID>` and stage 3 `--repo-id` must
  be the same name. `3_train.sh` re-registers the OpenPI config with the current repo-id each run.
- **Replay** (`../scripts/replay_demos.sh`) is GUI-only (needs a display).
- Status of the last validated run: `../STATUS.md` ("Synthetic Plumbing Validation Summary").
