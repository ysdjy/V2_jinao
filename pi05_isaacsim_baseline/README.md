# pi0.5 / OpenPI ⨉ IsaacSim/IsaacLab Baseline

> ### ▶ Start here: [`WORKFLOW/`](WORKFLOW/)
> The single operational front door for the whole flow — **teleop → collect → store →
> train → serve → eval**. One config (`WORKFLOW/pipeline.env`), numbered stage scripts
> (`1_collect.sh` … `5_eval.sh`). For other AIs: [`WORKFLOW/README.md`](WORKFLOW/README.md)
> (flow map + where each stage is implemented). For operating it yourself:
> [`WORKFLOW/COMMANDS.md`](WORKFLOW/COMMANDS.md) (照着敲的命令手册).

A minimal-but-complete pipeline to: drive a Franka in IsaacSim, collect teleop
demonstrations, convert them to LeRobot format, fine-tune **π0.5 (OpenPI)**, serve
the policy, and run a closed-loop evaluation in IsaacLab — with safe fallbacks at
every stage (mock policy, normalized-dataset fallback, headless mode).

> **Environment isolation is the core design rule.** IsaacLab keeps using its own
> conda env (`env_isaaclab`); OpenPI/π0.5 lives in an isolated `uv` venv
> (`.venv_openpi`). They communicate only over an HTTP policy server. Nothing in
> this project modifies `env_isaaclab`, the system, the drivers, or CUDA.

---

## Directory structure
```
pi05_isaacsim_baseline/
├── README.md  STATUS.md  TODO.md  run_all_smoke_test.sh
├── logs/                      env_check, install logs, MASTER_LOG.md, eval json, reports
├── configs/                   dataset mapping, safety limits, finetune + custom-task templates
├── adapters/
│   ├── policy_server/         schemas, HTTP server, client, mock + openpi backends
│   ├── action_adapters/       delta-EE / joint adapters + safety_filter
│   ├── data_conversion/       inspect_hdf5, hdf5_to_lerobot, normalized_to_lerobot, validate
│   ├── observation_adapters/  custom_task_observation template
│   ├── evaluation/            custom_success_metric template
│   └── real_robot/            FCI / D435 / FoundationPose STUBS
├── scripts/
│   ├── isaaclab/              export_observation.py, run_policy_in_isaaclab.py, isaac_obs_utils.py
│   ├── setup_openpi_env.sh    openpi_env_info.sh  check_isaaclab.sh  make_env_check.sh
│   ├── start_mock_server.sh   stop_server.sh  serve_pi05_checkpoint.sh
│   ├── collect_demos.sh       replay_demos.sh
│   ├── test_pi05_dryrun.{sh,py}  train_pi05.sh  register_openpi_config.py
│   ├── eval_pi05_in_isaaclab.sh  analyze_dataset.py  plot_trajectory.py  export_episode_video.py
│   ├── nightly_autorun.sh     update_status.py  run_tests.sh
├── data/  raw_hdf5/ lerobot/ processed/{rollouts,images,normalized_dataset}
├── policies/checkpoints/      training outputs
├── third_party/openpi/        cloned OpenPI (+ its .venv) ; symlinked as .venv_openpi
├── docs/                      architecture, migration, training, real-robot
└── tests/                     non-Isaac unit tests
```

## Quick start

### 0. One-shot smoke test
```bash
# Without IsaacSim (fast): env check + unit tests + mock server + dry-run
bash run_all_smoke_test.sh --no-isaac

# Full (also boots IsaacSim for a 1-episode mock rollout)
bash run_all_smoke_test.sh
```

### 1. Environments
- IsaacLab: your existing `env_isaaclab` (the scripts auto-`conda activate` it).
- OpenPI: `bash scripts/setup_openpi_env.sh` → creates `.venv_openpi` via `uv`.
  Verify: `bash scripts/openpi_env_info.sh` (→ `logs/openpi_env_info.txt`).

### 2. Collect demonstrations (your teleop device)
```bash
bash scripts/collect_demos.sh \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device <your_device> --num_demos 10 --enable_cameras
# HDF5 lands in data/raw_hdf5/franka_demo_<timestamp>.hdf5
```
- **Discard a failed demo:** press `R` (reset) during recording — only successful
  episodes are exported (`EXPORT_SUCCEEDED_ONLY`).
- **Replay:** `bash scripts/replay_demos.sh --task <id> --dataset_file <file>`
- **Inspect:** `python adapters/data_conversion/inspect_hdf5.py --input <file>`

### 3. Convert to LeRobot
```bash
python adapters/data_conversion/hdf5_to_lerobot.py \
    --input data/raw_hdf5/<file>.hdf5 \
    --output data/lerobot/franka_stack_cube_pi05 \
    --config configs/dataset_mapping_isaaclab_franka.yaml \
    --task_instruction "Stack the cubes with the Franka robot."
```
Always writes a normalized intermediate dataset; builds LeRobot if `lerobot` is
available (use the OpenPI venv python for that).

### 4. Fine-tune π0.5
```bash
bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 --steps 10 --batch-size 1   # smoke
bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 --full                       # real
```

### 5. Serve + evaluate
```bash
# serve (real checkpoint, else mock fallback)
bash scripts/serve_pi05_checkpoint.sh --config pi05_isaaclab_franka --ckpt <ckpt_dir> --port 8008
# closed-loop eval in IsaacLab
bash scripts/eval_pi05_in_isaaclab.sh --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --num_rollouts 10 --max_steps 200 --config pi05_isaaclab_franka --ckpt <ckpt_dir>
```
Outputs: `logs/eval_policy_*.json` (success rate, latency, control freq, clips).

## Migrating to your own task
See **docs/migration_to_custom_task.md** and `configs/custom_task_template.yaml`.
Real Franka + D435 + FoundationPose: **docs/real_franka_d435_migration.md**.

## Common errors
| symptom | fix |
|---------|-----|
| `No module named 'isaaclab'` when running a script | the scripts `conda activate env_isaaclab`; if running manually, activate it first |
| `uv sync` git "dubious ownership" | `git config --global --add safe.directory '*'` (done during setup) |
| policy server timeout in rollout | client auto-returns safe zero action; check the server log in `logs/` |
| LeRobot build skipped | run `adapters/data_conversion/normalized_to_lerobot.py` inside `.venv_openpi` |
| checkpoint download slow/blocked | training/serving fall back to mock; see STATUS.md |

See `docs/architecture.md` for the full data-flow and schemas.
