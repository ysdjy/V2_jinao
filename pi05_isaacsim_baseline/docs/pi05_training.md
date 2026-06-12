# Fine-tuning π0.5 on IsaacLab data

## Prereqs
- OpenPI venv installed (`scripts/setup_openpi_env.sh`; verify with `openpi_env_info.sh`).
- A LeRobot dataset built from your demos (`hdf5_to_lerobot.py`), default repo_id
  `franka_stack_cube_pi05` under `data/lerobot/`.

## How config registration works
OpenPI's training configs live in a python list `_CONFIGS` in
`third_party/openpi/src/openpi/training/config.py`. There is no plugin hook, so
`scripts/register_openpi_config.py` appends a `pi05_isaaclab_franka` `TrainConfig`
(reversibly, with a `.bak_pi05` backup and a marker block). It reuses
`LeRobotLiberoDataConfig` because our converter emits the same feature layout
(`image`, `wrist_image`, `state`, `actions`, `task`).

`train_pi05.sh` runs registration automatically. To remove it later:
```bash
.venv_openpi/bin/python scripts/register_openpi_config.py --unregister
```

## Smoke test (link validation, ~minutes)
```bash
bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 --steps 10 --batch-size 1
```
Success criteria: dataset loads, weights load from `pi05_base`, forward+backward
run, a checkpoint is saved, no shape mismatch.

## Real fine-tune
```bash
bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 --full   # steps=30000, batch=8
# or explicit:
bash scripts/train_pi05.sh --repo-id <id> --steps 30000 --batch-size 16
```

## Memory controls (2× RTX 8000, 48 GB each)
The smoke defaults are conservative. If you hit OOM:
- lower `--batch-size` (smoke uses 1)
- π0.5/OpenPI trains in **bf16** by default
- shard across both GPUs by passing `--fsdp-devices 2` (forwarded via EXTRA args)
- reduce `action_horizon` in the registered config (`register_openpi_config.py`)
- LoRA variant: see `pi0_fast_libero_low_mem_finetune` in OpenPI for the pattern
  (set `paligemma_variant="gemma_2b_lora"` + matching `freeze_filter`, `ema_decay=None`)

The script logs the final working config to `logs/train_pi05_<ts>.log`.

## Dataset resolution
LeRobot resolves datasets from `$HF_LEROBOT_HOME/<repo_id>`. `train_pi05.sh` sets
`HF_LEROBOT_HOME` to `data/lerobot` (or `--lerobot-root`). Ensure the dataset dir
name equals the `repo_id`.

## Norm stats
`compute_norm_stats.py <config>` runs before training (writes per-dataset
normalization). If it fails, the training step still runs and surfaces the real
error; re-run norm stats once the dataset is valid.

## Outputs
- checkpoints: `policies/checkpoints/pi05_isaaclab_<ts>/...`
- logs: `logs/train_pi05_<ts>.log`
Serve a checkpoint with `scripts/serve_pi05_checkpoint.sh --config pi05_isaaclab_franka --ckpt <dir>`.

## Base weights
`pi05_base` is loaded from `gs://openpi-assets/checkpoints/pi05_base/params`
(downloaded on first use). If download is blocked, training cannot initialize from
base — collect demos and retry when network allows; the mock pipeline remains
usable for everything except real weights.
