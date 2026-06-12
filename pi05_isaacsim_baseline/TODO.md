# TODO — by priority

## P1 — Collect real demonstrations (only you can; needs teleop device)
- [ ] `bash scripts/collect_demos.sh --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device <your_device> --num_demos 10 --enable_cameras`
- [ ] Verify: `ls -lh data/raw_hdf5/` and `python adapters/data_conversion/inspect_hdf5.py --input data/raw_hdf5/<file>.hdf5`
- [ ] Discard bad demos by pressing `R` during recording (only successes are exported).

## P2 — Convert to LeRobot
- [ ] `.venv_openpi/bin/python adapters/data_conversion/hdf5_to_lerobot.py --input data/raw_hdf5/<file>.hdf5 --output data/lerobot/franka_stack_cube_pi05 --config configs/dataset_mapping_isaaclab_franka.yaml --task_instruction "Stack the cubes with the Franka robot."`
- [ ] If field auto-mapping reports missing keys, edit `configs/dataset_mapping_isaaclab_franka.yaml` candidates to match `inspect_hdf5.py` output, or pass `--state-key/--action-key`.
- [ ] `python adapters/data_conversion/validate_lerobot_dataset.py --input data/processed/normalized_dataset/franka_stack_cube_pi05`
- [ ] `python scripts/analyze_dataset.py --input data/processed/normalized_dataset/franka_stack_cube_pi05`  (→ `logs/dataset_quality_report.md`)

## P3 — Finish real π0.5 dry-run
- [ ] `cat logs/pi05_dryrun_real.log` — if not `backend_used: openpi`, re-run:
      `.venv_openpi/bin/python scripts/test_pi05_dryrun.py --backend openpi --config pi05_droid --ckpt gs://openpi-assets/checkpoints/pi05_droid`
- [ ] (optional) install gsutil for faster downloads, or pre-fetch `pi05_base` for training.

## P4 — Fine-tune π0.5 (after a dataset exists)
- [ ] Smoke: `bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 --steps 10 --batch-size 1`
- [ ] Real: `bash scripts/train_pi05.sh --repo-id franka_stack_cube_pi05 --full`
- [ ] If OOM: drop batch size / add `--fsdp-devices 2`; see `docs/pi05_training.md`.

## P5 — Serve + evaluate the trained checkpoint
- [ ] `bash scripts/serve_pi05_checkpoint.sh --config pi05_isaaclab_franka --ckpt <ckpt_dir> --port 8008`
- [ ] `bash scripts/eval_pi05_in_isaaclab.sh --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_rollouts 10 --max_steps 200 --config pi05_isaaclab_franka --ckpt <ckpt_dir>`
- [ ] Read `logs/eval_policy_*.json` (success rate, latency, freq, clips).

## P6 — Visualization / quality
- [ ] `python scripts/plot_trajectory.py --rollout data/processed/rollouts/<file>.jsonl`
- [ ] `python scripts/export_episode_video.py --dataset data/processed/normalized_dataset/<name> --camera front_rgb --episode 0`
- [ ] (optional) `sudo apt-get install -y ffmpeg` for mp4 export (or rely on imageio).

## P7 — Migrate to your own task
- [ ] Copy `configs/custom_task_template.yaml`; fill task id / cameras / objects / success.
- [ ] Adapt `adapters/observation_adapters/custom_task_observation.py`.
- [ ] Follow `docs/migration_to_custom_task.md`.

## P8 — Real Franka + D435 + FoundationPose (later)
- [ ] Implement `adapters/real_robot/*_stub.py` per `docs/real_franka_d435_migration.md`.
- [ ] Keep real-time control in a SEPARATE process/env from training.

## Housekeeping / known gaps
- [ ] ffmpeg not installed (video export uses imageio fallback).
- [ ] `register_openpi_config.py` edits openpi's config.py (reversible, backup at
      `config.py.bak_pi05`; `--unregister` to remove).
- [ ] Camera capture path in `isaac_obs_utils._capture_images` assumes sensor keys
      `table_cam`/`wrist_cam`; adjust for your scene + run with `--enable_cameras`.
