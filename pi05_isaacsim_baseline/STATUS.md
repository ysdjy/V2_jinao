<!-- AUTO-SUMMARY -->
## Wake-up Summary
_Last updated by update_status.py_

* Project path: `/home1/banghai/Documents/IsaacLab/pi05_isaacsim_baseline`
* IsaacLab root: `/home1/banghai/Documents/IsaacLab`
* OpenPI installed: yes
* pi0.5 real model loaded (dry-run): no
* Mock policy server passed: yes
* IsaacLab rollout passed: yes
* Demo HDF5 found: no
* HDF5 inspect passed: no
* LeRobot conversion passed: no
* pi0.5 training smoke test passed: no

### Next 3 commands to run
```bash
bash pi05_isaacsim_baseline/scripts/make_env_check.sh
PYBIN=python3 bash pi05_isaacsim_baseline/scripts/start_mock_server.sh 8008 && bash pi05_isaacsim_baseline/scripts/test_pi05_dryrun.sh
bash pi05_isaacsim_baseline/scripts/collect_demos.sh --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device <your_device> --num_demos 10 --enable_cameras
```
<!-- AUTO-SUMMARY -->

## Synthetic Plumbing Validation Summary

**Result: the full collect → HDF5 → LeRobot → π0.5 train-smoke → checkpoint pipeline PASSED end-to-end (synthetic data).**

* Timestamp: 2026-06-12 13:08–13:37 (run TS 20260612_130811)
* Synthetic HDF5 generated: **yes** (3 episodes × 80 steps, scripted IK-Rel actions)
* HDF5 path: `data/raw_hdf5/synthetic_plumbing.hdf5` (654 KB)
* HDF5 inspect passed: **yes** — `actions (T,7)`, `obs/{eef_pos,eef_quat,joint_pos,gripper_pos}`, `initial_state/`, `states/`, 3 `data/demo_*`
* Inspect log: `logs/teleop_hdf5_inspect_20260612_130811.{txt,json}`
* LeRobot conversion passed: **yes** — 3 episodes / 240 frames, all mappings `found`, `missing: []`, `lerobot_built: true`, `synthesized_images: 224`
* LeRobot dataset path: `data/lerobot/synthetic_plumbing_check` (features: image, wrist_image, state[18], actions[7])
* Convert report: `logs/synthetic_convert_report_20260612_130811.json`
* π0.5 train smoke test passed: **yes** — LoRA (gemma_2b_lora + gemma_300m_lora), single GPU, batch=1, 10 steps, `train exit: 0`, losses e.g. 0.44/0.24/0.03/0.31, no shape mismatch
* Train log: `logs/train_pi05_synthetic_smoke_20260612_130811.log`
* Checkpoint generated: **yes**
* Checkpoint path: `policies/checkpoints/pi05_isaaclab_20260612_133228/pi05_isaaclab_franka/isaaclab_20260612_133228/9` (8.8 GB; params 6.0 GB + train_state + bundled norm_stats)
* Main blocker if failed: n/a (passed)
* Next step:
  * **Passed** → collect REAL teleop demos (human/keyboard or GELLO), then repeat conversion + training. Synthetic data does NOT produce a usable policy — it only proves the plumbing.

> **Important caveat:** the synthetic dataset is throwaway (scripted motion, `success=False`, zero/dummy images). A model trained on it will NOT perform the task. This run validated the engineering pipeline only.

### Project-side fixes made this run (no OpenPI internals / `.venv_openpi` modified; all reversible)
1. `scripts/register_openpi_config.py` — bracket-matching bug: anchored insertion on the real `_CONFIGS = [` assignment (was matching a docstring `_CONFIGS` + a `dict[...]` bracket).
2. `scripts/register_openpi_config.py` — config now: `wandb_enabled=False`, and **LoRA** low-mem (`gemma_2b_lora`/`gemma_300m_lora` + `freeze_filter` + `ema_decay=None`) so it fits a 48 GB GPU.
3. `scripts/train_pi05.sh` — `compute_norm_stats --config-name` (tyro), single-GPU pin for batch=1 on a 2-GPU box, `--no-wandb-enabled`, and an `--fsdp N` option.
4. `adapters/data_conversion/hdf5_to_lerobot.py` — `--synthesize-images SIZE` to inject zero `image`/`wrist_image` for a state-only THROWAWAY dataset (pi0.5/libero layout requires images). Never for real training.

> Why LoRA / why not batch=1 full-finetune: full pi0.5 (~3B + AdamW states) needs ~50 GB and OOMs even sharded across 2×48 GB at batch=2. OpenPI also requires `batch_size % num_GPUs == 0`. LoRA freezes the backbone → trains tiny adapters → fits one 48 GB GPU at batch=1 (matches "freeze / train expert only"). For a real full-finetune, use bigger/more GPUs and `--fsdp`/even batch.


---

# STATUS — detail

## What is real / verified tonight
| capability | state | evidence |
|------------|-------|----------|
| Project structure + docs | ✅ | this tree, README, docs/ |
| Env detection report | ✅ | `logs/env_check.txt`, `logs/openpi_env_info.txt` |
| OpenPI/π0.5 install (isolated uv venv) | ✅ real | imports incl. jax(2 GPUs), torch cuda, lerobot |
| Unit tests (schema/safety/server/hdf5) | ✅ real | `logs/run_tests_*.log` ALL PASSED |
| Mock policy server + client fallback | ✅ real | test + live `/health` |
| π0.5 dry-run (mock) | ✅ real | `logs/pi05_dryrun.txt` |
| **IsaacLab closed-loop rollout** | ✅ real | `logs/eval_policy_*.json`, `rollout_*.jsonl` (60 steps) |
| Safety filter (clamps/NaN/workspace/reset) | ✅ real | unit tests, wired into rollout |
| HDF5 inspect / convert tools | ✅ (synthetic) | tools done; awaiting real demos |
| Training scripts + config injection | ✅ scripts | `train_pi05.sh`, `register_openpi_config.py` |
| Serve + eval scripts | ✅ scripts | `serve_pi05_checkpoint.sh`, `eval_pi05_in_isaaclab.sh` |
| Real-robot / FoundationPose interfaces | ✅ stubs | `adapters/real_robot/` |

## Real π0.5 (DROID checkpoint) — ✅ VERIFIED
- `pi05_droid` checkpoint downloaded (~12 GB via gcsfs, public bucket; `gsutil` not
  installed) and ran a **real forward pass on GPU**.
- `logs/pi05_dryrun.txt`: `backend_used: openpi`, action chunk **[15, 8]**
  (action_horizon=15 × DROID 8-dim), avg latency ~919 ms (JAX incl. JIT on first calls).
- Re-run (now cached): `.venv_openpi/bin/python scripts/test_pi05_dryrun.py --backend openpi --config pi05_droid --ckpt gs://openpi-assets/checkpoints/pi05_droid`
- Serve over HTTP: `.venv_openpi/bin/python adapters/policy_server/server.py --backend openpi --config pi05_droid --ckpt gs://openpi-assets/checkpoints/pi05_droid --port 8008`
- NOTE: raw DROID action space ≠ the Stack task's `ik_rel` 7D. For a meaningful closed
  loop, fine-tune on YOUR demos (`pi05_isaaclab_franka` config) so the action space
  matches the env. The mock backend already proves the loop mechanics; the real model
  proves the OpenPI inference stack works on this machine.

## Fallbacks currently in effect
- Policy backend defaults to **mock** until a real checkpoint/trained model is wired.
- LeRobot conversion writes a **normalized intermediate** even if the LeRobot build
  step is unavailable; finish later with `normalized_to_lerobot.py` in `.venv_openpi`.
- `export_episode_video.py` uses **imageio** (ffmpeg not installed).

## Things you should know
- Scripts that call `isaaclab.sh` now `conda activate env_isaaclab` automatically.
- The Stack task pulls in your customized scene (Cabinet/CoffeeMachine USD warnings in
  the log are from your existing env config; rollout still completes fine).
- Only system-level change made: git `safe.directory '*'`
  (revert: `git config --global --unset safe.directory '*'`).
