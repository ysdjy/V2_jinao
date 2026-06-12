"""Idempotently register a `pi05_isaaclab_franka` finetune config into OpenPI.

OpenPI's config registry is a python list `_CONFIGS` in
src/openpi/training/config.py. There is no plugin hook, so the supported way to
add a custom finetune config is to append a TrainConfig entry. This helper does
that automatically and reversibly:

  * backs up config.py to config.py.bak_pi05 (once)
  * inserts our TrainConfig just before the closing `]` of `_CONFIGS`
  * guarded by a marker so re-running is a no-op

Our LeRobot dataset (built by hdf5_to_lerobot.py) uses the libero-style feature
layout {image, wrist_image, state, actions, task}, so we reuse
LeRobotLiberoDataConfig with our own repo_id.

Usage:
  .venv_openpi/bin/python scripts/register_openpi_config.py \
      --repo-id franka_stack_cube_pi05 [--name pi05_isaaclab_franka] [--unregister]
"""

from __future__ import annotations

import argparse
import os
import shutil

MARKER = "# >>> pi05_isaaclab_baseline auto-config >>>"
END_MARKER = "# <<< pi05_isaaclab_baseline auto-config <<<"


def _config_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "third_party", "openpi", "src", "openpi", "training", "config.py")


def _entry(name: str, repo_id: str) -> str:
    # LoRA / low-memory fine-tune: freeze the backbone, train only low-rank adapters.
    # Full pi0.5 fine-tune (~3B params + AdamW states) OOMs even on 2x48GB; LoRA shrinks
    # the trainable set ~100x so it fits on a single 48GB GPU with batch_size=1.
    # Mirrors OpenPI's `pi0_libero_low_mem_finetune`. (ema_decay=None: no EMA for LoRA.)
    return f"""    {MARKER}
    TrainConfig(
        name="{name}",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="{repo_id}",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=1,
        num_train_steps=10,
        save_interval=10,
        log_interval=1,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
    ),
    {END_MARKER}
"""


def register(name: str, repo_id: str):
    path = _config_path()
    with open(path) as f:
        src = f.read()
    if MARKER in src:
        print(f"[register] '{name}' already registered (marker present). No-op.")
        return
    if not os.path.exists(path + ".bak_pi05"):
        shutil.copy2(path, path + ".bak_pi05")
        print(f"[register] backed up -> {path}.bak_pi05")

    # Find the actual `_CONFIGS = [` (or `_CONFIGS: list[...] = [`) ASSIGNMENT and its
    # matching closing bracket. Anchoring on `=\s*\[` avoids false matches on the word
    # "_CONFIGS" inside docstrings/comments or on unrelated `[` like `dict[str, ...]`.
    import re

    m = re.search(r"^_CONFIGS\b[^\n]*?=\s*\[", src, re.MULTILINE)
    if not m:
        raise RuntimeError("could not find the `_CONFIGS = [` list assignment")
    open_br = m.end() - 1  # index of the list's opening '['
    depth = 0
    close_br = -1
    for i in range(open_br, len(src)):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                close_br = i
                break
    if close_br < 0:
        raise RuntimeError("could not locate end of _CONFIGS list")
    new_src = src[:close_br] + _entry(name, repo_id) + src[close_br:]
    with open(path, "w") as f:
        f.write(new_src)
    print(f"[register] inserted config '{name}' (repo_id={repo_id}) into {path}")


def unregister():
    path = _config_path()
    with open(path) as f:
        src = f.read()
    if MARKER not in src:
        print("[register] nothing to remove")
        return
    s = src.find("    " + MARKER)
    e = src.find(END_MARKER, s) + len(END_MARKER) + 1
    new_src = src[:s] + src[e:]
    with open(path, "w") as f:
        f.write(new_src)
    print("[register] removed auto-config block")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pi05_isaaclab_franka")
    p.add_argument("--repo-id", default="franka_stack_cube_pi05")
    p.add_argument("--unregister", action="store_true")
    args = p.parse_args()
    if args.unregister:
        unregister()
    else:
        register(args.name, args.repo_id)


if __name__ == "__main__":
    main()
