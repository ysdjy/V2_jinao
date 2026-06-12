"""Convert IsaacLab HDF5 demonstrations -> (1) a normalized intermediate dataset
(always) and (2) a LeRobot dataset (if `lerobot` is importable).

Robust to field-name differences: candidate keys come from
configs/dataset_mapping_isaaclab_franka.yaml, overridable via CLI. If a field
can't be matched the script does NOT crash -- it reports what was found / missing
and still writes the normalized dataset for everything it could extract.

Normalized intermediate layout (data/processed/normalized_dataset/<name>/):
  metadata.json          field mapping, units, fps, task, action space
  episodes.jsonl         one line per episode: {episode_index, length, ...}
  states.npy             (N_total, state_dim)
  actions.npy            (N_total, action_dim)
  episode_index.npy      (N_total,)  which episode each frame belongs to
  images/<cam>/ep<ee>_step<ssssss>.png   (only if cameras present)

Usage:
  python adapters/data_conversion/hdf5_to_lerobot.py \
      --input data/raw_hdf5/demo.hdf5 \
      --output data/lerobot/franka_stack_cube_pi05 \
      --config configs/dataset_mapping_isaaclab_franka.yaml \
      --task_instruction "Stack the cubes with the Franka robot."
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional


def _load_cfg(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _first_existing(ep_group, candidates) -> Optional[str]:
    for c in candidates or []:
        if c in ep_group:
            return c
    return None


def _resolve_component(ep_group, candidates) -> Optional[str]:
    return _first_existing(ep_group, candidates)


def convert(
    input_path: str,
    output_path: str,
    cfg: dict,
    task_instruction: str,
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    image_keys: Optional[dict] = None,
    normalized_root: str = "data/processed/normalized_dataset",
    build_lerobot: bool = True,
    synth_images: int = 0,
) -> dict:
    import h5py
    import numpy as np

    fps = int(cfg.get("fps", 30))
    image_resize = tuple(cfg.get("image_resize", [256, 256]))
    state_components = cfg.get("state_components", {})
    action_candidates = [action_key] if action_key else cfg.get("action_key_candidates", ["actions"])
    img_cfg = image_keys or cfg.get("image_key_candidates", {})
    ep_group_name = cfg.get("episode_group", "data")

    report: dict[str, Any] = {"input": input_path, "found": {}, "missing": [], "episodes": 0, "frames": 0}

    f = h5py.File(input_path, "r")
    if ep_group_name not in f:
        # Some files put demos at the root.
        root = f
    else:
        root = f[ep_group_name]
    demo_keys = [k for k in root.keys()]
    demo_keys.sort(key=lambda s: (len(s), s))
    if not demo_keys:
        raise RuntimeError(f"No episodes found under '{ep_group_name}'")

    # Resolve mapping from the FIRST episode.
    ep0 = root[demo_keys[0]]
    a_key = _first_existing(ep0, action_candidates)
    report["found"]["actions"] = a_key
    if a_key is None:
        report["missing"].append("actions")

    # state components
    comp_keys = {}
    for comp, cands in state_components.items():
        k = _resolve_component(ep0, cands)
        comp_keys[comp] = k
        report["found"][f"state.{comp}"] = k
        if k is None:
            report["missing"].append(f"state.{comp}")
    if state_key:  # explicit single state key overrides component assembly
        report["found"]["state_override"] = state_key

    cam_keys = {}
    for cam, cands in img_cfg.items():
        k = _first_existing(ep0, cands)
        cam_keys[cam] = k
        report["found"][f"image.{cam}"] = k

    # ------------------------------------------------------------------ #
    # Extract everything.
    os.makedirs(output_path, exist_ok=True)
    name = os.path.basename(output_path.rstrip("/"))
    norm_dir = os.path.join(normalized_root, name)
    os.makedirs(norm_dir, exist_ok=True)
    img_root = os.path.join(norm_dir, "images")

    all_states, all_actions, all_epidx = [], [], []
    episodes_meta = []
    have_images = any(v for v in cam_keys.values())

    for ei, dk in enumerate(demo_keys):
        ep = root[dk]
        if a_key is None or a_key not in ep:
            continue
        actions = np.asarray(ep[a_key], dtype=np.float32)
        T = actions.shape[0]

        # build state
        if state_key and state_key in ep:
            states = np.asarray(ep[state_key], dtype=np.float32)
        else:
            parts = []
            for comp in ["ee_position", "ee_quat", "gripper", "joint_pos"]:
                k = comp_keys.get(comp)
                if k and k in ep:
                    arr = np.asarray(ep[k], dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    parts.append(arr[:T])
            states = np.concatenate(parts, axis=1) if parts else np.zeros((T, 0), np.float32)

        all_states.append(states)
        all_actions.append(actions)
        all_epidx.append(np.full((T,), ei, dtype=np.int64))

        # images
        if have_images:
            for cam, k in cam_keys.items():
                if not k or k not in ep:
                    continue
                cam_dir = os.path.join(img_root, cam)
                os.makedirs(cam_dir, exist_ok=True)
                imgs = ep[k]
                try:
                    from PIL import Image

                    for t in range(T):
                        im = np.asarray(imgs[t])
                        if im.dtype != np.uint8:
                            im = np.clip(im, 0, 255).astype(np.uint8)
                        Image.fromarray(im).resize(image_resize).save(
                            os.path.join(cam_dir, f"ep{ei:04d}_step{t:06d}.png")
                        )
                except Exception as e:  # noqa
                    report.setdefault("image_errors", []).append(f"{cam}: {e}")

        episodes_meta.append({"episode_index": ei, "demo_key": dk, "length": int(T)})

    f.close()

    if not all_actions:
        raise RuntimeError("No usable episodes extracted (action key missing in all demos).")

    states_cat = np.concatenate(all_states, axis=0)
    actions_cat = np.concatenate(all_actions, axis=0)
    epidx_cat = np.concatenate(all_epidx, axis=0)
    report["episodes"] = len(episodes_meta)
    report["frames"] = int(actions_cat.shape[0])

    # write normalized
    np.save(os.path.join(norm_dir, "states.npy"), states_cat)
    np.save(os.path.join(norm_dir, "actions.npy"), actions_cat)
    np.save(os.path.join(norm_dir, "episode_index.npy"), epidx_cat)
    with open(os.path.join(norm_dir, "episodes.jsonl"), "w") as fp:
        for e in episodes_meta:
            fp.write(json.dumps(e) + "\n")
    meta = {
        "name": name,
        "source_hdf5": input_path,
        "fps": fps,
        "task_instruction": task_instruction,
        "num_episodes": len(episodes_meta),
        "num_frames": int(actions_cat.shape[0]),
        "state_dim": int(states_cat.shape[1]),
        "action_dim": int(actions_cat.shape[1]),
        "action_space": "env-native (IsaacLab action vector as recorded)",
        "units": {"position": "m", "rotation": "rad/quat", "gripper": "normalized or width"},
        "field_mapping": report["found"],
        "missing_fields": report["missing"],
        "has_images": have_images,
        "cameras": {k: v for k, v in cam_keys.items() if v},
    }
    with open(os.path.join(norm_dir, "metadata.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    report["normalized_dir"] = norm_dir

    # ------------------------------------------------------------------ #
    # LeRobot build (best effort)
    if build_lerobot:
        try:
            _build_lerobot(name, output_path, cfg, task_instruction, episodes_meta, all_states,
                           all_actions, cam_keys, img_root, image_resize, fps, synth_images)
            report["lerobot_dir"] = output_path
            report["lerobot_built"] = True
            if synth_images > 0 and not (cam_keys.get("front_rgb") or cam_keys.get("wrist_rgb")):
                report["synthesized_images"] = synth_images
        except Exception as e:  # noqa
            report["lerobot_built"] = False
            report["lerobot_error"] = str(e)

    return report


def _build_lerobot(name, output_path, cfg, task_instruction, episodes_meta, all_states,
                   all_actions, cam_keys, img_root, image_resize, fps, synth_images=0):
    """Build a real LeRobotDataset. Raises if lerobot is unavailable.

    synth_images > 0: when the source HDF5 has NO real camera data, inject zero-filled
    `image`/`wrist_image` features of size (synth_images, synth_images, 3). This lets a
    state-only THROWAWAY dataset match the pi0.5/libero data-config layout (which hard-
    requires image + wrist_image) so the training plumbing can be exercised. The zeros are
    NOT real observations and must never be used to train a deployable policy.
    """
    import numpy as np

    # lerobot import path differs across versions.
    LeRobotDataset = None
    for modpath in ("lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"):
        try:
            mod = __import__(modpath, fromlist=["LeRobotDataset"])
            LeRobotDataset = getattr(mod, "LeRobotDataset")
            break
        except Exception:
            continue
    if LeRobotDataset is None:
        raise ImportError("lerobot not importable in this environment")

    state_dim = all_states[0].shape[1]
    action_dim = all_actions[0].shape[1]
    H, W = image_resize
    have_front = bool(cam_keys.get("front_rgb"))
    have_wrist = bool(cam_keys.get("wrist_rgb"))

    # Synthesize zero images only when there is NO real camera data (throwaway plumbing).
    synth = synth_images > 0 and not (have_front or have_wrist)
    if synth:
        H = W = int(synth_images)

    features = {
        "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
    }
    if have_front or synth:
        features["image"] = {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]}
    if have_wrist or synth:
        features["wrist_image"] = {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]}

    import shutil

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    ds = LeRobotDataset.create(
        repo_id=name,
        root=output_path,
        robot_type="panda",
        fps=fps,
        features=features,
    )

    from PIL import Image

    for ei, (states, actions) in enumerate(zip(all_states, all_actions)):
        T = actions.shape[0]
        for t in range(T):
            frame = {
                "state": states[t].astype(np.float32),
                "actions": actions[t].astype(np.float32),
                "task": task_instruction,
            }
            if have_front or synth:
                p = os.path.join(img_root, "front_rgb", f"ep{ei:04d}_step{t:06d}.png")
                frame["image"] = (np.asarray(Image.open(p)) if (have_front and os.path.exists(p))
                                  else np.zeros((H, W, 3), np.uint8))
            if have_wrist or synth:
                p = os.path.join(img_root, "wrist_rgb", f"ep{ei:04d}_step{t:06d}.png")
                frame["wrist_image"] = (np.asarray(Image.open(p)) if (have_wrist and os.path.exists(p))
                                        else np.zeros((H, W, 3), np.uint8))
            ds.add_frame(frame)
        ds.save_episode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--task_instruction", default=None)
    p.add_argument("--state-key", default=None)
    p.add_argument("--action-key", default=None)
    p.add_argument("--no-lerobot", action="store_true", help="only build the normalized dataset")
    p.add_argument("--report-out", default=None)
    p.add_argument("--synthesize-images", type=int, default=0, metavar="SIZE",
                   help="If the HDF5 has no cameras, inject zero-filled image+wrist_image of SIZExSIZE "
                        "so a state-only THROWAWAY dataset matches the pi0.5/libero layout for plumbing tests. "
                        "0 = off. Never use the result to train a real policy.")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    task = args.task_instruction or cfg.get("default_task_instruction", "Perform the manipulation task.")
    report = convert(
        args.input, args.output, cfg, task,
        state_key=args.state_key, action_key=args.action_key,
        build_lerobot=not args.no_lerobot,
        synth_images=args.synthesize_images,
    )
    print(json.dumps(report, indent=2))
    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)
    if report.get("missing"):
        print("\n[WARN] Missing fields:", report["missing"])
        print("       Edit configs/dataset_mapping_isaaclab_franka.yaml candidates or pass --state-key/--action-key.")


if __name__ == "__main__":
    main()
