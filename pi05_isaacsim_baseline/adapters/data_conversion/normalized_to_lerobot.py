"""Second-stage converter: normalized intermediate dataset -> LeRobot dataset.

Use this when hdf5_to_lerobot.py produced a normalized dataset but the LeRobot
build failed (e.g. lerobot not installed at conversion time). Run it later inside
the OpenPI venv where lerobot is available.

Usage:
  python adapters/data_conversion/normalized_to_lerobot.py \
      --input data/processed/normalized_dataset/franka_stack_cube_pi05 \
      --output data/lerobot/franka_stack_cube_pi05
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="normalized dataset dir")
    p.add_argument("--output", required=True, help="output LeRobot dir")
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.input, "metadata.json")))
    states = np.load(os.path.join(args.input, "states.npy"))
    actions = np.load(os.path.join(args.input, "actions.npy"))
    epidx = np.load(os.path.join(args.input, "episode_index.npy"))
    img_root = os.path.join(args.input, "images")
    fps = int(meta.get("fps", 30))
    task = meta.get("task_instruction", "Perform the manipulation task.")
    cams = meta.get("cameras", {})
    have_front = "front_rgb" in cams
    have_wrist = "wrist_rgb" in cams

    LeRobotDataset = None
    for modpath in ("lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"):
        try:
            mod = __import__(modpath, fromlist=["LeRobotDataset"])
            LeRobotDataset = getattr(mod, "LeRobotDataset")
            break
        except Exception:
            continue
    if LeRobotDataset is None:
        raise ImportError("lerobot not importable; run inside the OpenPI venv")

    # detect image size
    H = W = 256
    features = {
        "state": {"dtype": "float32", "shape": (states.shape[1],), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (actions.shape[1],), "names": ["actions"]},
    }
    if have_front:
        features["image"] = {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]}
    if have_wrist:
        features["wrist_image"] = {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]}

    import shutil

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    ds = LeRobotDataset.create(repo_id=os.path.basename(args.output), root=args.output,
                               robot_type="panda", fps=fps, features=features)

    from PIL import Image

    n_eps = int(epidx.max()) + 1 if len(epidx) else 0
    for ei in range(n_eps):
        mask = epidx == ei
        s_ep = states[mask]
        a_ep = actions[mask]
        for t in range(len(a_ep)):
            frame = {"state": s_ep[t].astype(np.float32), "actions": a_ep[t].astype(np.float32), "task": task}
            if have_front:
                fp = os.path.join(img_root, "front_rgb", f"ep{ei:04d}_step{t:06d}.png")
                frame["image"] = np.asarray(Image.open(fp)) if os.path.exists(fp) else np.zeros((H, W, 3), np.uint8)
            if have_wrist:
                fp = os.path.join(img_root, "wrist_rgb", f"ep{ei:04d}_step{t:06d}.png")
                frame["wrist_image"] = np.asarray(Image.open(fp)) if os.path.exists(fp) else np.zeros((H, W, 3), np.uint8)
            ds.add_frame(frame)
        ds.save_episode()
    print(f"LeRobot dataset written to {args.output} ({n_eps} episodes)")


if __name__ == "__main__":
    main()
