"""Plot EE trajectory + gripper curve from a rollout JSONL or a normalized dataset.

Usage:
  python scripts/plot_trajectory.py --rollout data/processed/rollouts/rollout_XXX.jsonl
  python scripts/plot_trajectory.py --dataset data/processed/normalized_dataset/<name> --episode 0
Saves PNGs next to the input (or under logs/plots/). Falls back to a text summary
if matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os


def _from_rollout(path):
    eexyz, grip = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            obs = r.get("observation_summary", {})
            if obs.get("ee_position"):
                eexyz.append(obs["ee_position"])
            grip.append(obs.get("gripper_width", 0.0))
    return eexyz, grip


def _from_dataset(path, episode):
    import numpy as np

    states = np.load(os.path.join(path, "states.npy"))
    epidx = np.load(os.path.join(path, "episode_index.npy"))
    mask = epidx == episode
    s = states[mask]
    # state layout from converter: ee_pos(3), ee_quat(4), gripper(1), joints...
    eexyz = s[:, :3].tolist() if s.shape[1] >= 3 else []
    grip = s[:, 7].tolist() if s.shape[1] > 7 else [0.0] * len(s)
    return eexyz, grip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    if args.rollout:
        eexyz, grip = _from_rollout(args.rollout)
        stem = os.path.splitext(os.path.basename(args.rollout))[0]
    elif args.dataset:
        eexyz, grip = _from_dataset(args.dataset, args.episode)
        stem = f"{os.path.basename(args.dataset)}_ep{args.episode}"
    else:
        raise SystemExit("provide --rollout or --dataset")

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = args.outdir or os.path.join(proj, "logs", "plots")
    os.makedirs(outdir, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if eexyz:
            xs = [p[0] for p in eexyz]
            ys = [p[1] for p in eexyz]
            zs = [p[2] for p in eexyz]
            fig = plt.figure(figsize=(10, 4))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.plot(xs, ys, zs, marker=".")
            ax1.set_title("EE trajectory")
            ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(grip)
            ax2.set_title("gripper width / state"); ax2.set_xlabel("step")
            fp = os.path.join(outdir, f"{stem}_traj.png")
            fig.tight_layout(); fig.savefig(fp, dpi=120)
            print(f"[plot] saved {fp}")
        else:
            print("[plot] no EE positions found in input")
    except Exception as e:  # noqa
        print(f"[plot] matplotlib unavailable ({e}); text summary:")
        print(f"  frames={len(grip)} ee_samples={len(eexyz)}")
        if eexyz:
            print(f"  ee start={eexyz[0]} end={eexyz[-1]}")


if __name__ == "__main__":
    main()
