"""Dataset quality report for a normalized dataset (data/processed/normalized_dataset/<name>).

Writes logs/dataset_quality_report.md. Works with numpy only.
Usage:
  python scripts/analyze_dataset.py --input data/processed/normalized_dataset/franka_stack_cube_pi05
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(proj, "logs", "dataset_quality_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    meta = {}
    mp = os.path.join(args.input, "metadata.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    states = np.load(os.path.join(args.input, "states.npy"))
    actions = np.load(os.path.join(args.input, "actions.npy"))
    epidx = np.load(os.path.join(args.input, "episode_index.npy"))
    n_eps = int(epidx.max()) + 1 if len(epidx) else 0
    ep_lengths = [int((epidx == i).sum()) for i in range(n_eps)]

    def stats(a):
        return {
            "shape": list(a.shape),
            "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None,
            "mean": float(a.mean()) if a.size else None,
            "nan": int(np.isnan(a).sum()),
            "inf": int(np.isinf(a).sum()),
        }

    s_stats, a_stats = stats(states), stats(actions)

    img_root = os.path.join(args.input, "images")
    img_counts = {}
    if os.path.isdir(img_root):
        for cam in os.listdir(img_root):
            d = os.path.join(img_root, cam)
            if os.path.isdir(d):
                img_counts[cam] = len([f for f in os.listdir(d) if f.endswith(".png")])

    lines = [
        "# Dataset Quality Report",
        "",
        f"- Source: `{args.input}`",
        f"- Task: {meta.get('task_instruction','?')}",
        f"- Episodes: {n_eps}",
        f"- Total frames: {len(epidx)}",
        f"- FPS: {meta.get('fps','?')}",
        f"- Episode lengths: min={min(ep_lengths) if ep_lengths else 0} "
        f"max={max(ep_lengths) if ep_lengths else 0} "
        f"mean={sum(ep_lengths)/len(ep_lengths):.1f}" if ep_lengths else "- Episode lengths: n/a",
        "",
        "## State",
        f"- shape {s_stats['shape']}, range [{s_stats['min']}, {s_stats['max']}], "
        f"NaN={s_stats['nan']} Inf={s_stats['inf']}",
        "## Actions",
        f"- shape {a_stats['shape']}, range [{a_stats['min']}, {a_stats['max']}], "
        f"NaN={a_stats['nan']} Inf={a_stats['inf']}",
        "## Images",
        f"- {img_counts if img_counts else 'none (state-only dataset)'}",
        "",
        "## Field mapping (from metadata)",
        "```json",
        json.dumps(meta.get("field_mapping", {}), indent=2),
        "```",
    ]
    flags = []
    if s_stats["nan"] or s_stats["inf"] or a_stats["nan"] or a_stats["inf"]:
        flags.append("NaN/Inf present in arrays")
    if img_counts and len(set(img_counts.values())) > 1:
        flags.append(f"image frame counts differ between cameras: {img_counts}")
    lines += ["", "## Flags", ("- " + "\n- ".join(flags)) if flags else "- none"]

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[analyze] wrote {out}")


if __name__ == "__main__":
    main()
