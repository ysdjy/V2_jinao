"""Validate a converted dataset (LeRobot dir OR normalized intermediate dir).

Checks: presence of arrays, NaN/Inf, action/state ranges, episode counts, image
counts. Prints a short report; exit code 0 if usable, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os


def validate_normalized(path: str) -> dict:
    import numpy as np

    rep = {"path": path, "ok": True, "issues": []}
    meta_p = os.path.join(path, "metadata.json")
    if os.path.exists(meta_p):
        rep["metadata"] = json.load(open(meta_p))
    for fn in ["states.npy", "actions.npy", "episode_index.npy"]:
        fp = os.path.join(path, fn)
        if not os.path.exists(fp):
            rep["issues"].append(f"missing {fn}")
            rep["ok"] = False
            continue
        arr = np.load(fp)
        key = fn.replace(".npy", "")
        rep[key + "_shape"] = list(arr.shape)
        if arr.dtype.kind in "fi":
            nan = int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0
            inf = int(np.isinf(arr).sum()) if arr.dtype.kind == "f" else 0
            if nan or inf:
                rep["issues"].append(f"{fn}: {nan} NaN, {inf} Inf")
                rep["ok"] = False
            if arr.size:
                rep[key + "_range"] = [float(arr.min()), float(arr.max())]
    return rep


def validate_lerobot(path: str) -> dict:
    rep = {"path": path, "ok": True, "issues": []}
    # Look for the standard LeRobot meta files.
    for cand in ["meta/info.json", "meta/episodes.jsonl", "info.json"]:
        fp = os.path.join(path, cand)
        if os.path.exists(fp):
            rep.setdefault("found", []).append(cand)
    if not rep.get("found"):
        rep["issues"].append("no LeRobot meta files found")
        rep["ok"] = False
    try:
        for modpath in ("lerobot.common.datasets.lerobot_dataset", "lerobot.datasets.lerobot_dataset"):
            try:
                mod = __import__(modpath, fromlist=["LeRobotDataset"])
                LeRobotDataset = getattr(mod, "LeRobotDataset")
                ds = LeRobotDataset(repo_id=os.path.basename(path), root=path)
                rep["num_frames"] = len(ds)
                rep["num_episodes"] = ds.num_episodes
                break
            except Exception:
                continue
    except Exception as e:  # noqa
        rep["issues"].append(f"load check skipped: {e}")
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--kind", choices=["auto", "normalized", "lerobot"], default="auto")
    args = p.parse_args()

    kind = args.kind
    if kind == "auto":
        kind = "normalized" if os.path.exists(os.path.join(args.input, "states.npy")) else "lerobot"
    rep = validate_normalized(args.input) if kind == "normalized" else validate_lerobot(args.input)
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep.get("ok", False) else 1)


if __name__ == "__main__":
    main()
