"""Recursively print the structure of an IsaacLab HDF5 demonstration file.

Prints groups, datasets, shapes, dtypes, attrs, and a few sample values, so you
can author/verify the field mapping in configs/dataset_mapping_isaaclab_franka.yaml.

Usage:
  python adapters/data_conversion/inspect_hdf5.py --input data/raw_hdf5/demo.hdf5
  python adapters/data_conversion/inspect_hdf5.py --input demo.hdf5 --max-depth 6 --samples 2
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional


def inspect(path: str, max_depth: int = 8, samples: int = 1) -> dict[str, Any]:
    import h5py
    import numpy as np

    summary: dict[str, Any] = {"file": path, "datasets": {}, "groups": [], "attrs": {}}

    def fmt_attrs(obj):
        out = {}
        for k, v in obj.attrs.items():
            try:
                out[k] = v.item() if hasattr(v, "item") and getattr(v, "size", 2) == 1 else (
                    v.tolist() if hasattr(v, "tolist") else str(v)
                )
            except Exception:
                out[k] = str(v)
        return out

    def visit(name, obj, depth):
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            summary["groups"].append(name)
            a = fmt_attrs(obj)
            print(f"{indent}[G] {name}/  attrs={a if a else ''}")
            if a:
                summary["attrs"][name] = a
        elif isinstance(obj, h5py.Dataset):
            ds_info = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            a = fmt_attrs(obj)
            if a:
                ds_info["attrs"] = a
            sample_repr = ""
            try:
                if obj.size > 0 and len(obj.shape) <= 2 and obj.dtype.kind in "fiu":
                    flat = np.array(obj[(0,) if len(obj.shape) >= 1 else ()]).ravel()[:6]
                    sample_repr = "  e.g. " + np.array2string(flat, precision=4, max_line_width=120)
                    ds_info["sample"] = flat.tolist()
            except Exception as e:  # noqa
                sample_repr = f"  (sample read failed: {e})"
            print(f"{indent}[D] {name}  shape={obj.shape} dtype={obj.dtype}{sample_repr}")
            summary["datasets"][name] = ds_info

    with h5py.File(path, "r") as f:
        top_attrs = fmt_attrs(f)
        if top_attrs:
            print(f"[ROOT attrs] {top_attrs}")
            summary["attrs"]["/"] = top_attrs
        # Recursive walk
        def walk(group, prefix="", depth=0):
            if depth > max_depth:
                return
            for key in group.keys():
                obj = group[key]
                full = f"{prefix}/{key}" if prefix else key
                visit(full, obj, depth)
                if isinstance(obj, h5py.Group):
                    walk(obj, full, depth + 1)

        walk(f)

        # IsaacLab convention: data/demo_0/... ; report episode count
        if "data" in f:
            demos = list(f["data"].keys())
            print(f"\n[INFO] /data contains {len(demos)} episode(s): {demos[:5]}{' ...' if len(demos) > 5 else ''}")
            summary["num_episodes"] = len(demos)
            summary["episode_keys"] = demos
            if demos:
                first = f["data"][demos[0]]
                print(f"[INFO] first episode '{demos[0]}' keys: {list(first.keys())}")
                if "obs" in first:
                    print(f"[INFO] obs keys: {list(first['obs'].keys())}")
                    summary["obs_keys"] = list(first["obs"].keys())
                if "actions" in first:
                    print(f"[INFO] actions shape: {first['actions'].shape}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--json-out", default=None, help="optional path to dump the summary as JSON")
    args = p.parse_args()
    summary = inspect(args.input, args.max_depth, args.samples)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved summary] {args.json_out}")


if __name__ == "__main__":
    main()
