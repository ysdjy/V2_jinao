#!/usr/bin/env python3
"""Dump the top-level scene objects of a saved SceneLayoutModule scene.

Why this exists
---------------
The SceneLayoutModule JSON manifest only records objects under the env namespace
(``/World/envs/env_0``). Appliances the user adds at the stage root -- e.g.
``/coffeemachine`` and ``/microwave_flattened`` -- are NOT in the manifest. To sync
those into a downstream task env cfg we have to read the saved USD snapshot directly.

This script reads a saved ``scene_v0_*.usd`` (default: the newest one, i.e. what
``scene_layout_ui.py --load_latest_saved`` would open) and prints, for every
top-level prim, the authored local transform (translate / orient wxyz / scale) plus
any USD reference/payload asset path. Those are exactly the numbers needed to update
``stack_joint_pos_env_cfg.py`` (pos / rot / scale / usd_path).

It deliberately does NOT launch the Isaac/kit app; it only loads the USD runtime
(pxr) that ships inside the isaacsim pip package, so it is fast (~1s).

Usage
-----
    python SceneLayoutModule/inspect_saved_scene.py
    python SceneLayoutModule/inspect_saved_scene.py --usd <path to scene_v0_*.usd>

If ``pxr`` is not importable directly, re-run under the IsaacLab python:
    ./isaaclab.sh -p SceneLayoutModule/inspect_saved_scene.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAVED_DIR = REPO / "SceneLayoutModule" / "saved_scenes"


def _ensure_pxr() -> None:
    """Make ``pxr`` importable by locating omni.usd.libs inside the isaacsim wheel."""
    try:
        import pxr  # noqa: F401

        return
    except Exception:
        pass

    try:
        import isaacsim  # noqa: F401

        site = Path(isaacsim.__file__).resolve().parent
    except Exception:
        return  # let the later import raise a clear error

    matches = sorted(site.glob("extscache/omni.usd.libs-*/pxr"))
    if not matches:
        return
    pxr_root = matches[0].parent  # the dir that contains the importable ``pxr`` package
    bin_dir = pxr_root / "bin"
    conda_lib = Path(sys.prefix) / "lib"
    sys.path.insert(0, str(pxr_root))
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    extra = f"{bin_dir}:{conda_lib}"
    if extra not in ld:
        # Re-exec once with the shared libs on LD_LIBRARY_PATH (it must be set before
        # the process starts for the .so loader to pick it up).
        os.environ["LD_LIBRARY_PATH"] = f"{extra}:{ld}" if ld else extra
        os.environ["_INSPECT_REEXEC"] = "1"
        if os.environ.get("_INSPECT_REEXEC_DONE") != "1":
            os.environ["_INSPECT_REEXEC_DONE"] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])


def _latest_saved() -> Path | None:
    cands = sorted(glob.glob(str(SAVED_DIR / "scene_v0_*.usd")), key=os.path.getmtime)
    return Path(cands[-1]) if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=str, default=None, help="Saved scene USD (default: newest in saved_scenes/).")
    args = ap.parse_args()

    _ensure_pxr()
    from pxr import Usd, UsdGeom  # noqa: E402

    usd_path = Path(args.usd).expanduser() if args.usd else _latest_saved()
    if usd_path is None or not usd_path.exists():
        raise SystemExit(f"No saved scene USD found (looked in {SAVED_DIR}).")

    print(f"[inspect] scene USD: {usd_path}")
    stage = Usd.Stage.Open(str(usd_path))

    # Scene objects live in two places: under the env namespace (/World/envs/env_0/*,
    # also covered by the JSON manifest) and at the stage root (appliances the user adds,
    # e.g. /coffeemachine, /microwave_flattened -- NOT in the manifest). Show both.
    env_root = stage.GetPrimAtPath("/World/envs/env_0")
    targets = list(env_root.GetChildren()) if env_root and env_root.IsValid() else []
    skip_names = {
        "Render", "OmniverseKit_Persp", "OmniverseKit_Front", "OmniverseKit_Top",
        "OmniverseKit_Right", "OmniKit_Viewport_LightRig", "Replicator", "physicsScene",
        "World", "GroundPlane", "light",
    }
    for child in stage.GetPseudoRoot().GetChildren():
        if child.GetName() not in skip_names:
            targets.append(child)

    for prim in targets:
        path = str(prim.GetPath())
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName().split(":")[-1]: op.Get() for op in xf.GetOrderedXformOps()} if prim.IsA(UsdGeom.Xformable) else {}
        refs = []
        for spec in prim.GetPrimStack():
            for r in spec.referenceList.GetAddedOrExplicitItems():
                if r.assetPath:
                    refs.append(("ref", r.assetPath))
            for r in spec.payloadList.GetAddedOrExplicitItems():
                if r.assetPath:
                    refs.append(("payload", r.assetPath))
        print("=" * 72)
        print(f"PRIM {path}  ({prim.GetTypeName()})")
        if "translate" in ops:
            t = ops["translate"]
            print(f"  pos   = ({t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f})")
        if "orient" in ops and ops["orient"] is not None:
            q = ops["orient"]
            # pxr Quat: GetReal()=w, GetImaginary()=(x,y,z)
            try:
                w = q.GetReal()
                im = q.GetImaginary()
                print(f"  rot   = (w={w:.6f}, x={im[0]:.6f}, y={im[1]:.6f}, z={im[2]:.6f})")
            except Exception:
                print(f"  rot   = {q}")
        if "scale" in ops:
            s = ops["scale"]
            print(f"  scale = ({s[0]:.5f}, {s[1]:.5f}, {s[2]:.5f})")
        for kind, ap_ in refs:
            print(f"  {kind}: {ap_}")


if __name__ == "__main__":
    main()
