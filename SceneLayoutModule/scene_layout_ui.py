# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone V0 scene layout UI.

This entry is intentionally a layout tool, not a controller. It loads the project
scene, keeps the robot still, lets the user move/add prims in Isaac Sim, and
provides one-click save buttons for the edited USD stage plus a JSON transform
manifest.

Run:
    ./isaaclab.sh -p SceneLayoutModule/scene_layout_ui.py --num_envs 1
"""

from __future__ import annotations

"""Launch Omniverse Toolkit first."""

import argparse
import json
import re
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V0 scene layout tool for the project Franka scene.")
parser.set_defaults(disable_fabric=True)
parser.add_argument("--disable_fabric", action="store_true", dest="disable_fabric", help="Use USD I/O for layout editing.")
parser.add_argument("--enable_fabric", action="store_false", dest="disable_fabric", help="Enable Fabric for advanced debugging.")
parser.add_argument("--disable_collision_debug_vis", action="store_true", default=False, help="Disable collider overlays.")
parser.add_argument("--seed", type=int, default=1, help="Deterministic scene seed.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. This layout UI supports 1.")
parser.add_argument(
    "--save_dir",
    type=str,
    default="SceneLayoutModule/saved_scenes",
    help="Directory for saved USD stages and JSON manifests.",
)
parser.add_argument("--load_usd", type=str, default=None, help="Open a previously saved V0 USD stage instead of creating the task scene.")
parser.add_argument(
    "--load_latest_saved",
    action="store_true",
    default=False,
    help="Open the newest scene_v0_*.usd from --save_dir instead of creating the task scene.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Stack-Cube-Franka-JointPolicy-v0",
    help="Gym task id used as the V0 base scene.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import carb
import gymnasium as gym
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab.sim.utils.stage import open_stage
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

ENV_NS = "/World/envs/env_0"
DEFAULT_DYNAMIC_ASSETS = {
    "CoffeeMachine_103046": "SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd",
    "Cabinet_44853": "SapienAssetPipeline/usd_assets/Cabinet_44853/cabinet.usd",
    "Fridge_12252": "SapienAssetPipeline/usd_assets/Fridge_12252/fridge.usd",
    "Knife_101054": "SapienAssetPipeline/usd_assets/Knife_101054/knife.usd",
    "Microwave_7320": "SapienAssetPipeline/usd_assets/Microwave_7320/microwave_referenceable.usd",
}


def enable_collision_debug_visualization() -> None:
    settings = carb.settings.get_settings()
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 2)
    settings.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)


class LayoutWindow:
    def __init__(self, save_dir: Path):
        import omni.ui as ui

        self.ui = ui
        self.save_dir = save_dir
        self.asset_names = sorted(_discover_usd_assets()) or list(DEFAULT_DYNAMIC_ASSETS.keys())
        self.selected_asset_name = self.asset_names[0]
        self.asset_entry_model = ui.SimpleStringModel(self.selected_asset_name)
        self.status_labels: dict[str, object] = {}
        self.last_saved_usd = "None"
        self.last_saved_manifest = "None"
        self.last_added = "None"
        self.window = ui.Window("V0 Scene Layout", width=520, height=420)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Scene")
                with ui.HStack(spacing=6):
                    ui.Button("Save USD", clicked_fn=self.save_usd)
                    ui.Button("Save JSON", clicked_fn=self.save_manifest)
                    ui.Button("Save Both", clicked_fn=self.save_both)
                ui.Label("Add USD Asset")
                self.asset_model = ui.ComboBox(0, *self.asset_names).model
                self.asset_model.add_item_changed_fn(self._on_asset_changed)
                ui.StringField(model=self.asset_entry_model)
                with ui.HStack(spacing=6):
                    ui.Button("Refresh Assets", clicked_fn=self.refresh_assets)
                    ui.Button("Add Asset", clicked_fn=self.add_selected_asset)
                ui.Label("After adding or moving objects in the viewport, click Save Both.")
                for key in (
                    "task",
                    "stage_identifier",
                    "editable_root",
                    "selected_asset",
                    "last_added",
                    "last_saved_usd",
                    "last_saved_manifest",
                    "save_dir",
                ):
                    self.status_labels[key] = ui.Label(f"{key}:")

    def _on_asset_changed(self, model, item):
        index = model.get_item_value_model().as_int
        if 0 <= index < len(self.asset_names):
            self.selected_asset_name = self.asset_names[index]
            self.asset_entry_model.set_value(self.selected_asset_name)

    def refresh_assets(self):
        asset_map = _discover_usd_assets()
        self.asset_names = sorted(asset_map)
        self.selected_asset_name = self.asset_names[0] if self.asset_names else ""
        self.asset_entry_model.set_value(self.selected_asset_name)

    def add_selected_asset(self):
        asset_map = _discover_usd_assets()
        asset_key = self.asset_entry_model.as_string.strip() or self.selected_asset_name
        usd_path = Path(asset_key).expanduser()
        if not usd_path.is_absolute():
            usd_path = _repo_root() / usd_path
        asset_name = usd_path.parent.name if usd_path.is_file() else asset_key
        usd_path = usd_path if usd_path.is_file() else asset_map.get(asset_key)
        if usd_path is None:
            self.last_added = f"missing asset: {asset_key}"
            return
        prim_path = _add_usd_reference(asset_name, usd_path)
        self.last_added = prim_path

    def save_usd(self):
        self.last_saved_usd = str(_save_stage(self.save_dir))

    def save_manifest(self):
        self.last_saved_manifest = str(_save_manifest(self.save_dir))

    def save_both(self):
        self.save_usd()
        self.save_manifest()

    def update(self):
        stage = omni.usd.get_context().get_stage()
        values = {
            "task": args_cli.task,
            "stage_identifier": stage.GetRootLayer().identifier if stage is not None else "None",
            "editable_root": ENV_NS,
            "selected_asset": self.selected_asset_name,
            "last_added": self.last_added,
            "last_saved_usd": self.last_saved_usd,
            "last_saved_manifest": self.last_saved_manifest,
            "save_dir": str(self.save_dir),
        }
        for key, label in self.status_labels.items():
            label.text = f"{key}: {values[key]}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}"


def _safe_prim_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _discover_usd_assets() -> dict[str, Path]:
    repo = _repo_root()
    assets: dict[str, Path] = {}
    for name, relative in DEFAULT_DYNAMIC_ASSETS.items():
        path = repo / relative
        if path.is_file():
            assets[name] = path
    for usd_file in sorted((repo / "SapienAssetPipeline" / "usd_assets").glob("*/*.usd")):
        assets.setdefault(usd_file.parent.name, usd_file)
    return assets


def _resolve_load_usd(save_dir: Path) -> Path | None:
    if args_cli.load_usd:
        load_path = Path(args_cli.load_usd).expanduser()
        if not load_path.is_absolute():
            load_path = _repo_root() / load_path
        if not load_path.is_file():
            raise FileNotFoundError(f"Saved USD does not exist: {load_path}")
        return load_path.resolve()
    if not args_cli.load_latest_saved:
        return None
    candidates = sorted(save_dir.glob("scene_v0_*.usd"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No saved scene_v0_*.usd files found in: {save_dir}")
    return candidates[-1].resolve()


def _pause_timeline_for_layout() -> None:
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        if hasattr(timeline, "pause"):
            timeline.pause()
        else:
            timeline.stop()


def _next_free_prim_path(stage: Usd.Stage, base_path: str) -> str:
    if not stage.GetPrimAtPath(base_path).IsValid():
        return base_path
    index = 1
    while stage.GetPrimAtPath(f"{base_path}_{index:02d}").IsValid():
        index += 1
    return f"{base_path}_{index:02d}"


def _add_usd_reference(asset_name: str, usd_path: Path) -> str:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    base_name = _safe_prim_name(asset_name)
    prim_path = _next_free_prim_path(stage, f"{ENV_NS}/{base_name}")
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd_path.resolve()))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.25, 0.05))
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    print(f"[SceneV0] added {usd_path} -> {prim_path}", flush=True)
    return prim_path


def _save_stage(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = (save_dir / f"scene_v0_{_timestamp()}.usd").resolve()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    _pause_timeline_for_layout()
    if not stage.Export(str(out_path)):
        raise RuntimeError(f"Failed to save USD stage: {out_path}")
    print(f"[SceneV0] saved USD: {out_path}", flush=True)
    return out_path


def _get_world_transform(prim: Usd.Prim) -> dict[str, list[float]]:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    return {
        "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
        "rotation_wxyz": [
            float(rotation.GetReal()),
            float(rotation.GetImaginary()[0]),
            float(rotation.GetImaginary()[1]),
            float(rotation.GetImaginary()[2]),
        ],
    }


def _value_to_json(value):
    if value is None:
        return None
    if isinstance(value, (Gf.Vec3d, Gf.Vec3f, Gf.Vec3h)):
        return [float(value[0]), float(value[1]), float(value[2])]
    if isinstance(value, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
        return [float(value.GetReal()), *[float(item) for item in value.GetImaginary()]]
    try:
        return [float(item) for item in value]
    except TypeError:
        return str(value)


def _get_xform_ops(prim: Usd.Prim) -> dict[str, object]:
    xform = UsdGeom.Xformable(prim)
    result = {
        "translate": [0.0, 0.0, 0.0],
        "orient_wxyz": [1.0, 0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "ordered_ops": [],
    }
    for op in xform.GetOrderedXformOps():
        value = _value_to_json(op.Get())
        op_record = {"name": op.GetName(), "type": str(op.GetOpType()), "value": value}
        result["ordered_ops"].append(op_record)
        if op.GetName() == "xformOp:translate":
            result["translate"] = value
        elif op.GetName() == "xformOp:orient":
            result["orient_wxyz"] = value
        elif op.GetName() == "xformOp:scale":
            result["scale"] = value
    return result


def _get_listop_items(prim: Usd.Prim, metadata_key: str) -> list[object]:
    listop = prim.GetMetadata(metadata_key)
    if listop is None:
        return []
    items = []
    if hasattr(listop, "GetAddedOrExplicitItems"):
        items.extend(listop.GetAddedOrExplicitItems())
    for attr_name in ("prependedItems", "addedItems", "explicitItems", "appendedItems"):
        items.extend(getattr(listop, attr_name, []))
    deduped = []
    seen = set()
    for item in items:
        key = (str(getattr(item, "assetPath", "")), str(getattr(item, "primPath", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _get_subtree_assets(prim: Usd.Prim) -> list[dict[str, str]]:
    assets = []
    seen = set()
    for subtree_prim in Usd.PrimRange(prim):
        records = []
        if subtree_prim.HasAuthoredReferences():
            records.extend(("reference", item) for item in _get_listop_items(subtree_prim, "references"))
        if subtree_prim.HasAuthoredPayloads():
            records.extend(("payload", item) for item in _get_listop_items(subtree_prim, "payload"))
        for kind, item in records:
            asset_path = str(getattr(item, "assetPath", ""))
            prim_path = str(getattr(item, "primPath", ""))
            key = (subtree_prim.GetPath().pathString, kind, asset_path, prim_path)
            if key in seen:
                continue
            seen.add(key)
            assets.append(
                {
                    "owner_prim": subtree_prim.GetPath().pathString,
                    "kind": kind,
                    "asset_path": asset_path,
                    "prim_path": prim_path,
                }
            )
    return assets


def _has_api(prim: Usd.Prim, schema) -> bool:
    try:
        return prim.HasAPI(schema)
    except TypeError:
        return False


def _get_physics_summary(prim: Usd.Prim) -> list[dict[str, str]]:
    records = []
    for subtree_prim in Usd.PrimRange(prim):
        if _has_api(subtree_prim, UsdPhysics.RigidBodyAPI):
            records.append({"kind": "RigidBodyAPI", "path": subtree_prim.GetPath().pathString})
        if _has_api(subtree_prim, UsdPhysics.CollisionAPI):
            records.append({"kind": "CollisionAPI", "path": subtree_prim.GetPath().pathString})
        if _has_api(subtree_prim, UsdPhysics.ArticulationRootAPI):
            records.append({"kind": "ArticulationRootAPI", "path": subtree_prim.GetPath().pathString})
        if subtree_prim.IsA(UsdPhysics.Joint):
            records.append({"kind": subtree_prim.GetTypeName(), "path": subtree_prim.GetPath().pathString})
    return records


def _iter_layout_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    roots = []
    env_prim = stage.GetPrimAtPath(ENV_NS)
    if env_prim.IsValid():
        roots.extend(child for child in env_prim.GetChildren() if child.IsA(UsdGeom.Xformable))
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        for child in world_prim.GetChildren():
            path = child.GetPath().pathString
            if path.startswith("/World/envs") or path.startswith("/World/Visuals"):
                continue
            if child.IsA(UsdGeom.Xformable):
                roots.append(child)
    return roots


def _save_manifest(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    objects = []
    for prim in _iter_layout_prims(stage):
        imageable = UsdGeom.Imageable(prim) if prim.IsA(UsdGeom.Imageable) else None
        objects.append(
            {
                "path": prim.GetPath().pathString,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "active": prim.IsActive(),
                "loaded": prim.IsLoaded(),
                "visibility": str(imageable.ComputeVisibility(Usd.TimeCode.Default())) if imageable else "n/a",
                "xform": _get_xform_ops(prim) if prim.IsA(UsdGeom.Xformable) else None,
                "subtree_assets": _get_subtree_assets(prim),
                "applied_schemas": [str(item) for item in prim.GetAppliedSchemas()],
                "physics": _get_physics_summary(prim),
                "world_transform": _get_world_transform(prim),
            }
        )
    out_path = (save_dir / f"scene_v0_{_timestamp()}.json").resolve()
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "scene-layout-manifest-v1",
                "task": args_cli.task,
                "env_root": ENV_NS,
                "stage_identifier": stage.GetRootLayer().identifier,
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                "restore_contract": {
                    "preferred_restore": "Load the saved USD snapshot directly when possible.",
                    "programmatic_restore": "Create each object at path and apply xform.translate, xform.orient_wxyz, and xform.scale in ordered_ops order.",
                    "robot_joint_positions": "Not part of this manifest contract.",
                    "external_assets": "All subtree_assets asset_path entries must be available to the restoring project.",
                },
                "objects": objects,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[SceneV0] saved manifest: {out_path}", flush=True)
    return out_path


def main():
    if args_cli.num_envs != 1:
        raise ValueError("scene_v0_layout_ui supports --num_envs 1.")

    env = None
    save_dir = (_repo_root() / args_cli.save_dir).resolve()
    load_usd = _resolve_load_usd(save_dir)
    if load_usd is not None:
        if not open_stage(str(load_usd)):
            raise RuntimeError(f"Failed to open saved USD stage: {load_usd}")
        print(f"[SceneV0] loaded USD: {load_usd}", flush=True)
    else:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env_cfg.seed = args_cli.seed
        if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
            env_cfg.events.randomize_cube_positions = None
        env_cfg.viewer.eye = (2.0, -2.0, 1.4)
        env_cfg.viewer.lookat = (0.45, 0.0, 0.15)
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset(seed=args_cli.seed)
    _pause_timeline_for_layout()

    if not args_cli.headless and not args_cli.disable_collision_debug_vis:
        enable_collision_debug_visualization()

    window = None if args_cli.headless else LayoutWindow(save_dir)

    while simulation_app.is_running():
        _pause_timeline_for_layout()
        if window is not None:
            window.update()
        simulation_app.update()

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
