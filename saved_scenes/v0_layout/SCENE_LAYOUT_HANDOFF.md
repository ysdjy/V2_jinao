# Scene Layout V0 Handoff

This document describes the independent scene-layout workflow in this IsaacLab
workspace. It is written for another AI or engineer that needs to consume a
saved scene and recreate it inside a separate task.

## Purpose

The scene-layout tool is only for editing and exporting scene geometry state:
asset paths, prim paths, positions, orientations, scales, and coarse physics or
articulation indicators. It does not define a training task, reward function,
observation space, policy, controller, or robot joint state.

The intended workflow is:

1. Open the V0 layout tool.
2. Move, rotate, scale, or add scene assets in Isaac Sim.
3. Click `Save Both`.
4. Give the saved USD and JSON manifest to another AI.
5. The other AI restores the scene inside its own independent task.

## Files

- Layout UI entrypoint:
  `scripts/environments/state_machine/scene_v0_layout_ui.py`
- Default output directory:
  `saved_scenes/v0_layout/`
- Saved scene snapshot:
  `scene_v0_<timestamp>.usd`
- Programmatic restore manifest:
  `scene_v0_<timestamp>.json`
- Human-readable report:
  `scene_v0_<timestamp>_report.md`

The currently confirmed saved scene is:

- `saved_scenes/v0_layout/scene_v0_20260611_191056.usd`
- `saved_scenes/v0_layout/scene_v0_20260611_191056_report.md`

## Independence Contract

This layout module must stay independent from downstream training tasks.

The layout module may:

- Load the current project scene as a visual/layout base.
- Pause physics while the user edits the scene.
- Add USD assets as references.
- Export a USD snapshot.
- Export a JSON manifest that describes what was edited.

The layout module must not:

- Register a new training task.
- Modify the downstream task's reward, observations, actions, or terminations.
- Depend on another AI's code.
- Require another task to import this script at runtime.

Downstream tasks should treat the saved USD/JSON as data, not as code.

## Running The Layout Tool

Use the existing IsaacLab environment on this machine:

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/scene_v0_layout_ui.py --num_envs 1
```

Load the latest saved layout:

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/scene_v0_layout_ui.py --num_envs 1 --load_latest_saved
```

Load a specific saved layout:

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/scene_v0_layout_ui.py --num_envs 1 --load_usd saved_scenes/v0_layout/scene_v0_20260611_191056.usd
```

## Save Outputs

Clicking `Save USD` writes a full stage snapshot:

```text
saved_scenes/v0_layout/scene_v0_<timestamp>.usd
```

Clicking `Save JSON` writes a restore manifest:

```text
saved_scenes/v0_layout/scene_v0_<timestamp>.json
```

Clicking `Save Both` writes both. This is the preferred operation.

## JSON Manifest Contract

The JSON manifest uses:

```json
{
  "schema_version": "scene-layout-manifest-v1",
  "task": "...",
  "env_root": "/World/envs/env_0",
  "up_axis": "Z",
  "meters_per_unit": 1.0,
  "objects": []
}
```

Each object contains:

- `path`: target USD prim path.
- `name`: prim name.
- `type`: USD prim type.
- `active`, `loaded`, `visibility`: scene state.
- `xform.translate`: authored local position `[x, y, z]`.
- `xform.orient_wxyz`: authored local orientation quaternion `[w, x, y, z]`.
- `xform.scale`: authored local scale `[x, y, z]`.
- `xform.ordered_ops`: exact authored xform op order and values.
- `subtree_assets`: references and payloads found under the object subtree.
- `applied_schemas`: top-level applied USD schemas.
- `physics`: coarse physics/articulation/collision indicators.
- `world_transform`: world-space transform summary for inspection.

For programmatic restore, use `xform.ordered_ops` or at minimum apply:

1. `xform.translate`
2. `xform.orient_wxyz`
3. `xform.scale`

to the object at `path`.

Robot joint positions are intentionally not part of this contract.

## Recommended Restore Strategy For Another AI

The downstream AI has two valid options.

Option A, easiest and most faithful:

1. Copy the saved USD plus all referenced asset directories.
2. Open the saved USD as a scene asset or reference it into the downstream task.
3. Add that task's own robot/action/observation/reward logic separately.

Option B, programmatic reconstruction:

1. Load the JSON manifest.
2. For each object in `objects`, create or reference the asset at `path`.
3. Resolve `subtree_assets` so referenced USD/payload files are available.
4. Apply `xform.ordered_ops` exactly.
5. Ignore robot joint positions unless the downstream task explicitly sets them.

## External Asset Requirement

The saved USD is not guaranteed to be a self-contained bundle. The downstream AI
must preserve external asset paths or copy the assets and rewrite references.

For the confirmed saved scene, important asset roots include:

- `Connection/assets/Isaac/IsaacLab/Robots/FrankaEmika/`
- `simv2/USD/Cabinet_44853/`
- `simv2/USD/Knife_101054/`
- `SapienAssetPipeline/usd_assets/CoffeeMachine_103046/`

If these are missing, the saved scene may load but some meshes, collision shapes,
or articulation data may be unresolved.

## Known Scene Notes

The confirmed scene has a coffee machine prim at:

```text
/coffeemachine
```

not under:

```text
/World/envs/env_0
```

If a downstream IsaacLab task assumes all objects live under the env namespace,
move or reference this prim under `/World/envs/env_0` during restore, or explicitly
handle `/coffeemachine` as a global scene object.

During local loading, Isaac Sim warned about some internal visual references for
the coffee machine and cabinet. The scene still loaded, but a downstream AI should
check that all referenced asset directories are available.

## What Another AI Should Not Infer

The saved layout does not define:

- Control algorithm.
- Policy checkpoint.
- Reward function.
- Observation manager.
- Termination conditions.
- Robot joint positions.
- Episode reset logic.

Those belong to the downstream training task. This module only provides scene
layout data that the downstream task can consume.
