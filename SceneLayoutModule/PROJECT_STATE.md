# Scene Layout Project State

This IsaacLab workspace now contains an independent V0 scene-layout workflow.
It exists only to edit a scene and export the edited layout so another AI can
restore it in a separate task.

## Current Goal

The user wants to edit assets in a scene, including position, orientation, scale,
and newly added USD assets. After clicking save, another AI should be able to
quickly restore the saved scene in its own task without depending on this
project's controllers or training code.

## Independent Tool

Entrypoint:

```text
SceneLayoutModule/scene_layout_ui.py
```

This script is intentionally independent from the active state-machine debugging
scripts. It does not modify:

- `scripts/environments/state_machine/skill_test_ui_joint.py`
- downstream reward logic
- downstream observation/action/termination managers
- policy or controller code

It loads a base scene, pauses the timeline for layout editing, lets the user add
USD assets, and saves the result as data.

## How To Run

The working conda environment on this machine is:

```bash
conda activate env_isaaclab
```

Start a fresh editable scene:

```bash
./isaaclab.sh -p SceneLayoutModule/scene_layout_ui.py --num_envs 1
```

Load the latest saved scene:

```bash
./isaaclab.sh -p SceneLayoutModule/scene_layout_ui.py --num_envs 1 --load_latest_saved
```

Load a specific saved scene:

```bash
./isaaclab.sh -p SceneLayoutModule/scene_layout_ui.py --num_envs 1 --load_usd SceneLayoutModule/saved_scenes/scene_v0_20260611_191056.usd
```

## Save Contract

Use `Save Both` in the UI. It creates:

- `SceneLayoutModule/saved_scenes/scene_v0_<timestamp>.usd`
- `SceneLayoutModule/saved_scenes/scene_v0_<timestamp>.json`

The USD is the most faithful scene snapshot. The JSON is the programmatic
restore manifest for another AI.

The JSON schema version is:

```text
scene-layout-manifest-v1
```

Each object record contains:

- target `path`
- authored `translate`
- authored `orient_wxyz`
- authored `scale`
- exact ordered USD xform ops
- subtree reference and payload asset paths
- applied schemas
- physics/collision/articulation indicators
- world transform summary

Robot joint positions are intentionally excluded.

## Existing Confirmed Saved Scene

The confirmed saved scene from the user's manual edit is:

```text
SceneLayoutModule/saved_scenes/scene_v0_20260611_191056.usd
```

A human-readable report for it is:

```text
SceneLayoutModule/saved_scenes/scene_v0_20260611_191056_report.md
```

The handoff document for this workflow is:

```text
SceneLayoutModule/HANDOFF.md
```

## How Another AI Should Consume It

Preferred restore path:

1. Copy the saved USD.
2. Copy all referenced asset directories.
3. Load or reference the saved USD inside the downstream task.
4. Add that task's own robot/action/observation/reward/termination logic.

Programmatic restore path:

1. Load the JSON manifest.
2. Create/reference each object at `objects[*].path`.
3. Apply `objects[*].xform.ordered_ops` exactly.
4. Ensure every `objects[*].subtree_assets[*].asset_path` is available.
5. Ignore robot joint positions unless the downstream task defines them.

## Important Asset Roots

The current saved scene uses assets from:

```text
Connection/assets/Isaac/IsaacLab/Robots/FrankaEmika/
simv2/USD/Cabinet_44853/
simv2/USD/Knife_101054/
SapienAssetPipeline/usd_assets/CoffeeMachine_103046/
```

The saved USD is not guaranteed to be self-contained. If another machine lacks
these directories, it may load the scene with missing visual or collision pieces.

## Boundary

This project should be treated as a scene-data producer. Downstream training
projects should be scene-data consumers. They should read the saved USD/JSON
outputs and reconstruct the scene inside their own environments, without
importing or depending on this layout script.
