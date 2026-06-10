# KitchenV0

Clean Isaac Lab extension for the first kitchen scene prototype.

Current focus:

- Calibrate a clean Franka + refrigerator scene.
- Open the refrigerator door with a state machine.
- Keep microwave, cabinet and knife assets available but out of the active scene.

Original multi-object V0 goal:

- Franka at the center.
- Refrigerator on the robot's left, with a revolute door joint.
- Microwave on the robot's right, loaded from the local PartNet URDF with a revolute door joint.
- Cabinet behind the robot, using the local Sektion cabinet USD with drawers.
- Knife in front of the robot.
- No controller/state machine yet. The scene should load and run with zero actions.

## Install

From the IsaacLab repository root:

```bash
conda activate env_isaaclab
bash KitchenV0/scripts/setup_env.sh
```

The extension resolves assets in this order:

1. `KITCHEN_V0_ASSETS_DIR`, if set.
2. `KitchenV0/assets`.
3. `Connection/assets`, `Connection/USD/7320`, and `Connection/USD/101054` as a compatibility fallback.

`KitchenV0/assets` is already populated with the small offline assets needed by v0. The PartNet assets are cleaned with `KitchenV0/tools/clean_partnet_asset.py`, which extends the older `Connection/tools/clean_partnet_urdf.py` flow by also removing URDF mesh references that point to missing OBJ files.

## Run V0 Scene

Fridge-only calibration scene:

```bash
./isaaclab.sh -p KitchenV0/scripts/zero_agent.py --task Kitchen-Fridge-Franka-IK-Abs-Play-v0 --num_envs 1
```

Open the fridge with the state machine:

```bash
./isaaclab.sh -p KitchenV0/scripts/state_machine/open_fridge_sm.py --num_envs 1
```

Original combined kitchen scene:

```bash
./isaaclab.sh -p KitchenV0/scripts/zero_agent.py --task Kitchen-V0-Franka-IK-Abs-Play-v0 --num_envs 1
```

Run the full sequential scaffold in one environment:

```bash
./isaaclab.sh -p KitchenV0/scripts/kitchen_sequence_sm.py --num_envs 1 --fridge_angle 45
```

This v0 sequence does not reset between subtasks. It opens the fridge, opens the microwave, opens the bottom drawer, places the knife into the drawer, closes the drawer, then closes the microwave inside the same scene/episode. In v0 these are direct asset-level commands; the next version should replace each state with Franka IK motions.

Quick registration check without launching a full simulation:

```bash
bash KitchenV0/scripts/smoke_test.sh
```

Use the joint-position variant when you only need robot joint actions:

```bash
./isaaclab.sh -p KitchenV0/scripts/zero_agent.py --task Kitchen-V0-Franka-Play-v0 --num_envs 1
```

## State-Machine Targets

The scene config exposes constants for the future state machine:

- `FRIDGE_DOOR_JOINT = "joint_0"`
- `MICROWAVE_DOOR_JOINT = "joint_0"`
- `CABINET_BOTTOM_DRAWER_JOINT = "drawer_bottom_joint"`
- `FRIDGE_OPEN_15_DEG`, `FRIDGE_OPEN_45_DEG`
- `MICROWAVE_OPEN_45_DEG`
- `CABINET_BOTTOM_DRAWER_OPEN_POS`

`KitchenV0/scripts/kitchen_sequence_sm.py` already defines the full order:

1. open fridge to 15 or 45 degrees,
2. open microwave,
3. open the bottom drawer,
4. move knife into the drawer,
5. close the drawer,
6. close microwave.
