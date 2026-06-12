# Data Conversion: IsaacLab HDF5 → LeRobot

## Tools
- `inspect_hdf5.py` — recursively print groups/datasets/shapes/dtypes/attrs + samples.
- `hdf5_to_lerobot.py` — convert demos to (1) a **normalized intermediate** dataset
  (always) and (2) a **LeRobot** dataset (if `lerobot` importable).
- `normalized_to_lerobot.py` — second-stage build (run later inside the OpenPI venv).
- `validate_lerobot_dataset.py` — sanity-check either output kind.

Field mapping is **not hardcoded** — candidates come from
`configs/dataset_mapping_isaaclab_franka.yaml`; override with `--state-key` /
`--action-key`. Missing fields are reported, not fatal.

## Typical flow
```bash
# 1. see what's inside (use the IsaacLab env python which has h5py)
python adapters/data_conversion/inspect_hdf5.py --input data/raw_hdf5/demo.hdf5

# 2. convert (writes normalized always; LeRobot if lerobot is available)
python adapters/data_conversion/hdf5_to_lerobot.py \
    --input data/raw_hdf5/demo.hdf5 \
    --output data/lerobot/franka_stack_cube_pi05 \
    --config configs/dataset_mapping_isaaclab_franka.yaml \
    --task_instruction "Stack the cubes with the Franka robot."

# 3. if LeRobot build was skipped (no lerobot at convert time), do it in the OpenPI venv
.venv_openpi/bin/python adapters/data_conversion/normalized_to_lerobot.py \
    --input data/processed/normalized_dataset/franka_stack_cube_pi05 \
    --output data/lerobot/franka_stack_cube_pi05

# 4. validate
python adapters/data_conversion/validate_lerobot_dataset.py \
    --input data/processed/normalized_dataset/franka_stack_cube_pi05
```

## Normalized intermediate layout
```
data/processed/normalized_dataset/<name>/
  metadata.json     field mapping, units, fps, task, action/state dims
  episodes.jsonl    one line per episode
  states.npy        (N, state_dim)
  actions.npy       (N, action_dim)
  episode_index.npy (N,)
  images/<cam>/ep####_step######.png   (only if cameras present)
```

LeRobot features produced: `image`, `wrist_image`, `state`, `actions`, `task`
(OpenPI expects proprio in `state`, actions in `actions`).
