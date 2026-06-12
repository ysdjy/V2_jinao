# `gello_isaac_teleop` — GELLO → IsaacLab Franka teleop module

A small, reusable module that connects a **GELLO leader arm** to a **Franka** robot inside
**IsaacLab/IsaacSim**, for teleoperation and demonstration data collection. It is
**scene-agnostic**: the scene/objects can change between tasks — as long as the env contains a
Franka articulation (`robot`) with `panda_joint.*` joints, this module drives it.

> Built from the validated teleop logic in `skill_test_ui_joint_gello.py`, refactored into an
> importable package so any project can reuse it.

---

## What it does (and the 3 fixes baked in)

- Reads GELLO joints over Dynamixel serial in a **background thread** (never stalls your sim loop).
- Produces a **safe Franka joint command**: low-pass smoothing + joint-limit clip + rate limit +
  NaN/dropout guards + **no startup jump** (`q_cmd` starts at the robot's current pose).
- Maps the **gripper** (raw GELLO angle in degrees → binary open/close with hysteresis).

Fixes vs. the earlier ad-hoc script:
1. **Following no longer slows down at low sim FPS.** Smoothing (`smoothing_tau`, seconds) and
   rate limit (`max_joint_vel`, rad/s) are scaled by the real per-step `dt`, so the arm keeps up
   whether the GUI runs at 4 Hz or 60 Hz. (Old code used a fixed per-step delta → crawled at low FPS.)
2. **Frequency:** the serial read is decoupled in a thread (~57 Hz) so it never gates the loop;
   the remaining limit is IsaacSim render/physics — see *Performance*.
3. **Gripper works by default** (`enable_gripper=True`) with the calibrated 211°/169° mapping.

---

## Requirements

- IsaacLab + the env's conda env (`env_isaaclab`); a Franka joint-position env.
- The GELLO deps in that env: `pip install -e projects/gello_franka_teleop/third_party/gello_software`
  (and its `DynamixelSDK/python`), plus `pyyaml`. (Already installed for `env_isaaclab`.)
- A configured `configs/gello_franka.yaml` (port + calibrated offsets/signs/gripper).
- **Serial permission:** the GELLO port is in the `dialout` group. If your shell lacks it, run the
  launch command via `sg dialout -c "<cmd>"` (or re-login once after `usermod -aG dialout`).

---

## Quick start

### A) Joint-position env — one-call facade (recommended)
```python
import sys; sys.path.insert(0, "projects/gello_franka_teleop")   # make the module importable
from gello_isaac_teleop import GelloFrankaTeleop, GelloTeleopConfig

teleop = GelloFrankaTeleop(env, GelloTeleopConfig(
    gello_config="projects/gello_franka_teleop/configs/gello_franka.yaml",
    enable_gripper=True,          # gripper on by default
    max_joint_vel=2.5,            # following speed cap (rad/s)
    smoothing_tau=0.08,           # smoothing time constant (s)
))
teleop.start()                    # opens device, seeds q_cmd at current Franka q (no jump)
while sim_running:
    action = teleop.step()        # reads GELLO, returns the env action (joint targets + gripper)
    env.step(action)
teleop.stop()
# after an env.reset(): teleop.reseat()   # re-seat q_cmd at the new robot pose
```

### B) Non-joint-position env (e.g. IK-relative) — bring your own action mapping
```python
q_cmd, gripper_cmd = teleop.compute_joint_targets()   # 7 arm targets (rad) + binary gripper
# map (q_cmd, gripper_cmd) to your env's action space yourself, then env.step(...)
```
> For an **IK-Rel** task you also need GELLO-joints → forward-kinematics → EE-pose-delta. That
> bridge is not in this module yet (this module commands *joint* targets).

### C) Device only (no IsaacLab) — read GELLO anywhere
```python
from gello_isaac_teleop import GelloReader, ThreadedGelloReader
r = ThreadedGelloReader(GelloReader("…/gello_franka.yaml"), hz=60).start()
arm_q, gripper_deg, ok, read_ms, read_hz = r.get_latest()
```

### Runnable demo / smoke test
```bash
# GUI (watch the Franka follow GELLO):
sg dialout -c 'DISPLAY=:1 ./isaaclab.sh -p \
  projects/gello_franka_teleop/gello_isaac_teleop/examples/run_gello_teleop_demo.py \
  --task Isaac-Stack-Cube-Franka-JointPolicy-v0'

# Headless bounded smoke:
sg dialout -c './isaaclab.sh -p \
  projects/gello_franka_teleop/gello_isaac_teleop/examples/run_gello_teleop_demo.py \
  --task <any Franka joint-pos env> --headless --max_steps 200'
```

---

## API summary

| symbol | purpose |
|--------|---------|
| `GelloFrankaTeleop(env, cfg)` | facade; `.start() .step()/.compute_joint_targets() .reseat() .telemetry() .stop()` |
| `GelloTeleopConfig` | config dataclass (paths, gains, gripper, env-binding overrides) |
| `IsaacFrankaBinding(env, …)` | find Franka arm joints, read q/limits, build joint action (generic introspection) |
| `TeleopController(num_arm, lo, hi, cfg)` | framework-agnostic smoothing/safety; `.reset() .step()` |
| `ControllerCfg` / `GripperMapCfg` | gains / gripper mapping |
| `GelloReader` / `ThreadedGelloReader` | device reading (no IsaacLab dep) |

Key `GelloTeleopConfig` fields: `gello_config`, `read_hz=60`, `smoothing_tau=0.08`,
`max_joint_vel=2.5`, `abs_step_cap=0.6`, `enable_gripper=True`,
`gripper_open_raw=211/close_raw=169/deadband=3/invert=False`, plus env-binding overrides
(`robot_key="robot"`, `arm_joint_pattern="panda_joint.*"`, `arm_action_term`, `gripper_action_term`).

`telemetry()` returns: `real_loop_hz, read_hz, read_ms, q_gello, q_target, q_cmd, franka_q,
gripper_raw, gripper_cmd, gripper_state`.

---

## Using it for data collection

This module commands the robot; pair it with your project's recorder. Two patterns:
- **Custom collection loop:** in your loop, `action = teleop.step()`, `env.step(action)`, and let
  your env's recorder (e.g. IsaacLab `ActionStateRecorderManagerCfg`) write HDF5.
- **IsaacLab `record_demos.py`:** its teleop device system expects a `DeviceBase` returning an SE3
  twist; to plug GELLO in there you'd wrap this module in a thin `DeviceBase` adapter (extension
  point — not included). For joint-action collection, the custom-loop pattern is simplest.

---

## Performance (the "too slow / low Hz" question)

- Following speed is now **frame-rate independent** — at low FPS each step moves more
  (`max_joint_vel × dt`, capped by `abs_step_cap`). Raise `max_joint_vel` for snappier following,
  lower `smoothing_tau` for less lag (at the cost of smoothness).
- The remaining frame-rate limit is **IsaacSim render/physics**, not GELLO. To raise sim Hz:
  - close other GPU jobs (a second heavy job halves your FPS),
  - prefer a lighter scene for teleoperation/testing,
  - run on a dedicated GPU (`--device cuda:1`), reduce viewport resolution, or test headless.

---

## Troubleshooting

| symptom | fix |
|---------|-----|
| `fake driver` / `Operation not permitted` on serial | run via `sg dialout -c "…"`, or re-login after adding to `dialout` |
| a joint moves the **wrong way** | flip its sign in `configs/gello_franka.yaml` → `joint_signs` (never hardcode in code) |
| gripper won't close | check `gripper_open_raw/close_raw` vs. your calibration; squeeze fully so raw crosses `threshold ± deadband`; try `gripper_invert` |
| `No joint-position arm action term found` | the env is IK/other action space — use `compute_joint_targets()` + your own mapping |
| arm lunges at startup | it won't — `q_cmd` seeds at the robot pose; if a big startup gap is warned, hold GELLO near the robot's pose |
