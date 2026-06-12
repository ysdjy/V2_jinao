# Action Adapters

Convert a canonical `Action` (delta-EE 7D or joint) into the action vector the
IsaacLab env expects, after passing through the safety filter.

| file | purpose |
|------|---------|
| `delta_ee_to_isaac_action.py` | delta-EE → IK-Rel (7D) or IK-Abs (8D, integrated) |
| `joint_action_adapter.py` | joint_position / joint_delta → 8D joint+gripper |
| `safety_filter.py` | clamp/scrub before sim: position/rot norms, gripper, joints, workspace, NaN/Inf, error-streak reset |

Env-kind detection (by task name) lives in `scripts/isaaclab/isaac_obs_utils.py`:
- `*IK-Rel*` → `ik_rel` (7D: `[dx,dy,dz, rx,ry,rz, grip]`)
- `*IK-Abs*` → `ik_abs` (8D: `[x,y,z, qw,qx,qy,qz, grip]`)
- `*Joint*`  → `joint` (8D: `[q1..q7, grip]`)

Safety limits are in `configs/safety_limits.yaml`. Defaults: position ≤0.03 m/step
(L2), rotation ≤0.15 rad/step, gripper∈[-1,1], joint clamps, workspace box,
5 consecutive policy errors → episode reset. Every clip is counted and reported in
the eval summary (`num_safety_clips`).
