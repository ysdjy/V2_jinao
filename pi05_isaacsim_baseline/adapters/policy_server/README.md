# Policy Server

Isolates IsaacLab (sim/collect/eval) from OpenPI (model/training). They talk over
plain HTTP JSON so they can live in different python environments.

```
IsaacLab conda env                    OpenPI venv (.venv_openpi)
  run_policy_in_isaaclab.py             server.py --backend openpi
        │  Observation JSON  POST /infer        │
        └──────────────►  HTTP :8008  ◄─────────┘
                          Action JSON
```

## Files
| file | role | deps |
|------|------|------|
| `schemas.py` | unified Observation / Action dataclasses + units | stdlib |
| `server.py` | HTTP server, `/health` + `/infer`, mock/openpi backends | stdlib (mock) |
| `client.py` | IsaacLab-side client, safe fallback on timeout/error | stdlib |
| `mock_policy.py` | deterministic safe actions, no model | stdlib(+numpy opt) |
| `openpi_policy.py` | wraps a real pi0/pi0.5 policy | openpi venv |

## Run

Mock (any python):
```bash
python adapters/policy_server/server.py --backend mock --port 8008
curl localhost:8008/health
```

Real pi0.5 (OpenPI venv):
```bash
.venv_openpi/bin/python adapters/policy_server/server.py \
    --backend openpi --config pi05_isaaclab_franka \
    --ckpt policies/checkpoints/.../10 --port 8008
```

## Conventions (must match the action adapters)
- positions/lengths: **meters**; rotations: **radians**; quaternion order **XYZW**.
- gripper *action*: normalized **[-1, 1]** (-1 close, +1 open).
- canonical action vector: `[dx, dy, dz, rx, ry, rz, gripper]` (rotation = axis-angle).
