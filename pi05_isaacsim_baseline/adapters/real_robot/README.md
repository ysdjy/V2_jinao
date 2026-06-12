# Real Robot Adapters (STUBS)

Interface definitions only — **no hardware control here**. They show how the same
observation/action schema used in sim maps onto a real Franka + D435 +
FoundationPose, so sim→real stays aligned.

| stub | defines |
|------|---------|
| `franka_fci_action_stub.py` | policy action → (future) Franka FCI command, reusing the sim safety filter |
| `d435_observation_stub.py` | RealSense D435 RGB-D → `front_rgb`/`wrist_rgb` ImageRefs |
| `foundationpose_object_stub.py` | FoundationPose 6D pose → `observation.objects[]` |

**Do not** run real-robot control inside the training/conda env. Use a dedicated
real-time process. See `docs/real_franka_d435_migration.md` for the full rationale,
the FR3 FCI setup, and safety requirements.
