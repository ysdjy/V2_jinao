"""gello_isaac_teleop — connect a GELLO leader arm to an IsaacLab Franka (scene-agnostic).

Public API:
    GelloFrankaTeleop, GelloTeleopConfig   — one-call facade for a sim loop
    TeleopController, ControllerCfg, GripperMapCfg   — framework-agnostic smoothing/safety
    GelloReader, ThreadedGelloReader       — device reading (no IsaacLab dependency)
    IsaacFrankaBinding                     — env glue (find Franka, build joint action)

See USAGE.md for the integration guide.
"""

from .controller import ControllerCfg, GripperMapCfg, TeleopController
from .reader import GelloReader, ThreadedGelloReader

# IsaacLab-dependent pieces are imported lazily so `reader`/`controller` work without isaaclab.
__all__ = [
    "GelloFrankaTeleop",
    "GelloTeleopConfig",
    "TeleopController",
    "ControllerCfg",
    "GripperMapCfg",
    "GelloReader",
    "ThreadedGelloReader",
    "IsaacFrankaBinding",
]


def __getattr__(name):  # lazy import to avoid requiring isaaclab/torch for the device-only pieces
    if name in ("GelloFrankaTeleop", "GelloTeleopConfig"):
        from .teleop import GelloFrankaTeleop, GelloTeleopConfig
        return {"GelloFrankaTeleop": GelloFrankaTeleop, "GelloTeleopConfig": GelloTeleopConfig}[name]
    if name == "IsaacFrankaBinding":
        from .isaac_binding import IsaacFrankaBinding
        return IsaacFrankaBinding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
