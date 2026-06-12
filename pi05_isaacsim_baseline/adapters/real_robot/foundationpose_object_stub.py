"""STUB: FoundationPose 6D object pose -> Observation.objects entries.

Defines the contract for injecting estimated 6D object poses into the unified
observation, both for sim (privileged ground-truth) and real (FoundationPose).

Real implementation TODO:
  * run FoundationPose given RGB-D + object mesh + initial mask
  * output pose_in_camera (4x4) and pose_in_world (apply extrinsics)
  * fill confidence; keep `name` stable across frames for tracking
"""

from __future__ import annotations

from typing import Any


def make_object_entry(
    name: str,
    position_world,        # [x,y,z] m
    quat_world_xyzw,       # [x,y,z,w]
    confidence: float = 1.0,
    pose_in_camera=None,   # optional 4x4 (flattened, row-major)
    mesh_path: str | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "position": [float(v) for v in position_world],
        "quat": [float(v) for v in quat_world_xyzw],
        "confidence": float(confidence),
        "pose_in_camera": list(pose_in_camera) if pose_in_camera is not None else None,
        "mesh_path": mesh_path,
        "mask_path": mask_path,
    }


class FoundationPoseStub:
    def estimate(self, rgb, depth, intrinsics, mesh_path, init_mask) -> dict[str, Any]:
        raise NotImplementedError(
            "Run FoundationPose here; return make_object_entry(...). "
            "Then append the result to observation['objects']."
        )
