"""Object mesh + object-asset config loading.

Reads ``configs/object_assets.yaml`` and resolves per-object mesh / mask paths.
``trimesh`` is only imported when an actual mesh load is requested, so the
config side of this module is usable on the IsaacLab side too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..utils.config import load_yaml, resolve_path


@dataclass
class ObjectAsset:
    name: str
    mesh_path: str
    default_mask_path: str | None = None
    diameter: float | None = None
    symmetry: object | None = None


def load_object_assets(config_path: str) -> dict:
    """Load object_assets.yaml -> {name: ObjectAsset} with resolved paths."""
    cfg = load_yaml(config_path)
    base = os.path.dirname(os.path.abspath(config_path))
    # configs/ lives directly under the project root.
    project_root = os.path.dirname(base)
    objects = cfg.get("objects", {}) or {}
    out = {}
    for name, spec in objects.items():
        spec = spec or {}
        mesh = spec.get("mesh_path")
        mask = spec.get("default_mask_path")
        out[name] = ObjectAsset(
            name=name,
            mesh_path=resolve_path(mesh, project_root) if mesh else None,
            default_mask_path=resolve_path(mask, project_root) if mask else None,
            diameter=spec.get("diameter"),
            symmetry=spec.get("symmetry"),
        )
    return out


def get_object_asset(config_path: str, object_name: str) -> ObjectAsset:
    """Return the ObjectAsset for ``object_name`` with a clear error if absent."""
    assets = load_object_assets(config_path)
    if object_name not in assets:
        raise KeyError(
            f"object '{object_name}' not found in {config_path}. "
            f"Available: {sorted(assets.keys())}"
        )
    return assets[object_name]


def load_mesh(mesh_path: str):
    """Load a mesh with trimesh, with a clear error if the file is missing."""
    if mesh_path is None:
        raise ValueError("mesh_path is None; set it in configs/object_assets.yaml")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(
            f"mesh not found: {mesh_path}\n"
            "Place your target object's CAD mesh under assets/meshes/ and point "
            "configs/object_assets.yaml at it (units must be METERS for "
            "FoundationPose)."
        )
    try:
        import trimesh
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "trimesh is required to load meshes (FoundationPose side). "
            "Install with: pip install trimesh"
        ) from e
    mesh = trimesh.load(mesh_path, force="mesh")
    return mesh
