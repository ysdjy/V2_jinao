"""YAML config loading helpers with clear errors and path resolution."""

from __future__ import annotations

import os

import yaml

# Project root = the directory that contains this package and `configs/`.
# utils/config.py -> utils -> franka_d435_foundationpose (pkg) -> project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


def project_path(*parts) -> str:
    """Join paths relative to the project root."""
    return os.path.join(PROJECT_ROOT, *parts)


def load_yaml(path: str) -> dict:
    """Load a YAML file into a dict, raising a clear error if missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str, base: str | None = None) -> str:
    """Resolve ``path``; relative paths are taken w.r.t. ``base`` or PROJECT_ROOT."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    base = base or PROJECT_ROOT
    return os.path.abspath(os.path.join(base, path))


def resolve_cli_path(path: str) -> str:
    """Resolve a path typed by the user on the command line.

    Unlike :func:`resolve_path` (which resolves config-relative paths against the
    project root), CLI paths are resolved against the current working directory —
    matching shell intuition. The documented commands run from the IsaacLab root
    and pass paths like ``franka_d435_foundationpose/outputs/...``.
    """
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def default_config_path(name: str) -> str:
    """Path of a file inside the project's ``configs/`` directory."""
    return project_path("configs", name)
