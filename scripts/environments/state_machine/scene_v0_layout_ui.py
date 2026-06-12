# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility entrypoint for the standalone scene layout module."""

from __future__ import annotations

import runpy
from pathlib import Path


MODULE_ENTRYPOINT = Path(__file__).resolve().parents[3] / "SceneLayoutModule" / "scene_layout_ui.py"


if __name__ == "__main__":
    runpy.run_path(str(MODULE_ENTRYPOINT), run_name="__main__")
