# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""KitchenV0: custom Isaac Lab kitchen tasks.

Importing this package registers all Gym environments defined by the project.
"""

from .assets_paths import verify_assets

verify_assets()

from . import tasks  # noqa: E402,F401
