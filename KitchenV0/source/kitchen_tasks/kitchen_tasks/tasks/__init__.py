# Copyright (c) 2026. KitchenV0 project.
# SPDX-License-Identifier: BSD-3-Clause

"""Task definitions. Importing subpackages registers Gym environments."""

from .fridge_scene.config import franka as _fridge_scene_franka  # noqa: F401
from .kitchen_scene.config import franka as _kitchen_scene_franka  # noqa: F401
