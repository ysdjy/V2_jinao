# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Task definitions. Importing the subpackages registers the gym environments."""

from .open_drawer.config import franka  # noqa: F401  (triggers gym.register)
from .open_fridge.config import franka as _fridge_franka  # noqa: F401  (triggers gym.register)
from .multi_skill.config import franka as _multi_franka  # noqa: F401  (triggers gym.register)
