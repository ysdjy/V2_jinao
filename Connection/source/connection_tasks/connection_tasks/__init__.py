# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

"""Connection: custom Isaac Lab tasks.

Importing this package registers all gym environments defined by the project.
"""

from . import tasks  # noqa: F401  (triggers gym.register of all tasks)
