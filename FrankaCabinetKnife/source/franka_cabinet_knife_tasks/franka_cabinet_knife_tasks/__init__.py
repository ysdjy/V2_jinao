# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""FrankaCabinetKnife Isaac Lab tasks.

Importing this package verifies the local assets and registers the Gym tasks.
"""

from .assets_paths import verify_assets

verify_assets()

from . import tasks  # noqa: E402,F401

