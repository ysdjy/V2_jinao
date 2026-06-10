# Copyright (c) 2026. Connection project.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_handle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The vector from the end-effector to the fridge door handle."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    handle_tf_data: FrameTransformerData = env.scene["fridge_frame"].data
    return handle_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    return ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins
