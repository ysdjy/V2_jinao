# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selected-drawer MDP terms (obs / reward / event / termination) for the custom drawer RL task.

Every episode a target drawer is sampled (among the functional drawers). All observation, reward and
success terms reference only the SELECTED drawer's joint + handle frame, so a single policy learns to
open whichever drawer is selected. The selected joint id / handle-frame row are stored on the env by
the reset event and read by the term functions.

Obs layout matches the official Franka open-drawer policy (31-d, same order) so an official
checkpoint can be loaded for fine-tuning:
    joint_pos_rel(9) | joint_vel_rel(9) | sel_drawer_joint_pos(1) | sel_drawer_joint_vel(1) |
    sel_handle_pos - tcp(3) | last_action(8)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from .custom_drawer_config import DRAWER_TARGETS, FUNCTIONAL_DRAWERS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

DRAWER_FRAMES_SENSOR = "drawer_frames"  # FrameTransformer with one frame per functional drawer (in FUNCTIONAL_DRAWERS order)


# ----------------------------------------------------------------------------------------
# selected-drawer per-env state (set by the reset event, read by obs/reward/termination)
# ----------------------------------------------------------------------------------------
def _func_joint_ids(env) -> torch.Tensor:
    """Cabinet joint ids for the functional drawers, in FUNCTIONAL_DRAWERS order (cached on env)."""
    if getattr(env, "_func_joint_ids", None) is None:
        cabinet = env.scene["cabinet"]
        names = list(cabinet.data.joint_names)
        ids = [names.index(DRAWER_TARGETS[d]["joint_name"]) for d in FUNCTIONAL_DRAWERS]
        env._func_joint_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
    return env._func_joint_ids


def _ensure_selected(env) -> None:
    if getattr(env, "_sel_frame_row", None) is None:
        env._sel_frame_row = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._sel_joint_id = _func_joint_ids(env)[0].repeat(env.num_envs).clone()


def reset_select_drawer(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Reset all cabinet drawers to closed and sample a new selected drawer for ``env_ids``."""
    _ensure_selected(env)
    cabinet = env.scene["cabinet"]
    func_ids = _func_joint_ids(env)

    # close all cabinet drawer joints for these envs
    joint_pos = cabinet.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    cabinet.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # sample a functional drawer per env
    rows = torch.randint(0, len(FUNCTIONAL_DRAWERS), (len(env_ids),), device=env.device)
    env._sel_frame_row[env_ids] = rows
    env._sel_joint_id[env_ids] = func_ids[rows]


def _arange(env) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device)


def _selected_handle_pos(env) -> torch.Tensor:
    _ensure_selected(env)
    pos = env.scene[DRAWER_FRAMES_SENSOR].data.target_pos_w  # (N, F, 3)
    return pos[_arange(env), env._sel_frame_row]


def _selected_handle_quat(env) -> torch.Tensor:
    _ensure_selected(env)
    quat = env.scene[DRAWER_FRAMES_SENSOR].data.target_quat_w  # (N, F, 4)
    return quat[_arange(env), env._sel_frame_row]


def _selected_drawer_joint_pos(env) -> torch.Tensor:
    _ensure_selected(env)
    return env.scene["cabinet"].data.joint_pos[_arange(env), env._sel_joint_id]


def _selected_drawer_joint_vel(env) -> torch.Tensor:
    _ensure_selected(env)
    return env.scene["cabinet"].data.joint_vel[_arange(env), env._sel_joint_id]


def _tcp_pos(env) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :]


# ----------------------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------------------
def selected_drawer_joint_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _selected_drawer_joint_pos(env).unsqueeze(-1)


def selected_drawer_joint_vel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _selected_drawer_joint_vel(env).unsqueeze(-1)


def selected_rel_ee_drawer_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """selected_handle_pos - tcp_pos (matches official rel_ee_drawer_distance, but for the selected drawer)."""
    return _selected_handle_pos(env) - _tcp_pos(env)


# ----------------------------------------------------------------------------------------
# rewards (selected versions of the official cabinet rewards)
# ----------------------------------------------------------------------------------------
def selected_approach_ee_handle(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    distance = torch.norm(_selected_handle_pos(env) - _tcp_pos(env), dim=-1, p=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def selected_align_ee_handle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = _selected_handle_quat(env)
    ee_mat = matrix_from_quat(ee_quat)
    h_mat = matrix_from_quat(handle_quat)
    handle_x, handle_y = h_mat[..., 0], h_mat[..., 1]
    ee_x, ee_z = ee_mat[..., 0], ee_mat[..., 2]
    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def _fingertips(env):
    f = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    return f[..., 0, :], f[..., 1, :]


def selected_align_grasp_around_handle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    handle_pos = _selected_handle_pos(env)
    f0, f1 = _fingertips(env)
    return (torch.minimum(f0[:, 2], f1[:, 2]) < handle_pos[:, 2]) & (
        torch.maximum(f0[:, 2], f1[:, 2]) > handle_pos[:, 2]
    )


def selected_approach_gripper_handle(env: "ManagerBasedRLEnv", offset: float = 0.04) -> torch.Tensor:
    handle_pos = _selected_handle_pos(env)
    f0, f1 = _fingertips(env)
    d0 = torch.abs(f0[:, 2] - handle_pos[:, 2])
    d1 = torch.abs(f1[:, 2] - handle_pos[:, 2])
    graspable = (torch.minimum(f0[:, 2], f1[:, 2]) < handle_pos[:, 2]) & (
        torch.maximum(f0[:, 2], f1[:, 2]) > handle_pos[:, 2]
    )
    return graspable * ((offset - d0) + (offset - d1))


def selected_grasp_handle(
    env: "ManagerBasedRLEnv", threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    distance = torch.norm(_selected_handle_pos(env) - _tcp_pos(env), dim=-1, p=2)
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return (distance <= threshold) * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def selected_open_drawer_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    drawer_pos = _selected_drawer_joint_pos(env)
    graspable = selected_align_grasp_around_handle(env).float()
    return (graspable + 1.0) * drawer_pos


def selected_multi_stage_open_drawer(env: "ManagerBasedRLEnv") -> torch.Tensor:
    drawer_pos = _selected_drawer_joint_pos(env)
    graspable = selected_align_grasp_around_handle(env).float()
    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.20) * graspable
    open_hard = (drawer_pos > 0.30) * graspable
    return open_easy + open_medium + open_hard


def wrong_drawer_open_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize opening any functional drawer that is NOT the selected one."""
    _ensure_selected(env)
    cabinet = env.scene["cabinet"]
    func_ids = _func_joint_ids(env)
    total = torch.zeros(env.num_envs, device=env.device)
    for k in range(len(FUNCTIONAL_DRAWERS)):
        jid = func_ids[k]
        pos = cabinet.data.joint_pos[:, jid].clamp(min=0.0)
        is_other = env._sel_joint_id != jid
        total = total + is_other.float() * pos
    return total


# ----------------------------------------------------------------------------------------
# termination
# ----------------------------------------------------------------------------------------
def selected_drawer_opened(env: "ManagerBasedRLEnv", threshold: float = 0.20) -> torch.Tensor:
    return _selected_drawer_joint_pos(env) >= threshold
