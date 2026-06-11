# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position variant of the modified Franka stack scene for the joint-action state machine.

This config is intentionally close to :class:`stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg`
but changes the arm :class:`JointPositionActionCfg` ``scale`` to ``1.0`` so that the action
contract matches Isaac Lab's official Franka open-drawer PPO policy (which trains with
``scale=1.0, use_default_offset=True``).

The custom cabinet's ``BottomHandleProxy`` is NOT a separate rigid body, so it cannot be the
target of a FrameTransformer. The drawer observation adapter therefore derives the handle world
pose from the cabinet ``link_1`` body pose plus the (scaled) proxy local offset. See
``skill_runtime/drawer_obs_adapter.py``.

All skills (grasp / place via internal IK, open-drawer via the learned policy) drive this single
joint-position environment.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_joint_pos_env_cfg


@configclass
class FrankaCubeStackJointPolicyEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Joint-position stack scene tuned to match the official drawer-policy action contract."""

    def __post_init__(self):
        # post init of parent (builds the full stack + cabinet + knife scene)
        super().__post_init__()

        # Match the official open-drawer PPO action scale (1.0) so a learned joint policy
        # and the internal IK skills share one action contract on this env.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
