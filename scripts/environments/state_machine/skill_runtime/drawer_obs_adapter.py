"""Build the official Franka open-drawer policy observation from the live scene.

Mirrors ``cabinet_env_cfg.py:ObservationsCfg.PolicyCfg`` (concatenate_terms=True), in order:

    1. joint_pos            = robot joint_pos_rel              (9)
    2. joint_vel            = robot joint_vel_rel              (9)
    3. cabinet_joint_pos    = drawer joint joint_pos_rel       (1)
    4. cabinet_joint_vel    = drawer joint joint_vel_rel       (1)
    5. rel_ee_drawer_distance = handle_pos_w - tcp_pos_w       (3)
    6. actions              = last_action                      (8)
                                                       total = 31

We compute ``handle_pos - tcp_pos`` ourselves: from the ``cabinet_frame`` FrameTransformer
(``drawer_handle_top``) when present, otherwise from the cabinet ``link_1`` body pose combined with
the ``BottomHandleProxy`` local offset.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from .drawer_target_config import DRAWER_TARGETS
from .scene_state_provider import SceneState

# bottom-drawer handle proxy local offset on link_1 (see stack_joint_pos_env_cfg.py).
# The cabinet USD is spawned with scale 0.62, which propagates to this child prim's local
# translate, so we scale the authored offset to recover the world offset from link_1.
_HANDLE_PROXY_LOCAL_OFFSET = (0.11946, 0.01491, 1.06183)
_CABINET_SCALE = 0.62


class DrawerObsAdapter:
    def __init__(self, env, env_id: int = 0, drawer_joint_name: str = "joint_0"):
        self.env = env
        self.env_id = env_id
        self.scene = env.unwrapped.scene
        self.device = self.scene.device
        self.drawer_joint_name = drawer_joint_name
        self.cabinet = self.scene["cabinet"]
        names = list(getattr(self.cabinet.data, "joint_names", []))
        if drawer_joint_name in names:
            self._drawer_joint_id = names.index(drawer_joint_name)
        else:
            ids, _ = self.cabinet.find_joints(drawer_joint_name)
            if not ids:
                raise RuntimeError(f"drawer joint '{drawer_joint_name}' not found; available={names}")
            self._drawer_joint_id = int(ids[0])
        self._has_cabinet_frame = "cabinet_frame" in self.scene.keys()
        self.last_obs_dim: int | None = None

    def _handle_pos_w(self) -> torch.Tensor:
        if self._has_cabinet_frame:
            frame = self.scene["cabinet_frame"]
            return frame.data.target_pos_w[:, 0, :]
        # fallback: link_1 body pose + local handle offset
        body_names = list(getattr(self.cabinet.data, "body_names", []))
        link_idx = next((i for i, n in enumerate(body_names) if "link_1" in n), 0)
        link_pos = self.cabinet.data.body_pos_w[:, link_idx]
        link_quat = self.cabinet.data.body_quat_w[:, link_idx]
        offset_local = tuple(v * _CABINET_SCALE for v in _HANDLE_PROXY_LOCAL_OFFSET)
        offset = torch.tensor(offset_local, device=self.device).repeat(link_pos.shape[0], 1)
        handle_pos, _ = math_utils.combine_frame_transforms(link_pos, link_quat, offset)
        return handle_pos

    def build(self, state: SceneState | None = None) -> torch.Tensor:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]

        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel

        cab_jp = (
            self.cabinet.data.joint_pos[:, self._drawer_joint_id]
            - self.cabinet.data.default_joint_pos[:, self._drawer_joint_id]
        ).unsqueeze(-1)
        cab_jv = (
            self.cabinet.data.joint_vel[:, self._drawer_joint_id]
            - self.cabinet.data.default_joint_vel[:, self._drawer_joint_id]
        ).unsqueeze(-1)

        tcp_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        handle_pos_w = self._handle_pos_w()
        rel_ee_drawer = handle_pos_w - tcp_pos_w

        last_action = self.env.unwrapped.action_manager.action

        obs = torch.cat(
            (joint_pos_rel, joint_vel_rel, cab_jp, cab_jv, rel_ee_drawer, last_action), dim=-1
        )
        self.last_obs_dim = int(obs.shape[-1])
        return obs


class SelectedDrawerObsAdapter:
    """Deployment-side builder of the 31-d selected-drawer obs, matching the training env.

    The custom RL env (custom_drawer_mdp) builds obs from its drawer_frames FrameTransformer
    (zero offset on link_0 / link_2) + the per-env selected joint. The deployment env
    (Isaac-Stack-Cube-Franka-JointPolicy-v0) has no drawer_frames sensor, so here the selected
    handle = the drawer link body world pose (link origin == zero-offset frame), and the selected
    joint is resolved from the central target->joint config. Order matches training exactly:
        joint_pos_rel(9) | joint_vel_rel(9) | sel_joint_pos(1) | sel_joint_vel(1) |
        (sel_handle - tcp)(3) | last_action(8)  -> 31
    """

    def __init__(self, env, target_drawer: str, env_id: int = 0):
        cfg = DRAWER_TARGETS[target_drawer]
        self.env = env
        self.target_drawer = target_drawer
        self.env_id = env_id
        self.scene = env.unwrapped.scene
        self.device = self.scene.device
        self.cabinet = self.scene["cabinet"]
        self.joint_name = cfg["joint_name"]
        self.link_name = cfg["link_name"]

        jnames = list(getattr(self.cabinet.data, "joint_names", []))
        self._joint_id = jnames.index(self.joint_name)
        bnames = list(getattr(self.cabinet.data, "body_names", []))
        self._link_idx = next((i for i, n in enumerate(bnames) if self.link_name == n), None)
        if self._link_idx is None:
            self._link_idx = next((i for i, n in enumerate(bnames) if self.link_name in n), 0)
        self.last_obs_dim: int | None = None

    def selected_handle_pos_w(self) -> torch.Tensor:
        return self.cabinet.data.body_pos_w[:, self._link_idx]

    def selected_drawer_joint_pos(self) -> float:
        return float(self.cabinet.data.joint_pos[self.env_id, self._joint_id])

    def build(self, state: SceneState | None = None) -> torch.Tensor:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]

        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel

        cab_jp = (
            self.cabinet.data.joint_pos[:, self._joint_id]
            - self.cabinet.data.default_joint_pos[:, self._joint_id]
        ).unsqueeze(-1)
        cab_jv = (
            self.cabinet.data.joint_vel[:, self._joint_id]
            - self.cabinet.data.default_joint_vel[:, self._joint_id]
        ).unsqueeze(-1)

        tcp_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        rel = self.selected_handle_pos_w() - tcp_pos_w

        last_action = self.env.unwrapped.action_manager.action
        obs = torch.cat((joint_pos_rel, joint_vel_rel, cab_jp, cab_jv, rel, last_action), dim=-1)
        self.last_obs_dim = int(obs.shape[-1])
        return obs
