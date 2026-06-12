"""IsaacLab binding — scene-agnostic glue between the teleop module and a Franka env.

Works with ANY IsaacLab manager-based env that contains a Franka articulation named ``robot``
with ``panda_joint.*`` arm joints (the scene/objects may differ between tasks). It locates the
arm joints, reads current joint positions and limits, and — for joint-position-action envs —
builds the env action tensor from absolute arm targets + a binary gripper command.

If the env's arm action is NOT joint-position (e.g. an IK-relative task), :meth:`make_action`
raises a clear error; use the controller's ``q_cmd`` with your own action mapping instead.
"""

from __future__ import annotations

import torch

# Known-safe Franka Panda arm joint limits (rad), used only if the asset doesn't expose them.
_FALLBACK_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
_FALLBACK_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


class IsaacFrankaBinding:
    """Binds to a Franka articulation inside an IsaacLab env (env_id 0 by default)."""

    def __init__(
        self,
        env,
        *,
        env_id: int = 0,
        robot_key: str = "robot",
        arm_joint_pattern: str = "panda_joint.*",
        arm_action_term: str | None = None,
        gripper_action_term: str | None = None,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.scene = self.unwrapped.scene
        self.env_id = env_id
        self.device = self.scene.device
        self.robot = self.scene[robot_key]
        ids, _ = self.robot.find_joints(arm_joint_pattern)
        self.arm_joint_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.num_arm = int(self.arm_joint_ids.numel())
        self._arm_term_name = arm_action_term
        self._gripper_term_name = gripper_action_term
        self._layout = None

    # ---- state reads ----
    def read_arm_q(self) -> torch.Tensor:
        """Current absolute arm joint positions [num_arm]."""
        return self.robot.data.joint_pos[self.env_id, self.arm_joint_ids].clone()

    def read_arm_limits(self):
        """Return (lower[num_arm], upper[num_arm], source_str) on the sim device."""
        for attr in ("joint_pos_limits", "soft_joint_pos_limits"):
            limits = getattr(self.robot.data, attr, None)
            if limits is None:
                continue
            try:
                sel = limits[self.env_id][self.arm_joint_ids]
                lower = sel[:, 0].to(self.device).clone()
                upper = sel[:, 1].to(self.device).clone()
                if lower.numel() == self.num_arm and bool(torch.all(upper > lower)):
                    return lower, upper, f"robot.data.{attr}"
            except Exception:  # noqa: BLE001
                continue
        lower = torch.tensor(_FALLBACK_LOWER[: self.num_arm], dtype=torch.float32, device=self.device)
        upper = torch.tensor(_FALLBACK_UPPER[: self.num_arm], dtype=torch.float32, device=self.device)
        return lower, upper, "hardcoded Franka Panda"

    # ---- action build (joint-position envs) ----
    def _resolve_layout(self) -> dict:
        if self._layout is not None:
            return self._layout
        am = self.unwrapped.action_manager
        active = list(am.active_terms)

        def _is_joint_pos(term) -> bool:
            return type(term).__name__.lower().startswith("jointposition") or (
                hasattr(term, "_scale") and hasattr(term, "_offset") and hasattr(term, "_joint_ids")
                and "inversekinematics" not in type(term).__name__.lower()
            )

        # find arm term
        arm_name = self._arm_term_name
        if arm_name is None:
            arm_name = "arm_action" if "arm_action" in active else None
        if arm_name is None:
            for name in active:
                if _is_joint_pos(am.get_term(name)):
                    arm_name = name
                    break
        if arm_name is None:
            raise RuntimeError(
                "No joint-position arm action term found (env action space may be IK-relative). "
                "Use the controller q_cmd with your own action mapping instead of make_action()."
            )
        arm_term = am.get_term(arm_name)
        if not _is_joint_pos(arm_term):
            raise RuntimeError(
                f"Arm action term '{arm_name}' is {type(arm_term).__name__}, not joint-position. "
                "This env needs a custom action mapping; use the controller q_cmd directly."
            )

        # arm block offset within the concatenated action vector
        arm_start = 0
        for name in active:
            term = am.get_term(name)
            if term is arm_term:
                break
            arm_start += term.action_dim
        arm_dim = arm_term.action_dim

        def _vec(v):
            if not isinstance(v, torch.Tensor):
                return torch.full((arm_dim,), float(v), device=self.device)
            return v[self.env_id].to(self.device) if v.dim() > 1 else v.to(self.device)

        scale = _vec(arm_term._scale)
        offset = _vec(arm_term._offset)

        # gripper term: prefer name, else the first binary term after the arm block
        grip_name = self._gripper_term_name
        if grip_name is None:
            grip_name = "gripper_action" if "gripper_action" in active else None
        gripper_index = None
        if grip_name is not None:
            idx = 0
            for name in active:
                if name == grip_name:
                    gripper_index = idx
                    break
                idx += am.get_term(name).action_dim
        else:
            gripper_index = arm_start + arm_dim  # convention: gripper right after arm

        self._layout = {
            "total_dim": am.total_action_dim,
            "arm_start": arm_start,
            "arm_dim": arm_dim,
            "gripper_index": gripper_index,
            "scale": scale,
            "offset": offset,
        }
        print(
            f"[gello-isaac] joint_action_layout total={self._layout['total_dim']} arm_start={arm_start} "
            f"arm_dim={arm_dim} gripper_index={gripper_index}",
            flush=True,
        )
        return self._layout

    def make_action(self, q_cmd, gripper_cmd: float) -> torch.Tensor:
        """Build a ``(num_envs, total_action_dim)`` env action from arm targets + binary gripper.

        Only valid for joint-position-action envs (raises otherwise).
        """
        layout = self._resolve_layout()
        q = torch.as_tensor(q_cmd, dtype=torch.float32, device=self.device).reshape(-1)
        if q.numel() != layout["arm_dim"]:
            raise ValueError(f"q_cmd must have {layout['arm_dim']} entries, got {q.numel()}")
        raw_arm = (q - layout["offset"]) / layout["scale"]
        num_envs = self.unwrapped.num_envs
        action = torch.zeros((num_envs, layout["total_dim"]), device=self.device)
        action[:, layout["arm_start"] : layout["arm_start"] + layout["arm_dim"]] = raw_arm
        if layout["gripper_index"] is not None and layout["gripper_index"] < layout["total_dim"]:
            action[:, layout["gripper_index"]] = float(gripper_cmd)
        return action

    @property
    def supports_joint_action(self) -> bool:
        try:
            self._resolve_layout()
            return True
        except Exception:  # noqa: BLE001
            return False
