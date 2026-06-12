"""OpenPI / pi0.5 policy backend.

This wraps a real OpenPI policy so the HTTP server can serve pi0 / pi0.5 actions.
It MUST be run inside the isolated OpenPI venv (where `openpi` + JAX/torch are
installed). It is imported lazily so the mock backend never needs these deps.

Two construction paths are supported:
  1. from a trained checkpoint:   OpenPIPolicy(config_name=..., ckpt_dir=...)
  2. from a default env policy:   OpenPIPolicy(env_default="aloha_sim")  (debug)

The OpenPI policy `.infer(obs)` expects a flat dict whose keys depend on the
training config's data transforms. We adapt our unified Observation into the
common OpenPI layout used by the example configs:
    {"state": <np.ndarray>, "image": <HxWx3 uint8>, "wrist_image": ..., "prompt": str}
If your finetune config uses different keys, edit `_to_openpi_obs` below.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("openpi_policy")


class OpenPIPolicy:
    backend_name = "openpi"

    def __init__(
        self,
        config_name: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
        env_default: Optional[str] = None,
        default_prompt: str = "",
        image_size: int = 224,
        state_dim: int = 8,
        obs_format: Optional[str] = None,
    ):
        self.default_prompt = default_prompt
        self.image_size = image_size
        self.state_dim = state_dim
        # obs_format decides which input-key layout the policy's data transform expects.
        # "droid"  -> observation/{exterior_image_1_left,wrist_image_left,joint_position,gripper_position}
        # "libero" -> {image, wrist_image, state, prompt}  (also our finetune configs)
        if obs_format is None:
            name = (config_name or env_default or "").lower()
            obs_format = "droid" if "droid" in name else "libero"
        self.obs_format = obs_format
        self._policy = self._build(config_name, ckpt_dir, env_default)

    # ------------------------------------------------------------------ #
    def _build(self, config_name, ckpt_dir, env_default):
        from openpi.policies import policy_config as _policy_config  # noqa
        from openpi.training import config as _config  # noqa

        if config_name and ckpt_dir:
            logger.info("Loading OpenPI policy config=%s ckpt=%s", config_name, ckpt_dir)
            train_cfg = _config.get_config(config_name)
            return _policy_config.create_trained_policy(train_cfg, ckpt_dir)

        if env_default:
            # Mirror scripts/serve_policy.py default checkpoints.
            from scripts.serve_policy import DEFAULT_CHECKPOINT, EnvMode  # type: ignore

            mode = EnvMode(env_default)
            ckpt = DEFAULT_CHECKPOINT[mode]
            logger.info("Loading OpenPI default policy for env=%s", env_default)
            train_cfg = _config.get_config(ckpt.config)
            return _policy_config.create_trained_policy(train_cfg, ckpt.dir)

        raise ValueError("Provide (config_name+ckpt_dir) or env_default")

    # ------------------------------------------------------------------ #
    def _decode_image(self, ref: dict[str, Any]) -> np.ndarray:
        mode = (ref or {}).get("mode", "none")
        if mode == "none" or ref is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        if mode == "path":
            from PIL import Image

            img = Image.open(ref["path"]).convert("RGB").resize((self.image_size, self.image_size))
            return np.asarray(img, dtype=np.uint8)
        if mode == "base64":
            import base64
            import io

            from PIL import Image

            raw = base64.b64decode(ref["base64"])
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize((self.image_size, self.image_size))
            return np.asarray(img, dtype=np.uint8)
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _to_openpi_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        robot = obs.get("robot", {}) or {}
        # state = [ee_pos(3), ee_quat(4? -> use first), gripper] padded/truncated to state_dim.
        jp = list(robot.get("joint_positions", []))
        ee = list(robot.get("ee_position", [0, 0, 0]))
        gw = [float(robot.get("gripper_width", 0.0))]
        state = (ee + jp + gw)[: self.state_dim]
        while len(state) < self.state_dim:
            state.append(0.0)

        images = obs.get("images", {}) or {}
        front = self._decode_image(images.get("front_rgb"))
        wrist = self._decode_image(images.get("wrist_rgb"))

        prompt = obs.get("task_instruction") or self.default_prompt
        return {
            "state": np.asarray(state, dtype=np.float32),
            "image": front,
            "wrist_image": wrist,
            "prompt": prompt,
        }

    def _to_droid_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        robot = obs.get("robot", {}) or {}
        jp = list(robot.get("joint_positions", []))[:7]
        while len(jp) < 7:
            jp.append(0.0)
        gw = float(robot.get("gripper_width", 0.0))
        images = obs.get("images", {}) or {}
        front = self._decode_image(images.get("front_rgb"))
        wrist = self._decode_image(images.get("wrist_rgb"))
        prompt = obs.get("task_instruction") or self.default_prompt
        return {
            "observation/exterior_image_1_left": front,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": np.asarray(jp, dtype=np.float32),
            "observation/gripper_position": np.asarray([gw], dtype=np.float32),
            "prompt": prompt,
        }

    # ------------------------------------------------------------------ #
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        op_obs = self._to_droid_obs(obs) if self.obs_format == "droid" else self._to_openpi_obs(obs)
        result = self._policy.infer(op_obs)
        actions = np.asarray(result["actions"], dtype=np.float32)  # (horizon, adim) or (adim,)
        if actions.ndim == 1:
            actions = actions[None, :]
        first = actions[0]
        adim = first.shape[0]
        # Convention: first 3 = delta xyz, next 3 = delta rot, last = gripper.
        dpos = first[:3].tolist()
        drot = first[3:6].tolist() if adim >= 6 else [0.0, 0.0, 0.0]
        grip = float(first[-1]) if adim >= 1 else 0.0
        return {
            "action_type": "delta_ee_pose",
            "delta_ee_position": dpos,
            "delta_ee_rot": drot,
            "gripper": grip,
            "chunk": actions.tolist(),
            "raw_model_output": first.tolist(),
        }
