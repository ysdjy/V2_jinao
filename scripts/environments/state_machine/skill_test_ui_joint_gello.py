# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GELLO-teleop variant of skill_test_ui_joint.py (stage 2).

This is a COPY of ``skill_test_ui_joint.py`` with one addition: a ``--control_mode gello``
that lets a physical GELLO leader arm drive the Franka in the SAME task scene
(``Isaac-Stack-Cube-Franka-JointPolicy-v0``). The original button-driven skill test is
preserved verbatim under ``--control_mode ui`` (the default).

  * ``--control_mode ui``    : identical behaviour to skill_test_ui_joint.py.
  * ``--control_mode gello`` : reads GELLO arm joints (single process, ``import gello``) and
    maps q_gello[0:7] -> Franka joint targets via ``provider.make_joint_action_from_q_des``.
    Safety: low-pass filter + per-step max delta + joint-limit clip + NaN/dim checks +
    hold-last-q on read failure + no startup jump (q_cmd starts at the current Franka q).
    SkillExecutor does NOT emit actions in gello mode (it cannot fight the teleop).

The original ``skill_test_ui_joint.py`` is left untouched.

UI mode (unchanged):
    ./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui_joint_gello.py \
        --num_envs 1 --control_mode ui --seed 1

GELLO mode:
    ./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui_joint_gello.py \
        --num_envs 1 --control_mode gello \
        --gello_config projects/gello_franka_teleop/configs/gello_franka.yaml \
        --gello_hz 30 --gello_alpha 0.2 --gello_max_delta 0.03 \
        --show_affordance_debug --grasp_backend joint_ik --place_backend joint_ik \
        --drawer_backend scripted_joint --seed 1
    (the Franka serial port lives in the dialout group; if your shell lacks it, launch via
     ``sg dialout -c "<the command above>"``.)
"""

from __future__ import annotations

"""Launch Omniverse Toolkit first."""

import argparse
import json
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Sim native UI for joint-action Franka skills.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--disable_collision_debug_vis", action="store_true", default=False, help="Disable collider overlays.")
parser.add_argument("--show_affordance_debug", action="store_true", default=False, help="Show affordance debug frames.")
parser.add_argument("--auto_start", action="store_true", default=False, help="Auto-start the --skill (opt-in; off by default).")
parser.add_argument("--skill", default="grasp", choices=["grasp", "place", "open_drawer", "close_drawer"], help="Skill.")
parser.add_argument("--target", default="cube_2", choices=["cube_1", "cube_2", "cube_3", "knife"], help="Grasp target.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum sim steps before exit. 0 means no limit.")
parser.add_argument("--seed", type=int, default=1, help="Deterministic skill-test seed.")
parser.add_argument("--grasp_backend", type=str, default="joint_ik", choices=["joint_ik"], help="Grasp backend.")
parser.add_argument("--place_backend", type=str, default="joint_ik", choices=["joint_ik"], help="Place backend.")
parser.add_argument(
    "--drawer_backend",
    type=str,
    default="none",
    choices=["none", "scripted_joint", "official_joint_policy", "custom_selected_policy"],
    help="Drawer backend. Default 'none' (Open Drawer disabled) so scripted_joint is not mistaken "
    "for a learned skill. scripted_joint = direct drawer-joint baseline (robot does NOT physically "
    "pull). custom_selected_policy = learned policy. Never auto-runs.",
)
parser.add_argument("--drawer_policy_path", type=str, default=None, help="TorchScript policy.pt for the learned drawer backend.")
parser.add_argument("--log_every", type=int, default=30, help="Low-frequency debug log interval (sim steps).")
parser.add_argument(
    "--cube_grasp_z_offset",
    type=float,
    default=0.0,
    help="TCP z offset from cube center for grasp debugging, in [-0.010, 0.015] m.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Skill UI supports 1.")
# --- GELLO teleop (stage 2) options ---
parser.add_argument(
    "--control_mode",
    type=str,
    default="ui",
    choices=["ui", "gello"],
    help="'ui' = original button-driven skill test (unchanged). 'gello' = GELLO joints drive the Franka arm.",
)
parser.add_argument(
    "--gello_config",
    type=str,
    default="projects/gello_franka_teleop/configs/gello_franka.yaml",
    help="Path to the GELLO yaml config (port / offsets / signs / gripper).",
)
parser.add_argument("--gello_hz", type=float, default=60.0, help="Background GELLO read rate (Hz). Does NOT throttle the sim loop.")
parser.add_argument("--gello_alpha", type=float, default=0.5, help="Low-pass factor: q_cmd = (1-a)*q_cmd + a*q_target.")
parser.add_argument("--gello_max_delta", type=float, default=0.08, help="Max per-step joint change (rad) applied to q_cmd.")
parser.add_argument(
    "--gello_start_tolerance",
    type=float,
    default=0.8,
    help="If startup |q_gello - q_franka| exceeds this (rad, any joint), warn about a large initial gap.",
)
parser.add_argument("--gello_no_gripper", action="store_true", default=False, help="Read only 7 arm joints; ignore GELLO gripper.")
parser.add_argument("--gello_print_every", type=int, default=30, help="Print telemetry every N sim steps in gello mode.")
# --- GELLO gripper mapping (binary open/close with hysteresis) ---
parser.add_argument("--gello_enable_gripper", action="store_true", default=False, help="Map GELLO gripper to the Franka binary gripper. If off, gripper stays OPEN.")
parser.add_argument("--gello_gripper_open_raw", type=float, default=211.0, help="GELLO gripper raw angle (deg) when fully OPEN.")
parser.add_argument("--gello_gripper_close_raw", type=float, default=169.0, help="GELLO gripper raw angle (deg) when fully CLOSED.")
parser.add_argument("--gello_gripper_invert", action="store_true", default=False, help="Invert gripper open/close mapping.")
parser.add_argument("--gello_gripper_deadband", type=float, default=3.0, help="Hysteresis deadband (deg) around the open/close threshold.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import carb
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from skill_runtime.base_skill import pose_tensor
from skill_runtime.debug_visualizer import DebugVisualizer, OBJECT_AXIS_LENGTH, OBJECT_AXIS_WIDTH
from skill_runtime.drawer_obs_adapter import DrawerObsAdapter
from skill_runtime.ik_joint_adapter import IKJointAdapter
from skill_runtime.joint_debug_logger import JointDebugLogger
from skill_runtime.official_drawer_policy import OfficialDrawerPolicyWrapper
from skill_runtime.scene_state_provider import SceneStateProvider
from skill_runtime.simple_scene_layout import SimpleLayoutResult, SimpleSceneLayoutManager
from skill_runtime.skill_executor import JointBackendConfig, SkillExecutor
from skill_runtime.skill_request import SkillRequest
from skill_runtime.skill_types import ExecutionStatus, SkillType
from skill_runtime.target_registry import TargetRegistry
from skill_runtime.ui_controller import UIController


TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"

SKILL_LABELS = [
    ("Grasp", SkillType.GRASP),
    ("Place", SkillType.PLACE),
    ("Open Drawer", SkillType.OPEN_DRAWER),
    ("Close Drawer", SkillType.CLOSE_DRAWER),
]

# 下抽屉 / 中抽屉 / 上抽屉
DRAWER_TARGETS = [
    ("下抽屉 (bottom_drawer)", "bottom_drawer"),
    ("中抽屉 (middle_drawer)", "middle_drawer"),
    ("上抽屉 (top_drawer)", "top_drawer"),
]

DEFAULT_PLACE_POINTS = {
    "point_a": [0.42, 0.10, 0.00],
    "point_b": [0.55, 0.10, 0.00],
    "point_c": [0.68, 0.10, 0.00],
}
POINT_X_MIN = 0.30
POINT_X_MAX = 0.75
POINT_Y_MIN = -0.05
POINT_Y_MAX = 0.25
POINT_Z_MIN = 0.00
POINT_Z_MAX = 0.60
PLACE_POINT_VISUAL_Z_OFFSET = 0.005
PLACE_POINT_AXIS_LENGTH = 0.035
PLACE_POINT_AXIS_WIDTH = 0.002
PLACE_MOVE_STEPS = [("1 cm", 0.010), ("5 cm", 0.050)]


def enable_collision_debug_visualization():
    settings = carb.settings.get_settings()
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 2)
    settings.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)


class SkillTestWindow:
    def __init__(self, controller: UIController, executor: SkillExecutor, registry: TargetRegistry):
        import omni.ui as ui

        self.ui = ui
        self.controller = controller
        self.executor = executor
        self.registry = registry
        self.target_keys = [key for key, _ in registry.display_targets()]
        self.drawer_target_keys = [key for _, key in DRAWER_TARGETS]
        self.selected_drawer = self.drawer_target_keys[0]
        self.place_point_keys = list(DEFAULT_PLACE_POINTS.keys())
        self.place_points = {key: list(value) for key, value in DEFAULT_PLACE_POINTS.items()}
        self.selected_place_point = "point_a"
        self.place_move_step = PLACE_MOVE_STEPS[0][1]
        self.status_labels = {}
        self.window = ui.Window("Franka Skill Test (Joint)", width=460, height=720)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Skill")
                self.skill_model = ui.ComboBox(0, *[label for label, _ in SKILL_LABELS]).model
                self.skill_model.add_item_changed_fn(self._on_skill_changed)
                ui.Label("Grasp target")
                self.target_model = ui.ComboBox(1, *[label for _, label in registry.display_targets()]).model
                self.target_model.add_item_changed_fn(self._on_target_changed)
                ui.Label("Drawer target")
                self.drawer_model = ui.ComboBox(0, *[label for label, _ in DRAWER_TARGETS]).model
                self.drawer_model.add_item_changed_fn(self._on_drawer_changed)
                with ui.HStack(spacing=6):
                    ui.Button("Start", clicked_fn=self._start)
                    ui.Button("Stop", clicked_fn=self.controller.request_stop)
                    ui.Button("Resume", clicked_fn=self.controller.request_resume)
                    ui.Button("Reset", clicked_fn=self.controller.request_reset)
                ui.Label("Place point")
                self.place_point_model = ui.ComboBox(0, *self.place_point_keys).model
                self.place_point_model.add_item_changed_fn(self._on_place_point_changed)
                ui.Label("Place move step")
                self.place_step_model = ui.ComboBox(0, *[label for label, _ in PLACE_MOVE_STEPS]).model
                self.place_step_model.add_item_changed_fn(self._on_place_step_changed)
                with ui.VStack(spacing=4, height=0):
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Up (+Z)", clicked_fn=lambda: self._move_place_point(0.0, 0.0, 1.0))
                        ui.Label("")
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Forward (+X)", clicked_fn=lambda: self._move_place_point(1.0, 0.0, 0.0))
                        ui.Label("")
                    with ui.HStack(spacing=6):
                        ui.Button("Left (+Y)", clicked_fn=lambda: self._move_place_point(0.0, 1.0, 0.0))
                        ui.Button("Reset Point", clicked_fn=self._reset_place_point)
                        ui.Button("Right (-Y)", clicked_fn=lambda: self._move_place_point(0.0, -1.0, 0.0))
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Backward (-X)", clicked_fn=lambda: self._move_place_point(-1.0, 0.0, 0.0))
                        ui.Label("")
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Down (-Z)", clicked_fn=lambda: self._move_place_point(0.0, 0.0, -1.0))
                        ui.Label("")
                for key in (
                    "selected_skill",
                    "selected_target",
                    "selected_drawer",
                    "selected_place_point",
                    "place_point_x",
                    "place_point_y",
                    "place_point_z",
                    "place_move_step",
                    "held_object",
                    "active_skill",
                    "backend",
                    "control_mode",
                    "paused",
                    "latched_gripper_command",
                    "runtime_status",
                    "state",
                    "elapsed",
                    "position_error",
                    "orientation_error_deg",
                    "gripper_width",
                    "layout_seed",
                    "reset_index",
                    "layout_valid",
                    "cabinet_root_pose",
                    "cube_1_pose",
                    "cube_2_pose",
                    "cube_3_pose",
                    "knife_pose",
                    "target_pose",
                    "drawer_control_mode",
                    "drawer_joint_name",
                    "drawer_joint_position",
                    "drawer_joint_target",
                    "drawer_runtime_status",
                    "handle_pose",
                    "last_failure",
                    "last_result",
                ):
                    self.status_labels[key] = ui.Label(f"{key}:")

    def _on_skill_changed(self, model, item):
        index = model.get_item_value_model().as_int
        self.controller.selected_skill = SKILL_LABELS[index][1]

    def _on_target_changed(self, model, item):
        index = model.get_item_value_model().as_int
        self.controller.selected_target = self.target_keys[index]

    def _on_drawer_changed(self, model, item):
        index = model.get_item_value_model().as_int
        self.selected_drawer = self.drawer_target_keys[index]

    def _on_place_point_changed(self, model, item):
        index = model.get_item_value_model().as_int
        self.selected_place_point = self.place_point_keys[index]

    def _on_place_step_changed(self, model, item):
        index = model.get_item_value_model().as_int
        self.place_move_step = PLACE_MOVE_STEPS[index][1]

    def _move_place_point(self, x_dir: float, y_dir: float, z_dir: float):
        point = self.place_points[self.selected_place_point]
        point[0] = _clamp(point[0] + x_dir * self.place_move_step, POINT_X_MIN, POINT_X_MAX)
        point[1] = _clamp(point[1] + y_dir * self.place_move_step, POINT_Y_MIN, POINT_Y_MAX)
        point[2] = _clamp(point[2] + z_dir * self.place_move_step, POINT_Z_MIN, POINT_Z_MAX)

    def _reset_place_point(self):
        self.place_points[self.selected_place_point] = list(DEFAULT_PLACE_POINTS[self.selected_place_point])

    def _held_object_name(self) -> str:
        held_object = self.executor.held_object
        if held_object is not None:
            return held_object.object_name
        return "None"

    def _start(self):
        if self.controller.selected_skill == SkillType.PLACE:
            selected_xyz = list(self.place_points[self.selected_place_point])
            held_name = self.executor.held_object.object_name if self.executor.held_object is not None else None
            self.controller.queue_request(
                _make_request(
                    SkillType.PLACE,
                    held_name,
                    place_point_name=self.selected_place_point,
                    place_point_xyz=selected_xyz,
                )
            )
            return
        if self.controller.selected_skill in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            backend = self.executor.backend.drawer_backend
            target = self.selected_drawer
            if backend == "none":
                print(
                    "[UI] drawer_backend='none': Open/Close Drawer is disabled. Relaunch with "
                    "--drawer_backend scripted_joint (baseline) or custom_selected_policy (learned).",
                    flush=True,
                )
                return
            if backend == "scripted_joint":
                print(
                    "[BASELINE WARNING] scripted_joint directly commands drawer_joint_target; "
                    "robot arm will hold still.",
                    flush=True,
                )
            if backend == "custom_selected_policy" and target == "bottom_drawer":
                print(
                    "[UI][WARNING] bottom_drawer is currently locked / non-functional; "
                    "train/test top and middle first.",
                    flush=True,
                )
            self.controller.queue_request(_make_request(self.controller.selected_skill, target))
            return
        target = self.controller.selected_target
        self.controller.queue_request(_make_request(self.controller.selected_skill, target))

    def update(self, state, executor: SkillExecutor, layout_result: SimpleLayoutResult | None):
        result = executor.last_result
        active = executor.active_skill
        plan = getattr(getattr(active, "runtime", None), "filtered_plan", None)
        target_pose_state = getattr(plan, "target_pose", None) or getattr(plan, "handle_pose", None)
        target_pose = target_pose_state.as_pose_tensor() if target_pose_state is not None else None
        handle_pose_state = getattr(plan, "handle_pose", None)
        drawer_runtime = getattr(active, "runtime", None)
        drawer_joint_name = getattr(drawer_runtime, "drawer_joint_name", "joint_0")
        drawer_joint_position = None
        cabinet = state.objects.get("cabinet")
        if cabinet is not None:
            drawer_joint_position = cabinet.joint_pos.get(drawer_joint_name)
        elapsed = 0.0
        if active is not None:
            elapsed = max(0.0, state.sim_time - getattr(active.runtime, "start_time", state.sim_time))
        orientation_error = getattr(getattr(active, "runtime", None), "final_error_ori", None)
        drawer_joint_target = getattr(drawer_runtime, "drawer_joint_target", None)
        point = self.place_points[self.selected_place_point]
        latched = executor.latched_command
        values = {
            "selected_skill": self.controller.selected_skill.value,
            "selected_target": self.controller.selected_target,
            "selected_drawer": self.selected_drawer,
            "selected_place_point": self.selected_place_point,
            "place_point_x": f"{point[0]:.3f}",
            "place_point_y": f"{point[1]:.3f}",
            "place_point_z": f"{point[2]:.3f}",
            "place_move_step": f"{self.place_move_step:.3f}",
            "held_object": self._held_object_name(),
            "active_skill": None if active is None else active.__class__.__name__,
            "backend": None if active is None else getattr(active, "backend", None),
            "control_mode": "joint",
            "paused": executor.paused,
            "latched_gripper_command": None if latched is None else f"{latched.gripper_command:.1f}",
            "runtime_status": executor.runtime_status,
            "state": executor.current_state_name,
            "elapsed": f"{elapsed:.2f}",
            "position_error": str(getattr(getattr(active, "runtime", None), "final_error_pos", None)),
            "orientation_error_deg": None if orientation_error is None else f"{math.degrees(orientation_error):.2f}",
            "gripper_width": f"{state.robot.gripper_width:.5f}",
            "layout_seed": None if layout_result is None else layout_result.seed,
            "reset_index": None if layout_result is None else layout_result.reset_index,
            "layout_valid": None if layout_result is None else True,
            "cabinet_root_pose": None if layout_result is None else layout_result.object_poses.get("cabinet"),
            "cube_1_pose": None if layout_result is None else layout_result.object_poses.get("cube_1"),
            "cube_2_pose": None if layout_result is None else layout_result.object_poses.get("cube_2"),
            "cube_3_pose": None if layout_result is None else layout_result.object_poses.get("cube_3"),
            "knife_pose": None if layout_result is None else layout_result.object_poses.get("knife"),
            "target_pose": _short_pose(target_pose),
            "drawer_control_mode": getattr(drawer_runtime, "drawer_control_mode", None),
            "drawer_joint_name": drawer_joint_name,
            "drawer_joint_position": None if drawer_joint_position is None else f"{drawer_joint_position:.5f}",
            "drawer_joint_target": None if drawer_joint_target is None else f"{drawer_joint_target:.5f}",
            "drawer_runtime_status": getattr(drawer_runtime, "state", None),
            "handle_pose": _short_pose(handle_pose_state.as_pose_tensor() if handle_pose_state is not None else None),
            "last_failure": None if result is None else result.failure_reason,
            "last_result": None if result is None else result.final_status.value,
        }
        for key, label in self.status_labels.items():
            label.text = f"{key}: {values[key]}"


def _short_pose(pose: torch.Tensor | None) -> str:
    if pose is None:
        return "None"
    values = pose.detach().cpu().tolist()
    return "[" + ", ".join(f"{v:.3f}" for v in values[:3]) + "]"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _skill_type_from_arg(value: str) -> SkillType:
    return SkillType(value)


def _make_request(
    skill_type: SkillType,
    target: str | None,
    place_point_name: str = "point_a",
    place_point_xyz: list[float] | None = None,
) -> SkillRequest:
    if skill_type == SkillType.PLACE:
        selected_xyz = list(DEFAULT_PLACE_POINTS[place_point_name] if place_point_xyz is None else place_point_xyz)
        return SkillRequest(
            request_id=f"{skill_type.value}_{target or 'none'}_{time.time_ns()}",
            skill_type=skill_type,
            source_object=target,
            destination_type="point",
            destination_object=place_point_name,
            parameters={
                "target_frame": "env_local",
                "target_surface_xyz": [selected_xyz[0], selected_xyz[1], selected_xyz[2]],
            },
        )
    if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
        return SkillRequest(
            request_id=f"{skill_type.value}_{target}_{time.time_ns()}",
            skill_type=skill_type,
            source_object=None,
            destination_type="drawer",
            destination_object=target or "bottom_drawer",
            parameters={"drawer_link": "link_1"},
        )
    return SkillRequest(
        request_id=f"{skill_type.value}_{target}_{time.time_ns()}",
        skill_type=skill_type,
        source_object=target if skill_type == SkillType.GRASP else None,
    )


def _command_to_action(provider: SceneStateProvider, command, state):
    if command.control_mode == "joint":
        if command.raw_joint_action is not None:
            return provider.make_joint_action_from_raw(command.raw_joint_action)
        if command.joint_target is not None:
            return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
        return provider.make_hold_joint_action(state, None)
    return provider.make_action(command.tcp_pose_w, command.gripper_command)


def _settle_layout(env, provider: SceneStateProvider) -> None:
    provider.reset_cabinet_joint("joint_0", 0.0)
    state = provider.get_state()
    hold_action = provider.make_hold_joint_action(state, 1.0)
    for _ in range(5):
        env.step(hold_action)


def _apply_drawer_joint_command(provider: SceneStateProvider, command, baseline_warned: dict) -> None:
    if command.drawer_joint_target is None:
        return
    if not baseline_warned["done"]:
        print(
            "[BASELINE] scripted_joint is directly commanding drawer joint, not learned physical pulling.",
            flush=True,
        )
        baseline_warned["done"] = True
    provider.set_cabinet_joint_target(command.drawer_joint_name or "joint_0", command.drawer_joint_target)


# Fallback Franka Panda arm joint limits (rad), used only if we cannot read them from the asset.
_FRANKA_Q_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
_FRANKA_Q_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


class GelloReader:
    """Reads GELLO arm joints (+ optional gripper) using gello's own ``DynamixelRobot``.

    Returns plain numpy from :meth:`read`; the caller turns it into a torch target. Mirrors the
    proven logic in ``projects/gello_franka_teleop/scripts/read_gello_joints.py`` (fake-driver
    rejection + gripper fallback) so behaviour is consistent across the two entry points.
    """

    def __init__(self, config_path: str, use_gripper: bool = True, override_port: str | None = None):
        import numpy as np
        import yaml
        from gello.robots.dynamixel import DynamixelRobot

        self._np = np
        cfg_file = __import__("pathlib").Path(config_path)
        if not cfg_file.is_absolute():
            # resolve relative to the IsaacLab repo root (two levels above scripts/.../)
            cfg_file = (__import__("pathlib").Path(__file__).resolve().parents[3] / config_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"GELLO config not found: {cfg_file}")
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

        port = override_port or cfg.get("port")
        if not port or port == "None":
            raise RuntimeError(
                "GELLO config has no 'port'. Run projects/gello_franka_teleop/scripts/detect_gello_port.sh "
                "and fill it in."
            )
        joint_ids = list(cfg["joint_ids"])
        joint_offsets = list(cfg["joint_offsets"])
        joint_signs = list(cfg["joint_signs"])
        self.num_arm = len(joint_ids)
        baudrate = int(cfg.get("baudrate", 57600))

        gcfg = cfg.get("gripper") or {}
        gripper_config = None
        if use_gripper and gcfg.get("enabled", False):
            gripper_config = (int(gcfg["id"]), float(gcfg["open_value"]), float(gcfg["close_value"]))

        start = np.array(cfg.get("start_joints", [0.0] * self.num_arm), dtype=float)
        if start.shape[0] != self.num_arm:
            raise ValueError(f"start_joints must have {self.num_arm} entries, got {start.shape[0]}")
        start_full = np.concatenate([start, [0.0]]) if gripper_config is not None else start

        def _build(gc, st):
            robot = DynamixelRobot(
                joint_ids=joint_ids,
                joint_offsets=joint_offsets,
                joint_signs=joint_signs,
                real=True,
                port=port,
                baudrate=baudrate,
                gripper_config=gc,
                start_joints=st,
            )
            # gello silently falls back to a fake (all-zeros) driver if the serial port can't be
            # opened (usually a dialout permission issue). Reject it loudly instead of teleoping garbage.
            if getattr(robot._driver, "_is_fake", False):
                raise PermissionError(
                    "GELLO serial port did not actually open (gello fell back to its fake driver). "
                    "Check dialout permission, e.g. relaunch via: sg dialout -c \"<command>\"."
                )
            return robot

        try:
            self._robot = _build(gripper_config, start_full)
            self.gripper_enabled = gripper_config is not None
        except PermissionError:
            raise
        except Exception as exc:  # gripper servo may be missing -> degrade to 7 arm joints
            print(f"[GELLO] gripper init failed ({exc}); reading 7 arm joints only.", flush=True)
            self._robot = _build(None, start)
            self.gripper_enabled = False
        self.port = port

    def read(self):
        """Return ``(q_arm: np.ndarray[7], gripper_raw_deg: float | None, ok: bool)``.

        Reads the raw Dynamixel angles ONCE (serial I/O), applies the calibrated
        offsets/signs to get arm q, and returns the gripper as a RAW angle in degrees
        (matching the calibration's ``gripper open/close (degrees)`` ~211/169). We bypass
        ``get_joint_state`` here to (a) get the raw gripper degrees from the same read and
        (b) skip gello's internal smoothing, which only adds latency on top of our own filter.
        """
        np = self._np
        try:
            raw = np.asarray(self._robot._driver.get_joints(), dtype=float)  # radians, len = num_arm (+1)
        except Exception:  # noqa: BLE001 - keep teleop alive on a transient read error
            return None, None, False
        n = self.num_arm
        if raw.ndim != 1 or raw.shape[0] < n:
            return None, None, False
        offsets = np.asarray(self._robot._joint_offsets, dtype=float)
        signs = np.asarray(self._robot._joint_signs, dtype=float)
        arm_q = (raw[:n] - offsets[:n]) * signs[:n]
        if np.any(np.isnan(arm_q)):
            return None, None, False
        gripper_deg = None
        if self.gripper_enabled and raw.shape[0] > n:
            gripper_deg = float(np.rad2deg(raw[n]))
        return arm_q, gripper_deg, True


class ThreadedGelloReader:
    """Runs a :class:`GelloReader` in a background thread so the blocking serial read never
    stalls ``env.step``. The sim loop just grabs the most recent sample (non-blocking).

    Serial reads release the GIL, so the thread genuinely overlaps with sim/render work.
    """

    def __init__(self, reader: GelloReader, hz: float):
        import threading

        self._reader = reader
        self._period = 1.0 / max(1.0, float(hz))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="gello-reader", daemon=True)
        # shared state (guarded by _lock)
        self._latest_q = None        # np.ndarray[7], most recent GOOD arm read
        self._latest_gripper = None  # float deg or None, most recent GOOD gripper
        self._last_ok = False        # did the most recent read attempt succeed?
        self._read_ms = 0.0          # duration of the most recent read (ms)
        self._read_hz = 0.0          # EMA of actual read rate
        # passthrough metadata
        self.gripper_enabled = reader.gripper_enabled
        self.num_arm = reader.num_arm
        self.port = reader.port

    def start(self):
        self._thread.start()

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            q, gripper, ok = self._reader.read()
            t1 = time.perf_counter()
            with self._lock:
                if ok and q is not None:
                    self._latest_q = q
                    self._latest_gripper = gripper
                self._last_ok = ok
                self._read_ms = (t1 - t0) * 1000.0
                dt = t1 - last
                last = t1
                if dt > 0:
                    inst = 1.0 / dt
                    self._read_hz = inst if self._read_hz == 0.0 else 0.9 * self._read_hz + 0.1 * inst
            sleep_t = self._period - (t1 - t0)
            if sleep_t > 0:
                self._stop.wait(sleep_t)

    def get_latest(self):
        """Return ``(q_arm | None, gripper_deg | None, last_ok, read_ms, read_hz)`` (non-blocking)."""
        with self._lock:
            q = None if self._latest_q is None else self._latest_q.copy()
            return q, self._latest_gripper, self._last_ok, self._read_ms, self._read_hz

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def _read_arm_joint_limits(provider: SceneStateProvider):
    """Read Franka arm joint limits from the asset; fall back to known Panda limits.

    Returns ``(lower[7], upper[7], source_str)`` as tensors on the sim device.
    """
    device = provider.device
    fallback = (
        torch.tensor(_FRANKA_Q_LOWER, dtype=torch.float32, device=device),
        torch.tensor(_FRANKA_Q_UPPER, dtype=torch.float32, device=device),
        "hardcoded Franka Panda",
    )
    try:
        robot = provider.scene["robot"]
        arm_ids = provider._arm_joint_ids
        for attr in ("joint_pos_limits", "soft_joint_pos_limits"):
            limits = getattr(robot.data, attr, None)
            if limits is None:
                continue
            sel = limits[provider.env_id][arm_ids]  # [7, 2]
            lower = sel[:, 0].to(device).clone()
            upper = sel[:, 1].to(device).clone()
            if lower.numel() == 7 and bool(torch.all(upper > lower)):
                return lower, upper, f"robot.data.{attr}"
    except Exception as exc:  # noqa: BLE001
        print(f"[GELLO] could not read joint limits from asset ({exc}); using fallback.", flush=True)
    return fallback


def main():
    if args_cli.num_envs != 1:
        raise ValueError("skill_test_ui_joint currently supports --num_envs 1 so one UI maps to one scene.")

    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None
    if hasattr(env_cfg.scene, "cabinet") and hasattr(env_cfg.scene.cabinet, "actuators"):
        if "drawers" in env_cfg.scene.cabinet.actuators:
            env_cfg.scene.cabinet.actuators["drawers"].stiffness = 10.0
            env_cfg.scene.cabinet.actuators["drawers"].damping = 1.0
    if hasattr(env_cfg.scene, "knife") and hasattr(env_cfg.scene.knife, "actuators"):
        if "blade_lock" in env_cfg.scene.knife.actuators:
            env_cfg.scene.knife.actuators["blade_lock"].stiffness = 100.0
            env_cfg.scene.knife.actuators["blade_lock"].damping = 10.0
    env_cfg.viewer.eye = (2.0, -2.0, 1.4)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.15)

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    if not args_cli.headless and not args_cli.disable_collision_debug_vis:
        enable_collision_debug_visualization()

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    registry = TargetRegistry(env.unwrapped.device, cube_grasp_z_offset=args_cli.cube_grasp_z_offset)
    adapter = IKJointAdapter(env)

    drawer_policy = None
    drawer_obs_adapter = None
    if args_cli.drawer_backend in ("official_joint_policy", "custom_selected_policy"):
        drawer_policy = OfficialDrawerPolicyWrapper(args_cli.drawer_policy_path, device=env.unwrapped.device)
    if args_cli.drawer_backend == "official_joint_policy":
        drawer_obs_adapter = DrawerObsAdapter(env, drawer_joint_name="joint_0")

    backend = JointBackendConfig(
        mode="joint",
        grasp_backend=args_cli.grasp_backend,
        place_backend=args_cli.place_backend,
        drawer_backend=args_cli.drawer_backend,
        adapter=adapter,
        drawer_policy=drawer_policy,
        drawer_obs_adapter=drawer_obs_adapter,
        drawer_env=env,
        arm_joint_ids=provider._arm_joint_ids,
        drawer_joint_name="joint_0",
    )
    executor = SkillExecutor(registry, backend=backend)
    debug_logger = JointDebugLogger(every_steps=args_cli.log_every)
    baseline_warned = {"done": False}

    controller = UIController()
    controller.selected_skill = _skill_type_from_arg(args_cli.skill)
    controller.selected_target = args_cli.target
    visualizer = DebugVisualizer(enabled=not args_cli.headless or args_cli.show_affordance_debug)
    window = None if args_cli.headless else SkillTestWindow(controller, executor, registry)

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    sim_time = 0.0
    provider.set_sim_time(sim_time)

    layout_result = layout_manager.reset_layout(reset_index=0)
    _settle_layout(env, provider)

    # auto_start is opt-in only; by default nothing runs until the user clicks a button.
    # (disabled in gello mode: the leader arm drives the robot, not scripted skills.)
    if args_cli.auto_start and args_cli.control_mode != "gello":
        target = "bottom_drawer" if controller.selected_skill in (
            SkillType.OPEN_DRAWER,
            SkillType.CLOSE_DRAWER,
        ) else controller.selected_target
        controller.queue_request(_make_request(controller.selected_skill, target))

    # ---- GELLO teleop setup (only for --control_mode gello) ----
    gello_reader = None
    q_cmd = None
    q_target = None
    q_lower = q_upper = None
    gello_alpha = float(args_cli.gello_alpha)
    gello_max_delta = float(args_cli.gello_max_delta)
    gello_last_loop_t = time.perf_counter()
    gello_loop_hz = 0.0
    gello_last_gripper = None
    # gripper binary state machine (with hysteresis)
    gripper_state = "open"  # commanded state: "open" -> +1.0, "close" -> -1.0
    gripper_threshold = 0.5 * (args_cli.gello_gripper_open_raw + args_cli.gello_gripper_close_raw)
    if args_cli.control_mode == "gello":
        print("[GELLO] control_mode=gello: initializing GELLO leader arm ...", flush=True)
        _base_reader = GelloReader(args_cli.gello_config, use_gripper=not args_cli.gello_no_gripper)
        q_lower, q_upper, limit_src = _read_arm_joint_limits(provider)
        print(f"[GELLO] arm joint limits from: {limit_src}", flush=True)
        print(f"[GELLO]   lower = {[round(v, 3) for v in q_lower.tolist()]}", flush=True)
        print(f"[GELLO]   upper = {[round(v, 3) for v in q_upper.tolist()]}", flush=True)
        # Start q_cmd / q_target at the CURRENT Franka q -> guarantees no startup jump.
        init_state = provider.get_state()
        q_cmd = provider.arm_joint_pos(init_state).to(provider.device).float()
        q_target = q_cmd.clone()
        q_g0, _g0, ok0 = _base_reader.read()
        if ok0:
            q_g0_t = torch.as_tensor(q_g0, dtype=torch.float32, device=provider.device)
            gap = float(torch.max(torch.abs(q_g0_t - q_cmd)).item())
            print(f"[GELLO] startup gap max|q_gello - q_franka| = {gap:.3f} rad", flush=True)
            if gap > float(args_cli.gello_start_tolerance):
                print(
                    f"[GELLO][WARN] large startup gap ({gap:.3f} > {args_cli.gello_start_tolerance} rad). "
                    f"No jump will occur: the arm ramps to the GELLO pose at <= {gello_max_delta} rad/step. "
                    "For a gentler start, hold GELLO near the Franka home pose before launching.",
                    flush=True,
                )
        else:
            print("[GELLO][WARN] initial GELLO read failed; will hold q_cmd until reads succeed.", flush=True)
        # Start the background reader thread (decouples serial latency from env.step).
        gello_reader = ThreadedGelloReader(_base_reader, hz=args_cli.gello_hz)
        gello_reader.start()
        grip_mode = "ENABLED (binary)" if args_cli.gello_enable_gripper else "disabled (kept OPEN)"
        print(
            f"[GELLO] port={gello_reader.port} | gripper_read={'on' if gello_reader.gripper_enabled else 'off'} | "
            f"gripper_control={grip_mode} | threshold={gripper_threshold:.1f} deg "
            f"(open_raw={args_cli.gello_gripper_open_raw}, close_raw={args_cli.gello_gripper_close_raw}, "
            f"deadband={args_cli.gello_gripper_deadband}, invert={args_cli.gello_gripper_invert})",
            flush=True,
        )
        print(f"[GELLO] background read rate target = {args_cli.gello_hz} Hz (sim loop is NOT throttled).", flush=True)

    step_count = 0
    layout_reset_index = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            provider.set_sim_time(sim_time)
            state = provider.get_state()
            command = None
            if args_cli.control_mode == "gello":
                # GELLO leader drives the arm. SkillExecutor is NOT stepped (it must not fight teleop).
                # Only the Reset button is honoured (re-layout + re-seat q_cmd at the new Franka q).
                pending = controller.pop()
                if pending is not None and pending.command == "reset":
                    env.reset(seed=args_cli.seed)
                    provider.reset_cabinet_joint("joint_0", 0.0)
                    layout_reset_index += 1
                    layout_result = layout_manager.reset_layout(reset_index=layout_reset_index)
                    _settle_layout(env, provider)
                    state = provider.get_state()
                    q_cmd = provider.arm_joint_pos(state).to(provider.device).float()
                    q_target = q_cmd.clone()
                    print("[GELLO] scene reset; q_cmd re-initialized to current Franka q.", flush=True)

                # Non-blocking grab of the latest background sample (serial read runs in the thread).
                q_gello, gripper_raw, read_ok, read_ms, read_hz = gello_reader.get_latest()
                if q_gello is not None:
                    q_target = torch.clamp(                                      # joint-limit clip
                        torch.as_tensor(q_gello, dtype=torch.float32, device=provider.device), q_lower, q_upper
                    )
                    if gripper_raw is not None:
                        gello_last_gripper = gripper_raw
                # else: reuse previous q_target (already initialized to the Franka q).
                # Filter + rate-limit toward q_target EVERY frame (this is what removes the lag).
                q_filtered = (1.0 - gello_alpha) * q_cmd + gello_alpha * q_target  # low-pass
                q_delta = torch.clamp(q_filtered - q_cmd, -gello_max_delta, gello_max_delta)  # rate limit
                q_cmd = q_cmd + q_delta

                # --- gripper: binary command with hysteresis (deadband) ---
                gripper_cmd = 1.0  # default OPEN
                if args_cli.gello_enable_gripper and gello_last_gripper is not None:
                    raw = gello_last_gripper
                    db = args_cli.gello_gripper_deadband
                    if not args_cli.gello_gripper_invert:
                        if gripper_state == "open" and raw < gripper_threshold - db:
                            gripper_state = "close"
                        elif gripper_state == "close" and raw > gripper_threshold + db:
                            gripper_state = "open"
                    else:
                        if gripper_state == "open" and raw > gripper_threshold + db:
                            gripper_state = "close"
                        elif gripper_state == "close" and raw < gripper_threshold - db:
                            gripper_state = "open"
                    gripper_cmd = 1.0 if gripper_state == "open" else -1.0
                actions = provider.make_joint_action_from_q_des(q_cmd, gripper_cmd)

                # real (sim) loop rate
                now = time.perf_counter()
                dt_loop = now - gello_last_loop_t
                gello_last_loop_t = now
                if dt_loop > 0:
                    inst_hz = 1.0 / dt_loop
                    gello_loop_hz = inst_hz if gello_loop_hz == 0.0 else 0.9 * gello_loop_hz + 0.1 * inst_hz
                if args_cli.gello_print_every > 0 and step_count % args_cli.gello_print_every == 0:
                    cur_q = provider.arm_joint_pos(state).to(provider.device).float()
                    track_err = (q_cmd - cur_q)
                    qg_s = "n/a" if q_gello is None else "[" + ", ".join(f"{v:+.3f}" for v in q_gello) + "]"
                    qt_s = "[" + ", ".join(f"{v:+.3f}" for v in q_target.tolist()) + "]"
                    qc_s = "[" + ", ".join(f"{v:+.3f}" for v in q_cmd.tolist()) + "]"
                    cf_s = "[" + ", ".join(f"{v:+.3f}" for v in cur_q.tolist()) + "]"
                    te_s = "[" + ", ".join(f"{v:+.3f}" for v in track_err.tolist()) + "]"
                    grip_s = "n/a" if gello_last_gripper is None else f"{gello_last_gripper:.2f}"
                    act0 = "[" + ", ".join(f"{v:+.3f}" for v in actions[0].tolist()) + "]"
                    flag = "" if read_ok else "  <last read FAILED: reusing q_target>"
                    print(
                        f"[GELLO] real_loop_hz={gello_loop_hz:5.1f} gello_read_hz={read_hz:5.1f} "
                        f"read_ms={read_ms:5.1f}{flag}\n"
                        f"        q_gello={qg_s}\n"
                        f"        q_target={qt_s}\n"
                        f"        q_cmd  ={qc_s}\n"
                        f"        franka ={cf_s}\n"
                        f"        track_err(q_cmd-franka)={te_s}\n"
                        f"        gripper_raw={grip_s} gripper_cmd={gripper_cmd:+.1f} "
                        f"gripper_state={gripper_state} threshold={gripper_threshold:.1f}\n"
                        f"        actions[0]={act0}",
                        flush=True,
                    )
            else:
                pending = controller.pop()
                handled_control_command = False
                if pending is not None:
                    if pending.command == "start" and pending.request is not None:
                        executor.start(pending.request, state)
                    elif pending.command == "stop":
                        command = executor.pause(state)
                        actions = _command_to_action(provider, command, state)
                        handled_control_command = True
                    elif pending.command == "resume":
                        executor.resume(state)
                    elif pending.command == "reset":
                        executor.reset()
                        env.reset(seed=args_cli.seed)
                        provider.reset_cabinet_joint("joint_0", 0.0)
                        layout_reset_index += 1
                        layout_result = layout_manager.reset_layout(reset_index=layout_reset_index)
                        _settle_layout(env, provider)
                        state = provider.get_state()
                        actions = provider.make_hold_joint_action(state, 1.0)
                        handled_control_command = True

                if not handled_control_command:
                    command = executor.step(state, sim_dt)
                    _apply_drawer_joint_command(provider, command, baseline_warned)
                    actions = _command_to_action(provider, command, state)
            if command is not None:
                debug_logger.maybe_log(step_count, executor, command, state)
            env.step(actions)

            place_points = window.place_points if window is not None else DEFAULT_PLACE_POINTS
            _update_debug_visuals(visualizer, state, executor, place_points)
            if window is not None:
                window.update(state, executor, layout_result)

            sim_time += sim_dt
            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                if executor.status == ExecutionStatus.RUNNING:
                    executor.pause(provider.get_state())
                break
            if args_cli.headless and args_cli.auto_start and executor.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.NOT_IMPLEMENTED,
                ExecutionStatus.STOPPED,
            }:
                break

    if gello_reader is not None:
        gello_reader.stop()
    env.close()


def _update_debug_visuals(
    visualizer: DebugVisualizer,
    state,
    executor: SkillExecutor,
    place_points: dict[str, list[float]] | None = None,
):
    visualizer.update_pose("current_tcp", pose_tensor(state.robot.tcp_pose), use_coordinate_arrows=True)
    for object_name in ("cube_1", "cube_2", "cube_3", "knife"):
        obj = state.objects.get(object_name)
        if obj is not None:
            visualizer.update_pose(
                f"object_{object_name}",
                pose_tensor(obj.pose),
                axis_length=OBJECT_AXIS_LENGTH,
                axis_width=OBJECT_AXIS_WIDTH,
            )
    if place_points:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=state.env_origin_w.device)
        visual_offset = torch.tensor(
            [0.0, 0.0, PLACE_POINT_VISUAL_Z_OFFSET],
            dtype=torch.float32,
            device=state.env_origin_w.device,
        )
        for point_name, point_xyz in place_points.items():
            point_local = torch.tensor(point_xyz, dtype=torch.float32, device=state.env_origin_w.device)
            marker_pos_w = state.env_origin_w + point_local + visual_offset
            visualizer.update_pose(
                f"place_{point_name}",
                torch.cat((marker_pos_w, quat), dim=-1),
                axis_length=PLACE_POINT_AXIS_LENGTH,
                axis_width=PLACE_POINT_AXIS_WIDTH,
            )
    active = executor.active_skill
    runtime = getattr(active, "runtime", None)
    if runtime is None:
        return
    plan = getattr(runtime, "filtered_plan", None)
    if plan is not None and hasattr(plan, "handle_pose"):
        visualizer.update_pose("drawer_handle", pose_tensor(plan.handle_pose))
        visualizer.update_pose("drawer_pre_target", pose_tensor(plan.pre_handle_pose))
        action_pose = plan.push_target_pose if getattr(active, "request", None) and active.request.skill_type == SkillType.CLOSE_DRAWER else plan.pull_pose
        visualizer.update_pose("drawer_action_target", pose_tensor(action_pose))
    if getattr(runtime, "last_command_pose", None) is not None:
        visualizer.update_pose("current_stage_target", pose_tensor(runtime.last_command_pose), use_coordinate_arrows=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt received; shutting down cleanly.", flush=True)
    finally:
        simulation_app.close()
