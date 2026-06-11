# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic skill-test entry for the modified Franka stack scene.

GUI:
    ./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui.py --num_envs 1

Headless:
    ./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui.py \
        --headless --num_envs 1 --auto_start --skill grasp --target cube_1 --max_steps 1200
"""

from __future__ import annotations

"""Launch Omniverse Toolkit first."""

import argparse
import json
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Sim native UI for structured Franka skills.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--disable_collision_debug_vis", action="store_true", default=False, help="Disable collider overlays.")
parser.add_argument("--show_affordance_debug", action="store_true", default=False, help="Show affordance debug frames.")
parser.add_argument("--auto_start", action="store_true", default=False, help="Start the selected skill automatically.")
parser.add_argument("--skill", default="grasp", choices=["grasp", "place", "open_drawer", "close_drawer"], help="Skill.")
parser.add_argument(
    "--target", default="cube_2", choices=["cube_1", "cube_2", "cube_3", "knife"], help="Grasp target."
)
parser.add_argument("--max_steps", type=int, default=0, help="Maximum sim steps before exit. 0 means no limit.")
parser.add_argument("--seed", type=int, default=1, help="Deterministic skill-test seed.")
parser.add_argument("--randomize_test_pose", action="store_true", default=False, help="Apply small test pose perturbations.")
parser.add_argument("--layout_only", action="store_true", default=False, help="Only sample and validate layout.")
parser.add_argument("--layout_trials", type=int, default=1, help="Number of layout-only trials to run.")
parser.add_argument("--show_layout_debug", action="store_true", default=False, help="Show layout debug overlays.")
parser.add_argument(
    "--cube_grasp_z_offset",
    type=float,
    default=0.0,
    help="TCP z offset from cube center for grasp debugging, in [-0.010, 0.015] m.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Skill UI supports 1.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import carb
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from skill_runtime.base_skill import pose_tensor
from skill_runtime.debug_visualizer import DebugVisualizer, OBJECT_AXIS_LENGTH, OBJECT_AXIS_WIDTH
from skill_runtime.scene_state_provider import SceneStateProvider
from skill_runtime.simple_scene_layout import SimpleLayoutResult, SimpleSceneLayoutManager
from skill_runtime.skill_executor import SkillExecutor
from skill_runtime.skill_request import SkillRequest
from skill_runtime.skill_types import ExecutionStatus, SkillType
from skill_runtime.target_registry import TargetRegistry
from skill_runtime.ui_controller import UIController


SKILL_LABELS = [
    ("Grasp", SkillType.GRASP),
    ("Place", SkillType.PLACE),
    ("Open Drawer", SkillType.OPEN_DRAWER),
    ("Close Drawer", SkillType.CLOSE_DRAWER),
]

DRAWER_TARGETS = [
    ("Bottom Drawer", "bottom_drawer"),
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
        self.window = ui.Window("Franka Skill Test", width=460, height=700)
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
                    "cabinet_min_z",
                    "cube_1_pose",
                    "cube_2_pose",
                    "cube_3_pose",
                    "knife_pose",
                    "minimum_pair_clearance",
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
        target = self.selected_drawer if self.controller.selected_skill in (
            SkillType.OPEN_DRAWER,
            SkillType.CLOSE_DRAWER,
        ) else self.controller.selected_target
        self.controller.queue_request(_make_request(self.controller.selected_skill, target))

    def update(self, state, executor: SkillExecutor, layout_result: SimpleLayoutResult | None):
        result = executor.last_result
        active = executor.active_skill
        plan = getattr(getattr(active, "runtime", None), "filtered_plan", None)
        target_pose_state = getattr(plan, "target_pose", None) or getattr(plan, "handle_pose", None)
        target_pose = target_pose_state.as_pose_tensor() if target_pose_state is not None else None
        handle_pose_state = getattr(plan, "handle_pose", None)
        drawer_joint_position = None
        cabinet = state.objects.get("cabinet")
        if cabinet is not None:
            drawer_joint_position = cabinet.joint_pos.get("joint_0")
        elapsed = 0.0
        if active is not None:
            elapsed = max(0.0, state.sim_time - getattr(active.runtime, "start_time", state.sim_time))
        orientation_error = getattr(getattr(active, "runtime", None), "final_error_ori", None)
        drawer_runtime = getattr(active, "runtime", None)
        drawer_joint_name = getattr(drawer_runtime, "drawer_joint_name", "joint_0")
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
            "cabinet_min_z": None,
            "cube_1_pose": None if layout_result is None else layout_result.object_poses.get("cube_1"),
            "cube_2_pose": None if layout_result is None else layout_result.object_poses.get("cube_2"),
            "cube_3_pose": None if layout_result is None else layout_result.object_poses.get("cube_3"),
            "knife_pose": None if layout_result is None else layout_result.object_poses.get("knife"),
            "minimum_pair_clearance": None,
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
            destination_object=target,
            parameters={
                "drawer_link": "link_1",
                "drawer_joint": "joint_0",
            },
        )
    return SkillRequest(
        request_id=f"{skill_type.value}_{target}_{time.time_ns()}",
        skill_type=skill_type,
        source_object=target if skill_type == SkillType.GRASP else None,
        destination_object="cabinet" if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER) else None,
    )


def _print_layout_summary(layout_result: SimpleLayoutResult) -> None:
    print(
        json.dumps(
            {
                "layout_seed": layout_result.seed,
                "reset_index": layout_result.reset_index,
                "object_poses": layout_result.object_poses,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _settle_layout(env, provider: SceneStateProvider) -> None:
    provider.reset_cabinet_joint("joint_0", 0.0)
    state = provider.get_state()
    hold_action = provider.make_action(state.robot.tcp_pose, 1.0)
    for _ in range(5):
        env.step(hold_action)


def _apply_drawer_joint_command(provider: SceneStateProvider, command) -> None:
    if command.drawer_joint_target is None:
        return
    provider.set_cabinet_joint_target(command.drawer_joint_name or "joint_0", command.drawer_joint_target)


def _hold_drawer_joint(provider: SceneStateProvider, state) -> None:
    cabinet = state.objects.get("cabinet")
    if cabinet is None:
        return
    joint_pos = cabinet.joint_pos.get("joint_0")
    if joint_pos is None:
        return
    provider.set_cabinet_joint_target("joint_0", joint_pos)


def main():
    if args_cli.num_envs != 1:
        raise ValueError("skill_test_ui currently supports --num_envs 1 so one UI maps to one scene.")

    torch.manual_seed(args_cli.seed)
    env_cfg: StackEnvCfg = parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Abs-v0",
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

    env = gym.make("Isaac-Stack-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    if not args_cli.headless and not args_cli.disable_collision_debug_vis:
        enable_collision_debug_visualization()

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    registry = TargetRegistry(env.unwrapped.device, cube_grasp_z_offset=args_cli.cube_grasp_z_offset)
    executor = SkillExecutor(registry)
    controller = UIController()
    controller.selected_skill = _skill_type_from_arg(args_cli.skill)
    controller.selected_target = args_cli.target
    visualizer = DebugVisualizer(enabled=not args_cli.headless or args_cli.show_affordance_debug)
    window = None if args_cli.headless else SkillTestWindow(controller, executor, registry)

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    sim_time = 0.0
    provider.set_sim_time(sim_time)

    if args_cli.layout_only:
        for trial in range(args_cli.layout_trials):
            provider.set_sim_time(0.0)
            layout_result = layout_manager.reset_layout(reset_index=trial)
            _print_layout_summary(layout_result)
        env.close()
        return

    layout_result = layout_manager.reset_layout(reset_index=0)
    _settle_layout(env, provider)

    if args_cli.auto_start:
        target = "bottom_drawer" if controller.selected_skill in (
            SkillType.OPEN_DRAWER,
            SkillType.CLOSE_DRAWER,
        ) else controller.selected_target
        controller.queue_request(_make_request(controller.selected_skill, target))

    step_count = 0
    layout_reset_index = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            provider.set_sim_time(sim_time)
            state = provider.get_state()
            pending = controller.pop()
            handled_control_command = False
            if pending is not None:
                if pending.command == "start" and pending.request is not None:
                    executor.start(pending.request, state)
                elif pending.command == "stop":
                    command = executor.pause(state)
                    _hold_drawer_joint(provider, state)
                    actions = provider.make_action(command.tcp_pose_w, command.gripper_command)
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
                    actions = provider.hold_action(state, 1.0)
                    handled_control_command = True

            if not handled_control_command:
                command = executor.step(state, sim_dt)
                _apply_drawer_joint_command(provider, command)
                actions = provider.make_action(command.tcp_pose_w, command.gripper_command)
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
    visualizer.update_pose("current_stage_target", pose_tensor(runtime.last_command_pose), use_coordinate_arrows=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
