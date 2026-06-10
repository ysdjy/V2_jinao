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
from skill_runtime.debug_visualizer import DebugVisualizer
from skill_runtime.scene_state_provider import SceneStateProvider
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
        self.status_labels = {}
        self.window = ui.Window("Franka Skill Test", width=420, height=520)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Skill")
                self.skill_model = ui.ComboBox(0, *[label for label, _ in SKILL_LABELS]).model
                self.skill_model.add_item_changed_fn(self._on_skill_changed)
                ui.Label("Grasp target")
                self.target_model = ui.ComboBox(1, *[label for _, label in registry.display_targets()]).model
                self.target_model.add_item_changed_fn(self._on_target_changed)
                with ui.HStack(spacing=6):
                    ui.Button("Start", clicked_fn=self._start)
                    ui.Button("Stop", clicked_fn=self.controller.request_stop)
                    ui.Button("Reset", clicked_fn=self.controller.request_reset)
                for key in (
                    "selected_skill",
                    "selected_target",
                    "runtime_status",
                    "state",
                    "elapsed",
                    "position_error",
                    "orientation_error",
                    "gripper_width",
                    "target_pose",
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

    def _start(self):
        self.controller.request_start()

    def update(self, state, executor: SkillExecutor):
        result = executor.last_result
        active = executor.active_skill
        plan = getattr(getattr(active, "runtime", None), "filtered_plan", None)
        target_pose = plan.target_pose.as_pose_tensor() if plan else None
        elapsed = 0.0
        if active is not None:
            elapsed = max(0.0, state.sim_time - getattr(active.runtime, "start_time", state.sim_time))
        values = {
            "selected_skill": self.controller.selected_skill.value,
            "selected_target": self.controller.selected_target,
            "runtime_status": executor.status.value,
            "state": executor.current_state_name,
            "elapsed": f"{elapsed:.2f}",
            "position_error": str(getattr(getattr(active, "runtime", None), "final_error_pos", None)),
            "orientation_error": str(getattr(getattr(active, "runtime", None), "final_error_ori", None)),
            "gripper_width": f"{state.robot.gripper_width:.5f}",
            "target_pose": _short_pose(target_pose),
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


def _skill_type_from_arg(value: str) -> SkillType:
    return SkillType(value)


def _make_request(skill_type: SkillType, target: str) -> SkillRequest:
    return SkillRequest(
        request_id=f"{skill_type.value}_{target}_{time.time_ns()}",
        skill_type=skill_type,
        source_object=target if skill_type == SkillType.GRASP else None,
        destination_object="cabinet" if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER) else None,
    )


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
    env_cfg.viewer.eye = (2.0, -2.0, 1.4)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.15)

    env = gym.make("Isaac-Stack-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    if not args_cli.headless and not args_cli.disable_collision_debug_vis:
        enable_collision_debug_visualization()

    provider = SceneStateProvider(env)
    provider.reset_scene_deterministic()
    registry = TargetRegistry(env.unwrapped.device)
    executor = SkillExecutor(registry)
    controller = UIController()
    controller.selected_skill = _skill_type_from_arg(args_cli.skill)
    controller.selected_target = args_cli.target
    visualizer = DebugVisualizer(enabled=not args_cli.headless or args_cli.show_affordance_debug)
    window = None if args_cli.headless else SkillTestWindow(controller, executor, registry)

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    sim_time = 0.0
    provider.set_sim_time(sim_time)
    state = provider.get_state()
    actions = provider.make_action(state.robot.tcp_pose, 1.0)

    if args_cli.auto_start:
        controller.request_start(controller.selected_skill, controller.selected_target)

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            provider.set_sim_time(sim_time)
            state = provider.get_state()
            pending = controller.pop()
            if pending is not None:
                if pending.command == "start" and pending.request is not None:
                    executor.start(pending.request, state)
                elif pending.command == "stop":
                    command = executor.stop(state)
                    actions = provider.make_action(command.tcp_pose_w, command.gripper_command)
                elif pending.command == "reset":
                    executor.stop(state)
                    env.reset(seed=args_cli.seed)
                    provider.reset_scene_deterministic()
                    executor.reset()
                    state = provider.get_state()
                    actions = provider.make_action(state.robot.tcp_pose, 1.0)

            command = executor.step(state, sim_dt)
            actions = provider.make_action(command.tcp_pose_w, command.gripper_command)
            env.step(actions)

            _update_debug_visuals(visualizer, state, executor)
            if window is not None:
                window.update(state, executor)

            sim_time += sim_dt
            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                if executor.status == ExecutionStatus.RUNNING:
                    executor.stop(provider.get_state())
                break
            if args_cli.headless and args_cli.auto_start and executor.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.NOT_IMPLEMENTED,
                ExecutionStatus.STOPPED,
            }:
                break

    env.close()


def _update_debug_visuals(visualizer: DebugVisualizer, state, executor: SkillExecutor):
    visualizer.update_pose("current_tcp", pose_tensor(state.robot.tcp_pose))
    active = executor.active_skill
    runtime = getattr(active, "runtime", None)
    if runtime is None:
        return
    plan = runtime.filtered_plan
    if plan is not None:
        visualizer.update_pose("target_object", pose_tensor(plan.target_pose))
        visualizer.update_pose("grasp_frame", pose_tensor(plan.grasp_pose))
        visualizer.update_pose("pre_grasp_frame", pose_tensor(plan.pre_grasp_pose))
    visualizer.update_pose("active_command", pose_tensor(runtime.last_command_pose))
    visualizer.update_pose("locked_lift_frame", pose_tensor(runtime.locked_lift_pose))


if __name__ == "__main__":
    main()
    simulation_app.close()
