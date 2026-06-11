# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-action state-machine entry (method B).

Creates a *joint-position* env (``Isaac-Stack-Cube-Franka-JointPolicy-v0``) and runs a sequence of
skills through the unified :class:`SkillExecutor`. Every skill emits a joint action:

  * GRASP  -> GraspJointSkill  (internal DLS IK -> q_des)
  * PLACE  -> PlaceJointSkill   (internal DLS IK -> q_des)
  * OPEN_DRAWER -> OfficialDrawerJointSkill (official PPO policy -> raw joint action)
                   or ScriptedDrawerJointSkill baseline (--drawer_backend scripted_joint)

Does NOT touch the original IK-Abs ``skill_test_ui.py``.

Example (scripted drawer baseline, headless):
    ./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py \
        --num_envs 1 --sequence grasp:cube_1,place:point_a,open_drawer:bottom_drawer \
        --grasp_backend joint_ik --place_backend joint_ik --drawer_backend scripted_joint \
        --seed 1 --max_steps 3000 --headless
"""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Joint-action state machine for Franka skills.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (this entry supports 1).")
parser.add_argument("--seed", type=int, default=1, help="Deterministic seed.")
parser.add_argument(
    "--sequence",
    type=str,
    default="grasp:cube_1,place:point_a,open_drawer:bottom_drawer",
    help="Comma-separated skill:target list, e.g. grasp:cube_1,place:point_a,open_drawer:bottom_drawer",
)
parser.add_argument("--grasp_backend", type=str, default="joint_ik", choices=["joint_ik"], help="Grasp backend.")
parser.add_argument("--place_backend", type=str, default="joint_ik", choices=["joint_ik"], help="Place backend.")
parser.add_argument(
    "--drawer_backend",
    type=str,
    default="official_joint_policy",
    choices=["official_joint_policy", "scripted_joint"],
    help="Drawer backend.",
)
parser.add_argument("--drawer_policy_path", type=str, default=None, help="TorchScript policy.pt for official drawer.")
parser.add_argument("--drawer_joint_name", type=str, default="joint_0", help="Cabinet drawer joint to read for success.")
parser.add_argument("--drawer_success_threshold", type=float, default=0.20, help="Drawer-open success threshold (m).")
parser.add_argument("--max_steps", type=int, default=3000, help="Global hard cap on sim steps.")
parser.add_argument("--log_every", type=int, default=30, help="Low-frequency debug log interval (sim steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--show_affordance_debug", action="store_true", default=False, help="Show affordance debug frames.")
parser.add_argument(
    "--cube_grasp_z_offset", type=float, default=0.0, help="TCP z offset from cube center, in [-0.010, 0.015] m."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest follows after the app is launched."""

from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from skill_runtime.drawer_obs_adapter import DrawerObsAdapter
from skill_runtime.drawer_target_config import DRAWER_TARGETS
from skill_runtime.ik_joint_adapter import IKJointAdapter
from skill_runtime.joint_debug_logger import JointDebugLogger
from skill_runtime.official_drawer_policy import OfficialDrawerPolicyWrapper
from skill_runtime.scene_state_provider import SceneStateProvider
from skill_runtime.simple_scene_layout import SimpleSceneLayoutManager
from skill_runtime.skill_executor import JointBackendConfig, SkillExecutor
from skill_runtime.skill_request import SkillRequest
from skill_runtime.skill_types import ExecutionStatus, SkillType
from skill_runtime.target_registry import TargetRegistry


TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"
JOINT_LOG_PATH = Path("logs/skill_tests/joint_sequence_results.jsonl")
TERMINAL = {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED, ExecutionStatus.NOT_IMPLEMENTED}

DEFAULT_PLACE_POINTS = {
    "point_a": [0.42, 0.10, 0.00],
    "point_b": [0.55, 0.10, 0.00],
    "point_c": [0.68, 0.10, 0.00],
}


def _parse_sequence(text: str) -> list[tuple[SkillType, str]]:
    items: list[tuple[SkillType, str]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        skill_str, _, target = token.partition(":")
        items.append((SkillType(skill_str.strip()), target.strip()))
    return items


def _make_request(skill_type: SkillType, target: str) -> SkillRequest:
    rid = f"{skill_type.value}_{target or 'none'}_{time.time_ns()}"
    if skill_type == SkillType.PLACE:
        xyz = DEFAULT_PLACE_POINTS.get(target, DEFAULT_PLACE_POINTS["point_a"])
        return SkillRequest(
            request_id=rid,
            skill_type=skill_type,
            source_object=None,
            destination_type="point",
            destination_object=target,
            parameters={"target_frame": "env_local", "target_surface_xyz": list(xyz)},
        )
    if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
        return SkillRequest(
            request_id=rid,
            skill_type=skill_type,
            source_object=None,
            destination_type="drawer",
            destination_object=target or "bottom_drawer",
            # joint is resolved from the central target->joint config inside the drawer skill
            parameters={},
        )
    return SkillRequest(request_id=rid, skill_type=skill_type, source_object=target)


def _command_to_action(provider: SceneStateProvider, command, state):
    if command.control_mode == "joint":
        if command.raw_joint_action is not None:
            return provider.make_joint_action_from_raw(command.raw_joint_action)
        if command.joint_target is not None:
            return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
        return provider.make_hold_joint_action(state, None)
    # legacy IK fallback (not expected in this entry)
    return provider.make_action(command.tcp_pose_w, command.gripper_command)


def _append_record(record: dict):
    JOINT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOINT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _settle(env, provider, sim_dt):
    provider.reset_cabinet_joint("joint_0", 0.0)
    state = provider.get_state()
    action = provider.make_hold_joint_action(state, 1.0)
    for _ in range(5):
        env.step(action)


def main():
    if args_cli.num_envs != 1:
        raise ValueError("skill_sequence_joint supports --num_envs 1.")

    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
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

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    registry = TargetRegistry(env.unwrapped.device, cube_grasp_z_offset=args_cli.cube_grasp_z_offset)
    adapter = IKJointAdapter(env)

    drawer_policy = None
    drawer_obs_adapter = None
    if args_cli.drawer_backend == "official_joint_policy":
        # raises immediately with guidance if the path is missing/not found (no silent fallback)
        drawer_policy = OfficialDrawerPolicyWrapper(args_cli.drawer_policy_path, device=env.unwrapped.device)
        drawer_obs_adapter = DrawerObsAdapter(env, drawer_joint_name=args_cli.drawer_joint_name)

    backend = JointBackendConfig(
        mode="joint",
        grasp_backend=args_cli.grasp_backend,
        place_backend=args_cli.place_backend,
        drawer_backend=args_cli.drawer_backend,
        adapter=adapter,
        drawer_policy=drawer_policy,
        drawer_obs_adapter=drawer_obs_adapter,
        arm_joint_ids=provider._arm_joint_ids,
        drawer_joint_name=args_cli.drawer_joint_name,
        drawer_success_threshold=args_cli.drawer_success_threshold,
    )
    executor = SkillExecutor(registry, backend=backend)
    debug_logger = JointDebugLogger(every_steps=args_cli.log_every)
    baseline_warned = {"done": False}

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    sim_time = 0.0
    provider.set_sim_time(sim_time)

    layout_manager.reset_layout(reset_index=0)
    _settle(env, provider, sim_dt)

    sequence = _parse_sequence(args_cli.sequence)
    sequence_id = f"seq_{int(args_cli.seed)}_{time.time_ns()}"
    print(f"[skill_sequence_joint] sequence_id={sequence_id} items={[ (s.value,t) for s,t in sequence]}", flush=True)

    step_count = 0
    for index, (skill_type, target) in enumerate(sequence):
        request = _make_request(skill_type, target)
        provider.set_sim_time(sim_time)
        state = provider.get_state()
        skill_start_sim = sim_time
        skill_start_wall = time.time()
        executor.start(request, state)
        active = executor.active_skill
        print(
            f"[skill_sequence_joint] START skill={skill_type.value} target={target} "
            f"backend={getattr(active, 'backend', 'n/a')} class={active.__class__.__name__ if active else None}",
            flush=True,
        )

        while simulation_app.is_running() and step_count < args_cli.max_steps:
            provider.set_sim_time(sim_time)
            state = provider.get_state()
            with torch.inference_mode():
                command = executor.step(state, sim_dt)
                if command.drawer_joint_target is not None:
                    if not baseline_warned["done"]:
                        print(
                            "[BASELINE] scripted_joint is directly commanding drawer joint, "
                            "not learned physical pulling.",
                            flush=True,
                        )
                        baseline_warned["done"] = True
                    provider.set_cabinet_joint_target(
                        command.drawer_joint_name or "joint_0", command.drawer_joint_target
                    )
                action = _command_to_action(provider, command, state)
                debug_logger.maybe_log(step_count, executor, command, state)
                env.step(action)
            sim_time += sim_dt
            step_count += 1
            if executor.status in TERMINAL:
                break

        final_state = provider.get_state()
        cabinet = final_state.objects.get("cabinet")
        # success joint follows the target drawer (top=joint_0, middle=joint_2, bottom=joint_1)
        success_joint = args_cli.drawer_joint_name
        if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            success_joint = DRAWER_TARGETS.get(target, {}).get("joint_name", args_cli.drawer_joint_name)
        drawer_joint_pos = None if cabinet is None else cabinet.joint_pos.get(success_joint)
        runtime = getattr(active, "runtime", None)
        record = {
            "sequence_id": sequence_id,
            "index": index,
            "skill_type": skill_type.value,
            "backend": getattr(active, "backend", None),
            "target": target,
            "start_time": round(skill_start_sim, 4),
            "end_time": round(sim_time, 4),
            "elapsed": round(sim_time - skill_start_sim, 4),
            "wall_elapsed": round(time.time() - skill_start_wall, 4),
            "final_status": executor.status.value,
            "failure_reason": None if executor.last_result is None else executor.last_result.failure_reason,
            "final_robot_joint_pos": [round(float(v), 5) for v in final_state.robot.joint_pos.tolist()],
            "final_tcp_pose": [round(float(v), 5) for v in final_state.robot.tcp_pose.as_pose_tensor().tolist()],
            "drawer_joint_pos": None if drawer_joint_pos is None else round(float(drawer_joint_pos), 5),
            "policy_path": getattr(drawer_policy, "policy_path", None) if skill_type == SkillType.OPEN_DRAWER else None,
            "obs_shape": list(getattr(runtime, "obs_shape", None)) if getattr(runtime, "obs_shape", None) else None,
            "action_shape": list(getattr(runtime, "action_shape", None))
            if getattr(runtime, "action_shape", None)
            else None,
        }
        _append_record(record)
        print(f"[skill_sequence_joint] END {record}", flush=True)

        if step_count >= args_cli.max_steps:
            print("[skill_sequence_joint] reached --max_steps, stopping sequence.", flush=True)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
