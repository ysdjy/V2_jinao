# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level cabinet / drawer-joint / handle / gripper diagnostics (no skills, no policy, no training).

Stage-1 gate for the selected-drawer policy work: before training we must confirm, for the custom
cabinet, the top/middle/bottom -> joint mapping, the open direction, whether any drawer is jammed or
penetrating at the closed pose, and whether the handle frames are correct. This tool scans the
drawer joints with the robot held still and writes results to
``logs/skill_tests/drawer_joint_scan_results.jsonl``.

It does NOT run any skill, does NOT load a policy, and does NOT fake drawer success.

Run:
    ./isaaclab.sh -p scripts/environments/state_machine/debug_drawer_joint_scan.py \
        --num_envs 1 --drawer_joint all --values 0.00,0.05,0.10,0.20,0.30 \
        --show_affordance_debug --seed 1
"""

from __future__ import annotations

import argparse

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cabinet/drawer/handle/gripper low-level diagnostics.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (this tool supports 1).")
parser.add_argument("--seed", type=int, default=1, help="Deterministic seed.")
parser.add_argument("--show_affordance_debug", action="store_true", default=False, help="Show debug frames.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric (use for valid USD reads).")
parser.add_argument("--dwell_seconds", type=float, default=1.0, help="Seconds to hold at each scanned joint value.")
parser.add_argument("--drawer_joint", type=str, default="all", help="Which joint to scan: joint_0|joint_1|joint_2|all.")
parser.add_argument("--values", type=str, default="0.00,0.05,0.10,0.20,0.30", help="Comma-separated joint values to scan.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest follows after the app is launched."""

import json
from pathlib import Path

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import gymnasium as gym

from runtime.base_skill import pose_tensor
from runtime.debug_visualizer import DebugVisualizer
from runtime.drawer_obs_adapter import DrawerObsAdapter
from runtime.scene_state_provider import SceneStateProvider
from runtime.simple_scene_layout import SimpleSceneLayoutManager

TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"
HANDLE_PROXY_PRIM = "/World/envs/env_0/Cabinet/link_1/BottomHandleProxy"
LOG_PATH = Path("logs/skill_tests/drawer_joint_scan_results.jsonl")
JAM_EPS = 0.01  # m: body displacement below this when commanding +0.20 => treated as jammed


def _round(seq, n=4):
    return [round(float(v), n) for v in seq]


def _append(record: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _usd_world_pose(stage, prim_path):
    try:
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        im = q.GetImaginary()
        return [float(t[0]), float(t[1]), float(t[2])], [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]
    except Exception:
        return None


def _print_cabinet_joint_info(provider, stage):
    cabinet = provider.scene["cabinet"]
    names = list(getattr(cabinet.data, "joint_names", []))
    body_names = list(getattr(cabinet.data, "body_names", []))
    limits = getattr(cabinet.data, "joint_pos_limits", None)
    default = getattr(cabinet.data, "default_joint_pos", None)
    body_pos = cabinet.data.body_pos_w[provider.env_id]
    print("\n========== CABINET INFO ==========", flush=True)
    print(f"cabinet_body_names = {body_names}", flush=True)
    print("body world z (height) -> bottom=lowest z:", flush=True)
    for i, n in enumerate(body_names):
        print(f"  body[{i}] {n}: pos_w={_round(body_pos[i].tolist(),3)}", flush=True)
    print(f"cabinet_joint_names = {names}", flush=True)
    for i, n in enumerate(names):
        lo_hi = None if limits is None else _round(limits[0, i].tolist())
        dpos = None if default is None else round(float(default[0, i]), 5)
        print(f"  joint[{i}] {n}: limits={lo_hi} default={dpos}", flush=True)
    print("--- USD joint prims (axis / limits) ---", flush=True)
    try:
        for prim in stage.Traverse():
            if "Joint" not in str(prim.GetTypeName()):
                continue
            nm = prim.GetName()
            if nm not in names:
                continue
            axis = prim.GetAttribute("physics:axis").Get() if prim.GetAttribute("physics:axis") else None
            lo = prim.GetAttribute("physics:lowerLimit").Get() if prim.GetAttribute("physics:lowerLimit") else None
            hi = prim.GetAttribute("physics:upperLimit").Get() if prim.GetAttribute("physics:upperLimit") else None
            print(f"  {prim.GetPath()} type={prim.GetTypeName()} axis={axis} lower={lo} upper={hi}", flush=True)
    except Exception as exc:
        print(f"  (USD traverse failed: {exc})", flush=True)
    print("==================================\n", flush=True)
    return names, body_names


def _settle_all_zero(env, provider, joint_names, n_steps):
    for jn in joint_names:
        provider.reset_cabinet_joint(jn, 0.0)
    state = provider.get_state()
    for _ in range(n_steps):
        for jn in joint_names:
            provider.set_cabinet_joint_target(jn, 0.0)
        env.step(provider.make_hold_joint_action(state, 1.0))


def _joint_body_mapping_probe(env, provider, joint_names, body_names, sim_dt):
    """For each joint: teleport to +0.20, read body motion IMMEDIATELY and AFTER settling.

    Distinguishes: works (both move) / jammed-collision (immediate moves, settle returns) /
    locked (neither moves).
    """
    cabinet = provider.scene["cabinet"]
    n_settle = max(1, int(0.6 / sim_dt))
    results = {}
    print("\n========== JOINT -> BODY MAPPING PROBE ==========", flush=True)
    for ji, jn in enumerate(joint_names):
        _settle_all_zero(env, provider, joint_names, n_settle)
        base = cabinet.data.body_pos_w[provider.env_id].clone()

        # teleport joint to 0.20, step ONCE (immediate)
        provider.reset_cabinet_joint(jn, 0.20)
        state = provider.get_state()
        provider.set_cabinet_joint_target(jn, 0.20)
        env.step(provider.make_hold_joint_action(state, 1.0))
        imm = cabinet.data.body_pos_w[provider.env_id].clone()
        imm_actual = float(cabinet.data.joint_pos[provider.env_id, ji])
        imm_delta = (imm - base).norm(dim=-1)
        imm_top = int(imm_delta.argmax())

        # keep holding target, settle
        for _ in range(n_settle):
            provider.set_cabinet_joint_target(jn, 0.20)
            env.step(provider.make_hold_joint_action(state, 1.0))
        settled = cabinet.data.body_pos_w[provider.env_id].clone()
        set_actual = float(cabinet.data.joint_pos[provider.env_id, ji])
        set_delta = (settled - base).norm(dim=-1)
        set_top = int(set_delta.argmax())
        moved_body = body_names[set_top] if float(set_delta[set_top]) > JAM_EPS else body_names[imm_top]
        disp = _round((settled[set_top] - base[set_top]).tolist())

        if float(set_delta[set_top]) > JAM_EPS:
            verdict = "OK (moves and stays)"
        elif float(imm_delta[imm_top]) > JAM_EPS:
            verdict = "JAMMED (teleports then collision pushes back to closed)"
        else:
            verdict = "LOCKED (joint coordinate does not move the body at all)"

        print(
            f"  {jn}=+0.20: actual_after_teleport={imm_actual:.4f} actual_after_settle={set_actual:.4f}\n"
            f"     immediate: body '{body_names[imm_top]}' |Δ|={float(imm_delta[imm_top]):.4f}\n"
            f"     settled:   body '{body_names[set_top]}' |Δ|={float(set_delta[set_top]):.4f} Δxyz={disp}\n"
            f"     -> moved_body={moved_body}  VERDICT={verdict}",
            flush=True,
        )
        results[jn] = {
            "moved_body": moved_body,
            "immediate_disp": float(imm_delta[imm_top]),
            "settled_disp": float(set_delta[set_top]),
            "actual_after_teleport": imm_actual,
            "actual_after_settle": set_actual,
            "verdict": verdict,
            "settled_disp_xyz": disp,
        }
        _append({"phase": "mapping_probe", "joint": jn, **results[jn]})
    _settle_all_zero(env, provider, joint_names, n_settle)
    print("=================================================\n", flush=True)
    return results


def _classify_drawers(provider, joint_names, mapping):
    """Map joint -> moved body -> height (bottom=lowest z). Print top/middle/bottom inference."""
    cabinet = provider.scene["cabinet"]
    body_names = list(getattr(cabinet.data, "body_names", []))
    body_pos = cabinet.data.body_pos_w[provider.env_id]
    info = []
    for jn in joint_names:
        mb = mapping.get(jn, {}).get("moved_body")
        if mb is None or mb not in body_names:
            continue
        z = float(body_pos[body_names.index(mb)][2])
        info.append((jn, mb, z, mapping[jn]["verdict"]))
    info.sort(key=lambda x: x[2])  # ascending z
    labels = ["bottom_drawer", "middle_drawer", "top_drawer"]
    print("========== DRAWER CLASSIFICATION (by body height) ==========", flush=True)
    mapping_out = {}
    for label, (jn, mb, z, verdict) in zip(labels, info):
        print(f"  {label}: joint={jn} body={mb} z={z:.3f}  ({verdict})", flush=True)
        mapping_out[label] = {"joint_name": jn, "link_name": mb, "body_z": round(z, 4), "verdict": verdict}
    _append({"phase": "classification", "mapping": mapping_out})
    print("============================================================\n", flush=True)
    return mapping_out


def _gripper_sign_test(env, provider, sim_dt):
    print("\n========== GRIPPER SIGN TEST ==========", flush=True)
    robot = provider.scene["robot"]
    n_steps = max(1, int(0.6 / sim_dt))
    out = {}
    for label, cmd in (("OPEN(+1.0)", 1.0), ("CLOSE(-1.0)", -1.0)):
        state = provider.get_state()
        action = provider.make_joint_action_from_q_des(provider.arm_joint_pos(state), cmd)
        raw = float(action[0, -1])
        for _ in range(n_steps):
            env.step(action)
        state = provider.get_state()
        fp = robot.data.joint_pos[provider.env_id, provider._finger_joint_ids].tolist()
        print(f"  {label}: raw_gripper={raw:+.1f} finger_pos={_round(fp,5)} width={state.robot.gripper_width:.5f}", flush=True)
        out[label] = {"raw": raw, "finger_pos": _round(fp, 5), "width": round(state.robot.gripper_width, 5)}
    _append({"phase": "gripper_sign", **out})
    print("=======================================\n", flush=True)


def _scan_joint(env, provider, obs_adapter, visualizer, stage, jn, values, sim_dt):
    cabinet = provider.scene["cabinet"]
    names = list(getattr(cabinet.data, "joint_names", []))
    body_names = list(getattr(cabinet.data, "body_names", []))
    ji = names.index(jn)
    dwell = max(1, int(args_cli.dwell_seconds / sim_dt))
    print(f"\n========== SCAN {jn} ==========", flush=True)
    for value in values:
        provider.reset_cabinet_joint(jn, value)
        state = provider.get_state()
        hold = provider.make_hold_joint_action(state, 1.0)
        for _ in range(dwell):
            provider.set_cabinet_joint_target(jn, value)
            env.step(hold)
        state = provider.get_state()
        cab = state.objects.get("cabinet")
        actual = float(cabinet.data.joint_pos[provider.env_id, ji])
        body_pos = cabinet.data.body_pos_w[provider.env_id]
        body_dump = {n: _round(body_pos[i].tolist(), 3) for i, n in enumerate(body_names)}
        tcp = state.robot.tcp_pose
        computed_handle = obs_adapter._handle_pos_w()[provider.env_id]
        usd_handle = _usd_world_pose(stage, HANDLE_PROXY_PRIM)
        rel = _round((computed_handle - tcp.pos_w).tolist())
        print(
            f"{jn} commanded={value:+.3f} actual={actual:+.5f}\n"
            f"    all_body_pos_w = {body_dump}\n"
            f"    tcp_pos_w={_round(tcp.pos_w.tolist())} handle_computed={_round(computed_handle.tolist())} "
            f"rel_ee_drawer={rel}",
            flush=True,
        )
        _append({
            "phase": "scan", "joint": jn, "commanded": round(value, 4), "actual": round(actual, 5),
            "all_body_pos_w": body_dump, "tcp_pos_w": _round(tcp.pos_w.tolist()),
            "handle_computed_pos_w": _round(computed_handle.tolist()),
            "handle_usd_pose": None if usd_handle is None else _round(usd_handle[0]),
            "rel_ee_drawer": rel,
        })
        visualizer.update_pose("scan_tcp", pose_tensor(tcp), use_coordinate_arrows=True)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=computed_handle.device)
        visualizer.update_pose("scan_handle_computed", torch.cat((computed_handle, quat)))
    print("================================\n", flush=True)


def main():
    if args_cli.num_envs != 1:
        raise ValueError("debug_drawer_joint_scan supports --num_envs 1.")
    values = [float(v) for v in args_cli.values.split(",") if v.strip()]

    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_cube_positions"):
        env_cfg.events.randomize_cube_positions = None
    if hasattr(env_cfg.scene, "cabinet") and hasattr(env_cfg.scene.cabinet, "actuators"):
        if "drawers" in env_cfg.scene.cabinet.actuators:
            env_cfg.scene.cabinet.actuators["drawers"].stiffness = 10.0
            env_cfg.scene.cabinet.actuators["drawers"].damping = 1.0
    env_cfg.viewer.eye = (2.0, -2.0, 1.4)
    env_cfg.viewer.lookat = (0.45, 0.0, 0.15)

    env = gym.make(TASK_ID, cfg=env_cfg)
    env.reset(seed=args_cli.seed)

    provider = SceneStateProvider(env)
    layout_manager = SimpleSceneLayoutManager(env=env, base_seed=args_cli.seed)
    obs_adapter = DrawerObsAdapter(env, drawer_joint_name="joint_0")
    visualizer = DebugVisualizer(enabled=not args_cli.headless or args_cli.show_affordance_debug)
    stage = provider.scene.stage

    sim_dt = env_cfg.sim.dt * env_cfg.decimation
    provider.set_sim_time(0.0)
    layout_manager.reset_layout(reset_index=0)
    state = provider.get_state()
    for _ in range(5):
        env.step(provider.make_hold_joint_action(state, 1.0))

    joint_names, body_names = _print_cabinet_joint_info(provider, stage)
    mapping = _joint_body_mapping_probe(env, provider, joint_names, body_names, sim_dt)
    _classify_drawers(provider, joint_names, mapping)
    _gripper_sign_test(env, provider, sim_dt)

    scan_joints = joint_names if args_cli.drawer_joint == "all" else [args_cli.drawer_joint]
    for jn in scan_joints:
        if jn in joint_names:
            _scan_joint(env, provider, obs_adapter, visualizer, stage, jn, values, sim_dt)

    print(f"[debug] results written to {LOG_PATH}", flush=True)
    if not args_cli.headless:
        print("[debug] holding viewport; close the window to exit.", flush=True)
        while simulation_app.is_running():
            state = provider.get_state()
            env.step(provider.make_hold_joint_action(state, 1.0))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
