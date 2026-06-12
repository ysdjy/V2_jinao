#!/usr/bin/env python3
"""read_gello_joints.py — 第一阶段核心脚本：实时读取并打印 GELLO 关节数据。

只做读取，不控制 Isaac Sim，不做 IK，不做 ROS。

用法：
    python scripts/read_gello_joints.py --config configs/gello_franka.yaml --hz 30
    python scripts/read_gello_joints.py --port /dev/ttyUSB0        # 手动覆盖端口
    python scripts/read_gello_joints.py --no-log                   # 不写 CSV

成功标准（第一阶段）：
    - 终端稳定输出 q1~q7
    - 移动每个 GELLO 关节，对应 q 连续变化
    - loop_hz 接近设定值（约 30Hz）
    - 没有 NaN
    - 能保存 CSV 日志
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    print("[ERROR] 缺少 pyyaml。请先运行 bash scripts/setup_gello_env.sh", file=sys.stderr)
    sys.exit(1)


# 项目根目录 = 本文件的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gello_franka.yaml"

# 单步关节跳变告警阈值（弧度）。正常人手摇动一般不会超过这个值。
JUMP_WARN_RAD = 1.0


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式不对（应为 yaml 映射）: {config_path}")
    return cfg


def build_robot(cfg: dict, port: str, use_gripper: bool):
    """创建 DynamixelRobot。返回 (robot, gripper_enabled, num_arm_joints)。

    直接用 gello 的 DynamixelRobot（GelloAgent.act 内部也只是调它的
    get_joint_state）。这样能干净地处理"带/不带夹爪"两条路径：use_gripper=True
    时若创建失败（通常是夹爪舵机读不到），调用方可回退到 use_gripper=False。
    """
    # 延迟 import：保证 yaml/端口错误能先给出友好提示
    from gello.robots.dynamixel import DynamixelRobot

    joint_ids = list(cfg["joint_ids"])
    joint_offsets = list(cfg["joint_offsets"])
    joint_signs = list(cfg["joint_signs"])
    num_arm = len(joint_ids)

    if len(joint_offsets) != num_arm or len(joint_signs) != num_arm:
        raise ValueError(
            f"配置维度不一致: joint_ids={num_arm}, "
            f"joint_offsets={len(joint_offsets)}, joint_signs={len(joint_signs)}"
        )

    gripper_cfg = cfg.get("gripper") or {}
    gripper_config = None
    if use_gripper and gripper_cfg.get("enabled", False):
        gripper_config = (
            int(gripper_cfg["id"]),
            float(gripper_cfg["open_value"]),
            float(gripper_cfg["close_value"]),
        )

    # start_joints 必须和 get_joint_state() 的维度一致（含 gripper 时 +1，
    # 多出来的那一维会被 DynamixelRobot 内部裁掉，填占位即可）
    start_joints = np.array(cfg.get("start_joints", [0.0] * num_arm), dtype=float)
    if len(start_joints) != num_arm:
        raise ValueError(f"start_joints 长度应为 {num_arm}，实际 {len(start_joints)}")
    if gripper_config is not None:
        start_joints = np.concatenate([start_joints, [0.0]])

    robot = DynamixelRobot(
        joint_ids=joint_ids,
        joint_offsets=joint_offsets,
        joint_signs=joint_signs,
        real=True,
        port=port,
        baudrate=int(cfg.get("baudrate", 57600)),
        gripper_config=gripper_config,
        start_joints=start_joints,
    )
    # gello 的 DynamixelDriver 在串口打不开时会"静默"回退到 fake 模式（_is_fake=True，
    # 返回全 0 假数据），让程序看起来正常其实没连真机。这里显式拦截。
    if getattr(robot._driver, "_is_fake", False) or \
            type(robot._driver).__name__.lower().startswith("fake"):
        raise PermissionError(
            "串口未真正打开，gello 回退到了 fake 驱动（多半是 permission denied）"
        )
    return robot, (gripper_config is not None), num_arm


def serial_error_help(port: str) -> str:
    return (
        "\n串口打开失败。请按顺序检查：\n"
        "  1) GELLO 的 USB 是否插好（换口/换线试试）\n"
        "  2) 运行: bash scripts/detect_gello_port.sh\n"
        "  3) dialout 权限: 把当前用户加入 dialout 组后重新登录\n"
        "       sudo usermod -aG dialout $USER   然后 newgrp dialout\n"
        f"  4) 检查 configs/gello_franka.yaml 里的 port（当前: {port}）\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="实时读取并打印 GELLO 关节数据（第一阶段）"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="手动覆盖串口端口（优先于配置文件）",
    )
    parser.add_argument(
        "--hz", type=float, default=None,
        help="打印/读取频率（默认读配置 print_hz，否则 30）",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="不写 CSV 日志（覆盖配置 log_to_file）",
    )
    args = parser.parse_args()

    # ---- 读取配置 ----
    try:
        cfg = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 读取配置失败: {e}", file=sys.stderr)
        return 1

    port = args.port or cfg.get("port")
    if not port or port == "None":
        print(
            "[ERROR] 没有指定串口端口。\n"
            "  先运行 bash scripts/detect_gello_port.sh，\n"
            "  把端口填进 configs/gello_franka.yaml 的 port，\n"
            "  或者用 --port 手动指定。",
            file=sys.stderr,
        )
        return 1

    if not Path(port).exists():
        print(f"[ERROR] 端口不存在: {port}", file=sys.stderr)
        print(serial_error_help(port), file=sys.stderr)
        return 1

    hz = args.hz if args.hz is not None else float(cfg.get("print_hz", 30))
    if hz <= 0:
        print(f"[ERROR] hz 必须为正数，得到 {hz}", file=sys.stderr)
        return 1
    period = 1.0 / hz

    log_to_file = bool(cfg.get("log_to_file", True)) and not args.no_log

    # ---- 创建 robot（gripper 失败则回退到只读 7 关节）----
    print(f"[info] 连接端口: {port}")
    gripper_enabled = False
    num_arm = len(cfg.get("joint_ids", [1, 2, 3, 4, 5, 6, 7]))
    try:
        robot, gripper_enabled, num_arm = build_robot(cfg, port, use_gripper=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "permission" in msg or "could not open" in msg or "serial" in msg:
            print(f"[ERROR] 无法打开串口: {e}", file=sys.stderr)
            print(serial_error_help(port), file=sys.stderr)
            return 1
        # 其它失败：很可能是 gripper 舵机读不到，尝试只读 7 个关节
        print(f"[warn] 带夹爪初始化失败（{e}）。尝试只读 7 个关节 ...", file=sys.stderr)
        try:
            robot, gripper_enabled, num_arm = build_robot(cfg, port, use_gripper=False)
            print("[warn] gripper unavailable，继续只读 7 个关节。", file=sys.stderr)
        except Exception as e2:  # noqa: BLE001
            print(f"[ERROR] 初始化失败: {e2}", file=sys.stderr)
            print(serial_error_help(port), file=sys.stderr)
            return 1

    expected_dim = num_arm + (1 if gripper_enabled else 0)

    # ---- 准备 CSV ----
    writer = None
    csv_file = None
    if log_to_file:
        log_dir = PROJECT_ROOT / cfg.get("log_dir", "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"gello_joints_{ts}.csv"
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["time", "loop_hz", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper"]
        )
        print(f"[info] 日志写入: {csv_path}")

    print(f"[info] 读取频率 ~{hz:.0f}Hz，gripper={'on' if gripper_enabled else 'off'}")
    print("[info] 按 Ctrl+C 退出。\n")

    # ---- 主循环 ----
    t_start = time.perf_counter()
    last_loop_t = t_start
    last_q = None
    loop_hz = 0.0
    try:
        while True:
            loop_t0 = time.perf_counter()

            try:
                state = np.asarray(robot.get_joint_state(), dtype=float)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 读取失败，跳过本帧: {e}", file=sys.stderr)
                time.sleep(period)
                continue

            # 维度检查
            if state.shape[0] != expected_dim:
                print(
                    f"[warn] 维度异常: 期望 {expected_dim}，得到 {state.shape[0]}",
                    file=sys.stderr,
                )

            q = state[:num_arm]
            gripper_val = state[num_arm] if gripper_enabled and state.shape[0] > num_arm else None

            # NaN 检查
            if np.any(np.isnan(state)):
                print("[warn] 检测到 NaN！请检查接线/供电。", file=sys.stderr)

            # 跳变检查
            if last_q is not None and last_q.shape == q.shape:
                jump = np.max(np.abs(q - last_q))
                if jump > JUMP_WARN_RAD:
                    print(
                        f"[warn] 关节跳变过大: {jump:.3f} rad（可能丢包/掉电）",
                        file=sys.stderr,
                    )
            last_q = q.copy()

            # 计算瞬时频率（指数平滑）
            now = time.perf_counter()
            dt = now - last_loop_t
            last_loop_t = now
            if dt > 0:
                inst_hz = 1.0 / dt
                loop_hz = inst_hz if loop_hz == 0 else 0.9 * loop_hz + 0.1 * inst_hz

            t_rel = now - t_start

            # 打印
            q_str = "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"
            g_str = f"{gripper_val:.3f}" if gripper_val is not None else "n/a"
            print(f"[t={t_rel:7.3f}s | hz={loop_hz:4.1f}] q={q_str} gripper={g_str}")

            # 写 CSV
            if writer is not None:
                row = [f"{t_rel:.4f}", f"{loop_hz:.2f}"]
                row += [f"{v:.6f}" for v in q]
                # 补齐到 7 个关节列
                while len(row) < 2 + 7:
                    row.append("")
                row.append(f"{gripper_val:.6f}" if gripper_val is not None else "")
                writer.writerow(row)

            # 频率控制
            elapsed = time.perf_counter() - loop_t0
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[info] 收到 Ctrl+C，正在安全退出 ...")
    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
            print("[info] CSV 已保存。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
