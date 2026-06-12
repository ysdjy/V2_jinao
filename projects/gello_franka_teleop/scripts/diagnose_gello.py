#!/usr/bin/env python3
"""diagnose_gello.py — 检查第一阶段环境是否就绪。

逐项 PASS/FAIL：
    - Python 版本
    - import gello
    - import dynamixel_sdk
    - 读取配置文件
    - 端口是否存在
    - 当前用户是否在 dialout 组
    - 能否创建 GelloAgent 并读一次关节

只做一次初始化 + 一次读取，不会长时间运行。

用法：
    python scripts/diagnose_gello.py --config configs/gello_franka.yaml
    python scripts/diagnose_gello.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gello_franka.yaml"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ->  {detail}"
    print(line)
    _results.append((name, ok, detail))
    return ok


def dialout_status() -> tuple[bool, str]:
    """返回 (本进程当前是否能用 dialout, 提示信息)。

    关键：要看【本进程的有效组】(os.getgroups)，而不是 /etc/group 里的静态成员关系。
    因为常见情况是：用户已加入 dialout，但当前登录会话是在加入之前开的，
    进程的有效组里没有 dialout —— 此时静态检查会"假阳性"，但实际打不开串口。
    """
    import grp

    try:
        eff_names = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    except Exception:  # noqa: BLE001
        eff_names = set()
    if "dialout" in eff_names or os.geteuid() == 0:
        return True, "本进程有效组含 dialout"

    # 进一步看静态成员关系，给出更精确的修复建议
    try:
        import pwd

        user = pwd.getpwuid(os.getuid()).pw_name
        static_member = user in (grp.getgrnam("dialout").gr_mem or [])
    except Exception:  # noqa: BLE001
        static_member = False

    if static_member:
        return False, (
            "已是 dialout 成员，但当前会话未刷新组；用 'sg dialout -c <命令>' 运行，"
            "或重新登录 / newgrp dialout"
        )
    return False, "不在 dialout 组：sudo usermod -aG dialout $USER 后重新登录"


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 GELLO 第一阶段环境")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", type=str, default=None, help="手动覆盖端口")
    args = parser.parse_args()

    print("=" * 50)
    print("GELLO 第一阶段环境诊断")
    print("=" * 50)

    # 1. Python 版本
    v = sys.version_info
    check(
        "Python 版本 >= 3.8",
        v >= (3, 8),
        f"{v.major}.{v.minor}.{v.micro}",
    )

    # 2. import gello
    try:
        import gello  # noqa: F401
        from gello.agents.gello_agent import DynamixelRobotConfig  # noqa: F401
        from gello.robots.dynamixel import DynamixelRobot  # noqa: F401

        check("import gello", True)
    except Exception as e:  # noqa: BLE001
        check("import gello", False, str(e))

    # 3. import dynamixel_sdk
    try:
        from dynamixel_sdk.port_handler import PortHandler  # noqa: F401

        check("import dynamixel_sdk", True)
    except Exception as e:  # noqa: BLE001
        check("import dynamixel_sdk", False, str(e))

    # 4. 读取配置
    cfg = None
    try:
        import yaml

        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        check("读取配置文件", isinstance(cfg, dict), str(args.config))
    except Exception as e:  # noqa: BLE001
        check("读取配置文件", False, str(e))

    # 5. 端口
    port = args.port or (cfg.get("port") if cfg else None)
    if not port or port == "None":
        check("端口已配置", False, "port 为空，请先运行 detect_gello_port.sh")
    else:
        check("端口存在", Path(port).exists(), str(port))

    # 6. dialout 组（看本进程有效组，而不是静态成员关系）
    dialout_ok, dialout_msg = dialout_status()
    check("本进程可用 dialout（能开串口）", dialout_ok, dialout_msg)

    # 7. 创建 GelloAgent 并读一次
    if port and port != "None" and Path(port).exists() and cfg:
        try:
            import numpy as np
            from gello.robots.dynamixel import DynamixelRobot

            joint_ids = list(cfg["joint_ids"])
            joint_offsets = list(cfg["joint_offsets"])
            joint_signs = list(cfg["joint_signs"])
            gcfg = cfg.get("gripper") or {}
            gripper_config = None
            if gcfg.get("enabled", False):
                gripper_config = (
                    int(gcfg["id"]),
                    float(gcfg["open_value"]),
                    float(gcfg["close_value"]),
                )
            robot = DynamixelRobot(
                joint_ids=joint_ids,
                joint_offsets=joint_offsets,
                joint_signs=joint_signs,
                real=True,
                port=port,
                baudrate=int(cfg.get("baudrate", 57600)),
                gripper_config=gripper_config,
            )
            # gello 的 DynamixelDriver 串口打不开会静默回退到 fake 模式（_is_fake=True，
            # 返回假数据）→ 必须判 FAIL
            is_fake = getattr(robot._driver, "_is_fake", False) or \
                type(robot._driver).__name__.lower().startswith("fake")
            state = np.asarray(robot.get_joint_state(), dtype=float)
            dim_ok = state.ndim == 1 and state.shape[0] >= len(joint_ids)
            if is_fake:
                check(
                    "创建 GelloAgent 并读取一次",
                    False,
                    "串口未真正打开，回退到 FakeDynamixelDriver（假数据）；多半是 dialout 权限",
                )
            else:
                check(
                    "创建 GelloAgent 并读取一次",
                    dim_ok,
                    f"读到 {state.shape[0]} 维真实数据: {np.round(state, 3).tolist()}",
                )
        except Exception as e:  # noqa: BLE001
            check("创建 GelloAgent 并读取一次", False, str(e))
    else:
        check("创建 GelloAgent 并读取一次", False, "端口/配置不满足，跳过实读")

    # 汇总
    print("=" * 50)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"结果: {passed}/{total} 通过")
    if passed == total:
        print("全部 PASS，可以运行 read_gello_joints.py 了。")
        return 0
    print("有 FAIL，请按上面的提示修复后重试。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
