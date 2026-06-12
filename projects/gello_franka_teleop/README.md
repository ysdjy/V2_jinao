# GELLO-Franka 遥操作开发（第一阶段）

在 IsaacLab 项目目录下的一个**独立子项目**，用于 GELLO 遥操作设备开发。

> **第一阶段只做一件事**：实时读取 GELLO 设备的关节数据，并在终端稳定打印
> `q1~q7` 和 gripper 状态。
>
> 本阶段**不**控制 Isaac Sim 的 Franka，**不**做 IK，**不**用 ROS，**不**做数据采集系统。
> 也**不会**修改 IsaacLab 的核心源码——所有东西都在本目录内，用独立虚拟环境 `.venv-gello`。

---

## 1. 项目目标

GELLO（USB 串口连接的 Dynamixel 主臂）→ 实时读取 7 个关节 + 夹爪 → 终端稳定打印。
这是后续 "GELLO → 控制 Isaac Sim / Isaac Lab 里的 Franka" 的第一步。

## 2. 目录结构

```
gello_franka_teleop/
├── README.md
├── configs/
│   └── gello_franka.yaml         # 端口、关节 offset/sign、夹爪、频率等配置
├── scripts/
│   ├── setup_gello_env.sh        # 一键建环境 + 装依赖
│   ├── detect_gello_port.sh      # 检测串口 + 权限
│   ├── calibrate_gello_offset.sh # 调用 gello 官方脚本校准 offset
│   ├── read_gello_joints.py      # 【核心】实时读取并打印关节
│   └── diagnose_gello.py         # 环境自检（PASS/FAIL）
├── third_party/
│   └── gello_software/           # 自动 clone 的 gello 官方仓库（含 DynamixelSDK 子模块）
├── logs/                         # CSV 日志 + offset 校准结果
└── docs/
    ├── stage1_read_joints.md
    └── next_stage_isaac_control_plan.md
```

## 3. 一键安装

```bash
cd ~/IsaacLab/projects/gello_franka_teleop   # 你的实际路径可能是 ~/Documents/IsaacLab/...
bash scripts/setup_gello_env.sh
```

这会：clone gello_software（已存在则跳过）→ 初始化 DynamixelSDK 子模块 →
创建 `.venv-gello`（优先 `uv`，没有就用 `python3 -m venv`）→
安装读取关节所需的最小依赖（`numpy` / `pyyaml` / `dynamixel_sdk` / `gello`）。

> 想额外安装 gello 的完整依赖（含 pyrealsense2/pin/ur-rtde 等重依赖，第一阶段用不到）：
> `bash scripts/setup_gello_env.sh --full`

## 4. 检测端口

```bash
bash scripts/detect_gello_port.sh
```

它会列出 `/dev/serial/by-id/`、`/dev/ttyUSB*`、`/dev/ttyACM*`，检查 dialout 权限，
并推荐一个稳定端口名。**推荐使用 `/dev/serial/by-id/...` 而不是 `/dev/ttyUSB0`**（更稳定，不会因插拔顺序变化）。

## 5. 修改配置

把检测到的端口手动填进 `configs/gello_franka.yaml`：

```yaml
port: /dev/serial/by-id/usb-FTDI_...-if00-port0
```

其它字段（`joint_offsets` / `joint_signs` / `gripper`）的默认值只是猜测，需要校准后再改。

## 6. 校准 offset

先把 GELLO 摆成配置里 `start_joints` 描述的标准姿态，然后：

```bash
bash scripts/calibrate_gello_offset.sh
```

它会调用 gello 官方的 `scripts/gello_get_offset.py`，把结果存到
`logs/offset_calibration_<时间戳>.txt`。**把终端输出里的
`best offsets function of pi: [...]` 手动填进 `configs/gello_franka.yaml` 的 `joint_offsets`。**
（脚本不会自动覆盖配置，避免写错。）

## 7. 实时读取关节

```bash
source .venv-gello/bin/activate

# 先自检
python scripts/diagnose_gello.py --config configs/gello_franka.yaml

# 实时读取（核心）
python scripts/read_gello_joints.py --config configs/gello_franka.yaml --hz 30
```

输出形如：

```
[t= 12.345s | hz=29.8] q=[+0.001, -0.020, +0.003, -1.571, +0.000, +1.570, +0.000] gripper=0.123
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--config PATH` | 配置文件路径 |
| `--port PATH`   | 手动覆盖端口（优先于配置） |
| `--hz N`        | 读取/打印频率，默认读配置 `print_hz` |
| `--no-log`      | 不写 CSV |

若 `log_to_file: true`，数据保存到 `logs/gello_joints_<时间戳>.csv`，表头：
`time, loop_hz, q1, q2, q3, q4, q5, q6, q7, gripper`。

## 8. 常见错误

| 现象 | 排查 |
|------|------|
| **找不到端口** | USB 没插好/线坏；`bash scripts/detect_gello_port.sh`；`dmesg \| tail` 看识别情况 |
| **Permission denied** | 不在 dialout 组：`sudo usermod -aG dialout $USER`，然后重新登录或 `newgrp dialout` |
| **cannot import gello** | 没装好：重新 `bash scripts/setup_gello_env.sh`；确认已 `source .venv-gello/bin/activate` |
| **Dynamixel 读数失败 / 维度异常** | 检查舵机供电、线序、`joint_ids` 是否匹配；波特率是否 57600 |
| **gripper 不可用** | 程序会自动回退到只读 7 个关节并打印 `gripper unavailable`；检查夹爪舵机 ID（默认 8）和供电 |
| **q 值不对/漂移** | 多半是没校准：先跑 `calibrate_gello_offset.sh` 再填 `joint_offsets` |

## 9. 第一阶段验收标准

1. 终端稳定输出 `q1~q7`
2. 移动每个 GELLO 关节，对应的 `q` 值**连续**变化
3. `loop_hz` 在 30Hz 左右
4. 没有 NaN
5. 能保存 CSV 日志

详见 `docs/stage1_read_joints.md`。下一阶段计划见 `docs/next_stage_isaac_control_plan.md`。
