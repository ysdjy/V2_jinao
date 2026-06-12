# 第一阶段：读取 GELLO 关节数据 — 操作说明与验收

目标：实时、稳定地从 GELLO 设备读取 7 个关节 + 夹爪，并在终端打印 / 存 CSV。
本阶段不碰 Isaac Sim、不做 IK、不用 ROS。

## 操作步骤

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1. 建环境 | `bash scripts/setup_gello_env.sh` | clone + 建 `.venv-gello` + 装最小依赖 |
| 2. 检测端口 | `bash scripts/detect_gello_port.sh` | 找串口、查 dialout 权限 |
| 3. 改配置 | 编辑 `configs/gello_franka.yaml` | 填 `port:` |
| 4. 校准 | `bash scripts/calibrate_gello_offset.sh` | 摆好姿态后跑，填回 `joint_offsets` |
| 5. 自检 | `python scripts/diagnose_gello.py --config configs/gello_franka.yaml` | 全部 PASS 再继续 |
| 6. 读取 | `python scripts/read_gello_joints.py --config configs/gello_franka.yaml --hz 30` | 核心 |

> 第 5、6 步前记得 `source .venv-gello/bin/activate`。

## 配置要点（configs/gello_franka.yaml）

- `port`：强烈建议用 `/dev/serial/by-id/...`，比 `/dev/ttyUSB0` 稳定。
- `joint_offsets`：必须是 `pi/2` 的整数倍；校准前填 0，校准后替换。
- `joint_signs`：每项 +1 或 -1，取决于 GELLO 机械结构。
- `gripper`：`enabled/id/open_value/close_value`；读不到会自动回退到只读 7 关节。
- `print_hz`：默认 30。
- `log_to_file` + `log_dir`：是否存 CSV 及存哪。

## 验收表格

| # | 验收项 | 期望 | 通过? |
|---|--------|------|-------|
| 1 | 终端稳定输出 q1~q7 | 每帧 7 个数字，不报错 | ☐ |
| 2 | 关节连续性 | 手动转动某个关节，对应 q 平滑变化，方向合理 | ☐ |
| 3 | 频率 | `hz` 显示在 ~30Hz | ☐ |
| 4 | 无 NaN | 不出现 `检测到 NaN` 告警 | ☐ |
| 5 | CSV 日志 | `logs/gello_joints_*.csv` 生成且有数据 | ☐ |
| 6 | 夹爪 | gripper 在 0~1 之间随开合变化（或明确提示 unavailable） | ☐ |
| 7 | 安全退出 | Ctrl+C 干净退出、CSV 已保存 | ☐ |

## 排错速查

- **端口不存在 / 打不开**：USB、线、`detect_gello_port.sh`、dialout 权限、配置 port。
- **import 失败**：确认 `source .venv-gello/bin/activate`，必要时重跑 setup。
- **q 漂移/方向反**：先校准 offset；方向反就翻 `joint_signs` 对应项的符号。
- **跳变告警**：偶发可能是丢包；频繁出现检查供电与线缆。
