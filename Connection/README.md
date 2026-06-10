# Connection

自包含的 Isaac Lab 项目集合。所有项目都放在这个文件夹里，资产、代码、脚本一体化，**换电脑零下载即可复现**。

- Isaac Sim 5.1 / Isaac Lab 2.3.x
- Conda 环境：`env_isaaclab`

## 快速开始（在新电脑上复现）

1. 安装好 Isaac Lab（含 Isaac Sim 5.1，conda 环境 `env_isaaclab`），并把 Isaac Lab 指向当前目录：
   ```bash
   conda activate env_isaaclab
   cd /path/to/IsaacLab
   ./isaaclab.sh --install
   ```
2. 把本 `Connection` 文件夹放进该 Isaac Lab 根目录下，然后一键安装本项目：
   ```bash
   bash Connection/setup.sh
   ```
   `setup.sh` 会自动：缺资产时下载离线 USD → 以可编辑模式安装 `connection_tasks` 扩展。
3. 验证（不跑完整仿真）：
   ```bash
   bash Connection/scripts/smoke_test.sh
   ```

## 运行三站任务：Franka 状态机操作冰箱 / 柜子 / 微波炉

这个任务在同一个 Isaac Lab env 内放置三个相邻 station，每个 station 有一台 Franka：

- 左侧：Franka 打开并关闭冰箱门。
- 中间：Franka 打开 `44853` 三抽屉柜子的最下面抽屉（`joint_1` / `link_1`），抓起刀并放入抽屉，再关闭抽屉。
- 右侧：Franka 打开并关闭微波炉门。

```bash
cd /path/to/IsaacLab
conda activate env_isaaclab

# GUI
./isaaclab.sh -p Connection/scripts/state_machine/multi_skill_sm.py --num_envs 1

# 无显示/远程服务器
./isaaclab.sh -p Connection/scripts/state_machine/multi_skill_sm.py --num_envs 1 --headless
```

如需只检查三站布局并录制短视频：

```bash
./isaaclab.sh -p Connection/tools/view_scene.py --task Connection-Multi-Skill-Franka-IK-Abs-v0 --num_envs 1
```

## 运行 V0 任务：Franka 状态机开抽屉

```bash
cd /path/to/IsaacLab
conda activate env_isaaclab

# GUI
./isaaclab.sh -p Connection/scripts/state_machine/open_drawer_sm.py --num_envs 8

# 无显示/远程服务器
./isaaclab.sh -p Connection/scripts/state_machine/open_drawer_sm.py --num_envs 8 --headless
```

状态机流程（warp，GPU 并行）：`REST → 接近把手前方 → 接近把手 → 抓握 → 拉开抽屉 → 释放`。

## 已注册任务

| 任务 ID | 控制方式 | 用途 |
|---------|---------|------|
| `Connection-Open-Drawer-Franka-v0` | 关节位置 | RL 训练（含 rsl_rl PPO 配置） |
| `Connection-Open-Drawer-Franka-Play-v0` | 关节位置 | RL play / 小场景 |
| `Connection-Open-Drawer-Franka-IK-Abs-v0` | 绝对位姿 IK | **状态机脚本使用** |
| `Connection-Open-Fridge-Franka-v0` | 关节位置 | 冰箱单任务 |
| `Connection-Open-Fridge-Franka-IK-Abs-v0` | 绝对位姿 IK | 冰箱状态机 |
| `Connection-Multi-Skill-Franka-v0` | 关节位置 | 三站任务 RL / 场景检查 |
| `Connection-Multi-Skill-Franka-IK-Abs-v0` | 绝对位姿 IK | **三站状态机脚本使用** |
| `Connection-Multi-Skill-Franka-IK-Abs-Play-v0` | 绝对位姿 IK | 三站小场景 |

## 目录结构

```text
Connection/
├── README.md
├── setup.sh                         # 一键安装（资产 + 扩展）
├── assets/                          # 离线 USD 资产（已提交，自包含）
│   └── Isaac/...                    #   Franka + Sektion Cabinet（镜像 Nucleus 相对结构）
├── scripts/
│   ├── download_assets.sh           # 备用：重新下载离线资产
│   ├── smoke_test.sh                # 不跑仿真的快速校验
│   └── state_machine/
│       ├── open_drawer_sm.py        # V0 状态机入口
│       └── multi_skill_sm.py        # 三站状态机入口
├── source/
│   └── connection_tasks/            # 可 pip install -e 的扩展
│       ├── pyproject.toml
│       ├── config/extension.toml
│       └── connection_tasks/
│           ├── assets_paths.py      # 本地资产路径解析
│           ├── robots/franka.py     # 指向本地 USD 的 Franka 配置
│           ├── robots/cabinet_44853.py
│           ├── robots/fridge.py
│           ├── robots/knife.py
│           ├── robots/microwave.py
│           └── tasks/               # 场景 / MDP / Franka 配置 / gym 注册
└── USD/                             # 原始 PartNet/URDF 资产
```

## 资产说明

- `assets/` 下的 USD 从官方 Nucleus 下载并**镜像相对目录结构**，USD 内部引用为相对路径，因此离线可直接解析。
- 三站任务还使用本地转换资产：
  - `assets/Props/Fridge_12252/fridge.usd`
  - `assets/Props/Cabinet_44853/cabinet.usd`
  - `assets/Props/Microwave_7320/microwave.usd`
  - `assets/Props/Knife_101054/knife.usd`
- 如果这些 USD 缺失，可从 `Connection/USD/<asset_id>/mobility.urdf` 重新生成：
  ```bash
  python Connection/tools/prepare_partnet_urdf.py Connection/USD/44853
  ./isaaclab.sh -p scripts/tools/convert_urdf.py Connection/USD/44853/mobility_isaac.urdf Connection/assets/Props/Cabinet_44853/cabinet.usd --fix-base --joint-stiffness 0.0 --joint-damping 3.0 --joint-target-type none --headless
  ```
  微波炉 `7320`、刀 `101054` 同理；刀转换时使用 `--merge-joints`。`prepare_partnet_urdf.py` 会清理不合法 prim 名和缺失 mesh 引用。
- 冰箱、柜子底部抽屉把手、微波炉把手都额外添加了不可见 Cuboid 碰撞代理，避免原始 mesh 转换后把手没有碰撞导致夹爪无法交互。
- 机器人与柜子的核心材质 `OmniPBR.mdl` 由 Isaac Sim 运行时自带，无需打包。
- 如需在另一台机器重新获取资产：`bash Connection/scripts/download_assets.sh`。
- 可用环境变量 `CONNECTION_ASSETS_DIR` 覆盖资产目录位置。
