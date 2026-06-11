# Franka Skill State Machine (方案 B：joint-action 统一状态机)

一个基于 Isaac Lab 的 Franka 技能状态机项目：状态机统一调度技能，所有技能最终都向同一个
**joint-position** 环境输出 joint action。抓取/放置内部用 IK（封装在技能内），开/关抽屉用 IK 物理
拉/推（不训练、不直接 set 关节）。也保留了"用 RL 训练开抽屉小脑"的代码（暂存，未来可用）。

> 本项目是 Isaac Lab fork 仓库的一部分，克隆整个仓库即可运行（环境注册在 `isaaclab_tasks` 内，资产在
> `simv2/USD` 与 `SapienAssetPipeline/usd_assets`）。conda 环境：`env_isaaclab`。

## 目录结构

```
projects/franka_skill_state_machine/
├── README.md                ← 本文件
├── 指令.txt                  ← 所有运行指令
├── skills/                  ← 4 个通用技能块（状态机引用它们）
│   ├── grasp_skill.py        GraspJointSkill：给目标物体 → IK 抓取 → 输出 joint action
│   ├── place_skill.py        PlaceJointSkill：给目标点 → IK 放置
│   ├── open_drawer_skill.py  OpenDrawerIKSkill：给目标抽屉 → 实时读把手 → 物理拉开
│   └── close_drawer_skill.py CloseDrawerIKSkill：给目标抽屉 → 实时读把手 → 物理推关
├── state_machine/
│   └── skill_executor.py     SkillExecutor + JointBackendConfig：调度/暂停/恢复/切换技能
├── runtime/                 ← 共享运行时（状态读取、IK、配置、可视化、内部规则状态机等）
│   ├── scene_state_provider.py   读场景状态 + 构造 joint action（q_des↔raw、hold）
│   ├── ik_joint_adapter.py       DLS IK：world TCP pose → q_des（含 NaN/限位/步长保护）
│   ├── drawer_target_config.py   抽屉中央配置（top=joint_0 / middle=joint_2 / bottom=joint_1）
│   ├── drawer_obs_adapter.py     SelectedDrawerObsAdapter：选定抽屉把手/关节读取 + 31维 obs
│   ├── drawer_ik_common.py       开/关门几何：开门方向、抓取朝向
│   ├── grasp_skill.py / place_skill.py / drawer_skill.py  内部规则状态机（被技能块复用）
│   ├── target_registry.py        抓取目标 affordance
│   ├── simple_scene_layout.py    场景布局 + 区域内随机初始化（拒绝采样，避免干涉）
│   ├── base_skill.py / skill_types.py / skill_request.py / skill_result.py
│   ├── debug_visualizer.py / joint_debug_logger.py / ui_controller.py
├── learned_drawer/          ← RL 学习开抽屉（暂存，未来用；非当前主路径）
│   ├── official_drawer_policy.py / official_drawer_joint_skill.py
│   ├── custom_drawer_joint_skill.py / scripted_drawer_joint_skill.py
│   └── export_*.py / finetune_custom_drawer_from_official.py
└── entries/                 ← 运行入口
    ├── skill_test_ui_joint.py     joint 版交互 UI（点按钮跑技能）
    ├── skill_sequence_joint.py    headless 顺序执行 sequence
    ├── skill_test_ui.py           旧 IK-Abs UI（回归测试）
    └── debug_*.py                 诊断/标定/抓门把/开抽屉等测试脚本
```

## 技能块设计（通用、状态机只给目标）

- **抓取 / 放置**：状态机给目标物体名 / 目标点（状态机负责读取 pose），技能块内部生成 TCP 目标、
  调 DLS IK 求 q_des、输出 joint action。
- **开 / 关抽屉**：状态机只给"目标抽屉"。技能块**每一步实时读取把手位姿+朝向**（开合过程中会变），
  从"把手相对柜体"几何推出开门方向与抓取朝向，靠夹爪物理抓住把手再拉/推。
  `drawer_joint_target` 恒为 None，**绝不** `set_cabinet_joint_target`。
  支持 top/middle（bottom 当前 asset 卡死，暂不可开）。

## 底层环境 / 动作契约

- 部署环境 `Isaac-Stack-Cube-Franka-JointPolicy-v0`（注册在 isaaclab_tasks）：高 PD Franka，
  arm `JointPositionAction scale=1.0, use_default_offset=True`，action 8 维（7 arm + 1 binary gripper）。
- gripper：`+1` 张开，`-1` 闭合。

## 状态（已验证）

- grasp:cube_1 / place:point_a：成功（joint IK）。
- open_drawer / close_drawer（`--drawer_backend ik_pull`）：top 抽屉物理开→0.20 / 关→0.03 成功；middle 同理。
- cube/knife 在用户指定四边形区域内随机初始化（拒绝采样 + 最小间距，无干涉）。
- 柜子位姿对齐用户场景 `(0.27402, 0.91583, 0.323)` yaw+90°；三个抽屉把手有可视碰撞体。

运行指令见 `指令.txt`。
