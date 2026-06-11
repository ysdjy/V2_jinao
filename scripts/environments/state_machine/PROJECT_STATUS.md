# 方案 B 项目状态 (Method-B joint-action state machine + selected-drawer policy)

最近更新：状态机统一 joint-action 路径已可用；抓取/放置已复现 IK-Abs 水平；
selected-drawer 统一 policy 训练/微调/部署链路已打通（sanity 级）；柜子位置已在训练与部署间统一。

---

## 1. 总体架构

```
状态机 (SkillExecutor, mode="joint")
  ├── GRASP  -> GraspJointSkill   (内部 DLS IK -> q_des)            ──┐
  ├── PLACE  -> PlaceJointSkill    (内部 DLS IK -> q_des)            ──┼─> joint-position env
  └── OPEN_DRAWER ->                                                  │   (Isaac-Stack-Cube-Franka-
        ├─ scripted_joint        ScriptedDrawerJointSkill (baseline) │    JointPolicy-v0)
        ├─ official_joint_policy OfficialDrawerJointSkill            │
        └─ custom_selected_policy CustomDrawerJointSkill ────────────┘
所有 skill 输出 joint action；学习 policy 输出 raw_joint_action，drawer_joint_target 恒 None。
```

- 部署环境：`Isaac-Stack-Cube-Franka-JointPolicy-v0`（custom Cabinet_44853 + Franka，
  high-PD 机器人 + arm JointPositionAction scale=1.0, use_default_offset=True）。
- 训练环境：`Isaac-Open-CustomDrawer-Selected-Franka-v0(/-Play)`（复用上面场景，去 cubes/knife，
  装 selected-drawer MDP）。两者共享同一动作契约与柜子位姿。

## 2. 柜子 / 抽屉关键事实（debug_drawer_joint_scan.py 确认）

| target | joint | link | body z | 状态 |
|--------|-------|------|--------|------|
| top_drawer | joint_0 | link_0 | 0.684 | ✅ 可动 (开向世界 −X) |
| middle_drawer | joint_2 | link_2 | 0.391 | ✅ 可动 (开向世界 −X) |
| bottom_drawer | joint_1 | link_1 | 0.111 | ❌ LOCKED (closed 处穿模) |

- prismatic, axis=Z(局部), limits [0,0.8], closed=0, open_direction=+1。gripper +1开/−1合（已验证）。
- 中央配置单一来源：`source/.../franka/custom_drawer_config.py`（`skill_runtime/drawer_target_config.py` re-export）。
- **柜子位置已统一**：训练 cfg 与部署 layout 都为 `(1.0, -0.8, 0.323)`。

## 3. 文件地图（本项目新增/改动）

**状态机 runtime** `scripts/environments/state_machine/skill_runtime/`
- `base_skill.py` SkillCommand(control_mode/joint_target/raw_joint_action)
- `scene_state_provider.py` make_joint_action_from_q_des/_from_raw/make_hold_joint_action
- `ik_joint_adapter.py` DLS IK -> q_des
- `grasp_joint_skill.py` / `place_joint_skill.py`
- `drawer_skill.py` (scripted) / `scripted_drawer_joint_skill.py`
- `official_drawer_joint_skill.py` / `official_drawer_policy.py` / `drawer_obs_adapter.py`(含 SelectedDrawerObsAdapter)
- `custom_drawer_joint_skill.py` (CustomDrawerJointSkill, learned selected)
- `drawer_target_config.py` (re-export central config)
- `joint_debug_logger.py` (低频终端日志)
- `skill_executor.py` (JointBackendConfig + backend 调度)
- `simple_scene_layout.py` (柜子摆放，已统一到 (1.0,-0.8))

**入口** `scripts/environments/state_machine/`
- `skill_test_ui.py`（旧 IK-Abs UI，未改，回归用）
- `skill_test_ui_joint.py`（joint UI，drawer_backend 默认 none）
- `skill_sequence_joint.py`（headless 序列）
- `debug_drawer_joint_scan.py`（抽屉诊断）
- `debug_custom_drawer_env.py`（RL env smoke）
- `export_official_drawer_policy.py` / `export_custom_drawer_selected_policy.py`
- `finetune_custom_drawer_from_official.py`

**RL 任务** `source/.../manipulation/stack/config/franka/`
- `stack_joint_policy_env_cfg.py`（部署 joint env）
- `custom_drawer_config.py` / `custom_drawer_mdp.py` / `custom_drawer_selected_env_cfg.py`
- `agents/rsl_rl_custom_drawer_ppo_cfg.py`，`__init__.py` 注册

## 4. 运行命令

```bash
conda activate env_isaaclab
# 旧 IK-Abs UI（回归）
./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui.py --num_envs 1 --show_affordance_debug --seed 1
# joint UI（drawer 默认 none；要测抽屉显式选 backend）
./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui_joint.py --num_envs 1 \
  --show_affordance_debug --grasp_backend joint_ik --place_backend joint_ik --drawer_backend none --seed 1
# joint 序列：抓取+放置
./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py --num_envs 1 \
  --sequence grasp:cube_1,place:point_a --grasp_backend joint_ik --place_backend joint_ik \
  --drawer_backend none --seed 1 --max_steps 2000
# 抽屉诊断
./isaaclab.sh -p scripts/environments/state_machine/debug_drawer_joint_scan.py --num_envs 1 \
  --drawer_joint all --values 0.00,0.05,0.10,0.20,0.30 --seed 1
# selected-drawer env smoke
./isaaclab.sh -p scripts/environments/state_machine/debug_custom_drawer_env.py --num_envs 1 --resets 10 --steps 100 --headless
# 短 sanity 训练
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-CustomDrawer-Selected-Franka-v0 --num_envs 128 --max_iterations 5 --headless
# 从官方 checkpoint fine-tune
./isaaclab.sh -p scripts/environments/state_machine/finetune_custom_drawer_from_official.py \
  --use_published_official_checkpoint --num_envs 128 --max_iterations 5 --headless --seed 1
# 导出 learned policy
./isaaclab.sh -p scripts/environments/state_machine/export_custom_drawer_selected_policy.py \
  --num_envs 1 --output_path logs/policies/custom_drawer_selected_policy.pt --headless
# 状态机调用 learned drawer policy
./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py --num_envs 1 \
  --sequence open_drawer:top_drawer --drawer_backend custom_selected_policy \
  --drawer_policy_path logs/policies/custom_drawer_selected_policy.pt --seed 1 --max_steps 1500
```

## 5. 测试状态

| 项 | 状态 |
|----|------|
| 旧 IK-Abs UI 回归 | ✅ |
| joint grasp:cube_1 / place:point_a | ✅ SUCCEEDED |
| 抽屉诊断（映射/锁死/gripper） | ✅ 确认 |
| scripted baseline top/middle | ✅；bottom 锁死(预期) |
| selected-drawer env smoke (obs31/act8/reward) | ✅ |
| 短训练 128env×5iter | ✅ reward 上升、保存 checkpoint |
| fine-tune-from-official 128env×5iter | ✅ shape 匹配 load + 续训 |
| 导出 custom policy.pt | ✅ |
| 状态机调用 custom_selected_policy（链路） | ✅ obs[1,31]/act[1,8]/raw joint/target=None/arm 动/top·middle 选择正确/bottom 拒绝 |
| 柜子统一位置 (1.0,-0.8) 训练+部署 | ✅ |

## 6. 待办 / 已知限制

1. **handle frame 精度**：当前 selected handle≈drawer link 原点(zero offset)；middle(link_2)原点在柜体后部，
   非真实把手中心 → 正式训练前需为每个 drawer 标定 front-face 偏移（要看 mesh），否则成功率受限。
2. **正式训练**：需用户确认后跑（num_envs≥1024, max_iter 100/200/400），并用真实 handle 偏移。
3. **bottom_drawer 锁死**：`functional=False`，未纳入；需修 asset 碰撞/closed 偏移/摆放。
4. **learned drawer 仍未真正打开抽屉**：当前仅 sanity policy；待正式训练 + handle 标定后再评估成功率。
5. close_drawer learned policy：架构预留，暂用 scripted。
