# 方案 B — Joint-action 状态机实现笔记

状态机统一调度技能，但所有技能最终都向同一个 **joint-position** 环境输出 joint action。
Grasp/Place 内部仍用 IK，但 IK 被封装在技能内部，技能输出的是 **q_des / raw joint action**。
Open-drawer 使用官方训练好的 Franka open-drawer PPO checkpoint，policy 直接输出 raw joint action。

## 1. 现有 action space 调研

### IK-Abs 抓取/放置入口（旧, 不改）
- 入口: `scripts/environments/state_machine/skill_test_ui.py`
- 环境: `Isaac-Stack-Cube-Franka-IK-Abs-v0`
  (`stack_ik_abs_env_cfg.py`: `DifferentialInverseKinematicsActionCfg`, body=`panda_hand`,
   body_offset=+0.1034 m Z (TCP), `ik_method="dls"`, command_type="pose", absolute)
- action = `[x, y, z, qw, qx, qy, qz, gripper]`, shape `(num_envs, 8)`，由 `SceneStateProvider.make_action` 构造。
- gripper: `1.0`=open, `-1.0`=close (BinaryJointPositionAction: `action < 0` → close)。

### Joint-position 环境（本次新增的执行路径）
- `Isaac-Stack-Cube-Franka-v0` (`stack_joint_pos_env_cfg.py`):
  `JointPositionActionCfg(joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)` + binary gripper。
  arm action dim = 7, gripper dim = 1 → total 8。
- 官方 open-drawer 环境 (`cabinet/.../joint_pos_env_cfg.py`): 同样的 JointPositionAction，但 **scale=1.0**, use_default_offset=True。
- 处理公式 (`JointAction.process_actions`): `q_target = raw * scale + offset`，
  `offset = default_joint_pos[arm]`（在 action term **构造时**克隆，stack 场景的 `set_default_joint_pose` reset 事件
  会改 `data.default_joint_pos`，但**不会**改 action term 已缓存的 `_offset`）。
  → 因此 raw 转换必须读 **action term 的 `_scale` / `_offset`**，不能直接用 `robot.data.default_joint_pos`。
- 反解: `raw_arm = (q_des - offset) / scale`。

### 为减少不确定性，新增独立 joint-policy env
- `Isaac-Stack-Cube-Franka-JointPolicy-v0`
  (`stack_joint_policy_env_cfg.py`): 复制 stack joint scene，arm `scale=1.0`, `use_default_offset=True`，
  与官方 drawer PPO 一致；并新增 `cabinet_frame` FrameTransformer 把 `BottomHandleProxy`
  映射成 `drawer_handle_top`，供 drawer observation 使用。

## 2. 官方 drawer policy observation（顺序很重要，concatenate_terms=True）
来源 `cabinet_env_cfg.py:ObservationsCfg.PolicyCfg`:
1. `joint_pos`  = `joint_pos_rel`(robot)            → 9  (7 arm + 2 finger)
2. `joint_vel`  = `joint_vel_rel`(robot)            → 9
3. `cabinet_joint_pos` = joint_pos_rel(cabinet drawer_top_joint) → 1
4. `cabinet_joint_vel` = joint_vel_rel(...)         → 1
5. `rel_ee_drawer_distance` = handle_pos_w - tcp_pos_w → 3
6. `actions` = last_action                          → 8
合计 = **31**。 action 输出 = **8**。

- `joint_*_rel` = `data.joint_pos - data.default_joint_pos`。
- 自定义 cabinet 没有 `drawer_top_joint`，映射到 `joint_0`（底层抽屉）。
- handle frame：用新增的 `cabinet_frame`（BottomHandleProxy）或回退到 link_1+offset 计算。

## 3. 成功条件
- open_drawer: 官方 cabinet 读 `drawer_top_joint`，自定义 cabinet 读 `joint_0`；joint pos >= **0.20** 视为成功。
- 学习 policy **禁止**直接 `set_cabinet_joint_target`，必须靠 Franka 物理交互打开。
- scripted_joint baseline (`DrawerSkill`) 允许直接 set joint，仅作显式 baseline (`--drawer_backend scripted_joint`)。

## 4. 新增 / 修改文件
新增:
- `source/.../stack/config/franka/stack_joint_policy_env_cfg.py` + 在 `__init__.py` 注册
- `skill_runtime/ik_joint_adapter.py`        — DLS IK 封装, 输出 q_des
- `skill_runtime/grasp_joint_skill.py`       — 复用 GraspSkill，TCP→q_des
- `skill_runtime/place_joint_skill.py`       — 复用 PlaceSkill，TCP→q_des
- `skill_runtime/scripted_drawer_joint_skill.py` — 复用 DrawerSkill，arm hold + 直接 set joint (baseline)
- `skill_runtime/official_drawer_policy.py`  — torchscript policy.pt 加载 + 推理
- `skill_runtime/drawer_obs_adapter.py`      — 从 env/state 构造官方 31 维 obs
- `skill_runtime/official_drawer_joint_skill.py` — state→obs→policy→raw joint action
- `scripts/environments/state_machine/skill_sequence_joint.py` — joint 版 headless 入口
- `scripts/environments/state_machine/export_official_drawer_policy.py` — 导出官方 policy.pt

修改 (向后兼容):
- `skill_runtime/base_skill.py`     — `SkillCommand` 扩展 control_mode/joint_target/raw_joint_action
- `skill_runtime/scene_state_provider.py` — make_joint_action_from_q_des / _from_raw / make_hold_joint_action
- `skill_runtime/skill_executor.py` — backend 配置 + joint 路径 + joint 模式 pause/hold/switch

旧 `skill_test_ui.py` 与 IK-Abs 功能保持不变。

## 5. Smoke test 验证结果 (num_envs=1, headless)

| # | 测试 | 结果 |
|---|------|------|
| 1 | py_compile 全部新增/修改文件 | PASS |
| 2 | joint env 创建 + hold | PASS (EXIT=0, `joint_action_layout total_dim=8 scale=1.0 offset=franka defaults`) |
| 3 | grasp joint IK | IK adapter 每步 `ik_success=True` 输出有限 q_des (PASS)。完整抓取在 joint env 下 PRE_GRASP 超时 (见已知问题) |
| 4 | place command 解析 | PASS (无 held object 时正确返回 `REQUEST_INVALID: NO_HELD_OBJECT`) |
| 5 | scripted drawer baseline | PASS (joint_0 → 0.2496 ≥ 0.20, succeeded) |
| 6 | official policy load + run | PASS：torchscript 加载成功，obs_shape=[1,31] / action_shape=[1,8] 与 env 一致；policy 驱动手臂运动；`drawer_joint_target` 全程为 None (未作弊) |
| 7 | grasp→place→open_drawer 顺序调度 | PASS (三个 skill 依次 START/END/status，state machine 顺序正确) |
| - | 旧 IK-Abs `skill_test_ui.py` 回归 | PASS (grasp cube_2 PRE_GRASP 误差 0.0009 m，正常推进到 close/verify) |

导出命令验证: `export_official_drawer_policy.py` 用缓存 checkpoint 成功导出 214KB `policy.pt` 后立即退出 (需要 `handle_deprecated_rsl_rl_cfg` 适配新版 rsl-rl cfg 结构)。

## 6. 已知问题 / 需要后续 fine-tune

1. **joint-IK grasp/place 跟踪精度**：joint-policy env 用 `FRANKA_PANDA_CFG`（普通 PD），
   而 IK-Abs env 用 `FRANKA_PANDA_HIGH_PD_CFG`（高刚度）。同样的 bounded TCP 目标，IK-Abs 能到 0.0009 m，
   joint env 在普通 PD 下手臂下垂、PRE_GRASP 残差 ~0.13 m 而超时。架构与 IK 求解正确（q_des 有限、单步 DLS）。
   后续：为 joint-policy env 单独调高 Franka arm actuator stiffness/damping，或放宽 grasp/place 阶段阈值。
   注意：不能直接换成 high-PD，否则破坏官方 drawer policy 的 PD 一致性。
2. **官方 drawer policy zero-shot 不能打开自定义抽屉**：官方 policy 在 Sektion cabinet + `drawer_top_joint` +
   特定 handle frame 上训练；本场景是自定义 cabinet（`joint_0` 底抽屉，handle 由 link_1+缩放 offset 估计），
   且 obs 的 `default_joint_pos`（stack reset 事件改成 stack pose）与训练时（franka 默认）不一致。
   结果：手臂朝估计的 handle 方向运动但没真正勾到把手，drawer 未打开 → 如实记录 `DRAWER_OPEN_TIMEOUT`，
   **接口保留、绝不直接 set drawer joint**。后续若要 zero-shot：让 handle frame / obs 的 default 与官方语义严格对齐，
   或在本场景上 fine-tune drawer policy。
3. **close_drawer learned policy**：暂用 `scripted_joint`；架构已预留（`OfficialDrawerJointSkill` 可换 close policy）。

## 7. 运行命令

```bash
# (a) 导出官方 drawer policy（导出后立即退出）
./isaaclab.sh -p scripts/environments/state_machine/export_official_drawer_policy.py \
    --task Isaac-Open-Drawer-Franka-Play-v0 --use_pretrained_checkpoint \
    --output_path logs/policies/official_open_drawer_policy.pt --headless

# (b) joint 状态机 + scripted drawer baseline
./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py --num_envs 1 \
    --sequence grasp:cube_1,place:point_a,open_drawer:bottom_drawer \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend scripted_joint \
    --seed 1 --max_steps 3000 --headless

# (c) joint 状态机 + 官方 drawer policy
./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py --num_envs 1 \
    --sequence grasp:cube_1,place:point_a,open_drawer:bottom_drawer \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend official_joint_policy \
    --drawer_policy_path logs/policies/official_open_drawer_policy.pt \
    --seed 1 --max_steps 3000 --headless
```
结果日志: `logs/skill_tests/joint_sequence_results.jsonl`（每个 skill 一条，含 backend/target/timing/status/joint_pos/obs+action shape）。
