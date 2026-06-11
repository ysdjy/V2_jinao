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

## 5b. 第二轮修复（可视化运行反馈后）

- **姿态怪异 / 抓取超时 已修复**：根因是 joint-policy env 之前用 `FRANKA_PANDA_CFG`（软 PD + 有重力），
  手臂下垂、IK 累积漂移成怪异姿态。改为与 IK-Abs 一致的 `FRANKA_PANDA_HIGH_PD_CFG`
  （shoulder/forearm stiffness 400、`disable_gravity=True`）。验证：
  - `grasp:cube_1` → **SUCCEEDED**（PRE_GRASP 残差 0.13 m → 0.004 m，完整走到 close/probe-lift/full-lift/HOLD）
  - `place:point_a` → **SUCCEEDED**（移动到 point_a 0.420/0.100/0.022 后释放）
  - 全程 `drawer_joint_pos=0.0`，抽屉未被触碰。
  注意：high-PD/无重力 与官方 drawer PPO 训练时的软 PD 不同；drawer policy 已暂缓，后续重接时再决定
  （独立 env 或抓取/放置阶段开 high-PD、drawer 阶段开软 PD）。
- **新增 joint UI 入口** `skill_test_ui_joint.py`：完全复制 `skill_test_ui.py` 的 UI/marker/可视化，
  改用 `Isaac-Stack-Cube-Franka-JointPolicy-v0` + joint backends；默认不自动执行，抽屉只在点击 Open Drawer 时启动；
  drawer 目标显示为 下抽屉/中抽屉/上抽屉(bottom/middle/top → joint_0/1/2)。`--show_affordance_debug` 保留。
- **低频终端日志** `skill_runtime/joint_debug_logger.py`：默认每 30 步打印 active skill / backend / target /
  OUTPUT=joint(q_des|raw)+shape+min/max+gripper / IK 的 target&current tcp+pos&ori err+ik_success /
  drawer 的 joint pos+target(是否 None)+obs/action shape+rel_ee_drawer。非 joint 输出会打印 WARNING。
- **scripted drawer baseline 守卫**：实际驱动抽屉关节时打印一次
  `[BASELINE] scripted_joint is directly commanding drawer joint, not learned physical pulling.`；
  official_joint_policy 路径 `drawer_joint_target` 恒为 None，绝不 `set_cabinet_joint_target`。

## 5c. 自定义 cabinet 低层诊断（debug_drawer_joint_scan.py）

运行：
```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/debug_drawer_joint_scan.py \
    --num_envs 1 --show_affordance_debug --seed 1
# headless 取数值用：加 --headless --disable_fabric
```
> 注意：fabric 开启时 `UsdGeom.XformCache` 读到的是 **authored（layout 之前）** 的 stale 位姿，
> 诊断取数值请加 `--disable_fabric`，并以 articulation `body_pos_w`（物理视图）为准。

诊断结论（自定义 Cabinet_44853）：
- **joint 类型/限位**：joint_0/1/2 均为 `PhysicsPrismaticJoint`，**axis=Z（局部）**，limits **[0.0, 0.8]**，default 0。
  下限=0 → 不能为负，**closed = 0**。
- **joint → body → 物理高度 映射**（关键，原映射是错的）：
  | joint | 移动的 body | body z (高度) | 物理含义 | 命令 +0.20 时位移 |
  |-------|-------------|---------------|----------|-------------------|
  | joint_0 | link_0 | 0.684 (最高) | **顶抽屉** | Δ=−0.20 m (世界 −X) ✓ 能动 |
  | joint_1 | link_1 | 0.111 (最低) | **底抽屉**(handle 所在) | **Δ=0.0000 卡死/穿模，不动** |
  | joint_2 | link_2 | 0.391 (中) | 中抽屉 | Δ=−0.20 m (世界 −X) ✓ 能动 |
- **打开方向**：能动的抽屉 joint 增大 → body 沿 **世界 −X**（朝机器人侧）拉出；joint 越大越开。
- **handle frame**：BottomHandleProxy 挂在 link_1（确实是最低=底抽屉，body 选择正确），
  但 link_1 由 **joint_1** 驱动且 joint_1 卡死不动 → handle 世界位姿永不变化。
  另：之前 `bottom_drawer → joint_0` 的映射是**错的**（joint_0 其实是顶抽屉）；
  obs adapter 默认 `drawer_joint_name="joint_0"` 读的也是顶抽屉。
- **gripper 符号**：**正确**。+1.0 → 张开(finger 0.04, width 0.08)；−1.0 → 闭合(finger ~0, width ~0)。
- **官方 policy 失败最可能原因**：**geometry / joint-mapping / handle 绑定到卡死 joint_1** 的组合问题——
  不是 gripper mapping（已确认正确），不是 obs 维度（31/8 匹配）。具体：
  (i) 底抽屉(link_1/joint_1)物理卡死，命令也拉不开（这就是 drawer_joint_pos 一直 ~0、机器人撞柜的根因）；
  (ii) 抽屉滑动轴是世界 −X，与官方 Sektion cabinet 训练几何/轴不同；
  (iii) scripted baseline 之前把 bottom 映射成 joint_0，实际开的是**顶抽屉**，并非底抽屉。

**建议的修复方向（待确认，本轮未改业务逻辑）**：
1. 把 drawer 目标→joint 映射改成 bottom=joint_1 / middle=joint_2 / top=joint_0（按物理高度）；
   obs adapter 的 `drawer_joint_name` 也相应改。
2. 先解决底抽屉 link_1/joint_1 卡死/穿模（检查该 drawer 的碰撞网格 / 初始位姿 / closed 偏移），
   否则任何 backend（scripted 或 learned）都拉不开底抽屉。
3. 官方 policy 的几何/轴与本柜不一致，zero-shot 仍不可期；建议先用能动的抽屉（joint_0/2）验证 pipeline。

## 5d. Selected-drawer policy — Stage 1 诊断（已确认，gating 项）

工具升级版 `debug_drawer_joint_scan.py`（`--drawer_joint all --values ... --disable_fabric`），
结果写 `logs/skill_tests/drawer_joint_scan_results.jsonl`。**确认结论**：

| target | joint | link | body z | 开合 | 状态 |
|--------|-------|------|--------|------|------|
| top_drawer | joint_0 | link_0 | 0.684 | +joint=开, 世界 −X | ✅ 正常 |
| middle_drawer | joint_2 | link_2 | 0.391 | +joint=开, 世界 −X | ✅ 正常 |
| bottom_drawer | joint_1 | link_1 | 0.111 | — | ❌ **LOCKED 卡死** |

- 全部 prismatic, axis=Z(局部), limits [0,0.8] → closed=0, open_direction=+1。
- **bottom_drawer(joint_1) 卡死**：直接 `write_joint_state(joint_1=0.20)` 一步后被压回 0
  (`actual_after_teleport=0.0000`)。drawer 驱动 stiffness=10 很弱，说明该抽屉在 closed pose 处
  **几何穿模/硬接触**，与目视"嵌进柜体"一致。
- gripper 符号正确：+1.0 张开(width 0.08)，−1.0 闭合(width ~0)。
- 统一映射已固化到 `skill_runtime/drawer_target_config.py`（单一来源），bottom_drawer 标记 `functional=False`。

**结论**：top/middle 可用于学习；bottom 在修复 asset 碰撞/closed 偏移/摆放前不可训练（符合"确认前不训练"）。

## 5e. Selected-drawer 统一 policy — Stage 2~5 已完成并验证

中央配置 `custom_drawer_config.py`(source, canonical) + `skill_runtime/drawer_target_config.py`(re-export)。

- **Stage 2 scripted baseline（已验证）**：修正映射后
  `open_drawer:top_drawer`→joint_0 SUCCEEDED(0.2496)，`middle_drawer`→joint_2 SUCCEEDED(0.2353)，
  `bottom_drawer`→joint_1 超时(LOCKED，符合预期)。每次驱动关节打印 `[BASELINE]`。
- **Stage 3 selected-drawer RL env（已验证 smoke）**：`Isaac-Open-CustomDrawer-Selected-Franka-v0`
  复用 JointPolicy 场景(高 PD + scale=1.0 joint action)，去掉 cubes/knife，加 `drawer_frames`
  FrameTransformer(link_0=top, link_2=middle)。reset 随机选 functional drawer，obs/reward/success 只看 selected。
  smoke: obs=[1,31]、action=[1,8]、reset 10 次 top/middle 各 5、随机 100 步 reward 有限、无崩溃。
  日志 `logs/skill_tests/custom_drawer_selected_smoke.jsonl`。
- **Stage 4 短 sanity 训练（已验证）**：`train.py --task Isaac-Open-CustomDrawer-Selected-Franka-v0
  --num_envs 128 --max_iterations 5` → reward 0.89→3.70，checkpoint 保存到 `logs/rsl_rl/custom_drawer_selected/`。
- **Stage 5 fine-tune-from-official（已验证）**：`finetune_custom_drawer_from_official.py
  --use_published_official_checkpoint --num_envs 128 --max_iterations 5` → 官方 checkpoint
  shape 匹配(obs31/act8/net[256,128,64])直接 load 成功，续训 5 iter reward 2.6→8.7；
  shape 不一致会明确报错不静默训练。

obs 维度/顺序与官方 open-drawer 一致(31)，便于加载官方 checkpoint。动作 joint-position(scale=1.0,
use_default_offset)，与状态机部署 env(JointPolicy)动作契约一致，policy raw action 可直接迁移。

### 待办（下一阶段，需显式确认/正式训练）
- **Stage 6 部署集成**：导出 `export_custom_drawer_selected_policy.py` → torchscript；新增
  `--drawer_backend custom_selected_policy` + `CustomDrawerJointSkill`(target_drawer→selected obs→policy→raw action,
  drawer_joint_target=None)；在 skill_test_ui_joint / skill_sequence_joint 中按 target_drawer 调用。**尚未实现**。
- **handle frame 精度**：当前 selected handle≈drawer link 原点(zero offset)。top(link_0)在前面尚可，
  **middle(link_2)原点在柜体后部 x≈1.14**，不是真实把手中心 → 直接训练 middle 成功率会低。
  正式训练前需为每个 drawer 标定 front-face handle 偏移(需要 mesh 检查)。
- **bottom_drawer 卡死**：仍 `functional=False`，未纳入训练，需先修 asset 碰撞/closed 偏移。
- **正式训练**(num_envs≥1024, max_iter 100/200/400)：需用户确认后再跑。

## 5f. Stage 6 部署集成（已完成并验证）+ cabinet 移动

- 新增 `export_custom_drawer_selected_policy.py`（从 logs/rsl_rl/custom_drawer_selected 最新或 --checkpoint
  导出 torchscript → `logs/policies/custom_drawer_selected_policy.pt`，导出后退出）。
- 新增 `skill_runtime/custom_drawer_joint_skill.py`（CustomDrawerJointSkill）+
  `SelectedDrawerObsAdapter`（部署侧 31 维 selected obs，handle=drawer link body 世界位姿，与训练一致）。
- executor 新增 backend `custom_selected_policy`；`skill_sequence_joint.py` / `skill_test_ui_joint.py`
  都支持 `--drawer_backend custom_selected_policy --drawer_policy_path ...`。
- UI 新增 `drawer_backend="none"`（默认，Open Drawer 禁用，避免误以为 scripted 是学习技能）；
  点 Open Drawer 时：scripted→打印 `[BASELINE WARNING] scripted_joint directly commands
  drawer_joint_target; robot arm will hold still.`；none→提示禁用；custom+bottom→打印锁死 WARNING。
- custom_selected_policy 路径：状态机只传 target_drawer；skill 内部选 joint/handle、构造 selected obs、
  policy 输出 raw joint action、**drawer_joint_target 恒 None、绝不 set_cabinet_joint_target**。
- 链路测试（5-iter sanity checkpoint 导出 policy.pt）：policy 加载✓、obs[1,31]✓、action[1,8]✓、
  OUTPUT=joint(raw_joint_action)✓、drawer_joint_target=None✓、arm 随 policy 运动✓、
  top→joint_0/上抽屉、middle→joint_2/中抽屉 selected obs 正确✓、bottom→WARNING+REQUEST_INVALID 拒绝✓。
  （sanity policy 未训练，抽屉未真正打开，符合预期。）

**cabinet 移动**：按要求柜子 +0.15 世界 X、−0.15 世界 Y。改了两处：
  `stack_joint_pos_env_cfg.py` init pos (0.85,−0.65)→(1.0,−0.8)（RL 训练 env），
  `simple_scene_layout.py` CABINET_LOCAL 0.78/0.52→0.93/0.37（部署 UI/sequence）。验证 handle 整体平移 +0.15X/−0.15Y。
  ⚠️ 注意：训练 env(cfg, y=−0.8) 与部署 layout(y=+0.37) 柜子位置**本就不一致**（历史遗留），
  这会影响 learned policy 的 train→deploy 迁移；正式训练前建议统一两者的柜子位姿。

## 6. 已知问题 / 需要后续 fine-tune

1. **官方 drawer policy zero-shot 不能打开自定义抽屉**：官方 policy 在 Sektion cabinet + `drawer_top_joint` +
   特定 handle frame 上训练；本场景是自定义 cabinet（`joint_0` 底抽屉，handle 由 link_1+缩放 offset 估计），
   且 obs 的 `default_joint_pos`（stack reset 事件改成 stack pose）与训练时（franka 默认）不一致。
   结果：手臂朝估计的 handle 方向运动但没真正勾到把手，drawer 未打开 → 如实记录 `DRAWER_OPEN_TIMEOUT`，
   **接口保留、绝不直接 set drawer joint**。后续若要 zero-shot：让 handle frame / obs 的 default 与官方语义严格对齐，
   或在本场景上 fine-tune drawer policy。
3. **close_drawer learned policy**：暂用 `scripted_joint`；架构已预留（`OfficialDrawerJointSkill` 可换 close policy）。

## 7. 运行命令

```bash
# (0) 旧 IK-Abs UI（回归测试入口，未改动）
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui.py \
    --num_envs 1 --show_affordance_debug --seed 1

# (1) 新 joint UI（手动点击，抽屉不会自动打开）
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/skill_test_ui_joint.py \
    --num_envs 1 --show_affordance_debug \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend scripted_joint --seed 1

# (2) joint sequence，只测抓取/放置（不碰抽屉）
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py \
    --num_envs 1 --sequence grasp:cube_1,place:point_a \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend scripted_joint \
    --seed 1 --max_steps 2000

# (3) joint sequence，显式测 scripted drawer baseline
conda activate env_isaaclab
./isaaclab.sh -p scripts/environments/state_machine/skill_sequence_joint.py \
    --num_envs 1 --sequence open_drawer:bottom_drawer --drawer_backend scripted_joint \
    --seed 1 --max_steps 1000

# --- 以下为暂缓的官方 drawer policy 路径 ---
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
