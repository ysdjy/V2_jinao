# 项目交接文档 (HANDOFF) — Franka 技能状态机（方案 B）

> 写给接手的 AI / 工程师，帮助你**快速、完整**地理解当前项目，直接接手后续任务。
> 当前最新提交：`111ca37e4e`（remote `v2_jinao` 的 `main` 分支）。
> 阅读顺序建议：本文件 → `README.md` → `指令.txt` → `JOINT_PLAN_NOTES.md`（详细演进与诊断）。

---

## 0. 一句话总览

在 Isaac Lab 上做一个 **Franka 技能状态机（方案 B）**：状态机统一调度技能，所有技能最终都向**同一个
joint-position 环境**输出 joint action。抓取/放置内部用 IK（封装在技能里）；开/关抽屉用 **IK 物理抓把手再拉/推**
（不训练、不直接 set 关节）。另有一套"用 RL 训练开抽屉小脑"的代码，暂存待用。

---

## 1. 仓库 / 环境 / 运行方式（务必先懂）

- 本仓库是 **Isaac Lab 的 fork**。远程：`v2_jinao` = `https://github.com/ysdjy/V2_jinao.git`，
  当前工作分支 `push-v2-skill-runtime`，**push 到该远程的 `main`**：`git push v2_jinao push-v2-skill-runtime:main`。
- conda 环境：**`env_isaaclab`**（不要新建环境）。
- 所有运行都用：`./isaaclab.sh -p <脚本.py> [args]`（在仓库根目录执行）。GUI 去掉 `--headless`。
- **规则**：commit/push 只在用户明确要求时做；测试都 `--num_envs 1 --headless` 控算力；长循环要 `--max_steps`。

### ⚠️ 运行调试的关键坑（先记住，能省大量时间）
1. **被 timeout/SIGKILL 杀掉的 Isaac 进程会丢失 Python 缓冲输出**（kit 不 flush）。只有**正常跑完**（env.close）的
   run 才能在日志里看到 `print` 输出。所以：给足 `timeout`，让 run 跑完再看日志；否则日志里只有 kit 的 app 日志、没有技能输出。
2. **fabric 开启时 `UsdGeom.XformCache` 读到的是 authored（旧）位姿**（stale）。要读真值用 articulation 的
   `body_pos_w`，或加 `--disable_fabric`。
3. Isaac 启动慢（~20s），一次 grasp+place+开关抽屉的 sequence 约 1–2 分钟 wall。
4. 杀进程别用 `pkill -f "skill_test_ui_joint"`（会匹配到你自己的命令行而自杀）；用 bracket 技巧
   `pkill -9 -f "[s]kill_test_ui_joint"`，且 kill 和 relaunch 分成**两条命令**。

---

## 2. 项目代码结构（已重组到一个文件夹）

主项目在 **`projects/franka_skill_state_machine/`**（克隆整个仓库即可运行；env 注册在 isaaclab_tasks 内）：

```
projects/franka_skill_state_machine/
├── README.md / 指令.txt / HANDOFF.md(本文件) / JOINT_PLAN_NOTES.md / PROJECT_STATUS.md
├── skills/                  ← 4 个通用技能块（状态机引用）
│   ├── grasp_skill.py        GraspJointSkill
│   ├── place_skill.py        PlaceJointSkill
│   ├── open_drawer_skill.py  OpenDrawerIKSkill
│   └── close_drawer_skill.py CloseDrawerIKSkill
├── state_machine/skill_executor.py   SkillExecutor + JointBackendConfig
├── runtime/                 ← 共享运行时
│   ├── scene_state_provider.py   读状态 + 构造 joint action
│   ├── ik_joint_adapter.py       DLS IK: world TCP pose → q_des
│   ├── drawer_ik_common.py       开门方向 + 抓把手朝向几何
│   ├── drawer_obs_adapter.py     DrawerObsAdapter / SelectedDrawerObsAdapter（把手/关节读取, 31维obs）
│   ├── drawer_target_config.py   re-export 中央抽屉配置（真正定义在 isaaclab_tasks，见 §4）
│   ├── target_registry.py        抓取 affordance（cube/knife 的抓取计划）
│   ├── simple_scene_layout.py    场景布局 + 区域随机初始化
│   ├── grasp_skill.py/place_skill.py/drawer_skill.py  内部"规则状态机"（被技能块复用）
│   ├── base_skill.py/skill_types.py/skill_request.py/skill_result.py
│   ├── debug_visualizer.py/joint_debug_logger.py/ui_controller.py
├── learned_drawer/          ← RL 学习开抽屉（暂存，非主路径）
│   ├── official_drawer_policy.py / official_drawer_joint_skill.py
│   ├── custom_drawer_joint_skill.py / scripted_drawer_joint_skill.py
│   └── export_*.py / finetune_custom_drawer_from_official.py
└── entries/                 ← 运行入口（自举 sys.path 到项目根）
    ├── skill_test_ui_joint.py     joint 交互 UI（主用）
    ├── skill_sequence_joint.py    headless sequence（主用）
    ├── skill_test_ui.py           旧 IK-Abs UI（回归）
    └── debug_drawer_joint_scan.py / debug_drawer_handle_calib.py /
        debug_handle_grasp_test.py / debug_open_drawer_ik.py / debug_custom_drawer_env.py
```

- **导入方案**：包内用绝对导入 `runtime.* / skills.* / state_machine.* / learned_drawer.*`；
  入口脚本顶部自举：`_sys.path.insert(0, <项目根>)`。**注意**：缩进的局部 import 别漏改（曾因此崩溃）。
- **兼容垫片**：旧路径 `scripts/environments/state_machine/skill_runtime/__init__.py` 是个 shim，
  把模块以旧 `skill_runtime.<name>` 名再暴露（用户的 `skill_test_ui_joint_gello.py` 靠它继续工作，别删）。

---

## 3. 底层环境 / 动作契约（核心）

- **部署环境**：`Isaac-Stack-Cube-Franka-JointPolicy-v0`（所有技能跑在它上面）。
  - 机器人 `FRANKA_PANDA_HIGH_PD_CFG`（**高刚度 PD + 机械臂 disable_gravity=True**）——为让 IK 抓取/放置跟踪精准、不下垂。
  - arm action：`JointPositionActionCfg(joint_names=["panda_joint.*"], scale=1.0, use_default_offset=True)`；
    gripper：`BinaryJointPositionActionCfg`。**action 8 维**（7 arm + 1 gripper）。**gripper: +1=张开, −1=闭合**（已验证）。
  - ⚠️ action 的 `_offset` 在 action term **构造时**缓存 = Franka cfg 默认关节角（不是 stack reset 事件改的姿态）。
    所以 q_des→raw 必须读 action term 的 `_scale`/`_offset`（`scene_state_provider._resolve_joint_action` 已做），
    不要用 `robot.data.default_joint_pos`。
- **抽屉执行器**：`ik_pull` 后端会把 cabinet drawer 执行器设成 **stiffness=0（自由滑动）**，让夹爪能物理拉开；
  其它后端 stiffness=10。
- **自动 reset 已禁用**：UI/sequence 入口里设 `episode_length_s=1e9` 并清掉 cube 终止项，交互时不会定时重置场景。

---

## 4. 场景（非常重要，详细）

场景定义在 **isaaclab_tasks**（不在项目文件夹里，因为要 gym 注册）：
`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/`
- `stack_joint_pos_env_cfg.py` —— 基础自定义场景（被 JointPolicy 继承）。物体与位姿：
  | 物体 | 资产 | 位置(world) | 朝向(wxyz) | scale |
  |------|------|-------------|-----------|-------|
  | Franka | panda_instanceable.usd | 原点 | 单位 | — |
  | cube_1(蓝)/cube_2(红)/cube_3(绿) | Cuboid 0.0406 | (0.4,0,0.0203)/(0.55,0.05,0.0203)/(0.6,-0.1,0.0203) | 单位 | — |
  | Cabinet | simv2/USD/Cabinet_44853/cabinet.usd | **(0.27402, 0.91583, 0.323)** | **(0.7071,0,0,0.7071)=绕Z +90°** | 0.62 |
  | Knife | simv2/USD/Knife_101054/knife.usd | (0.35,0.28,0.095) | 绕Z +90° | 0.12 |
  | CoffeeMachine | SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd | (0.43152,-0.48373,0.135) | 绕Z −90° | 0.2 |
  | 3 个把手碰撞代理 | Cuboid 细条 (0.10,0.028,0.028) | 见下 | 单位 | (随柜子0.62) |

- `stack_joint_policy_env_cfg.py` → 注册 `Isaac-Stack-Cube-Franka-JointPolicy-v0`（高PD机器人，scale=1.0）。
- `stack_ik_abs_env_cfg.py` → 注册 `Isaac-Stack-Cube-Franka-IK-Abs-v0`（旧 IK-Abs，仅 `skill_test_ui.py` 用）。
- `custom_drawer_*` + `agents/rsl_rl_custom_drawer_ppo_cfg.py` → 注册 RL 训练任务
  `Isaac-Open-CustomDrawer-Selected-Franka-v0(/-Play-v0)`（见 §7 learned_drawer）。

### 抽屉关键事实（已用 `debug_drawer_joint_scan.py` 确认）
| target | joint | link | body z(高度) | 状态 |
|--------|-------|------|-------------|------|
| top_drawer | joint_0 | link_0 | 0.684 | ✅ 可开关 |
| middle_drawer | joint_2 | link_2 | 0.391 | ✅ 可开关 |
| bottom_drawer | joint_1 | link_1 | 0.111 | ❌ **asset 物理卡死**（直接写关节状态都被弹回0，疑似 closed 处穿模） |
- 全部 prismatic, 局部 axis=Z, limits **[0,0.8]**, closed=0, open_direction=+1。
- 柜子绕Z+90°后，**抽屉在世界里沿 −Y 方向开**（朝机器人）。开门方向由 `drawer_ik_common.open_direction_world(cabinet_quat)`
  = `quat_apply(cab_quat, (-1,0,0))` 推出（三个抽屉一致）。
- **把手前表面偏移**（link-local，已用 `debug_drawer_handle_calib.py` 在旋转后场景标定，存在
  `custom_drawer_config.py:HANDLE_LOCAL_OFFSET`）：top `(-0.0733,-0.0053,0.0395)`、middle `(0.0306,0.0389,0.6824)`、
  bottom `(0.0741,0.022,0.6737)`。三个把手世界位都落在前表面 (x≈0.272, y≈0.53)，z 分别 0.679/0.430/0.133。
- 把手碰撞代理（可视、半透明、带碰撞）：TopHandleProxy/MiddleHandleProxy/BottomHandleProxy，
  局部位置 = `HANDLE_LOCAL_OFFSET / 0.62`（柜子 scale）。

### 部署时的场景布局（重要差异）
- UI/sequence 入口用 **`SimpleSceneLayoutManager`** 在 reset 时重排场景：
  - 柜子放到 `CABINET_LOCAL=(0.27402, 0.91583)`（与 cfg 一致；姿态来自 cfg 四元数 yaw+90°）。
  - **cube_1/2/3 + knife 在用户指定四边形区域内随机初始化**：角点
    `[(0.15,-0.30),(0.58,-0.30),(0.70,0.25),(0.22,0.37)]`，拒绝采样 + `min_separation=0.15`（避免干涉）。
    （仿照官方 `franka_stack_events.sample_object_poses`。）
  - 注意：随机区域 x 可低到 0.15，所以抓取工作空间 `WORKSPACE_X_MIN` 已放宽到 **0.13**（否则 cube 会被判 TARGET_UNSAFE）。
- 资产清单（场景报告）：`saved_scenes/v0_layout/scene_v0_*_report.md`。当前场景与该报告一致。

---

## 5. 四个技能块（通用，状态机只给目标）

所有技能输出 **joint command**（`SkillCommand(control_mode="joint", joint_target=q_des 或 raw_joint_action, gripper_command, drawer_joint_target=None)`）。

1. **GraspJointSkill** (`skills/grasp_skill.py`)：状态机给目标物体名（cube_1/2/3/knife）。内部复用规则状态机
   `runtime/grasp_skill.py(GraspSkill)` 算阶段（pre_grasp/descend/close/lift），每步把 TCP 目标用
   `IKJointAdapter` 求 q_des 输出。
   - **抓取姿态已与 cube 坐标系解耦**（`target_registry._cube_grasp_plan` + `_snapped_gripper_yaw`）：
     永远**俯视(top-down)**，yaw 取"当前夹爪 yaw 最近的 90°"（从需要手臂转动最小的那对侧面夹），cube 翻转也不会侧/下抓。
2. **PlaceJointSkill** (`skills/place_skill.py`)：状态机给目标点 xyz（point_a..d 等）。复用 `runtime/place_skill.py`。
3. **OpenDrawerIKSkill** (`skills/open_drawer_skill.py`)：状态机只给 target_drawer。阶段：
   `MOVE_TO_HOME`（先回默认姿态做 IK 好初值，**关键**，否则手腕会顶 joint6 限位）→ `MOVE_TO_PRE_GRASP`（把手外侧 0.12）
   → `APPROACH`（把手）→ `CLOSE_GRIPPER` → `PULL`（沿开门方向把把手往外拉，**每步实时读把手位姿**）→ 成功 drawer≥0.20。
4. **CloseDrawerIKSkill** (`skills/close_drawer_skill.py`)：同上但 `PUSH` 沿 −开门方向往柜里推 → drawer≤0.01。
   - 注意：close 的 pre-grasp 不外退（否则会把已夹/自由的抽屉拖更开）。

- 开/关门技能都靠 `runtime/drawer_ik_common.py`（开门方向、抓把手朝向：手指上下夹横把手、从外侧 −open_dir 接近）
  + `SelectedDrawerObsAdapter`（实时读把手世界位姿 = drawer link body pose + HANDLE_LOCAL_OFFSET；读 drawer 关节）。
- **绝不** `set_cabinet_joint_target`，`drawer_joint_target` 恒 None —— 抽屉只因物理拉/推而动。

---

## 6. 状态机 / 后端（state_machine/skill_executor.py）

- `SkillExecutor(registry, backend=JointBackendConfig(...))`，`backend.mode="joint"`。
- `JointBackendConfig` 字段：`grasp_backend="joint_ik"`、`place_backend="joint_ik"`、`drawer_backend`、
  `adapter`(IKJointAdapter)、`drawer_policy`、`drawer_obs_adapter`、`drawer_env`(env句柄)、`arm_joint_ids` 等。
- `drawer_backend` 取值：
  - `none`：禁用开关抽屉（UI 默认）。
  - `scripted_joint`：直接 set 抽屉关节的 baseline（机器人不动；会打印 `[BASELINE]`）。
  - `official_joint_policy` / `custom_selected_policy`：学习 policy（需 torchscript policy.pt，见 §7）。
  - **`ik_pull`：物理 IK 开/关门（OpenDrawerIKSkill/CloseDrawerIKSkill）——当前推荐、已验证可用。**
- 支持 pause / resume / switch；切换技能立即用新命令（不等旧技能跑完）。pause 时保持当前关节位置。

---

## 7. learned_drawer/（RL 学习开抽屉，暂存）

- 训练任务 `Isaac-Open-CustomDrawer-Selected-Franka-v0`：每个 episode 随机选一个 functional 抽屉，
  policy 只针对"选定抽屉"（obs 含 selected 把手相对位姿 + 选定抽屉关节），obs **31 维**、action 8 维，
  网络与官方 `Isaac-Open-Drawer-Franka` 一致（便于加载官方 checkpoint fine-tune）。
- 文件：`custom_drawer_mdp.py`(选定 obs/reward/event/termination)、`custom_drawer_selected_env_cfg.py`、
  `agents/rsl_rl_custom_drawer_ppo_cfg.py`（都在 isaaclab_tasks 内）；技能/导出在 learned_drawer/。
- 状态：只做过 **sanity 训练**（128env×5iter）+ 从官方 checkpoint fine-tune sanity；**未正式训练**。
  官方 policy **zero-shot 打不开**自定义柜子（几何/把手/obs 不一致）。**当前主路径是 IK（ik_pull），不是学习 policy。**
- 已知：把手 obs 近似 = drawer link 原点 + 标定偏移；正式训练前可进一步精修。

---

## 8. 运行指令（详见 `指令.txt`）

```bash
conda activate env_isaaclab
# joint 交互 UI（推荐 ik_pull）
./isaaclab.sh -p projects/franka_skill_state_machine/entries/skill_test_ui_joint.py \
  --num_envs 1 --show_affordance_debug --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull --seed 1
# headless：抓取+放置+开关上/中抽屉
./isaaclab.sh -p projects/franka_skill_state_machine/entries/skill_sequence_joint.py --num_envs 1 \
  --sequence grasp:cube_1,place:point_a,open_drawer:top_drawer,close_drawer:top_drawer \
  --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull --seed 1 --max_steps 3500 --headless
# 抽屉关节扫描 / 把手标定（诊断）
./isaaclab.sh -p projects/franka_skill_state_machine/entries/debug_drawer_joint_scan.py --num_envs 1 --drawer_joint all --values 0.00,0.05,0.10,0.20,0.30 --disable_fabric --headless
./isaaclab.sh -p projects/franka_skill_state_machine/entries/debug_drawer_handle_calib.py --num_envs 1 --disable_fabric --headless
```

---

## 9. 当前已验证可用

- ✅ grasp:cube_1 + place:point_a（joint IK，俯视抓取、解耦 cube 朝向）。
- ✅ open/close **top & middle** 抽屉（ik_pull 物理开关，无 policy，drawer_joint_target=None）；
  端到端：grasp→place→open top(0.206)→close top(0.0)；open middle(0.201)→close middle(0.008)。
- ✅ 关门基本无缝（top 0.0、middle 0.008）。
- ✅ cube/knife 在用户区域内随机初始化，无干涉。
- ✅ 场景与用户保存场景（含咖啡机）一致；三个把手有可视碰撞体。
- ✅ UI 不崩、不自动重置。

## 10. 已知问题 / 待办（接手重点）

1. **下抽屉(bottom/joint_1/link_1) asset 物理卡死**，物理拉不开（把手位姿已修正一致）。要真能开需修该抽屉的
   碰撞网格 / closed 偏移 / 摆放（在 `simv2/USD/Cabinet_44853` 或 env cfg）。
2. **资产归集（用户要求，未做）**：把场景 USD 都放到 `SapienAssetPipeline/usd_assets/`。
   注意：`usd_assets/` 下其实**已有** Cabinet_44853、Knife_101054、CoffeeMachine_103046、Microwave_7320、Fridge_12252，
   但 env cfg 仍引用 `simv2/USD/Cabinet_44853/...` 和 `simv2/USD/Knife_101054/...`。
   归集 = 把 `stack_joint_pos_env_cfg.py` 里 cabinet/knife 的 `usd_path`（`_simv2_usd_path(...)`）改指向
   `SapienAssetPipeline/usd_assets/...`（资产已就位），然后跑一次 env 加载测试。
3. **微波炉任务（用户新需求，未做）**：把咖啡机换成 `SapienAssetPipeline/usd_assets/Microwave_7320`（同位置附近），
   注意摆放姿态/尺寸让 Franka 够得到、给**微波炉门把手加碰撞体**，再做**开门/关门技能块**
   （和开/关抽屉技能一样：状态机给门把手 pose → 技能自己引导机器人开/关门）。微波炉是**转动门**（revolute），
   不是平移抽屉，开门技能需要绕铰链做圆弧轨迹（可参考 drawer 技能，但 PULL 改成绕门轴的弧线）。
4. learned_drawer 若要走学习路线：需正式训练 + 把手 obs 精修；且训练用普通 PD、部署用高PD+无重力，
   存在 PD 不一致（已在 JOINT_PLAN_NOTES 记录）。

## 11. 最近修过的坑（避免重复踩）

- 重组后崩溃：`sed` 只改了行首 `from .`，漏了**缩进的局部 import** → 已修；改动后务必 grep 检查 `^[ ]+from \.`。
- 场景定时刷新：env 的 `episode_length_s` 超时自动 reset → UI/sequence 已禁用。
- 抓取从侧/下：grasp 朝向曾绑 cube 坐标系 → 改为俯视 + 最近90°（解耦）。
- 开抽屉顶 joint6 限位：开/关门技能加 `MOVE_TO_HOME`（IK 好初值）→ top/middle 都能开了。
- 把手位姿不一致：开门方向改由柜子朝向推导 + 重标定三把手前表面偏移。

---

接手建议顺序（按用户优先级）：先做 **资产归集（小、低风险）** → 再做 **微波炉替换 + 门把手碰撞 + 开/关门技能**
→ 视情况修 **下抽屉卡死**。所有测试 `--num_envs 1 --headless`，跑完再看日志；改动经用户确认后再 push。
