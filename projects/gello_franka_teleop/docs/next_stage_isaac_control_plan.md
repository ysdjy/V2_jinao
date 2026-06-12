# 下一阶段计划：GELLO → Isaac Sim / Isaac Lab Franka（仅规划，先不实现）

> 本文件只描述方向，**第一阶段不实现以下任何一项**。
> 第一阶段的唯一目标是：稳定读到 GELLO 关节数据。

## 总体思路

GELLO 读到的 7 个关节角 `q[0:7]` 直接作为 Isaac Lab 里 Franka 的关节目标
（method B：joint-space 遥操作，不需要 IK）。夹爪单独映射。

本仓库已有的 joint-action 状态机经验（见项目记忆 `joint-action-state-machine`、
`joint-env-gotchas`）可复用：动作 offset 缓存、PD 刚度匹配等坑要注意。

## 计划步骤（按优先级）

1. **关节映射**
   - GELLO `q[0:7]` → Isaac Franka `joint_pos_target[0:7]`。
   - 注意两边关节顺序/正负方向是否一致，必要时加一层映射表。
   - 夹爪 [0,1] → Franka 夹爪开合（width 或 binary）。

2. **低通滤波**
   - 对 GELLO 读数做一阶低通 / EMA，减少抖动后再下发，避免 Franka 抽搐。
   - gello_software 本身在 `get_joint_state` 里已有 alpha 平滑，可叠加或替换。

3. **关节限幅**
   - 对目标关节做 Franka 关节上下限 clip。
   - 限制单步最大变化量（rate limit），防止突跳。

4. **急停 / 安全**
   - 键盘急停（如空格暂停下发、维持当前姿态）。
   - 检测 NaN / 超限 / 长时间无数据 → 立即冻结，不下发。

5. **接入 Isaac Lab**
   - 在 IsaacLab 里跑一个 Franka 场景，把过滤+限幅后的关节目标喂给 articulation。
   - 先空载验证跟随，再加物体交互。

6. **最后才做数据采集**
   - 跟随稳定后，再记录 (GELLO q, Franka state, action, 图像) 用于模仿学习。

## 不做的事（继续保持）

- 不引入 ROS。
- 不修改 IsaacLab 核心源码（控制逻辑放在本项目内，或通过既有入口接入）。
- 不在跟随未稳定前就做大规模数据采集。
