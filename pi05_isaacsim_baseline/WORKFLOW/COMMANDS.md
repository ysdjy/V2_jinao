# 操作手册（数据采集 → 训练 → 推理）

> 这是给你自己用的「照着敲」命令手册。整个流程只在这一个文件夹里操作：
> `pi05_isaacsim_baseline/WORKFLOW/`。所有参数都在 `pipeline.env` 里改，脚本不用动。
>
> 固定任务：`Isaac-Stack-Cube-Franka-IK-Rel-v0`，7 维动作 `[dx,dy,dz,drx,dry,drz,gripper]`。

---

## 0. 一次性准备

```bash
cd /home1/banghai/Documents/IsaacLab            # 永远先进 IsaacLab 根目录
```

改默认参数（任务、设备、采集数量、数据集名、训练步数等）：
```bash
# 编辑这个文件即可，下面所有命令都会读它
gedit pi05_isaacsim_baseline/WORKFLOW/pipeline.env   # 或用你习惯的编辑器
```
常改的几项：`TELEOP_DEVICE`（遥操作设备）、`NUM_DEMOS`（采几条）、`ENABLE_CAMERAS`、
`REPO_ID`（数据集名字）、`TRAIN_STEPS`（训练步数）。

---

## 1. 采集遥操作数据（你亲自操作）

```bash
bash pi05_isaacsim_baseline/WORKFLOW/1_collect.sh
```
- 启动后先**用鼠标点一下 3D 窗口**，让它获得键盘焦点。
- 键盘控制：`W/S`=X，`A/D`=Y，`Q/E`=Z，`Z/X`=转X，`T/G`=转Y，`C/V`=转Z，`K`=开合夹爪，`R`=重置/放弃这一条。
- 只有**成功**的回合才会被保存；采够 `NUM_DEMOS` 条成功后自动退出。
- 数据存到 `data/raw_hdf5/<名字>.hdf5`。

常用变体：
```bash
# 临时换设备 / 数量 / 文件名（不改 pipeline.env）
bash pi05_isaacsim_baseline/WORKFLOW/1_collect.sh --device spacemouse --num 20 --name franka_real_batch1
# 不要相机（只采状态，做快速链路验证）
bash pi05_isaacsim_baseline/WORKFLOW/1_collect.sh --cameras 0
```
> 数据质量比数量重要：动作要平滑、意图清晰、夹爪时机合理；废数据用 `R` 丢掉。

---

## 2. 检查数据（可选但建议）

```bash
bash pi05_isaacsim_baseline/WORKFLOW/inspect.sh
```
看输出里有没有：`actions (T,7)`、`obs/eef_pos`、`obs/eef_quat`、`obs/joint_pos`、
`obs/gripper_pos`，以及若干 `data/demo_*`。日志存到 `logs/inspect_*.txt`。

---

## 3. 转换成训练格式（HDF5 → LeRobot）

```bash
bash pi05_isaacsim_baseline/WORKFLOW/2_convert.sh
```
- 默认拿**最新**那个 HDF5，转换到 `data/lerobot/$REPO_ID/`。
- 看到 `missing: []`、`lerobot_built: true` 就对了。
- 如果采集时**没开相机**（状态-only），先这样跑（注入占位图片，仅用于跑通链路）：
```bash
SYNTH_IMAGES=224 bash pi05_isaacsim_baseline/WORKFLOW/2_convert.sh
```

指定输入/数据集名：
```bash
bash pi05_isaacsim_baseline/WORKFLOW/2_convert.sh --input data/raw_hdf5/franka_real_batch1.hdf5 --repo-id franka_real
```

---

## 4. 训练 π0.5（LoRA，单卡可跑）

先小步冒烟（验证链路，约几分钟，第一次会下载 ~10GB 基础权重）：
```bash
bash pi05_isaacsim_baseline/WORKFLOW/3_train.sh --steps 10
```
正式训练（步数调大）：
```bash
bash pi05_isaacsim_baseline/WORKFLOW/3_train.sh --repo-id franka_real --steps 3000
```
- 模型存到 `policies/checkpoints/pi05_isaaclab_<时间戳>/.../<步数>/`。
- 训练日志在 `logs/train_pi05_*.log`。
- 说明：当前用 **LoRA 低显存**配置，能在单张 48GB 卡上 `batch=1` 跑。要做**全量微调**
  需要更多显存，并用 `--fsdp 2`、偶数 batch（见 README「训练显存」一节）。

---

## 5. 启动推理服务（policy server）

```bash
bash pi05_isaacsim_baseline/WORKFLOW/4_serve.sh
```
- 默认加载**最新**的 checkpoint，监听 `http://127.0.0.1:8008`。
- 没有 checkpoint 时自动用 mock 策略（方便先把闭环跑通）。
- 健康检查：`curl http://127.0.0.1:8008/health`
- 强制用 mock：`bash pi05_isaacsim_baseline/WORKFLOW/4_serve.sh --mock`

---

## 6. 在 IsaacSim 里闭环评估

```bash
bash pi05_isaacsim_baseline/WORKFLOW/5_eval.sh
```
- 自动起服务 + 在 IsaacLab 里跑 `NUM_ROLLOUTS` 个回合，结束自动停服务。
- 结果（成功率、推理延迟、控制频率、安全裁剪次数）写到 `logs/eval_policy_*.json`。
- 只想用 mock 把闭环跑通：`bash pi05_isaacsim_baseline/WORKFLOW/5_eval.sh --mock`

停止服务（如果手动起过）：
```bash
bash pi05_isaacsim_baseline/WORKFLOW/stop.sh
```

---

## 一条龙（采完数据后，从转换到评估）

```bash
cd /home1/banghai/Documents/IsaacLab
bash pi05_isaacsim_baseline/WORKFLOW/2_convert.sh
bash pi05_isaacsim_baseline/WORKFLOW/3_train.sh --steps 3000
bash pi05_isaacsim_baseline/WORKFLOW/5_eval.sh        # 内部会自动起/停服务
```

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `No module named 'isaaclab'` | 脚本会自动 `conda activate env_isaaclab`；手动跑时先激活它 |
| 训练 OOM（显存不足） | 保持 LoRA + `batch=1`；要全量微调需更多显存（见 README） |
| 转换报 `missing: [...]` | 按提示改 `configs/dataset_mapping_isaaclab_franka.yaml` 的候选字段名 |
| 服务连不上 / 超时 | 客户端会自动返回安全零动作；看 `logs/` 里的 server 日志 |
| 训练用错了数据集 | `3_train.sh` 每次都会按当前 `--repo-id` 重新注册配置，确认 `--repo-id` 与 `data/lerobot/<名>` 一致 |
| 想换遥操作设备 | 改 `pipeline.env` 的 `TELEOP_DEVICE`；GELLO 需要先做 `Se3Gello` 设备（见 README） |

> 数据/模型/日志的存放位置，和每一步对应的真实脚本，见 `WORKFLOW/README.md`。
