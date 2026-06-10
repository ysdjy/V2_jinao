# FrankaCabinetKnife

一个独立 Isaac Lab 小项目：复制官方 `Isaac-Open-Drawer-Franka-IK-Abs-v0` 的 Franka 开柜子任务场景，并在机器人侧边放置已有的 `Knife_101054` USD 小刀。

场景保持官方对象命名：

- `robot`
- `cabinet`
- `cabinet_frame`
- `ee_frame`

状态机展示脚本执行完整流程：

1. Franka 打开官方 Sektion 柜子的顶部抽屉。
2. Franka 移动到机器人侧边拿起小刀。
3. Franka 将小刀放入打开的抽屉。
4. Franka 回到抽屉把手处并关闭抽屉。

## 安装

在 IsaacLab 根目录执行：

```bash
conda activate env_isaaclab
bash FrankaCabinetKnife/setup.sh
```

## 运行状态机展示

```bash
./isaaclab.sh -p FrankaCabinetKnife/scripts/state_machine/open_cabinet_knife_sm.py --num_envs 1
```

无显示环境：

```bash
./isaaclab.sh -p FrankaCabinetKnife/scripts/state_machine/open_cabinet_knife_sm.py --num_envs 1 --headless
```

快速检查：

```bash
bash FrankaCabinetKnife/scripts/smoke_test.sh
```

## 已注册任务

| 任务 ID | 用途 |
| --- | --- |
| `FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-v0` | 完整状态机展示默认任务 |
| `FrankaCabinetKnife-Open-Drawer-Franka-IK-Abs-Play-v0` | 小规模 play 配置 |

## 资产

本项目自带从 `Connection/assets` 复制的本地 USD：

- `assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd`
- `assets/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd`
- `assets/Props/Knife_101054/knife.usd`

可用 `FRANKA_CABINET_KNIFE_ASSETS_DIR` 覆盖资产目录。
