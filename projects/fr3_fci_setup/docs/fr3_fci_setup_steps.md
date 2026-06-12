# FR3 FCI 配置操作顺序

> 全程：脚本默认只读；只有 `01 --apply` 和 `07 --apply` 会产生实际作用，且都有拦截/确认。

```bash
cd /home1/banghai/Documents/IsaacLab/projects/fr3_fci_setup
```

## 阶段 A：网络

1. **检测网口（只读）**
   ```bash
   bash scripts/00_detect_network_interfaces.sh
   ```
   记下 `DEFAULT_ROUTE_INTERFACE`（上网口，禁止动）和 FCI 候选。

2. **手动确认并填配置**
   编辑 `configs/fr3_fci_network.yaml`：
   ```yaml
   confirmed_fci_interface: <你确认的FCI网口，例如 eno1>
   ```
   绝不能填默认路由网口。

3. **配置 FCI 网口（先 dry-run）**
   ```bash
   bash scripts/01_configure_fci_interface_nmcli.sh
   ```
   看清将执行的 nmcli 命令、确认拦截都通过。

4. **真正配置（确认无误后）**
   ```bash
   bash scripts/01_configure_fci_interface_nmcli.sh --apply
   ```
   执行后脚本会自动核对默认路由没变。

5. **ping 机器人**
   ```bash
   bash scripts/02_check_fci_ping.sh
   ```

6. **激活 FCI（手动在浏览器）**
   ```bash
   bash scripts/03_check_franka_desk.sh
   ```
   浏览器开 `https://172.16.0.2` → 登录 → 无错误 → 松抱闸 → Execution → Activate FCI。

## 阶段 B：libfranka 测试

7. **检查 libfranka**
   ```bash
   bash scripts/04_check_libfranka.sh
   bash scripts/05_find_libfranka_examples.sh
   ```

8. **read state 测试（只读优先）**
   ```bash
   bash scripts/06_run_fci_read_state_test.sh
   ```

9. **阻抗示例（会动！先打印再确认）**
   ```bash
   bash scripts/07_run_impedance_example_guarded.sh
   # 满足安全清单后：
   bash scripts/07_run_impedance_example_guarded.sh --apply   # 交互输入 RUN
   ```

## 验收
- ping `172.16.0.2` 通；默认路由网口未变、上网正常。
- Desk 能登录、FCI 能激活。
- `echo_robot_state` 能持续打印机器人状态。
- 官方阻抗示例能小幅运行、可随时急停。
