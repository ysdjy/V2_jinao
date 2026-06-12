# FR3 FCI 网络配置与 libfranka 基础测试

在这台双网口 Ubuntu 上**安全地**配置 Franka FR3 的 FCI 网络，ping 通机器人、激活 FCI，并跑通官方 libfranka 示例（read state + 阻抗控制）。

> ⚠️ **最高优先：绝不碰上网网口。** 本机两个有线网口，一个上网（默认路由），一个连 FR3 FCI。
> 所有脚本默认**只读**；唯一会改网络的 `01` 脚本默认 dry-run，必须 `--apply` 才执行，且对默认路由网口有多重拦截。
> 本阶段**不启动 Isaac Sim、不做真实-仿真联动**。

---

## 1. 项目目标
- 阶段 A：给 FCI 网口配 `172.16.0.3/24`，ping 通 `172.16.0.2`，在 Desk 激活 FCI。
- 阶段 B：跑通官方 libfranka 示例（先 read state，再小幅阻抗控制）。

## 2. 两个网口的风险
你有两个有线网口：
- **上网口（默认路由）**：当前是 `enp9s0f2np2`，IP `10.96.80.171/21`，承载 `default` 路由。**改它=断网**，禁止。
- **FCI 口**：当前强候选 `eno1`（有线、carrier UP、无 IP、非默认路由）。需要你确认。

## 3. 如何识别上网网口
```bash
ip route get 8.8.8.8     # 输出里的 dev xxx 就是上网/默认路由网口
ip route                 # default via ... dev xxx
```
`00_detect_network_interfaces.sh` 会自动把它标成 `DEFAULT_ROUTE_INTERFACE` 并提示“禁止修改”。

## 4. 为什么不能改默认路由网口
改它的 IP / 加错 gateway / flush 它，会立刻断网，甚至断开你正在用的远程连接。所以脚本会**动态**检测默认路由网口，凡是 `confirmed_fci_interface` 等于它、或它当前承载默认路由，一律拒绝执行。

## 5. 如何确认 FCI 网口
跑 `00` 脚本看候选；结合**网线物理走向**（哪根线插在 FR3 的 Control/FCI 口）和 MAC 地址确认。确认后填进 `configs/fr3_fci_network.yaml`：
```yaml
confirmed_fci_interface: eno1   # 换成你确认的名字，绝不能是上网网口
```

## 6. 如何配置 172.16.0.3/24
```bash
bash scripts/01_configure_fci_interface_nmcli.sh           # dry-run，只打印命令
bash scripts/01_configure_fci_interface_nmcli.sh --apply   # 确认后真正执行
```
它用 nmcli 建/改一个专用连接 `fr3-fci-<iface>`：`ipv4.method manual`、`addresses 172.16.0.3/24`、**不设 gateway / DNS**、`never-default yes`、`ipv6 disabled`。只动这一个网口，不抢默认路由。执行后会再次核对默认路由没变。

## 7. 如何 ping 机器人
```bash
bash scripts/02_check_fci_ping.sh        # ping -c 4 172.16.0.2
```

## 8. 打开 Desk 并激活 FCI
```bash
bash scripts/03_check_franka_desk.sh
```
然后浏览器开 `https://172.16.0.2` → 证书告警继续 → 登录 → 确认无错误 → 解锁关节/松抱闸 → 切 Execution → **手动 Activate FCI**。

## 9. 检查 libfranka
```bash
bash scripts/04_check_libfranka.sh        # dpkg / ldconfig / pkg-config / 找示例
bash scripts/05_find_libfranka_examples.sh
```
没装/没编译都**不会自动装**。装之前务必确认版本与 FR3 系统版本匹配（最好参照另一台已调通的电脑）。

## 10. 运行 read state 测试
```bash
bash scripts/06_run_fci_read_state_test.sh
```
优先 `echo_robot_state`（纯读取，不动机器人）；只有 `communication_test` 时会提示其进入零力矩控制环并要交互确认。

## 11. 安全运行阻抗控制示例
```bash
bash scripts/07_run_impedance_example_guarded.sh           # 仅打印 + 安全清单
bash scripts/07_run_impedance_example_guarded.sh --apply   # 交互确认(输入 RUN) 后才运行
```
**会让真实 FR3 运动**，运行前务必满足 `docs/safety_checklist.md` 全部条目，急停在手。

## 12. 常见错误
| 现象 | 排查 |
|------|------|
| ping 不通 | 网线插对端口了吗（FCI 口而非上网口）；FCI 网口 IP 是否 172.16.0.3/24；机器人是否开机；IP 是否仍 172.16.0.2；`nmcli connection up fr3-fci-<iface>` |
| Desk 打不开 | 先 ping 通；证书告警要点“继续”；确认 `https://`（不是 http） |
| FCI 未激活 | Desk 里切 Execution 后点 Activate FCI；机器人需无错误、已松抱闸 |
| libfranka version mismatch | libfranka 版本要匹配 FR3 系统版本；重装匹配版本 |
| command rejected | Desk 有错误 / 未松抱闸 / 未激活 FCI / 机器人不在可控状态 |
| another FCI client connected | 另一台电脑占用了 FCI；同一时间只能一个客户端，先断开它 |
| 配错了网口（误改上网口） | 见下方“误改恢复” |

## 13. 如果误改了上网网口（先排查，不要自动恢复）
```bash
# 查看现状（只读）
ip route
ip route get 8.8.8.8
nmcli connection show --active
nmcli device status
```
若发现上网网口被本项目的 `fr3-fci-*` 连接占用或默认路由没了，**手动**（确认后）处理，例如：
```bash
# 把误建的连接停掉（确认名字后再执行）
nmcli connection down fr3-fci-<错误网口>
nmcli connection delete fr3-fci-<错误网口>
# 让上网网口回到它原来的连接（名字用 nmcli connection show 查到的那个）
nmcli connection up "Wired connection 2"
```
> 这些命令本 README 只列出，不自动执行。执行前务必确认网口名字。

## 执行顺序速查
见 `docs/fr3_fci_setup_steps.md`。真实控制前必读 `docs/safety_checklist.md`，排错见 `docs/troubleshooting.md`。
