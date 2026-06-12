# FR3 FCI 排错

> 所有排错命令默认只读。涉及改网络的，先确认网口名字再手动执行，绝不碰上网网口。

## ping 不通 172.16.0.2
- 网线是否插在 FR3 的 **Control / FCI** 网口（不是普通上网口、不是 Shop 口）。
- 本机 FCI 网口 IP 是否已是 `172.16.0.3/24`：`ip -br addr show <iface>`。
- carrier 是否 UP：`cat /sys/class/net/<iface>/carrier`（1=UP）。
- 默认路由网口有没有被误改：`ip route get 8.8.8.8`（应仍是上网口）。
- 机器人是否开机、Desk 能否打开。
- 重新拉起连接：`nmcli connection up fr3-fci-<iface>`。

## Desk 打不开（https://172.16.0.2）
- 先确保 ping 通。
- 浏览器证书告警 → 选择“继续访问/高级”。
- 确认用的是 `https://` 不是 `http://`。
- 换浏览器 / 清缓存试试。

## FCI 未激活 / command rejected
- Desk 无错误、已松抱闸、模式在 Execution、已点 Activate FCI。
- 机器人处于可控状态（非手动引导模式、非错误锁定）。

## libfranka version mismatch
- libfranka 版本必须匹配 FR3 的 System Version。
- 参照另一台已调通 FCI 的电脑用的 libfranka 版本/tag，重装匹配版本（手动，确认后）。

## another FCI client connected / connection refused
- 同一时间只允许一个 FCI 客户端。先断开占用 FCI 的另一台电脑/进程，再重试。

## 误改了上网网口（最严重）
先只读查看：
```bash
ip route
ip route get 8.8.8.8
nmcli connection show --active
nmcli device status
```
确认问题后，**手动**（核对网口名字）恢复，例如：
```bash
nmcli connection down fr3-fci-<错误网口>
nmcli connection delete fr3-fci-<错误网口>
nmcli connection up "<上网口原来的连接名>"   # 用 nmcli connection show 查到的名字
```
> 本文件只列命令，不自动执行。

## 实时确认默认路由网口（任何时候）
```bash
ip route get 8.8.8.8 | grep -oE 'dev [^ ]+'
```
