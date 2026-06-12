#!/usr/bin/env bash
#
# 00_detect_network_interfaces.sh — 只读检测网口，绝不修改任何网络配置。
#
# 作用：列出所有网口、找出默认路由（上网）网口、列出每个有线网口的状态，
#       并给出 FR3 FCI 候选网口建议。最后让你手动确认并填进 configs。
#
# 本脚本不执行任何会改网络的命令（没有 ip addr、没有 nmcli modify、没有 sudo）。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================================="
echo "[00] FR3 FCI 网口检测（只读，不改网络）"
echo "=================================================="

echo ""
echo "---- ip -br link ----"
ip -br link
echo ""
echo "---- ip -br addr ----"
ip -br addr
echo ""
echo "---- nmcli device status ----"
nmcli device status 2>/dev/null || echo "(nmcli 不可用)"
echo ""
echo "---- nmcli connection show --active ----"
nmcli connection show --active 2>/dev/null || echo "(nmcli 不可用)"
echo ""
echo "---- ip route ----"
ip route
echo ""
echo "---- ip route get 8.8.8.8 ----"
ip route get 8.8.8.8 2>/dev/null || echo "(无法解析默认路由)"

# 动态找出默认路由网口（= 上网网口，禁止修改）
DEFAULT_ROUTE_INTERFACE="$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'dev [^ ]+' | awk '{print $2}' | head -1)"
if [ -z "${DEFAULT_ROUTE_INTERFACE}" ]; then
  DEFAULT_ROUTE_INTERFACE="$(ip route show default 2>/dev/null | grep -oE 'dev [^ ]+' | awk '{print $2}' | head -1)"
fi

echo ""
echo "=================================================="
echo "  DEFAULT_ROUTE_INTERFACE = ${DEFAULT_ROUTE_INTERFACE:-未知}"
echo "  ^ 这个网口是【上网 / 默认路由网口】，禁止修改！"
echo "=================================================="

echo ""
echo "---- 每个有线网口详情 ----"
# 遍历所有非 lo / 非虚拟网口
for IFACE in $(ls /sys/class/net | grep -vE '^(lo|docker|veth|br-|virbr|tailscale|wg|tun|tap)'); do
  # 只看以太网（有 device 链接的物理口）
  [ -e "/sys/class/net/${IFACE}/device" ] || continue
  MAC="$(cat /sys/class/net/${IFACE}/address 2>/dev/null)"
  CARRIER_RAW="$(cat /sys/class/net/${IFACE}/carrier 2>/dev/null || echo 0)"
  CARRIER="DOWN"; [ "${CARRIER_RAW}" = "1" ] && CARRIER="UP"
  IPS="$(ip -br addr show "${IFACE}" 2>/dev/null | awk '{$1=$2=""; print $0}' | xargs)"
  [ -z "${IPS}" ] && IPS="(无 IP)"
  IS_DEFAULT="no"; [ "${IFACE}" = "${DEFAULT_ROUTE_INTERFACE}" ] && IS_DEFAULT="YES <== 上网网口，禁止修改"
  echo ""
  echo "  interface : ${IFACE}"
  echo "  MAC       : ${MAC}"
  echo "  carrier   : ${CARRIER}"
  echo "  IP        : ${IPS}"
  echo "  默认路由  : ${IS_DEFAULT}"
done

echo ""
echo "---- FR3 FCI 候选网口 ----"
# 全部非默认路由的物理网口（FCI 口一定在这里面，无论 carrier 当前是否 UP）
NONDEFAULT=()
UP_CANDIDATES=()
for IFACE in $(ls /sys/class/net | grep -vE '^(lo|docker|veth|br-|virbr|tailscale|wg|tun|tap)'); do
  [ -e "/sys/class/net/${IFACE}/device" ] || continue
  [ "${IFACE}" = "${DEFAULT_ROUTE_INTERFACE}" ] && continue
  NONDEFAULT+=("${IFACE}")
  CARRIER_RAW="$(cat /sys/class/net/${IFACE}/carrier 2>/dev/null || echo 0)"
  [ "${CARRIER_RAW}" = "1" ] && UP_CANDIDATES+=("${IFACE}")
done

if [ "${#UP_CANDIDATES[@]}" -ge 1 ]; then
  echo "  carrier UP 的强候选：${UP_CANDIDATES[*]}"
  [ "${#UP_CANDIDATES[@]}" -gt 1 ] && echo "  多个，请按网线物理走向 / MAC 进一步确认。"
elif [ "${#NONDEFAULT[@]}" -ge 1 ]; then
  echo "  非默认路由物理网口（carrier 当前都是 DOWN）：${NONDEFAULT[*]}"
  echo "  ⚠ 这些很可能就是 FCI 口，但现在 carrier=DOWN。常见原因："
  echo "     - FR3 还没开机 / 网线没插好（最常见）；"
  echo "     - NetworkManager 把未配置的口置成 admin-down（配上静态 IP 后会变 UP）。"
  echo "  请先开机/插好网线，再重跑本脚本看 carrier 是否变 UP。"
else
  echo "  没找到任何非默认路由的物理网口。检查硬件。"
fi

echo ""
echo "=================================================="
echo "下一步（手动）："
echo "  1) 确认上面哪个 iface 是连到 FR3 Control/FCI 端口的网口。"
echo "  2) 把它填进 ${PROJECT_ROOT}/configs/fr3_fci_network.yaml ："
echo "         confirmed_fci_interface: <iface>"
echo "  3) 千万不要填默认路由网口 (${DEFAULT_ROUTE_INTERFACE:-未知})。"
echo "  4) 然后跑：bash scripts/01_configure_fci_interface_nmcli.sh   （先 dry-run）"
echo "=================================================="
