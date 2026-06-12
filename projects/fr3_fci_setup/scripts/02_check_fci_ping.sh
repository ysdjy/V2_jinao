#!/usr/bin/env bash
#
# 02_check_fci_ping.sh — ping FR3，确认链路通。只读，不改网络。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/fr3_fci_network.yaml"

get_yaml() {
  grep -E "^[[:space:]]*$1[[:space:]]*:" "${CONFIG}" | head -1 \
    | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//"
}

ROBOT_IP="$(get_yaml robot_ip)"
LOCAL_IP="$(get_yaml local_fci_ip)"
IFACE="$(get_yaml confirmed_fci_interface)"

echo "=================================================="
echo "[02] ping FR3: ${ROBOT_IP}"
echo "=================================================="
echo "  本机 FCI IP 应为: ${LOCAL_IP}  网口: ${IFACE}"
echo ""

if ping -c 4 -W 2 "${ROBOT_IP}"; then
  echo ""
  echo "[02] PASS：能 ping 通 ${ROBOT_IP}。"
  echo "  下一步： bash scripts/03_check_franka_desk.sh"
  exit 0
fi

echo ""
echo "[02] FAIL：ping 不通 ${ROBOT_IP}。排查（不要自动改网）："
echo "  1) 网线是否插在 FR3 的 Control / FCI 网口（不是普通上网口）。"
echo "  2) 本机 FCI 网口是否已是 ${LOCAL_IP}/24：  ip -br addr show ${IFACE}"
echo "  3) 是否误改了上网网口（先跑 00 脚本核对默认路由网口没变）。"
echo "  4) 机器人是否已开机、Desk 是否能打开。"
echo "  5) 机器人 IP 是否仍是 ${ROBOT_IP}。"
echo "  6) 试着重新拉起 FCI 连接：  nmcli connection up fr3-fci-${IFACE}"
echo "  7) 确认 ${IFACE} 的 carrier 是 UP：  cat /sys/class/net/${IFACE}/carrier"
exit 1
