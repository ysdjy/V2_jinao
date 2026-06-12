#!/usr/bin/env bash
#
# 01_configure_fci_interface_nmcli.sh — 给【已确认的】FR3 FCI 网口配置静态 IP 172.16.0.3/24。
#
# 安全设计：
#   * 默认 dry-run：只打印将要执行的命令，不动网络。加 --apply 才真正执行。
#   * 多重拦截：confirmed_fci_interface 为 null / 等于默认路由网口 / 在 forbidden 列表 / 不存在 -> 拒绝。
#   * 不设 gateway、不设 DNS、never-default=yes、ipv6 disabled -> 不会抢默认路由、不影响上网。
#   * 只操作这一个网口；其它网口完全不碰。
#   * 不用 ip addr flush。
#
# 用法：
#   bash scripts/01_configure_fci_interface_nmcli.sh           # dry-run（默认）
#   bash scripts/01_configure_fci_interface_nmcli.sh --apply   # 真正执行（需要 sudo 权限的 nmcli）

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/fr3_fci_network.yaml"

APPLY=0
for arg in "$@"; do
  case "${arg}" in
    --apply) APPLY=1 ;;
    *) echo "[01] 未知参数: ${arg}（忽略）" ;;
  esac
done

# --- 简易 YAML 取值（仅限本文件的扁平键）---
get_yaml() {
  grep -E "^[[:space:]]*$1[[:space:]]*:" "${CONFIG}" | head -1 \
    | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//"
}

[ -f "${CONFIG}" ] || { echo "[01][ERROR] 找不到配置 ${CONFIG}" >&2; exit 1; }

IFACE="$(get_yaml confirmed_fci_interface)"
LOCAL_IP="$(get_yaml local_fci_ip)"
PREFIX="$(get_yaml prefix)"
ROBOT_IP="$(get_yaml robot_ip)"

echo "=================================================="
echo "[01] 配置 FR3 FCI 网口（$([ "${APPLY}" -eq 1 ] && echo 真正执行 --apply || echo dry-run 只打印)）"
echo "=================================================="
echo "  confirmed_fci_interface = ${IFACE}"
echo "  local_fci_ip/prefix     = ${LOCAL_IP}/${PREFIX}"
echo "  robot_ip                = ${ROBOT_IP}"

# --- 拦截 1：未确认 ---
if [ -z "${IFACE}" ] || [ "${IFACE}" = "null" ]; then
  echo "" >&2
  echo "[01][STOP] confirmed_fci_interface 还是 null。" >&2
  echo "  请先运行 00_detect_network_interfaces.sh，确认 FCI 网口后填进 configs。" >&2
  exit 1
fi

# --- 动态找默认路由网口（不只信配置文件）---
DEFAULT_ROUTE_INTERFACE="$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'dev [^ ]+' | awk '{print $2}' | head -1)"
[ -z "${DEFAULT_ROUTE_INTERFACE}" ] && DEFAULT_ROUTE_INTERFACE="$(ip route show default 2>/dev/null | grep -oE 'dev [^ ]+' | awk '{print $2}' | head -1)"
echo "  默认路由网口(实时)      = ${DEFAULT_ROUTE_INTERFACE:-未知}"

# --- 拦截 2：等于默认路由网口 ---
if [ -n "${DEFAULT_ROUTE_INTERFACE}" ] && [ "${IFACE}" = "${DEFAULT_ROUTE_INTERFACE}" ]; then
  echo "" >&2
  echo "[01][REFUSE] confirmed_fci_interface (${IFACE}) 就是当前默认路由/上网网口！" >&2
  echo "  拒绝执行，避免断网。请重新确认 FCI 网口。" >&2
  exit 1
fi

# --- 拦截 3：在 forbidden 列表里 ---
if grep -E "^[[:space:]]*-[[:space:]]*${IFACE}([[:space:]]|#|$)" "${CONFIG}" >/dev/null 2>&1; then
  echo "" >&2
  echo "[01][REFUSE] ${IFACE} 在 forbidden_interfaces 列表里，拒绝执行。" >&2
  exit 1
fi

# --- 拦截 4：网口不存在 ---
if [ ! -e "/sys/class/net/${IFACE}" ]; then
  echo "" >&2
  echo "[01][ERROR] 网口 ${IFACE} 不存在。请检查名字是否填错。" >&2
  exit 1
fi

# --- 拦截 5：该网口当前是否承载了默认路由（双保险）---
if ip route show default 2>/dev/null | grep -qE "dev ${IFACE}([[:space:]]|$)"; then
  echo "" >&2
  echo "[01][REFUSE] ${IFACE} 当前承载着一条默认路由，拒绝修改。" >&2
  exit 1
fi

CON_NAME="fr3-fci-${IFACE}"

# 已存在同名连接 -> modify；否则 add
if nmcli -g NAME connection show 2>/dev/null | grep -qx "${CON_NAME}"; then
  ACTION="modify"
  CMD_MAIN=(nmcli connection modify "${CON_NAME}"
    ipv4.method manual
    ipv4.addresses "${LOCAL_IP}/${PREFIX}"
    ipv4.gateway ""
    ipv4.dns ""
    ipv4.never-default yes
    ipv6.method disabled)
else
  ACTION="add"
  CMD_MAIN=(nmcli connection add type ethernet ifname "${IFACE}" con-name "${CON_NAME}"
    ipv4.method manual
    ipv4.addresses "${LOCAL_IP}/${PREFIX}"
    ipv4.never-default yes
    ipv6.method disabled)
fi
CMD_UP=(nmcli connection up "${CON_NAME}")

echo ""
echo "  将要执行（${ACTION}）："
echo "    ${CMD_MAIN[*]}"
echo "    ${CMD_UP[*]}"
echo "  （不设 gateway、不设 DNS、never-default=yes -> 不影响上网/默认路由）"

if [ "${APPLY}" -ne 1 ]; then
  echo ""
  echo "[01] dry-run 结束。确认无误后执行： bash scripts/01_configure_fci_interface_nmcli.sh --apply"
  exit 0
fi

# ---- 真正执行 ----
echo ""
echo "[01] --apply：开始执行 nmcli ..."
"${CMD_MAIN[@]}"
"${CMD_UP[@]}"

echo ""
echo "---- 执行后 ip -br addr show ${IFACE} ----"
ip -br addr show "${IFACE}"
echo ""
echo "---- 执行后 ip route ----"
ip route
echo ""
NEW_DEFAULT="$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'dev [^ ]+' | awk '{print $2}' | head -1)"
echo "  默认路由网口（执行后）= ${NEW_DEFAULT:-未知}"
if [ -n "${DEFAULT_ROUTE_INTERFACE}" ] && [ "${NEW_DEFAULT}" != "${DEFAULT_ROUTE_INTERFACE}" ]; then
  echo "  [01][WARN] 默认路由网口发生了变化（${DEFAULT_ROUTE_INTERFACE} -> ${NEW_DEFAULT}）！请立即检查上网是否正常。" >&2
else
  echo "  [01] OK：默认路由网口未变，上网不受影响。"
fi
echo ""
echo "[01] 完成。下一步： bash scripts/02_check_fci_ping.sh"
