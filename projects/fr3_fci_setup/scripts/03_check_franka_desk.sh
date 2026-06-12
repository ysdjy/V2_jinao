#!/usr/bin/env bash
#
# 03_check_franka_desk.sh — 指引你打开 Franka Desk 并手动激活 FCI。
# 本脚本不会自动登录 Desk，只做一次只读的可达性探测 + 打印步骤。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/fr3_fci_network.yaml"

get_yaml() {
  grep -E "^[[:space:]]*$1[[:space:]]*:" "${CONFIG}" | head -1 \
    | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//"
}
ROBOT_IP="$(get_yaml robot_ip)"
DESK_URL="https://${ROBOT_IP}"

echo "=================================================="
echo "[03] Franka Desk / 激活 FCI 指引"
echo "=================================================="
echo ""
echo "  在浏览器打开：  ${DESK_URL}"
echo ""

# 只读探测 443 端口是否可连（不登录、不改任何东西）。用 bash 的 /dev/tcp，比 http_code 可靠。
echo "---- 只读探测 Desk 443 端口可达性（不登录）----"
if timeout 4 bash -c "exec 3<>/dev/tcp/${ROBOT_IP}/443" 2>/dev/null; then
  echo "  OK：${ROBOT_IP}:443 可连。浏览器打开 ${DESK_URL}，证书告警选择继续即可。"
else
  echo "  无法连到 ${ROBOT_IP}:443。先确认 02 的 ping 能通、机器人已开机、Desk 已启动。"
fi

echo ""
echo "---- 在 Desk 里手动完成 ----"
echo "  1) 浏览器证书告警 -> 选择继续访问 ${DESK_URL}。"
echo "  2) 登录 Franka Desk。"
echo "  3) 确认机器人没有报错（右上角无红色错误）。"
echo "  4) 解锁关节 / 松开抱闸（Unlock joints / release brakes）。"
echo "  5) 右侧模式切到 Execution。"
echo "  6) 手动点击 Activate FCI（激活 FCI）。"
echo ""
echo "  ※ FCI 激活后，再去运行 libfranka 测试（04 -> 05 -> 06）。"
echo "=================================================="
