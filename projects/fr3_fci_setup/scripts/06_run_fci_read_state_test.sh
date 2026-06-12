#!/usr/bin/env bash
#
# 06_run_fci_read_state_test.sh — 跑只读的 FCI 链路测试。
#
# 优先 echo_robot_state（纯读取机器人状态，不进控制环、不会动）。
# 若只有 communication_test（会进入零力矩控制环，机器人可能在重力补偿下轻微移动），
# 则需要交互确认后才运行。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/fr3_fci_network.yaml"

get_yaml() {
  grep -E "^[[:space:]]*$1[[:space:]]*:" "${CONFIG}" | head -1 \
    | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//"
}
ROBOT_IP="$(get_yaml robot_ip)"

find_one() { find "${HOME}" /opt /usr/local 2>/dev/null -name "$1" -type f 2>/dev/null | head -n 1; }

ECHO_BIN="$(find_one echo_robot_state)"
COMM_BIN="$(find_one communication_test)"

echo "=================================================="
echo "[06] FCI 读状态测试  (robot_ip=${ROBOT_IP})"
echo "=================================================="
echo ""
echo "运行前请确认："
echo "  * Desk 里已 Activate FCI；"
echo "  * 没有另一台电脑正占用 FCI（一次只能一个 FCI 客户端）；"
echo "  * 机器人无报错。"
echo ""

if [ -n "${ECHO_BIN}" ]; then
  echo "找到 echo_robot_state（只读，不会动机器人）："
  echo "  ${ECHO_BIN} ${ROBOT_IP}"
  echo ""
  echo "[06] 运行中（Ctrl+C 停止）..."
  "${ECHO_BIN}" "${ROBOT_IP}"
  RC=$?
elif [ -n "${COMM_BIN}" ]; then
  echo "未找到 echo_robot_state，只找到 communication_test："
  echo "  ${COMM_BIN} ${ROBOT_IP}"
  echo ""
  echo "  ⚠ communication_test 会进入【零力矩控制环】，机器人可能在重力补偿下轻微移动。"
  echo "  确认机器人周围安全、急停在手、Desk 无错误、FCI 已激活。"
  read -r -p "  确定运行 communication_test 吗？输入 yes 继续： " ANS
  if [ "${ANS}" = "yes" ]; then
    echo "[06] 运行中（Ctrl+C 停止）..."
    "${COMM_BIN}" "${ROBOT_IP}"
    RC=$?
  else
    echo "[06] 已取消。"
    exit 0
  fi
else
  echo "[06][ERROR] 没找到 echo_robot_state 或 communication_test。"
  echo "  先跑 04/05 确认 libfranka examples 是否已编译。"
  exit 1
fi

echo ""
if [ "${RC:-1}" -eq 0 ]; then
  echo "[06] PASS：FCI 链路 + libfranka 正常。可继续 07（阻抗示例，谨慎）。"
else
  echo "[06] 失败（返回码 ${RC:-?}）。可能原因："
  echo "  * FCI 没在 Desk 里激活；"
  echo "  * 机器人 Desk 仍处于错误/未松抱闸状态；"
  echo "  * IP 不通（先跑 02）；"
  echo "  * libfranka 版本与 FR3 系统版本不匹配；"
  echo "  * 另一台电脑仍占用 FCI。"
fi
exit "${RC:-1}"
