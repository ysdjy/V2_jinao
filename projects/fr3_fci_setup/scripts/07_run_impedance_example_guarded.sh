#!/usr/bin/env bash
#
# 07_run_impedance_example_guarded.sh — 受保护地运行【官方】阻抗控制示例。
#
# 这个示例会让真实 FR3 进入力矩控制并主动运动！必须极其小心。
#   * 默认只打印路径和命令，绝不自动运行。
#   * 必须加 --apply，并在交互确认安全清单后，才会运行。
#   * 只运行 libfranka 官方示例（joint/cartesian impedance），不运行任何自写控制器。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/fr3_fci_network.yaml"

APPLY=0
for arg in "$@"; do
  case "${arg}" in
    --apply) APPLY=1 ;;
    *) echo "[07] 未知参数: ${arg}（忽略）" ;;
  esac
done

get_yaml() {
  grep -E "^[[:space:]]*$1[[:space:]]*:" "${CONFIG}" | head -1 \
    | sed -E "s/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^\"//; s/\"$//; s/[[:space:]]*$//"
}
ROBOT_IP="$(get_yaml robot_ip)"

find_one() { find "${HOME}" /opt /usr/local 2>/dev/null -name "$1" -type f 2>/dev/null | head -n 1; }
JOINT_BIN="$(find_one joint_impedance_control)"
CART_BIN="$(find_one cartesian_impedance_control)"

echo "=================================================="
echo "[07] 阻抗控制示例（受保护，$([ "${APPLY}" -eq 1 ] && echo --apply || echo 仅打印)）"
echo "=================================================="
echo ""
echo "找到的官方示例："
[ -n "${JOINT_BIN}" ] && echo "  joint_impedance_control     : ${JOINT_BIN}" || echo "  joint_impedance_control     : (未找到)"
[ -n "${CART_BIN}" ]  && echo "  cartesian_impedance_control : ${CART_BIN}"  || echo "  cartesian_impedance_control : (未找到)"

# 选用：优先 cartesian_impedance_control（官方常用、扰动柔顺），否则 joint
RUN_BIN="${CART_BIN:-${JOINT_BIN}}"
if [ -z "${RUN_BIN}" ]; then
  echo ""
  echo "[07][ERROR] 没找到阻抗示例，先编译 libfranka examples（见 05）。"
  exit 1
fi
echo ""
echo "  将要运行： ${RUN_BIN} ${ROBOT_IP}"

echo ""
echo "================ 安全检查清单（运行前必须全部满足）================"
echo "  [ ] 机器人周围无障碍物、无人员在运动范围内"
echo "  [ ] 急停按钮在手边且可用"
echo "  [ ] 操作者已离开机器人运动范围"
echo "  [ ] Franka Desk 无错误、已松抱闸"
echo "  [ ] FCI 已激活"
echo "  [ ] 已先通过 06（echo_robot_state / communication_test）"
echo "  [ ] 只运行官方小幅示例，随时准备急停"
echo "=================================================================="

if [ "${APPLY}" -ne 1 ]; then
  echo ""
  echo "[07] 仅打印模式。确认安全后执行： bash scripts/07_run_impedance_example_guarded.sh --apply"
  exit 0
fi

echo ""
echo "  你正要让【真实 FR3】进入阻抗控制并运动。"
read -r -p "  以上安全清单是否全部满足？全部满足请输入大写 RUN： " ANS
if [ "${ANS}" != "RUN" ]; then
  echo "[07] 未确认（没输入 RUN），已取消。"
  exit 0
fi

echo ""
echo "[07] 运行官方阻抗示例（Ctrl+C 或急停停止）..."
echo "     ${RUN_BIN} ${ROBOT_IP}"
"${RUN_BIN}" "${ROBOT_IP}"
RC=$?
echo ""
echo "[07] 结束（返回码 ${RC}）。"
exit "${RC}"
