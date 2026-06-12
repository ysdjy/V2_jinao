#!/usr/bin/env bash
#
# calibrate_gello_offset.sh — 校准 GELLO 关节 offset
#
# 做的事情：
#   - 从 configs/gello_franka.yaml 读取 port / start_joints / joint_signs / gripper.enabled
#   - 调用 gello_software 自带的 scripts/gello_get_offset.py
#   - 把输出保存到 logs/offset_calibration_<时间戳>.txt
#
# 重要：
#   - 运行前请把 GELLO 摆成 configs 里 start_joints 描述的标准姿态。
#   - 本脚本【不会】自动覆盖 configs/gello_franka.yaml，
#     请你看终端输出，手动把 "best offsets function of pi" 填进 joint_offsets。
#
# 用法：
#   bash scripts/calibrate_gello_offset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${PROJECT_ROOT}/configs/gello_franka.yaml"
VENV_PY="${PROJECT_ROOT}/.venv-gello/bin/python"
OFFSET_SCRIPT="${PROJECT_ROOT}/third_party/gello_software/scripts/gello_get_offset.py"

if [ ! -x "${VENV_PY}" ]; then
  echo "[calib][ERROR] 找不到虚拟环境 Python: ${VENV_PY}" >&2
  echo "  请先运行: bash scripts/setup_gello_env.sh" >&2
  exit 1
fi
if [ ! -f "${OFFSET_SCRIPT}" ]; then
  echo "[calib][ERROR] 找不到 ${OFFSET_SCRIPT}" >&2
  echo "  请先运行: bash scripts/setup_gello_env.sh" >&2
  exit 1
fi
if [ ! -f "${CONFIG}" ]; then
  echo "[calib][ERROR] 找不到配置文件 ${CONFIG}" >&2
  exit 1
fi

# ---- 用 venv python 解析 yaml，导出为 shell 变量 ----
# 输出三行：PORT=... / START=... / SIGNS=... / GRIPPER=...
eval "$(
  "${VENV_PY}" - "${CONFIG}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
port = cfg.get("port")
start = cfg.get("start_joints") or []
signs = cfg.get("joint_signs") or []
gripper = bool((cfg.get("gripper") or {}).get("enabled", False))
print(f'PORT="{"" if port is None else port}"')
print(f'START="{" ".join(str(x) for x in start)}"')
print(f'SIGNS="{" ".join(str(int(x)) for x in signs)}"')
print(f'GRIPPER="{1 if gripper else 0}"')
PYEOF
)"

if [ -z "${PORT}" ] || [ "${PORT}" = "None" ]; then
  echo "[calib][ERROR] 配置里的 port 为空。" >&2
  echo "  请先运行: bash scripts/detect_gello_port.sh" >&2
  echo "  然后把端口填进 ${CONFIG} 的 port:" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"
# 时间戳（不依赖 GNU date 扩展）
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${PROJECT_ROOT}/logs/offset_calibration_${TS}.txt"

GRIPPER_FLAG="--gripper"
[ "${GRIPPER}" = "1" ] || GRIPPER_FLAG="--no-gripper"

echo "=================================================="
echo "[calib] port        : ${PORT}"
echo "[calib] start_joints : ${START}"
echo "[calib] joint_signs  : ${SIGNS}"
echo "[calib] gripper      : ${GRIPPER_FLAG}"
echo "[calib] 输出保存到    : ${OUT}"
echo "=================================================="
echo "[calib] 请确认 GELLO 已摆到 start_joints 姿态，3 秒后开始 ..."
sleep 3

# shellcheck disable=SC2086
"${VENV_PY}" "${OFFSET_SCRIPT}" \
  --port "${PORT}" \
  --start-joints ${START} \
  --joint-signs ${SIGNS} \
  ${GRIPPER_FLAG} 2>&1 | tee "${OUT}"

echo ""
echo "=================================================="
echo "[calib] 校准结果已保存: ${OUT}"
echo "[calib] 请把上面 'best offsets function of pi: [...]' 里的数值"
echo "        手动换算后填入 ${CONFIG} 的 joint_offsets。"
echo "        （脚本不会自动覆盖配置，避免写错。）"
echo "=================================================="
