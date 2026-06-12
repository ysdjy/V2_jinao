#!/usr/bin/env bash
#
# setup_gello_env.sh — 一键准备 GELLO-Franka 第一阶段环境
#
# 做的事情：
#   1. 如果 third_party/gello_software 不存在，则 clone（已存在则只检查状态，不重复 clone）
#   2. 初始化 DynamixelSDK 子模块（读取关节必需）
#   3. 创建独立虚拟环境 .venv-gello（优先 uv，没有则用 python3 -m venv）
#   4. 安装第一阶段读取关节所需的最小依赖：
#        - dynamixel_sdk（editable，来自子模块）
#        - gello 本体（editable）
#        - numpy / pyyaml
#   5. 可选 --full：额外安装 gello_software 的完整 requirements.txt
#        （包含 pyrealsense2 / pin / ur-rtde 等重依赖，第一阶段用不到，且容易编译失败）
#
# 用法：
#   bash scripts/setup_gello_env.sh          # 最小安装（推荐，最快）
#   bash scripts/setup_gello_env.sh --full   # 额外安装完整 requirements.txt
#
# 本脚本不会污染 IsaacLab 的 Python 环境，所有东西都装进 .venv-gello。

set -euo pipefail

# ---- 定位项目根目录（脚本所在目录的上一级）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GELLO_DIR="${PROJECT_ROOT}/third_party/gello_software"
DXLSDK_DIR="${GELLO_DIR}/third_party/DynamixelSDK/python"
VENV_DIR="${PROJECT_ROOT}/.venv-gello"
INSTALL_FULL=0

for arg in "$@"; do
  case "${arg}" in
    --full) INSTALL_FULL=1 ;;
    *) echo "[setup] 未知参数: ${arg}（忽略）" ;;
  esac
done

echo "=================================================="
echo "[setup] 项目根目录: ${PROJECT_ROOT}"
echo "=================================================="

# ---- 1. clone gello_software ----
if [ -d "${GELLO_DIR}/.git" ]; then
  echo "[setup] gello_software 已存在，跳过 clone。"
  git -C "${GELLO_DIR}" status -s -b | head -1 || true
else
  echo "[setup] clone gello_software ..."
  git clone https://github.com/wuphilipp/gello_software.git "${GELLO_DIR}"
fi

# 处理 root 拥有目录导致的 git "dubious ownership"
git config --global --add safe.directory "${GELLO_DIR}" 2>/dev/null || true

# ---- 2. 初始化 DynamixelSDK 子模块 ----
if [ -f "${DXLSDK_DIR}/setup.py" ]; then
  echo "[setup] DynamixelSDK 已就绪。"
else
  echo "[setup] 初始化 DynamixelSDK 子模块 ..."
  git -C "${GELLO_DIR}" submodule update --init --depth 1 third_party/DynamixelSDK
fi
if [ ! -f "${DXLSDK_DIR}/setup.py" ]; then
  echo "[setup][ERROR] 找不到 ${DXLSDK_DIR}/setup.py，DynamixelSDK 初始化失败。" >&2
  exit 1
fi

# ---- 3. 创建虚拟环境 ----
if [ -d "${VENV_DIR}" ]; then
  echo "[setup] 虚拟环境已存在: ${VENV_DIR}"
else
  if command -v uv >/dev/null 2>&1; then
    echo "[setup] 用 uv 创建虚拟环境 ..."
    uv venv --python 3.11 "${VENV_DIR}" || uv venv "${VENV_DIR}"
  else
    echo "[setup] 没有 uv，回退到 python3 -m venv ..."
    python3 -m venv "${VENV_DIR}"
  fi
fi

PY="${VENV_DIR}/bin/python"
PIP_INSTALL=("${PY}" -m pip install)
if command -v uv >/dev/null 2>&1; then
  # uv 装的 venv 默认没有 pip，用 uv pip 更稳
  PIP_INSTALL=(uv pip install --python "${PY}")
else
  echo "[setup] 升级 pip ..."
  "${PY}" -m pip install --upgrade pip
fi

# ---- 4. 安装最小依赖 ----
echo "[setup] 安装核心依赖（numpy / pyyaml）..."
"${PIP_INSTALL[@]}" numpy pyyaml

echo "[setup] 安装 DynamixelSDK (editable) ..."
"${PIP_INSTALL[@]}" -e "${DXLSDK_DIR}"

echo "[setup] 安装 gello (editable) ..."
"${PIP_INSTALL[@]}" -e "${GELLO_DIR}"

# ---- 5. 可选完整依赖 ----
if [ "${INSTALL_FULL}" -eq 1 ]; then
  echo "[setup] --full：安装 gello_software 完整 requirements.txt（可能较慢/部分失败）..."
  "${PIP_INSTALL[@]}" -r "${GELLO_DIR}/requirements.txt" || \
    echo "[setup][WARN] 完整 requirements.txt 部分安装失败，第一阶段读取关节不受影响。"
fi

echo ""
echo "=================================================="
echo "[setup] 完成。验证导入："
"${PY}" -c "import numpy, yaml; from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig; from dynamixel_sdk.port_handler import PortHandler; print('  OK: gello + dynamixel_sdk 可导入')"
echo "=================================================="
echo ""
echo "下一步："
echo "  1) bash scripts/detect_gello_port.sh        # 检测串口"
echo "  2) 把端口填入 configs/gello_franka.yaml 的 port:"
echo "  3) bash scripts/calibrate_gello_offset.sh   # 校准 offset"
echo "  4) source .venv-gello/bin/activate"
echo "  5) python scripts/diagnose_gello.py --config configs/gello_franka.yaml"
echo "  6) python scripts/read_gello_joints.py --config configs/gello_franka.yaml --hz 30"
