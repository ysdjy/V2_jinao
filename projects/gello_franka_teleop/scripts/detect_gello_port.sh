#!/usr/bin/env bash
#
# detect_gello_port.sh — 检测 GELLO 串口设备
#
# 做的事情：
#   - 列出 /dev/serial/by-id/  /dev/ttyUSB*  /dev/ttyACM*
#   - 检查当前用户是否在 dialout 组（串口权限）
#   - 推荐一个最稳定的端口名给你填进配置
#
# 本脚本【不会】自动执行 sudo usermod，只会提示命令。

set -uo pipefail

echo "=================================================="
echo "[detect] 检测 GELLO 串口设备"
echo "=================================================="

echo ""
echo "---- /dev/serial/by-id/ （推荐使用这里的稳定名字）----"
if [ -d /dev/serial/by-id ]; then
  ls -l /dev/serial/by-id/ 2>/dev/null || echo "  (空)"
else
  echo "  目录不存在（设备可能没插，或不是 USB 串口）"
fi

echo ""
echo "---- /dev/ttyUSB* ----"
ls -l /dev/ttyUSB* 2>/dev/null || echo "  (无)"

echo ""
echo "---- /dev/ttyACM* ----"
ls -l /dev/ttyACM* 2>/dev/null || echo "  (无)"

# ---- 权限检查 ----
echo ""
echo "---- dialout 权限检查 ----"
if id -nG "$USER" | tr ' ' '\n' | grep -qx dialout; then
  echo "  OK: 用户 '$USER' 已在 dialout 组。"
else
  echo "  [警告] 用户 '$USER' 不在 dialout 组，可能会 Permission denied。"
  echo "  请手动执行（需要你授权 sudo）："
  echo "      sudo usermod -aG dialout $USER"
  echo "  然后重新登录，或运行："
  echo "      newgrp dialout"
fi

# ---- 推荐端口 ----
echo ""
echo "---- 推荐端口 ----"
RECOMMENDED=""
if [ -d /dev/serial/by-id ]; then
  # 取第一个 by-id 设备的完整路径
  for f in /dev/serial/by-id/*; do
    [ -e "$f" ] || continue
    RECOMMENDED="$f"
    break
  done
fi
if [ -z "${RECOMMENDED}" ]; then
  for f in /dev/ttyUSB* /dev/ttyACM*; do
    [ -e "$f" ] || continue
    RECOMMENDED="$f"
    break
  done
fi

if [ -n "${RECOMMENDED}" ]; then
  echo "  建议把下面这一行填进 configs/gello_franka.yaml ："
  echo ""
  echo "      port: ${RECOMMENDED}"
  echo ""
else
  echo "  没检测到任何串口设备。请检查："
  echo "    1) GELLO 的 USB 是否插好"
  echo "    2) 换一个 USB 口 / 换一根数据线"
  echo "    3) dmesg | tail 看看插入时有没有识别到设备"
fi

echo "=================================================="
