#!/usr/bin/env bash
#
# 04_check_libfranka.sh — 检查系统里有没有 libfranka 和 franka 示例。只读，不安装任何东西。

set -uo pipefail

echo "=================================================="
echo "[04] 检查 libfranka 安装情况（只读，不安装）"
echo "=================================================="

echo ""
echo "---- dpkg 里的 franka 包 ----"
dpkg -l 2>/dev/null | grep -i franka || echo "  (dpkg 里没找到 franka 包)"

echo ""
echo "---- ldconfig 里的 libfranka 库 ----"
ldconfig -p 2>/dev/null | grep -i franka || echo "  (ldconfig 里没找到 libfranka)"

echo ""
echo "---- pkg-config 版本 ----"
pkg-config --modversion franka 2>/dev/null || echo "  (pkg-config 查不到 franka 版本)"

echo ""
echo "---- 在 HOME 下查找常见 franka 示例可执行文件 ----"
find "${HOME}" \
  \( -name "echo_robot_state" \
     -o -name "communication_test" \
     -o -name "cartesian_impedance_control" \
     -o -name "joint_impedance_control" \
     -o -name "print_joint_poses" \
     -o -name "generate_cartesian_pose_motion" \) \
  -type f 2>/dev/null | head -n 50 || true

echo ""
echo "=================================================="
echo "建议（不要自动安装，安装前先问我）："
echo "  * libfranka 版本必须与 FR3 的系统(System Version)匹配，否则会 version mismatch。"
echo "  * 最好参照另一台已经调通 FCI 的电脑上的 libfranka 版本来装。"
echo "  * 若需要安装/编译 libfranka 及其 examples，请先确认版本，再手动操作。"
echo "  下一步： bash scripts/05_find_libfranka_examples.sh"
echo "=================================================="
