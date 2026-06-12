#!/usr/bin/env bash
#
# 05_find_libfranka_examples.sh — 查找 libfranka 官方示例可执行文件路径。只读。

set -uo pipefail

echo "=================================================="
echo "[05] 查找 libfranka 官方示例"
echo "=================================================="

EXAMPLES=(
  echo_robot_state
  communication_test
  print_joint_poses
  joint_impedance_control
  cartesian_impedance_control
  generate_cartesian_pose_motion
)

# 搜索范围：HOME + 常见安装/编译目录
SEARCH_DIRS=("${HOME}")
for d in /opt /usr/local/bin /usr/local/libexec /usr/lib; do
  [ -d "${d}" ] && SEARCH_DIRS+=("${d}")
done

FOUND_ANY=0
for name in "${EXAMPLES[@]}"; do
  HITS="$(find "${SEARCH_DIRS[@]}" -name "${name}" -type f 2>/dev/null | head -n 10)"
  if [ -n "${HITS}" ]; then
    FOUND_ANY=1
    echo ""
    echo "  [${name}]"
    while IFS= read -r line; do
      MARK=""
      [ -x "${line}" ] && MARK=" (可执行)"
      echo "    ${line}${MARK}"
    done <<< "${HITS}"
  fi
done

if [ "${FOUND_ANY}" -eq 0 ]; then
  echo ""
  echo "  没找到任何示例可执行文件。"
  echo "  说明可能还没编译 libfranka examples。典型做法（确认版本后手动执行，不要我代跑）："
  echo "    git clone --recursive https://github.com/frankarobotics/libfranka.git"
  echo "    # 切到与 FR3 系统版本匹配的 tag"
  echo "    cd libfranka && mkdir build && cd build"
  echo "    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON .. && cmake --build ."
  echo "    # 示例会生成在 build/examples/ 下"
else
  echo ""
  echo "  下一步： bash scripts/06_run_fci_read_state_test.sh"
fi
echo "=================================================="
