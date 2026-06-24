#!/bin/bash
# 每隔 80～200 秒（随机）运行一次 run_multimodal_processor_tool_async_tests_skill_id.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/run_multimodal_processor_tool_async_tests_skill_ood.sh"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始执行: ${TEST_SCRIPT}"
  bash "$TEST_SCRIPT" || true
  # 80～200 秒之间随机取一个整数
  SEC=$((650 + RANDOM % 40))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 ${SEC} 秒后执行测试..."
  sleep "$SEC"
done
