#!/bin/bash

# Multimodal Processor Tool Async 测试脚本（skills + run_skill）
# 每个测试函数的输出会写入对应的日志文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_async"
SERVER_LOG="/tmp/tool_multimodal_processor_async.log"
SKILL_STORE_DIR="$(mktemp -d /tmp/verl_skills_async_XXXXXX)"
export VERL_SKILL_STORE_DIR="${SKILL_STORE_DIR}"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SERVER_PY="python -m verl_tool.servers.tool_server"
PORT="$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
port = s.getsockname()[1]
s.close()
print(port)
PY
)"

cleanup_port() {
    if [ -z "${PORT}" ]; then
        return
    fi
    echo "Cleaning up port ${PORT}..."
    fuser -k "${PORT}"/tcp 2>/dev/null || true
    netstat -tulpn 2>/dev/null | grep ":${PORT} " | awk '{print $7}' | cut -d'/' -f1 | xargs -r kill -9 2>/dev/null || true
    # 不再全局 pkill，避免误杀训练等其他环境的 tool_server 进程
    sleep 2
}

wait_for_server() {
    local log_file=$1
    local max_wait=30
    local waited=0

    echo "Waiting for server to start (max ${max_wait}s) on port ${PORT}..."

    while [ $waited -lt $max_wait ]; do
        if netstat -tuln 2>/dev/null | grep -q ":${PORT} "; then
            echo "✓ Server is listening on port ${PORT}"
            sleep 2
            return 0
        fi

        if grep -q "Uvicorn running on" "$log_file" 2>/dev/null; then
            echo "✓ Server startup detected in logs"
            sleep 2
            return 0
        fi

        if timeout 1 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
            echo "✓ Server accepting connections on port ${PORT}"
            sleep 2
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
        if [ $((waited % 5)) -eq 0 ]; then
            echo "  Still waiting... (${waited}/${max_wait}s)"
        fi
    done

    echo "ERROR: Server failed to start after ${max_wait}s"
    echo "Last 50 lines of log:"
    tail -n 50 "$log_file"
    return 1
}

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "async-no-ray: multimodal_processor_tool_adapt_skill"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

cleanup_port

${SERVER_PY} --tool_type multimodal_processor_tool_adapt_skill --port "${PORT}" --workers_per_tool 8 > "${SERVER_LOG}" 2>&1 &
PID_ASYNC=$!

if ! wait_for_server "${SERVER_LOG}"; then
    kill ${PID_ASYNC} 2>/dev/null || true
    exit 1
fi

BASE_URL="http://localhost:${PORT}/get_observation"

# 测试函数列表
declare -a TEST_FUNCTIONS=(
    "vlm:test_async_vlm"
    "vlm_235:test_async_vlm_235"
    "params:test_async_parameter_validation"
    "wolfram_call:test_async_wolfram_call"
    "mineru_content:test_async_mineru_content_call"
    "paddle_call:test_async_paddle_call"
    "easyocr_call:test_async_easyocr_call"
    "groundingdino_call:test_async_groundingdino_call"
    "unichart_call:test_async_unichart_call"
    "deplot_call:test_async_deplot_call"
    "chartmoe_call:test_async_chartmoe_call"
    "step3_call:test_async_step3_call"
    "sam3_text:test_async_sam3_text_call"
    "image_edit:test_async_image_edit_call"
    "skill_md_call:test_async_skill_md_call"
    "create_skill_real:test_async_create_skill_real_call"
    "run_skill_call:test_async_run_skill_call"
    "run_skill_missing_required_image_index:test_async_run_skill_missing_required_image_index"
    "run_skill_non_image_without_image_index:test_async_run_skill_non_image_without_image_index"
    "run_skill_entrypoint_level_image_requirement:test_async_run_skill_entrypoint_level_image_requirement"
    "run_skill_missing_stage1_spec_degrades:test_async_run_skill_missing_stage1_spec_degrades"
    "python_code_call:test_async_python_code_call"
    "python_code_params:test_async_python_code_params"
    "opencv_ops:test_async_opencv_ops"
    "opencv_params:test_async_opencv_params"
    "opencv_base64:test_async_opencv_base64"
    "env_image_accumulation:test_async_env_image_accumulation"
    "bad_function:test_async_bad_function"
)

FAIL_COUNT=0

# 运行所有测试
for test_info in "${TEST_FUNCTIONS[@]}"; do
    cmd="${test_info%%:*}"
    func_name="${test_info##*:}"
    log_file="${LOG_DIR}/${func_name}_${TIMESTAMP}.log"

    echo -e "${YELLOW}[${func_name}]${NC} -> ${log_file}"

    # 运行测试并保存输出到日志文件
    if python -m verl_tool.servers.tests.test_multimodal_processor_tool_async_skill "${cmd}" --url "${BASE_URL}" > "${log_file}" 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

kill ${PID_ASYNC} 2>/dev/null || true
cleanup_port

echo "=========================================="
echo "结束时间: $(date)"
echo "日志目录: ${LOG_DIR}"
echo "失败数量: ${FAIL_COUNT}"
echo "=========================================="

if [ "${FAIL_COUNT}" -gt 0 ]; then
    exit 1
fi

exit 0
