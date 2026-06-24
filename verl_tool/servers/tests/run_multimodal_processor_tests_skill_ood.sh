#!/bin/bash

# Multimodal Processor Tool 测试脚本（OOD pool + skills + run_skill）
# 每个测试函数的输出会写入对应的日志文件

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_multimodal_processor_tool_skill_ood.py"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

declare -a TEST_FUNCTIONS=(
    "init:test_tool_initialization"
    "params:test_parameter_validation"
    "wolfram_params:test_wolfram_parameter_validation"
    "call:test_model_call"
    "wolfram_call:test_wolfram_call"
    "unichart_call:test_unichart_call"
    "deplot_call:test_deplot_call"
    "chartmoe_call:test_chartmoe_call"
    "step3_call:test_step3_call"
    "python_code_call:test_python_code_call"
    "python_code_params:test_python_code_parameter_validation"
    "skill_md_call:test_skill_md_call"
    "create_skill_parse_ok:test_create_skill_parse_with_arguments"
    "create_skill_parse_reject_legacy:test_create_skill_parse_reject_top_level_description"
    "create_skill_real:test_create_skill_real_call"
    "skill_md_call_null_args:test_skill_md_call_with_null_arguments"
    "run_skill_call:test_run_skill_call"
    "run_skill_call_raw_base64:test_run_skill_call_with_raw_base64_image"
    "run_skill_invalid_long_string_image_input:test_run_skill_invalid_long_string_image_input"
    "run_skill_missing_required_image_index:test_run_skill_missing_required_image_index"
    "run_skill_non_image_without_image_index:test_run_skill_non_image_skill_without_image_index"
    "run_skill_entrypoint_level_image_requirement:test_run_skill_entrypoint_level_image_requirement"
    "run_skill_missing_stage1_spec_degrades:test_run_skill_missing_stage1_spec_degrades"
    "env_image_accumulation:test_env_image_accumulation"
)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Multimodal Processor Tool OOD 测试"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

FAIL_COUNT=0

for test_info in "${TEST_FUNCTIONS[@]}"; do
    cmd="${test_info%%:*}"
    func_name="${test_info##*:}"
    log_file="${LOG_DIR}/${func_name}_${TIMESTAMP}.log"

    echo -e "${YELLOW}[${func_name}]${NC} -> ${log_file}"

    if python "${TEST_SCRIPT}" "${cmd}" > "${log_file}" 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "=========================================="
echo "结束时间: $(date)"
echo "日志目录: ${LOG_DIR}"
echo "失败数量: ${FAIL_COUNT}"
echo "=========================================="

if [ "${FAIL_COUNT}" -gt 0 ]; then
    exit 1
fi

exit 0
