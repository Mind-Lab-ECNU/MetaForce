#!/bin/bash

# Multimodal Processor Tool 测试脚本（skills + run_skill）
# 每个测试函数的输出会写入对应的日志文件

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_multimodal_processor_tool_skill.py"
LOG_DIR="${SCRIPT_DIR}/logs"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 测试函数列表
declare -a TEST_FUNCTIONS=(
    "init:test_tool_initialization"
    "params:test_parameter_validation"
    "wolfram_params:test_wolfram_parameter_validation"
    "call:test_model_call"
    "vlm_235:test_vlm_235_call"
    "wolfram_call:test_wolfram_call"
    "sam3_text:test_sam3_text_call"
    "mineru_content:test_mineru_content_call"
    "paddle_call:test_paddle_call"
    "easyocr_call:test_easyocr_call"
    "groundingdino_call:test_groundingdino_call"
    "unichart_call:test_unichart_call"
    "deplot_call:test_deplot_call"
    "chartmoe_call:test_chartmoe_call"
    "step3_call:test_step3_call"
    "image_edit_call:test_image_edit_call"
    "skill_md_call:test_skill_md_call"
    "create_skill_parse_ok:test_create_skill_parse_with_arguments"
    "create_skill_parse_reject_legacy:test_create_skill_parse_reject_top_level_description"
    "create_skill_real:test_create_skill_real_call"
    "skill_md_call_null_args:test_skill_md_call_with_null_arguments"
    "run_skill_call:test_run_skill_call"
    "run_skill_missing_required_image_index:test_run_skill_missing_required_image_index"
    "run_skill_non_image_without_image_index:test_run_skill_non_image_skill_without_image_index"
    "run_skill_entrypoint_level_image_requirement:test_run_skill_entrypoint_level_image_requirement"
    "run_skill_missing_stage1_spec_degrades:test_run_skill_missing_stage1_spec_degrades"
    "python_code_call:test_python_code_call"
    "python_code_params:test_python_code_parameter_validation"
    "opencv_ops:test_opencv_operations"
    "opencv_params:test_opencv_parameter_validation"
    "opencv_base64:test_opencv_base64_image"
    "env_image_accumulation:test_env_image_accumulation"
)

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Multimodal Processor Tool 测试"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

FAIL_COUNT=0

# 运行所有测试
for test_info in "${TEST_FUNCTIONS[@]}"; do
    cmd="${test_info%%:*}"
    func_name="${test_info##*:}"
    log_file="${LOG_DIR}/${func_name}_${TIMESTAMP}.log"

    echo -e "${YELLOW}[${func_name}]${NC} -> ${log_file}"

    # 运行测试并保存输出到日志文件
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
