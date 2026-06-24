#!/usr/bin/env bash
# =============================================================================
# Batch eval launcher for explicit multimodal_orchestra checkpoint/base-model paths.
# - CHECKPOINT_PATHS and TARGET_STEPS are space-separated lists matched by index.
# - Checkpoint item:
#     CHECKPOINT_PATH points to an experiment directory and TARGET_STEP resolves to
#     ${CHECKPOINT_PATH}/global_step_${TARGET_STEP}/actor/huggingface
# - Base model item:
#     CHECKPOINT_PATH points directly to a HuggingFace model directory and TARGET_STEP
#     is one of: base | none | raw
# - EXP_NAMES is optional and overrides the auto-generated experiment name.
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# User-editable config
# -----------------------------------------------------------------------------
# Command examples:
# CHECKPOINT_PATHS="/path/exp3_xxx /path/exp5_yyy" \
# TARGET_STEPS="100 240" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt.sh
#
# CHECKPOINT_PATHS="/path/exp3_xxx /path/exp5_yyy" \
# TARGET_STEPS="100 240" \
# EXP_NAMES="exp3_gs100 exp5_gs240" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt.sh
#
# CHECKPOINT_PATHS="/path/exp3_xxx /path/exp5_yyy" \
# TARGET_STEPS="100 240" \
# EXP_NAMES="a b" \
# EVAL_SCRIPT="examples/train/multimodal_orchestra/single_node/eval_8gpu.sh" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt.sh
#
# Base model example:
# CHECKPOINT_PATHS="/inspire/.../Qwen3-VL-8B-Instruct /path/to/exp5_yyy" \
# TARGET_STEPS="base 240" \
# EXP_NAMES="base exp5_gs240" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt.sh
#
# Example:
# CHECKPOINT_PATHS_INPUT="/path/to/exp3_run /path/to/exp5_run"
# TARGET_STEPS_INPUT="100 240"
# EXP_NAMES_INPUT="exp3_gs100 exp5_gs240"
# CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/R1-Onevision-7B-RL"
# CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/verl_step_records/checkpoint/train/20260324_225047_exp15_multimodal_orchestra-fsdp2-agent-_inspire_hdd_project_ai4education_zhouaimin-p-zhouaimin_zc_verltools_verl_m_exp13-skill-reward-lr1e-6-step80-grpo-lr1e-6-skill-reward-c_and_f_s-0-cu_s-20-l_s-0-t_s-0-kl_l-0.002"
CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/VTool-Qwen2.5-7B"
# CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/PixelReasoner-RL-v1"
# CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/DeepEyesV2_7B_1031"
# CHECKPOINT_PATHS_INPUT=/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct
# CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/Qwen3-VL-32B-Instruct"
# CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/R1-VL-7B"
# TARGET_STEPS_INPUT="40"
TARGET_STEPS_INPUT="base"
# EXP_NAMES_INPUT="r1-onevision-7b-rl_ai2d_mathvista"
# EXP_NAMES_INPUT="exp15_step40_from_13_step80_ai2d_mathvista"
EXP_NAMES_INPUT="vtool-qwen2.5-7b_ai2d_mathvista"
# EXP_NAMES_INPUT="pixelreasoner=rl-v7-data"
# EXP_NAMES_INPUT="DeepEyesV2_7B_ai2d_mathvista"
# EXP_NAMES_INPUT="qwen3-32b-vl_ai2d_mathvista"
# EXP_NAMES_INPUT="r1-vl-7b"

CHECKPOINT_PATHS_STR="${CHECKPOINT_PATHS:-$CHECKPOINT_PATHS_INPUT}"
TARGET_STEPS_STR="${TARGET_STEPS:-$TARGET_STEPS_INPUT}"
EXP_NAMES_STR="${EXP_NAMES:-$EXP_NAMES_INPUT}"

EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/single_node/eval_8gpu.sh}"
TRAIN_DATA_PATH="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/train_merge_v6_tool_id/train_merge_v6_tool_id.parquet"
VAL_DATA_PATH='/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/test_merge_v3_ai2d_mathvista_tool_id/test_ood_merge_v3_ai2d_mathvista_tool_id.parquet'
EVAL_TOOL_VARIANT="id"
LATENCY_PENALTY_START_STEP="${LATENCY_PENALTY_START_STEP:-0}"
TOOL_PENALTY_START_STEP="${TOOL_PENALTY_START_STEP:-0}"
BATCH_LOG_DIR="${BATCH_LOG_DIR:-$(pwd)/logs/multimodal_orchestra_eval_batch}"

sanitize_label() {
    local raw_label="$1"
    local clean_label

    clean_label=$(printf '%s' "$raw_label" | tr '/: ' '___' | tr -cd '[:alnum:]_.-')
    if [ -z "$clean_label" ]; then
        clean_label="model"
    fi

    printf '%s' "$clean_label"
}

is_base_target_step() {
    local target_step="$1"
    case "$target_step" in
        base|none|raw)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_model_dir() {
    local checkpoint_path="$1"
    local target_step="$2"

    if is_base_target_step "$target_step"; then
        printf '%s' "$checkpoint_path"
    else
        printf '%s' "${checkpoint_path}/global_step_${target_step}/actor/huggingface"
    fi
}

setup_batch_logging() {
    local log_dir="$1"
    local log_path="$2"
    local tee_cmd="tee"

    mkdir -p "$log_dir"

    BATCH_LOG_PIPE="$(mktemp "${TMPDIR:-/tmp}/eval_batch_log_pipe.XXXXXX")"
    rm -f "$BATCH_LOG_PIPE"
    mkfifo "$BATCH_LOG_PIPE"

    if command -v stdbuf >/dev/null 2>&1; then
        tee_cmd="stdbuf -oL tee"
    fi

    # Use a FIFO instead of process substitution for better shell portability.
    eval "$tee_cmd -a \"\$log_path\" < \"\$BATCH_LOG_PIPE\"" &
    BATCH_LOG_TEE_PID=$!

    exec >"$BATCH_LOG_PIPE" 2>&1
    rm -f "$BATCH_LOG_PIPE"
}

cleanup_batch_logging() {
    local exit_code=$?

    if [ -n "${BATCH_LOG_TEE_PID:-}" ]; then
        exec >&- 2>&-
        wait "$BATCH_LOG_TEE_PID" 2>/dev/null || true
    fi

    exit "$exit_code"
}

batch_log_ts="$(date +%Y%m%d_%H%M%S)"
batch_log_variant="$(sanitize_label "${EVAL_TOOL_VARIANT:-unknown}")"
batch_log_path="${BATCH_LOG_DIR}/batch_eval_${batch_log_variant}_${EXP_NAMES_STR}_${batch_log_ts}.log"

setup_batch_logging "$BATCH_LOG_DIR" "$batch_log_path"
trap cleanup_batch_logging EXIT

echo "Batch eval log: $batch_log_path"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: EVAL_SCRIPT not found: $EVAL_SCRIPT"
    exit 1
fi

if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "ERROR: TRAIN_DATA_PATH must be set to a non-empty parquet path."
    exit 1
fi

if [ -z "$VAL_DATA_PATH" ]; then
    echo "ERROR: VAL_DATA_PATH must be set to the evaluation parquet path."
    exit 1
fi

if [ -z "$EVAL_TOOL_VARIANT" ]; then
    echo "ERROR: EVAL_TOOL_VARIANT must be set to all, id, or ood."
    exit 1
fi

if [ -z "$CHECKPOINT_PATHS_STR" ]; then
    echo "ERROR: CHECKPOINT_PATHS is required."
    exit 1
fi

if [ -z "$TARGET_STEPS_STR" ]; then
    echo "ERROR: TARGET_STEPS is required."
    exit 1
fi

read -r -a CHECKPOINT_PATHS_ARR <<< "$CHECKPOINT_PATHS_STR"
read -r -a TARGET_STEPS_ARR <<< "$TARGET_STEPS_STR"

HAS_EXP_NAMES=0
if [ -n "$EXP_NAMES_STR" ]; then
    read -r -a EXP_NAMES_ARR <<< "$EXP_NAMES_STR"
    HAS_EXP_NAMES=1
fi

if [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#TARGET_STEPS_ARR[@]}" ]; then
    echo "ERROR: CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]}) must match TARGET_STEPS count (${#TARGET_STEPS_ARR[@]})."
    exit 1
fi

if [ "$HAS_EXP_NAMES" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#EXP_NAMES_ARR[@]}" ]; then
    echo "ERROR: EXP_NAMES count (${#EXP_NAMES_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

echo "=========================================="
echo "Batch Eval Config"
echo "EVAL_SCRIPT: $EVAL_SCRIPT"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH:-<unset>}"
echo "VAL_DATA_PATH: ${VAL_DATA_PATH:-<unset>}"
echo "EVAL_TOOL_VARIANT: ${EVAL_TOOL_VARIANT:-<unset>}"
echo "CHECKPOINT_PATHS: $CHECKPOINT_PATHS_STR"
echo "TARGET_STEPS: $TARGET_STEPS_STR"
echo "EXP_NAMES: ${EXP_NAMES_STR:-<auto>}"
echo "NUM_ITEMS: ${#CHECKPOINT_PATHS_ARR[@]}"
echo "LATENCY_PENALTY_START_STEP: $LATENCY_PENALTY_START_STEP"
echo "TOOL_PENALTY_START_STEP: $TOOL_PENALTY_START_STEP"
echo "=========================================="

success_count=0
fail_count=0
failed_items=""

for ((idx = 0; idx < ${#CHECKPOINT_PATHS_ARR[@]}; idx++)); do
    checkpoint_path="${CHECKPOINT_PATHS_ARR[$idx]}"
    target_step="${TARGET_STEPS_ARR[$idx]}"
    if is_base_target_step "$target_step"; then
        auto_name="$(basename "$checkpoint_path")_base"
    else
        auto_name="$(basename "$checkpoint_path")_gs${target_step}"
    fi

    if [ "$HAS_EXP_NAMES" -eq 1 ]; then
        exp_name_raw="${EXP_NAMES_ARR[$idx]}"
    else
        exp_name_raw="$auto_name"
    fi

    exp_name="$(sanitize_label "$exp_name_raw")"

    echo ""
    echo "------------------------------------------"
    echo "Processing item $((idx + 1))/${#CHECKPOINT_PATHS_ARR[@]}: $exp_name"
    echo "CHECKPOINT_PATH: $checkpoint_path"
    echo "TARGET_STEP: $target_step"
    echo "------------------------------------------"

    if [ ! -d "$checkpoint_path" ]; then
        echo "WARN: checkpoint path not found: $checkpoint_path, skipping."
        fail_count=$((fail_count + 1))
        failed_items="${failed_items} ${exp_name}(missing_checkpoint_path)"
        continue
    fi

    model_dir="$(resolve_model_dir "$checkpoint_path" "$target_step")"
    if [ ! -d "$model_dir" ]; then
        if is_base_target_step "$target_step"; then
            echo "WARN: base model dir not found: $model_dir, skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} ${exp_name}(missing_base_model_dir)"
        else
            echo "WARN: checkpoint model dir not found: $model_dir, skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} ${exp_name}(missing_hf_dir)"
        fi
        continue
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    model_tag="$exp_name"
    run_id="batch_eval_${EVAL_TOOL_VARIANT}_${exp_name}_${run_ts}"

    echo "Resolved MODEL_NAME: $model_dir"
    echo "MODEL_TAG: $model_tag"
    echo "RUN_ID: $run_id"

    MODEL_NAME="$model_dir" \
    MODEL_TAG="$model_tag" \
    TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
    VAL_DATA_PATH="$VAL_DATA_PATH" \
    EVAL_TOOL_VARIANT="$EVAL_TOOL_VARIANT" \
    LATENCY_PENALTY_START_STEP="$LATENCY_PENALTY_START_STEP" \
    TOOL_PENALTY_START_STEP="$TOOL_PENALTY_START_STEP" \
    RUN_ID="$run_id" \
    bash "$EVAL_SCRIPT"
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo "${exp_name} finished successfully."
        success_count=$((success_count + 1))
    else
        echo "WARN: ${exp_name} failed with exit code $exit_code, continuing."
        fail_count=$((fail_count + 1))
        failed_items="${failed_items} ${exp_name}(exit_${exit_code})"
    fi
done

echo ""
echo "=========================================="
echo "Batch Eval Summary"
echo "Success: $success_count"
echo "Failed: $fail_count"
if [ "$fail_count" -gt 0 ]; then
    echo "Failed items:${failed_items}"
fi
echo "=========================================="

if [ "$fail_count" -gt 0 ]; then
    exit 1
fi

exit 0