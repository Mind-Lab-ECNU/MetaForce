#!/usr/bin/env bash
# =============================================================================
# Batch eval launcher for multimodal_orchestra checkpoints.
# - Edit EXP_LIST directly in this script.
# - TARGET_STEP controls which global_step_* to evaluate:
#     * empty TARGET_STEP: use latest global_step_* in exp dir
#     * non-empty TARGET_STEP: use global_step_${TARGET_STEP}
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# User-editable config
# -----------------------------------------------------------------------------
EXP_LIST=(0 3 5)
if [ -n "${EXP_LIST_OVERRIDE:-}" ]; then
    read -r -a EXP_LIST <<< "${EXP_LIST_OVERRIDE}"
fi
# Set TARGET_STEP_INPUT to a specific step like "100".
# Keep TARGET_STEP_INPUT="" to auto-pick the latest global_step_*.
TARGET_STEP_INPUT=""
TARGET_STEP="${TARGET_STEP:-$TARGET_STEP_INPUT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/verl_step_records/checkpoint/train}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/single_node/eval_8gpu.sh}"
EVAL_TRAIN_DATA_PATH=""
EVAL_DATA_PATH=""
EVAL_TOOL_VARIANT=""

# Force eval reward penalties to start from step 0.
LATENCY_PENALTY_START_STEP="${LATENCY_PENALTY_START_STEP:-0}"
TOOL_PENALTY_START_STEP="${TOOL_PENALTY_START_STEP:-0}"

echo "=========================================="
echo "Batch Eval Config"
echo "EVAL_SCRIPT: $EVAL_SCRIPT"
echo "CHECKPOINT_ROOT: $CHECKPOINT_ROOT"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "EVAL_TRAIN_DATA_PATH: ${EVAL_TRAIN_DATA_PATH:-<unset>}"
echo "EVAL_DATA_PATH: ${EVAL_DATA_PATH:-<unset>}"
echo "EVAL_TOOL_VARIANT: ${EVAL_TOOL_VARIANT:-<unset>}"
echo "TARGET_STEP: ${TARGET_STEP:-<latest>}"
echo "EXP_LIST: ${EXP_LIST[*]}"
echo "LATENCY_PENALTY_START_STEP: $LATENCY_PENALTY_START_STEP"
echo "TOOL_PENALTY_START_STEP: $TOOL_PENALTY_START_STEP"
echo "=========================================="

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: EVAL_SCRIPT not found: $EVAL_SCRIPT"
    exit 1
fi

if [ -z "$EVAL_TRAIN_DATA_PATH" ]; then
    echo "ERROR: EVAL_TRAIN_DATA_PATH must be set to a non-empty parquet path."
    exit 1
fi

if [ -z "$EVAL_DATA_PATH" ]; then
    echo "ERROR: EVAL_DATA_PATH must be set to the evaluation parquet path."
    exit 1
fi

if [ -z "$EVAL_TOOL_VARIANT" ]; then
    echo "ERROR: EVAL_TOOL_VARIANT must be set to all, id, or ood."
    exit 1
fi

resolve_latest_exp_dir() {
    local exp_idx="$1"
    local latest_dir
    latest_dir=$(ls -dt "${CHECKPOINT_ROOT}/exp${exp_idx}_"* 2>/dev/null | head -1 || true)
    printf '%s' "$latest_dir"
}

resolve_step_dir() {
    local exp_dir="$1"
    local target_step="$2"

    if [ -n "$target_step" ]; then
        printf '%s' "${exp_dir}/global_step_${target_step}"
        return
    fi

    local latest_step_dir
    latest_step_dir=$(ls -d "${exp_dir}"/global_step_* 2>/dev/null | sort -V | tail -1 || true)
    printf '%s' "$latest_step_dir"
}

success_count=0
fail_count=0
failed_items=""

for exp_idx in "${EXP_LIST[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Processing exp${exp_idx}"
    echo "------------------------------------------"

    resolved_model=""
    step_label=""

    if [ "$exp_idx" = "0" ]; then
        resolved_model="$BASE_MODEL_PATH"
        step_label="base"
    else
        exp_dir=$(resolve_latest_exp_dir "$exp_idx")
        if [ -z "$exp_dir" ] || [ ! -d "$exp_dir" ]; then
            echo "WARN: exp${exp_idx} directory not found under $CHECKPOINT_ROOT, skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} exp${exp_idx}(missing_exp_dir)"
            continue
        fi

        step_dir=$(resolve_step_dir "$exp_dir" "$TARGET_STEP")
        if [ -z "$step_dir" ] || [ ! -d "$step_dir" ]; then
            echo "WARN: step dir not found for exp${exp_idx} (TARGET_STEP=${TARGET_STEP:-latest}), skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} exp${exp_idx}(missing_step_dir)"
            continue
        fi

        hf_dir="${step_dir}/actor/huggingface"
        if [ ! -d "$hf_dir" ]; then
            echo "WARN: missing huggingface model dir: $hf_dir, skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} exp${exp_idx}(missing_hf_dir)"
            continue
        fi

        resolved_model="$hf_dir"
        step_label="$(basename "$step_dir")"
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    run_id="batch_eval_${EVAL_TOOL_VARIANT}_exp${exp_idx}_${step_label}_${run_ts}"
    model_tag="exp${exp_idx}-${step_label}-${EVAL_TOOL_VARIANT}"

    echo "Resolved MODEL_NAME: $resolved_model"
    echo "MODEL_TAG: $model_tag"
    echo "RUN_ID: $run_id"

    MODEL_NAME="$resolved_model" \
    MODEL_TAG="$model_tag" \
    EVAL_TRAIN_DATA_PATH="$EVAL_TRAIN_DATA_PATH" \
    EVAL_DATA_PATH="$EVAL_DATA_PATH" \
    EVAL_TOOL_VARIANT="$EVAL_TOOL_VARIANT" \
    LATENCY_PENALTY_START_STEP="$LATENCY_PENALTY_START_STEP" \
    TOOL_PENALTY_START_STEP="$TOOL_PENALTY_START_STEP" \
    RUN_ID="$run_id" \
    bash "$EVAL_SCRIPT"
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo "exp${exp_idx} finished successfully."
        success_count=$((success_count + 1))
    else
        echo "WARN: exp${exp_idx} failed with exit code $exit_code, continuing."
        fail_count=$((fail_count + 1))
        failed_items="${failed_items} exp${exp_idx}(exit_${exit_code})"
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
