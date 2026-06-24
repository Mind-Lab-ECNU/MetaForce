#!/usr/bin/env bash
# =============================================================================
# Batch eval launcher for standalone token-in/token-out multimodal_orchestra evaluation.
# - CHECKPOINT_PATHS and TARGET_STEPS are space-separated lists matched by index.
# - Each item resolves a local HuggingFace model dir:
#     CHECKPOINT_PATH/ + TARGET_STEP -> local MODEL_PATH
# - If TARGET_STEPS is omitted entirely, all items default to `base`.
# - MODEL_FAMILY can be one shared value for all items.
# - MODEL_FAMILIES can be a space-separated per-item list matched by index.
# - TOKENIZER_PATHS is optional; if omitted, each item defaults to resolved MODEL_PATH.
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_DATASETS_CACHE="/dev/shm/hf_cache_$$"
export HF_HOME="/dev/shm/hf_home_$$"
mkdir -p $HF_DATASETS_CACHE $HF_HOME
ROOT_DIR="$(pwd)"
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${ROOT_DIR}/swanlog_output_external_eval_token

# -----------------------------------------------------------------------------
# User-editable config
# -----------------------------------------------------------------------------
# Example:
# CHECKPOINT_PATHS="/path/exp3_run /path/exp5_run" \
# TARGET_STEPS="100 240" \
# MODEL_FAMILY="internvl" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_token_standalone.sh
#
# Example with mixed model families:
# CHECKPOINT_PATHS="/path/internvl /path/qwen_run" \
# TARGET_STEPS="base 240" \
# MODEL_FAMILIES="internvl qwen_vl" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_token_standalone.sh

EVAL_SCRIPT_INPUT=""
CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/InternVL3_5-14B-HF"
TARGET_STEPS_INPUT="base"
EXP_NAMES_INPUT="internvl-14b-v8-data"
MODEL_FAMILY_INPUT="interns1"
MODEL_FAMILIES_INPUT=""
TOKENIZER_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/InternVL3_5-14B-HF"
TRAIN_DATA_PATH_INPUT="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/train_merge_v6_tool_ood/train_merge_v6_tool_ood.parquet"
VAL_DATA_PATH_INPUT='/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/test_merge_v8_tool_id/test_merge_v8_tool_id.parquet'
EVAL_TOOL_VARIANT_INPUT="id"
LATENCY_PENALTY_START_STEP_INPUT=""
TOOL_PENALTY_START_STEP_INPUT=""
BATCH_LOG_DIR_INPUT=""

CHECKPOINT_PATHS_STR="${CHECKPOINT_PATHS:-$CHECKPOINT_PATHS_INPUT}"
TARGET_STEPS_STR="${TARGET_STEPS:-$TARGET_STEPS_INPUT}"
EXP_NAMES_STR="${EXP_NAMES:-$EXP_NAMES_INPUT}"
MODEL_FAMILY="${MODEL_FAMILY:-$MODEL_FAMILY_INPUT}"
MODEL_FAMILIES_STR="${MODEL_FAMILIES:-$MODEL_FAMILIES_INPUT}"
TOKENIZER_PATHS_STR="${TOKENIZER_PATHS:-$TOKENIZER_PATHS_INPUT}"

EVAL_SCRIPT="${EVAL_SCRIPT:-${EVAL_SCRIPT_INPUT:-${SCRIPT_DIR}/single_node/eval_8gpu_token_standalone.sh}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$TRAIN_DATA_PATH_INPUT}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$VAL_DATA_PATH_INPUT}"
EVAL_TOOL_VARIANT="${EVAL_TOOL_VARIANT:-$EVAL_TOOL_VARIANT_INPUT}"
LATENCY_PENALTY_START_STEP="${LATENCY_PENALTY_START_STEP:-${LATENCY_PENALTY_START_STEP_INPUT:-0}}"
TOOL_PENALTY_START_STEP="${TOOL_PENALTY_START_STEP:-${TOOL_PENALTY_START_STEP_INPUT:-0}}"
BATCH_LOG_DIR="${BATCH_LOG_DIR:-${BATCH_LOG_DIR_INPUT:-$(pwd)/logs/multimodal_orchestra_eval_batch_token_standalone_clevr_math}}"

sanitize_label() {
    local raw_label="$1"
    local clean_label

    clean_label=$(printf '%s' "$raw_label" | tr '/: ' '___' | tr -cd '[:alnum:]_.-')
    if [ -z "$clean_label" ]; then
        clean_label="model"
    fi

    printf '%s' "$clean_label"
}

render_progress_bar() {
    local current=$1
    local total=$2
    local width=${3:-24}
    local filled=0
    local bar=""
    local idx

    if [ "$total" -le 0 ]; then
        total=1
    fi

    filled=$((current * width / total))
    if [ "$filled" -gt "$width" ]; then
        filled=$width
    fi

    for ((idx = 0; idx < width; idx++)); do
        if [ "$idx" -lt "$filled" ]; then
            bar="${bar}#"
        else
            bar="${bar}-"
        fi
    done

    printf '[%s]' "$bar"
}

print_outer_progress() {
    local stage="$1"
    local current="$2"
    local total="$3"
    local label="$4"
    local success="$5"
    local failure="$6"
    local result="${7:-running}"
    local percent remaining

    if [ "$total" -le 0 ]; then
        total=1
    fi

    percent=$((current * 100 / total))
    remaining=$((total - current))
    if [ "$remaining" -lt 0 ]; then
        remaining=0
    fi

    echo "[OUTER_PROGRESS] stage=${stage} item=${current}/${total} percent=${percent}% bar=$(render_progress_bar "$current" "$total") label=${label} success=${success} failure=${failure} remaining=${remaining} result=${result}"
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

    BATCH_LOG_PIPE="$(mktemp "${TMPDIR:-/tmp}/eval_batch_token_standalone_log_pipe.XXXXXX")"
    rm -f "$BATCH_LOG_PIPE"
    mkfifo "$BATCH_LOG_PIPE"

    if command -v stdbuf >/dev/null 2>&1; then
        tee_cmd="stdbuf -oL tee"
    fi

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
batch_log_path="${BATCH_LOG_DIR}/batch_eval_token_standalone_${batch_log_variant}_${batch_log_ts}.log"

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

read -r -a CHECKPOINT_PATHS_ARR <<< "$CHECKPOINT_PATHS_STR"

if [ -z "$TARGET_STEPS_STR" ]; then
    TARGET_STEPS_ARR=()
    for ((idx = 0; idx < ${#CHECKPOINT_PATHS_ARR[@]}; idx++)); do
        TARGET_STEPS_ARR+=("base")
    done
else
    read -r -a TARGET_STEPS_ARR <<< "$TARGET_STEPS_STR"
fi

HAS_EXP_NAMES=0
if [ -n "$EXP_NAMES_STR" ]; then
    read -r -a EXP_NAMES_ARR <<< "$EXP_NAMES_STR"
    HAS_EXP_NAMES=1
fi

HAS_MODEL_FAMILIES=0
if [ -n "$MODEL_FAMILIES_STR" ]; then
    read -r -a MODEL_FAMILIES_ARR <<< "$MODEL_FAMILIES_STR"
    HAS_MODEL_FAMILIES=1
fi

HAS_TOKENIZER_PATHS=0
if [ -n "$TOKENIZER_PATHS_STR" ]; then
    read -r -a TOKENIZER_PATHS_ARR <<< "$TOKENIZER_PATHS_STR"
    HAS_TOKENIZER_PATHS=1
fi

if [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#TARGET_STEPS_ARR[@]}" ]; then
    echo "ERROR: CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]}) must match TARGET_STEPS count (${#TARGET_STEPS_ARR[@]})."
    exit 1
fi

if [ "$HAS_EXP_NAMES" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#EXP_NAMES_ARR[@]}" ]; then
    echo "ERROR: EXP_NAMES count (${#EXP_NAMES_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

if [ "$HAS_MODEL_FAMILIES" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#MODEL_FAMILIES_ARR[@]}" ]; then
    echo "ERROR: MODEL_FAMILIES count (${#MODEL_FAMILIES_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

if [ "$HAS_TOKENIZER_PATHS" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#TOKENIZER_PATHS_ARR[@]}" ]; then
    echo "ERROR: TOKENIZER_PATHS count (${#TOKENIZER_PATHS_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

echo "=========================================="
echo "Batch Eval Config"
echo "EVAL_SCRIPT: $EVAL_SCRIPT"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH:-<unset>}"
echo "VAL_DATA_PATH: ${VAL_DATA_PATH:-<unset>}"
echo "EVAL_TOOL_VARIANT: ${EVAL_TOOL_VARIANT:-<unset>}"
echo "CHECKPOINT_PATHS: $CHECKPOINT_PATHS_STR"
echo "TARGET_STEPS: ${TARGET_STEPS_STR:-<auto_base>}"
echo "EXP_NAMES: ${EXP_NAMES_STR:-<auto>}"
echo "MODEL_FAMILY: ${MODEL_FAMILY:-<unset>}"
echo "MODEL_FAMILIES: ${MODEL_FAMILIES_STR:-<single_MODEL_FAMILY_or_auto>}"
echo "TOKENIZER_PATHS: ${TOKENIZER_PATHS_STR:-<resolved_model_dir>}"
echo "NUM_ITEMS: ${#CHECKPOINT_PATHS_ARR[@]}"
echo "LATENCY_PENALTY_START_STEP: $LATENCY_PENALTY_START_STEP"
echo "TOOL_PENALTY_START_STEP: $TOOL_PENALTY_START_STEP"
echo "=========================================="

success_count=0
fail_count=0
failed_items=""
total_items=${#CHECKPOINT_PATHS_ARR[@]}

for ((idx = 0; idx < total_items; idx++)); do
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
    current_item=$((idx + 1))

    echo ""
    echo "------------------------------------------"
    echo "Processing item ${current_item}/${total_items}: $exp_name"
    echo "CHECKPOINT_PATH: $checkpoint_path"
    echo "TARGET_STEP: $target_step"
    echo "------------------------------------------"
    print_outer_progress "start" "$current_item" "$total_items" "$exp_name" "$success_count" "$fail_count" "running"

    if [ ! -d "$checkpoint_path" ]; then
        echo "WARN: checkpoint path not found: $checkpoint_path, skipping."
        fail_count=$((fail_count + 1))
        failed_items="${failed_items} ${exp_name}(missing_checkpoint_path)"
        print_outer_progress "finish" "$current_item" "$total_items" "$exp_name" "$success_count" "$fail_count" "missing_checkpoint_path"
        continue
    fi

    model_dir="$(resolve_model_dir "$checkpoint_path" "$target_step")"
    if [ ! -d "$model_dir" ]; then
        if is_base_target_step "$target_step"; then
            echo "WARN: base model dir not found: $model_dir, skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} ${exp_name}(missing_base_model_dir)"
            print_outer_progress "finish" "$current_item" "$total_items" "$exp_name" "$success_count" "$fail_count" "missing_base_model_dir"
        else
            echo "WARN: checkpoint model dir not found: $model_dir, skipping."
            fail_count=$((fail_count + 1))
            failed_items="${failed_items} ${exp_name}(missing_hf_dir)"
            print_outer_progress "finish" "$current_item" "$total_items" "$exp_name" "$success_count" "$fail_count" "missing_hf_dir"
        fi
        continue
    fi

    if [ "$HAS_MODEL_FAMILIES" -eq 1 ]; then
        item_model_family="${MODEL_FAMILIES_ARR[$idx]}"
    elif [ -n "$MODEL_FAMILY" ]; then
        item_model_family="$MODEL_FAMILY"
    else
        item_model_family="auto"
    fi

    if [ "$HAS_TOKENIZER_PATHS" -eq 1 ]; then
        tokenizer_path="${TOKENIZER_PATHS_ARR[$idx]}"
    else
        tokenizer_path="$model_dir"
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    run_id="batch_eval_token_standalone_${EVAL_TOOL_VARIANT}_${exp_name}_${run_ts}"

    echo "Resolved MODEL_PATH: $model_dir"
    echo "Resolved MODEL_FAMILY: $item_model_family"
    echo "Resolved TOKENIZER_PATH: $tokenizer_path"
    echo "RUN_ID: $run_id"

    MODEL_PATH="$model_dir" \
    MODEL_FAMILY="$item_model_family" \
    TOKENIZER_PATH="$tokenizer_path" \
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
        print_outer_progress "finish" "$current_item" "$total_items" "$exp_name" "$success_count" "$fail_count" "success"
    else
        echo "WARN: ${exp_name} failed with exit code $exit_code, continuing."
        fail_count=$((fail_count + 1))
        failed_items="${failed_items} ${exp_name}(exit_${exit_code})"
        print_outer_progress "finish" "$current_item" "$total_items" "$exp_name" "$success_count" "$fail_count" "exit_${exit_code}"
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
