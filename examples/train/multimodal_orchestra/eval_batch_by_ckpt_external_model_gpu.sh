#!/usr/bin/env bash
# =============================================================================
# Batch eval launcher for external-model multimodal_orchestra evaluation.
# - CHECKPOINT_PATHS and TARGET_STEPS are space-separated lists matched by index.
# - Each item resolves a local model dir exactly like eval_batch_by_ckpt.sh:
#     CHECKPOINT_PATH/ + TARGET_STEP -> local HF dir for tokenizer usage
# - If TARGET_STEPS is omitted entirely, all items default to `base`.
# - CHECKPOINT_PATHS in this script are mainly used to resolve local HuggingFace
#   directories for tokenizer / processor loading.
#   They are not sent to the external model server unless you intentionally let
#   MODEL_NAME default to the resolved local model dir string.
# - MODEL_NAMES is optional:
#     if set, each item is passed to external MODEL_NAME by index.
#     This means the `model` field sent to the external OpenAI/vLLM-compatible
#     server. It is NOT the same thing as EXP_NAMES.
#     if unset, MODEL_NAME defaults to the resolved local model dir string
#     (this only works if your external server accepts that exact string).
# - EXP_NAMES is only for local run naming / logging / RUN_ID labeling.
#   EXP_NAMES is never sent to the external model server.
# - MODEL_BASE_URLS is optional:
#     if set, each item is passed to external MODEL_BASE_URL by index
#     if unset, fallback to the single MODEL_BASE_URL
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_DATASETS_CACHE="/dev/shm/hf_cache_$$"
export HF_HOME="/dev/shm/hf_home_$$"
mkdir -p $HF_DATASETS_CACHE $HF_HOME
ROOT_DIR="$(pwd)"
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${ROOT_DIR}/swanlog_output_external_eval

# -----------------------------------------------------------------------------
# User-editable config
# -----------------------------------------------------------------------------
# Terminology to avoid ambiguity:
# - CHECKPOINT_PATHS:
#     local experiment/base-model paths used to resolve tokenizer / processor dirs
# - TARGET_STEPS:
#     how CHECKPOINT_PATHS are resolved into local HF dirs
# - MODEL_NAMES:
#     external served model names sent in API payload `model=...`
# - EXP_NAMES:
#     local labels only, used for logging / RUN_ID
#
# Example:
# CHECKPOINT_PATHS="/path/exp3_run /path/exp5_run" \
# TARGET_STEPS="100 240" \
# MODEL_BASE_URL="http://<vllm-host>:8000" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_external_model.sh
#
# Example with explicit served names:
# CHECKPOINT_PATHS="/path/exp3_run /path/exp5_run" \
# TARGET_STEPS="100 240" \
# MODEL_NAMES="served-exp3-gs100 served-exp5-gs240" \
# MODEL_BASE_URL="http://<vllm-host>:8000" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_external_model.sh
#
# Example with per-model base URLs:
# CHECKPOINT_PATHS="/path/exp3_run /path/exp5_run" \
# TARGET_STEPS="100 240" \
# MODEL_BASE_URLS="http://host-a:8000 http://host-b:8000" \
# MODEL_NAMES="served-exp3-gs100 served-exp5-gs240" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_external_model.sh
#
# Base model example with explicit TARGET_STEPS:
# CHECKPOINT_PATHS="/inspire/.../Qwen3-VL-8B-Instruct /path/to/exp5_run" \
# TARGET_STEPS="base 240" \
# MODEL_BASE_URL="http://<vllm-host>:8000" \
# MODEL_NAMES="Qwen3-VL-8B-Instruct served-exp5-gs240" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_external_model.sh
#
# Pure base-model example without TARGET_STEPS:
# CHECKPOINT_PATHS="/inspire/.../Qwen3-VL-8B-Instruct /inspire/.../Qwen2.5-VL-7B-Instruct" \
# MODEL_BASE_URL="http://<vllm-host>:8000" \
# MODEL_NAMES="Qwen3-VL-8B-Instruct Qwen2.5-VL-7B-Instruct" \
# TRAIN_DATA_PATH="/path/to/train.parquet" \
# VAL_DATA_PATH="/path/to/eval.parquet" \
# EVAL_TOOL_VARIANT="all" \
# bash examples/train/multimodal_orchestra/eval_batch_by_ckpt_external_model.sh

# Fill this block directly when you want to edit the script itself.
# Environment variables with the same names still override these defaults.
# Notes:
# - MODEL_BASE_URL_INPUT:
#     one shared external model service base URL for all items
#     if you only fill this variable, every checkpoint/model item will use
#     the same MODEL_BASE_URL
# - MODEL_BASE_URLS_INPUT:
#     space-separated per-item base URLs matched to CHECKPOINT_PATHS by index
#     if you only fill this variable, each checkpoint/model item will use its
#     own MODEL_BASE_URL by index:
#       CHECKPOINT_PATHS[0] -> MODEL_BASE_URLS_INPUT[0]
#       CHECKPOINT_PATHS[1] -> MODEL_BASE_URLS_INPUT[1]
#     count must match CHECKPOINT_PATHS
# - MODEL_BASE_URL_INPUT vs MODEL_BASE_URLS_INPUT:
#     fill MODEL_BASE_URL_INPUT when all items share one external server
#     fill MODEL_BASE_URLS_INPUT when different items are served by different servers
#     you can fill either one; both empty is invalid
# - MODEL_BASE_URLS_STR:
#     internal resolved string:
#       MODEL_BASE_URLS (env) -> fallback MODEL_BASE_URLS_INPUT (script)
#     you do not fill MODEL_BASE_URLS_STR directly
EVAL_SCRIPT_INPUT=
CHECKPOINT_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/InternVL3_5-38B-HF"
TARGET_STEPS_INPUT="base"
# EXP_NAMES_INPUT="gemini-3.1-pro-preview-v8-data"
EXP_NAMES_INPUT="gpt-5.4-clevr-math"
MODEL_BASE_URL_INPUT=""
# MODEL_BASE_URLS_INPUT="https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/125f91d5-c3ba-4150-836a-aa0b4727b4d3/3c0edfd2-b044-48fc-a64a-508a54744d78/proxy/30033/v1"
# MODEL_BASE_URLS_INPUT="https://api.innospark.cn/v1"
MODEL_BASE_URLS_INPUT="https://api.gptgod.online/v1"
# MODEL_NAMES_INPUT="gemini-3.1-pro-preview"
MODEL_NAMES_INPUT="gpt-5.4"
TOKENIZER_PATHS_INPUT="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/InternVL3_5-38B-HF"
TRAIN_DATA_PATH_INPUT='/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/train_merge_v6_tool_ood/train_merge_v6_tool_ood.parquet'
VAL_DATA_PATH_INPUT='/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/test_merge_v4_clevr_math_tool_ood/test_ood_merge_v4_clevr_math_tool_ood.parquet'
EVAL_TOOL_VARIANT_INPUT="ood"
LATENCY_PENALTY_START_STEP_INPUT=""
TOOL_PENALTY_START_STEP_INPUT=""
BATCH_LOG_DIR_INPUT=""

CHECKPOINT_PATHS_STR="${CHECKPOINT_PATHS:-$CHECKPOINT_PATHS_INPUT}"
TARGET_STEPS_STR="${TARGET_STEPS:-$TARGET_STEPS_INPUT}"
EXP_NAMES_STR="${EXP_NAMES:-$EXP_NAMES_INPUT}"
MODEL_BASE_URL="${MODEL_BASE_URL:-$MODEL_BASE_URL_INPUT}"
MODEL_BASE_URLS_STR="${MODEL_BASE_URLS:-$MODEL_BASE_URLS_INPUT}"
MODEL_NAMES_STR="${MODEL_NAMES:-$MODEL_NAMES_INPUT}"
TOKENIZER_PATHS_STR="${TOKENIZER_PATHS:-$TOKENIZER_PATHS_INPUT}"

EVAL_SCRIPT="${EVAL_SCRIPT:-${EVAL_SCRIPT_INPUT:-${SCRIPT_DIR}/single_node/eval_8gpu_external_model.sh}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$TRAIN_DATA_PATH_INPUT}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$VAL_DATA_PATH_INPUT}"
EVAL_TOOL_VARIANT="${EVAL_TOOL_VARIANT:-$EVAL_TOOL_VARIANT_INPUT}"
LATENCY_PENALTY_START_STEP="${LATENCY_PENALTY_START_STEP:-${LATENCY_PENALTY_START_STEP_INPUT:-0}}"
TOOL_PENALTY_START_STEP="${TOOL_PENALTY_START_STEP:-${TOOL_PENALTY_START_STEP_INPUT:-0}}"
BATCH_LOG_DIR="${BATCH_LOG_DIR:-${BATCH_LOG_DIR_INPUT:-$(pwd)/logs/multimodal_orchestra_eval_batch_external_model_gpu_closed}}"

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

    BATCH_LOG_PIPE="$(mktemp "${TMPDIR:-/tmp}/eval_batch_external_model_log_pipe.XXXXXX")"
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
batch_log_path="${BATCH_LOG_DIR}/batch_eval_external_model_${batch_log_variant}_${batch_log_ts}.log"

setup_batch_logging "$BATCH_LOG_DIR" "$batch_log_path"
trap cleanup_batch_logging EXIT

echo "Batch eval log: $batch_log_path"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: EVAL_SCRIPT not found: $EVAL_SCRIPT"
    exit 1
fi

if [ -z "$MODEL_BASE_URL" ] && [ -z "$MODEL_BASE_URLS_STR" ]; then
    echo "ERROR: either MODEL_BASE_URL or MODEL_BASE_URLS must be set."
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

HAS_MODEL_BASE_URLS=0
if [ -n "$MODEL_BASE_URLS_STR" ]; then
    read -r -a MODEL_BASE_URLS_ARR <<< "$MODEL_BASE_URLS_STR"
    HAS_MODEL_BASE_URLS=1
fi

HAS_MODEL_NAMES=0
if [ -n "$MODEL_NAMES_STR" ]; then
    read -r -a MODEL_NAMES_ARR <<< "$MODEL_NAMES_STR"
    HAS_MODEL_NAMES=1
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

if [ "$HAS_MODEL_BASE_URLS" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#MODEL_BASE_URLS_ARR[@]}" ]; then
    echo "ERROR: MODEL_BASE_URLS count (${#MODEL_BASE_URLS_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

if [ "$HAS_MODEL_NAMES" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#MODEL_NAMES_ARR[@]}" ]; then
    echo "ERROR: MODEL_NAMES count (${#MODEL_NAMES_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

if [ "$HAS_TOKENIZER_PATHS" -eq 1 ] && [ "${#CHECKPOINT_PATHS_ARR[@]}" -ne "${#TOKENIZER_PATHS_ARR[@]}" ]; then
    echo "ERROR: TOKENIZER_PATHS count (${#TOKENIZER_PATHS_ARR[@]}) must match CHECKPOINT_PATHS count (${#CHECKPOINT_PATHS_ARR[@]})."
    exit 1
fi

echo "=========================================="
echo "Batch Eval Config"
echo "EVAL_SCRIPT: $EVAL_SCRIPT"
echo "MODEL_BASE_URL: ${MODEL_BASE_URL:-<unset>}"
echo "MODEL_BASE_URLS: ${MODEL_BASE_URLS_STR:-<single_MODEL_BASE_URL>}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH:-<unset>}"
echo "VAL_DATA_PATH: ${VAL_DATA_PATH:-<unset>}"
echo "EVAL_TOOL_VARIANT: ${EVAL_TOOL_VARIANT:-<unset>}"
echo "CHECKPOINT_PATHS: $CHECKPOINT_PATHS_STR"
echo "TARGET_STEPS: ${TARGET_STEPS_STR:-<auto_base>}"
echo "EXP_NAMES: ${EXP_NAMES_STR:-<auto>}"
echo "MODEL_NAMES: ${MODEL_NAMES_STR:-<resolved_model_dir>}"
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

    if [ "$HAS_MODEL_NAMES" -eq 1 ]; then
        served_model_name="${MODEL_NAMES_ARR[$idx]}"
    else
        served_model_name="$model_dir"
    fi

    if [ "$HAS_MODEL_BASE_URLS" -eq 1 ]; then
        item_model_base_url="${MODEL_BASE_URLS_ARR[$idx]}"
    else
        item_model_base_url="$MODEL_BASE_URL"
    fi

    if [ "$HAS_TOKENIZER_PATHS" -eq 1 ]; then
        tokenizer_path="${TOKENIZER_PATHS_ARR[$idx]}"
    else
        tokenizer_path="$model_dir"
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    run_id="batch_eval_external_model_${EVAL_TOOL_VARIANT}_${exp_name}_${run_ts}"

    echo "Resolved local model dir: $model_dir"
    echo "Resolved MODEL_BASE_URL: $item_model_base_url"
    echo "Resolved MODEL_NAME: $served_model_name"
    echo "Resolved TOKENIZER_PATH: $tokenizer_path"
    echo "RUN_ID: $run_id"

    MODEL_BASE_URL="$item_model_base_url" \
    MODEL_NAME="$served_model_name" \
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
