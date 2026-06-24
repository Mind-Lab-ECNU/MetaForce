#!/usr/bin/env bash
# =============================================================================
# Multimodal Orchestra Eval Script - Single Node (Standalone Token Backend)
# =============================================================================
# 单节点评测脚本：
# - 模型推理走本机 vLLM standalone rollout actor
# - tool server 可外置，也可由脚本本地启动
# - 不走 main_ppo / Ray actor rollout trainer
#
# 用法示例:
# MODEL_PATH=/path/to/InternVL-or-Qwen-VL \
#   MODEL_FAMILY=internvl \
#   VAL_DATA_PATH=/path/to/eval.parquet \
#   bash examples/train/multimodal_orchestra/single_node/eval_8gpu_token_standalone.sh
# =============================================================================

set -euo pipefail
set -x

SCRIPT_START_EPOCH=$(date +%s)
JOB_START_EPOCH=0
JOB_END_EPOCH=0
server_pid=""

format_duration_hms() {
    local total_seconds=$1
    if [ "$total_seconds" -lt 0 ]; then
        total_seconds=0
    fi
    printf "%02d:%02d:%02d" \
        $((total_seconds / 3600)) \
        $(((total_seconds % 3600) / 60)) \
        $((total_seconds % 60))
}

is_true() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

print_timing_summary() {
    local now script_elapsed job_end job_elapsed
    now=$(date +%s)
    script_elapsed=$((now - SCRIPT_START_EPOCH))
    if [ "$script_elapsed" -lt 0 ]; then
        script_elapsed=0
    fi

    if [ "$JOB_START_EPOCH" -gt 0 ]; then
        job_end=$JOB_END_EPOCH
        if [ "$job_end" -le 0 ]; then
            job_end=$now
        fi
        job_elapsed=$((job_end - JOB_START_EPOCH))
        if [ "$job_elapsed" -lt 0 ]; then
            job_elapsed=0
        fi
        echo "[TIME] job_elapsed_sec=$job_elapsed job_elapsed_hms=$(format_duration_hms "$job_elapsed")"
    else
        echo "[TIME] job_elapsed_sec=0 job_elapsed_hms=00:00:00"
    fi

    echo "[TIME] script_elapsed_sec=$script_elapsed script_elapsed_hms=$(format_duration_hms "$script_elapsed")"
}

on_exit() {
    local exit_code=$?
    if [ -n "${server_pid:-}" ]; then
        kill -9 "$server_pid" 2>/dev/null || true
    fi
    print_timing_summary
    echo "[EXIT] exit_code=$exit_code"
}

trap on_exit EXIT

ROOT_DIR="$(pwd)"

VAL_DATA_PATH="${VAL_DATA_PATH:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
EVAL_TOOL_VARIANT="${EVAL_TOOL_VARIANT:-all}"
TOOL_SERVER_URL="${TOOL_SERVER_URL:-}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
SKILL_STORE_DIR="${SKILL_STORE_DIR:-}"
RUN_ID="${RUN_ID:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
N="${N:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-24576}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
MAX_ACTION_LENGTH="${MAX_ACTION_LENGTH:-8192}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:--$((8192 + 8192))}"
MAX_TURNS="${MAX_TURNS:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOOL_CALL_TIMEOUT="${TOOL_CALL_TIMEOUT:-160}"
TOOL_CALL_MAX_RETRIES="${TOOL_CALL_MAX_RETRIES:-4}"
MAX_CONCURRENT_TRAJECTORIES="${MAX_CONCURRENT_TRAJECTORIES:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PER_WORKER_MAX_CONCURRENCY="${PER_WORKER_MAX_CONCURRENCY:-8}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-32}"
LATENCY_PENALTY_START_STEP="${LATENCY_PENALTY_START_STEP:-0}"
TOOL_PENALTY_START_STEP="${TOOL_PENALTY_START_STEP:-0}"
REWARD_MANAGER="${REWARD_MANAGER:-multimodal_orchestra_eval_relaxed_answer}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-console,swanlab}"
PROJECT_NAME="${PROJECT_NAME:-$REWARD_MANAGER}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-10000}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"

if [ -z "$VAL_DATA_PATH" ]; then
    echo "VAL_DATA_PATH must be set." >&2
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH must be set to the local HF/vLLM model path." >&2
    exit 1
fi

if [ -z "$TOKENIZER_PATH" ]; then
    TOKENIZER_PATH="$MODEL_PATH"
fi

if [ -z "$RUN_ID" ]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)_token_eval"
fi

if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="$RUN_ID"
fi

RECORD_ROOT="${ROOT_DIR}/verl_step_records"
TOOL_LOG_DIR="${RECORD_ROOT}/tool_logs/val/${RUN_ID}"
DEFAULT_SKILL_STORE_DIR="${RECORD_ROOT}/skills/val/${RUN_ID}"
VAL_DATA_DIR="${RECORD_ROOT}/validation_data/val/${RUN_ID}"
SKILL_STORE_DIR="${SKILL_STORE_DIR:-$DEFAULT_SKILL_STORE_DIR}"

export VERL_RUN_ID="$RUN_ID"
export VERL_SKILL_STORE_DIR="$SKILL_STORE_DIR"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "$N_GPUS_PER_NODE" ] && [ "$N_GPUS_PER_NODE" -gt 1 ]; then
    visible_devices=""
    for ((gpu_idx = 0; gpu_idx < N_GPUS_PER_NODE; gpu_idx++)); do
        if [ -n "$visible_devices" ]; then
            visible_devices="${visible_devices},"
        fi
        visible_devices="${visible_devices}${gpu_idx}"
    done
    export CUDA_VISIBLE_DEVICES="$visible_devices"
fi

mkdir -p "$TOOL_LOG_DIR" "$SKILL_STORE_DIR" "$VAL_DATA_DIR"

if [ -z "$TOOL_SERVER_URL" ]; then
    host=$(hostname -i | awk '{print $1}')
    port=$(shuf -i 30000-31000 -n 1)
    TOOL_SERVER_URL="http://${host}:${port}/get_observation"
    case "${EVAL_TOOL_VARIANT}" in
        all)
            tool_type="multimodal_processor_tool_adapt_skill"
            ;;
        id)
            tool_type="multimodal_processor_tool_adapt_skill_id"
            ;;
        ood)
            tool_type="multimodal_processor_tool_adapt_skill_ood"
            ;;
        *)
            echo "Invalid EVAL_TOOL_VARIANT: ${EVAL_TOOL_VARIANT}" >&2
            exit 1
            ;;
    esac

    python -m verl_tool.servers.serve \
        --host "$host" \
        --port "$port" \
        --tool_type "$tool_type" \
        --workers_per_tool 64 &
    server_pid=$!
    sleep 10
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Tool server failed to start." >&2
        exit 1
    fi
fi

echo "=========================================="
echo "Run ID: $RUN_ID"
echo "VAL_DATA_PATH: $VAL_DATA_PATH"
echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_FAMILY: $MODEL_FAMILY"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "TOOL_SERVER_URL: $TOOL_SERVER_URL"
echo "REWARD_MANAGER: $REWARD_MANAGER"
echo "TRUST_REMOTE_CODE: $TRUST_REMOTE_CODE"
echo "LOGGER_BACKENDS: $LOGGER_BACKENDS"
echo "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES: $RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "TOOL_LOG_DIR: $TOOL_LOG_DIR"
echo "SKILL_STORE_DIR: $SKILL_STORE_DIR"
echo "VAL_DATA_DIR: $VAL_DATA_DIR"
echo "=========================================="

JOB_START_EPOCH=$(date +%s)
eval_args=(
    --val-data-path "$VAL_DATA_PATH"
    --train-data-path "$TRAIN_DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --model-path "$MODEL_PATH"
    --model-family "$MODEL_FAMILY"
    --tool-server-url "$TOOL_SERVER_URL"
    --reward-manager "$REWARD_MANAGER"
    --tool-log-dir "$TOOL_LOG_DIR"
    --skill-store-dir "$SKILL_STORE_DIR"
    --validation-data-dir "$VAL_DATA_DIR"
    --batch-size "$VAL_BATCH_SIZE"
    --n "$N"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --max-response-length "$MAX_RESPONSE_LENGTH"
    --max-action-length "$MAX_ACTION_LENGTH"
    --max-obs-length "$MAX_OBS_LENGTH"
    --max-turns "$MAX_TURNS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --tool-call-timeout "$TOOL_CALL_TIMEOUT"
    --tool-call-max-retries "$TOOL_CALL_MAX_RETRIES"
    --max-concurrent-trajectories "$MAX_CONCURRENT_TRAJECTORIES"
    --num-workers "$NUM_WORKERS"
    --per-worker-max-concurrency "$PER_WORKER_MAX_CONCURRENCY"
    --eval-tool-variant "$EVAL_TOOL_VARIANT"
    --filter-overlong-prompts-workers "$FILTER_OVERLONG_PROMPTS_WORKERS"
    --latency-penalty-start-step "$LATENCY_PENALTY_START_STEP"
    --tool-penalty-start-step "$TOOL_PENALTY_START_STEP"
    --logger-backends "$LOGGER_BACKENDS"
    --project-name "$PROJECT_NAME"
    --experiment-name "$EXPERIMENT_NAME"
    --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --rollout-max-num-seqs "$ROLLOUT_MAX_NUM_SEQS"
    --rollout-max-num-batched-tokens "$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
)

if [ -n "$N_GPUS_PER_NODE" ]; then
    eval_args+=(--n-gpus-per-node "$N_GPUS_PER_NODE")
fi

if is_true "$TRUST_REMOTE_CODE"; then
    eval_args+=(--trust-remote-code)
fi

python3 -m verl_tool.eval.standalone_token_agent_eval "${eval_args[@]}"
JOB_END_EPOCH=$(date +%s)
