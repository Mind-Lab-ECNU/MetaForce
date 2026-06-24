#!/usr/bin/env bash
# =============================================================================
# Multimodal Orchestra Eval Script - Single Node (External Model API)
# =============================================================================
# 单节点评测脚本：
# - 模型推理走外部 OpenAI/vLLM 兼容服务
# - tool server 可外置，也可由脚本本地启动
# - 不走 main_ppo / Ray actor rollout
#
# 用法示例:
# MODEL_BASE_URL=http://<vllm-host>:8000 \
#   MODEL_NAME=<served-model-name> \
#   TOKENIZER_PATH=/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct \
#   VAL_DATA_PATH=<your_eval_parquet> \
#   bash examples/train/multimodal_orchestra/single_node/eval_8gpu_external_model.sh
#
# 参数说明（以及与旧版 verl eval 脚本的大致对应关系）:
# - MODEL_BASE_URL:
#     外部 OpenAI/vLLM 兼容服务地址
#     对应旧链路里“框架内部启动的 rollout / vLLM server 地址”
# - MODEL_NAME:
#     发给外部服务的 model 字段
#     不是本地实验名，也不是 tokenizer 路径
#     旧版 eval_8gpu.sh 里没有一一对应的独立参数，因为旧链路直接加载本地模型
# - TOKENIZER_PATH:
#     本地 tokenizer / processor 路径
#     对应旧版的 actor_rollout_ref.model.path / critic.model.path 里的 tokenizer 来源
#     这里只用于本地分词、prompt 处理、reward 解码，不会在框架内加载整套模型权重
# - VAL_DATA_PATH:
#     评测 parquet
#     对应旧版 data.val_files
# - TRAIN_DATA_PATH:
#     兼容保留字段；当前 standalone external eval 不依赖它来启动 rollout
#     主要是为了和原来的批量脚本/调用习惯保持一致
# - EVAL_TOOL_VARIANT:
#     all / id / ood
#     对应旧版脚本里的 EVAL_TOOL_VARIANT
# - TOOL_SERVER_URL:
#     外部 tool server 地址；不传则脚本内自启
#     对应旧版 actor_rollout_ref.agent.tool_server_url
# - SKILL_STORE_DIR:
#     skill 生命周期目录
#     对应旧版 trainer.skill_store_dir
# - RUN_ID:
#     当前评测运行标识
#     对应旧版 VERL_RUN_ID / 输出目录命名
# - VAL_BATCH_SIZE:
#     standalone runner 读数据时每批样本数
#     近似对应旧版 data.val_batch_size
# - N:
#     每条样本重复评测多少次
#     在这个 standalone runner 里，语义上更接近旧版的 actor_rollout_ref.rollout.val_kwargs.n
#     也就是 validation repeat count
#     注意：旧版 eval_8gpu.sh 实际传的是 actor_rollout_ref.rollout.n=$n，
#     但 _validate() 真正读取的是 val_kwargs.n；旧脚本这里本身存在语义错位。
#     本脚本里 N 会被直接用于“每条样本重复跑 N 次”，不再经过 rollout.n / val_kwargs.n 的二次映射
# - MAX_PROMPT_LENGTH:
#     prompt 长度上限
#     对应旧版 data.max_prompt_length / actor_rollout_ref.agent.max_prompt_length
# - MAX_RESPONSE_LENGTH:
#     整条 response 预算上限（含多轮 agent 生成 + tool observation 拼接后的 response 部分）
#     对应旧版 data.max_response_length / actor_rollout_ref.agent.max_response_length
# - MAX_ACTION_LENGTH:
#     单轮模型输出上限
#     对应旧版 actor_rollout_ref.agent.max_action_length
# - MAX_OBS_LENGTH:
#     单轮 tool observation 回填上限
#     对应旧版 actor_rollout_ref.agent.max_obs_length
# - MAX_TURNS:
#     最大 agent 交互轮数
#     对应旧版 actor_rollout_ref.agent.max_turns
# - TEMPERATURE / TOP_P:
#     外部模型采样参数
#     对应旧版 actor_rollout_ref.rollout.val_kwargs.temperature / top_p
# - TOOL_CALL_TIMEOUT:
#     tool server 调用超时
#     对应旧版 actor_rollout_ref.agent.tool_call_timeout
# - TOOL_CALL_MAX_RETRIES:
#     tool server 调用重试次数
#     对应旧版 actor_rollout_ref.agent.tool_call_max_retries
# - MAX_CONCURRENT_TRAJECTORIES:
#     standalone runner 内部并发 trajectory 数
#     对应旧版 actor_rollout_ref.agent.max_concurrent_trajectories
# - FILTER_OVERLONG_PROMPTS_WORKERS:
#     数据集过滤超长 prompt 的并发 worker 数
#     对应旧版 data.filter_overlong_prompts_workers
# - LATENCY_PENALTY_START_STEP / TOOL_PENALTY_START_STEP:
#     reward manager 起效步数；在 standalone eval 里通常保持 0
#     对应旧版 reward_model.reward_kwargs.latency_penalty_start_step /
#     reward_model.reward_kwargs.tool_penalty_start_step
# - TRUST_REMOTE_CODE:
#     是否为 tokenizer / processor 开启 trust_remote_code
#     InternVL 这类模型通常必须打开，否则 AutoProcessor 可能加载失败
# - VLLM_API_KEY:
#     外部 vLLM OpenAI 接口鉴权 token
#     这里默认写死为 token-abc123，并同步导出给 MODEL_API_KEY / OPENAI_API_KEY
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
MODEL_BASE_URL="${MODEL_BASE_URL:-}"
MODEL_NAME="${MODEL_NAME:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct}"
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
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-$((8192 + 8192))}"
MAX_TURNS="${MAX_TURNS:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOOL_CALL_TIMEOUT="${TOOL_CALL_TIMEOUT:-160}"
TOOL_CALL_MAX_RETRIES="${TOOL_CALL_MAX_RETRIES:-4}"
# MAX_CONCURRENT_TRAJECTORIES="${MAX_CONCURRENT_TRAJECTORIES:-2}" # gpt
MAX_CONCURRENT_TRAJECTORIES="${MAX_CONCURRENT_TRAJECTORIES:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PER_WORKER_MAX_CONCURRENCY="${PER_WORKER_MAX_CONCURRENCY:-8}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-32}"
LATENCY_PENALTY_START_STEP="${LATENCY_PENALTY_START_STEP:-0}"
TOOL_PENALTY_START_STEP="${TOOL_PENALTY_START_STEP:-0}"
REWARD_MANAGER="${REWARD_MANAGER:-multimodal_orchestra_eval_relaxed_answer}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
# Set your API key via environment variable or replace with your own key
VLLM_API_KEY="${VLLM_API_KEY:-your-api-key-here}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-console,swanlab}"
PROJECT_NAME="${PROJECT_NAME:-$REWARD_MANAGER}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

export VLLM_API_KEY
export MODEL_API_KEY="${MODEL_API_KEY:-$VLLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$VLLM_API_KEY}"

if [ -z "$VAL_DATA_PATH" ]; then
    echo "VAL_DATA_PATH must be set." >&2
    exit 1
fi

if [ -z "$MODEL_BASE_URL" ]; then
    echo "MODEL_BASE_URL must be set to the external model service base URL." >&2
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME must be set to the external model identifier." >&2
    exit 1
fi

if [ -z "$RUN_ID" ]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)_external_model_eval"
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
        --workers_per_tool 32 &
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
echo "MODEL_BASE_URL: $MODEL_BASE_URL"
echo "MODEL_NAME: $MODEL_NAME"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "TOOL_SERVER_URL: $TOOL_SERVER_URL"
echo "REWARD_MANAGER: $REWARD_MANAGER"
echo "TRUST_REMOTE_CODE: $TRUST_REMOTE_CODE"
echo "LOGGER_BACKENDS: $LOGGER_BACKENDS"
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
    --model-base-url "$MODEL_BASE_URL"
    --model-name "$MODEL_NAME"
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
)

if is_true "$TRUST_REMOTE_CODE"; then
    eval_args+=(--trust-remote-code)
fi

python3 -m verl_tool.eval.external_model_agent_eval "${eval_args[@]}"
JOB_END_EPOCH=$(date +%s)
