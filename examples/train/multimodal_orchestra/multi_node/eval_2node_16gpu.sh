#!/usr/bin/env bash
# =============================================================================
# Multimodal Orchestra Eval Script - Multi Node (2 Nodes x 8 GPUs = 16 GPUs)
# =============================================================================
# 多节点评测脚本，使用 Ray 集群管理，uvicorn 启动工具服务器
#
# 启动方式:
#   1. 在 head 节点运行此脚本，并设置 NODE_RANK=0
#   2. 在 worker 节点运行此脚本，并设置 NODE_RANK=1, MASTER_ADDR=<head_ip>
#
# 环境变量:
#   NODE_RANK: 节点编号 (0=head, 1=worker)
#   MASTER_ADDR: head 节点 IP 地址 (worker 节点需要设置)
#   MASTER_PORT: Ray head 端口 (默认 6379)
#   TOOL_SERVER_HOST: 工具服务器主机 (默认使用 head 节点)
#   TOOL_SERVER_PORT: 工具服务器端口 (默认 5000)
#
# 示例:
#   # 在 head 节点 (假设 IP 为 192.168.1.100):
#   NODE_RANK=0 bash '/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu.sh'
#
#   # 在 worker 节点:
#   NODE_RANK=1 MASTER_ADDR=252.0.137.164 bash '/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu.sh'
# =============================================================================

set -x

SCRIPT_START_EPOCH=$(date +%s)
JOB_START_EPOCH=0
JOB_END_EPOCH=0

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
    print_timing_summary
    echo "[EXIT] exit_code=$exit_code"
}

trap on_exit EXIT

# =============================================================================
# 节点配置
# =============================================================================
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}


# 获取本机 IP
LOCAL_IP=$(hostname -i | awk '{print $1}')

# 如果是 head 节点，MASTER_ADDR 设为本机 IP
if [ "$NODE_RANK" -eq 0 ]; then
    MASTER_ADDR=${MASTER_ADDR:-$LOCAL_IP}
fi

# 检查 worker 节点是否设置了 MASTER_ADDR
if [ "$NODE_RANK" -ne 0 ] && [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: Worker node requires MASTER_ADDR to be set!"
    echo "Usage: NODE_RANK=1 MASTER_ADDR=<head_ip> bash eval_2node_16gpu.sh"
    exit 1
fi

echo "=========================================="
echo "Node Configuration"
echo "NODE_RANK: $NODE_RANK"
echo "LOCAL_IP: $LOCAL_IP"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "=========================================="

# =============================================================================
# 基础路径配置
# =============================================================================
ROOT_DIR="$(pwd)"
DATA_ROOT="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category"

# =============================================================================
# SwanLab 本地配置
# =============================================================================
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${ROOT_DIR}/swanlog_output

# =============================================================================
# 评测数据配置 (运行前直接在这里填写)
# =============================================================================
EVAL_TRAIN_DATA_PATH=""
EVAL_DATA_PATH=""
EVAL_TOOL_VARIANT=""

if [ -z "$EVAL_TRAIN_DATA_PATH" ]; then
    echo "EVAL_TRAIN_DATA_PATH must be set to a non-empty parquet path because the trainer always builds train_dataset." >&2
    exit 1
fi

if [ -z "$EVAL_DATA_PATH" ]; then
    echo "EVAL_DATA_PATH must be set to the evaluation parquet path." >&2
    exit 1
fi

if [ -z "$EVAL_TOOL_VARIANT" ]; then
    echo "EVAL_TOOL_VARIANT must be set to all, id, or ood." >&2
    exit 1
fi

# val_only eval still expects train_files in trainer config.
train_data="$EVAL_TRAIN_DATA_PATH"
val_data="$EVAL_DATA_PATH"

# =============================================================================
# 模型和训练参数配置
# =============================================================================
model_name="${MODEL_NAME:-/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct}"
rl_alg=grpo
n_gpus_per_node=8
n_nodes=2
n=8
batch_size=256  # 多节点可以使用更大的 batch size
ppo_mini_batch_size=256

# 序列长度配置
max_prompt_length=$((8192*3))
max_response_length=$((8192*2 + 4096))
max_action_length=8192
max_obs_length=8192
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# 采样参数
temperature=1.0
top_p=1.0

# Agent配置
enable_agent=True
strategy="fsdp2"
action_stop_tokens='</tool_call>'
max_turns=10
TOOL_CALL_TIMEOUT=${TOOL_CALL_TIMEOUT:-100}

# KL和熵系数
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6

# Reward Manager配置
reward_manager=multimodal_orchestra
LATENCY_PENALTY_START_STEP=${LATENCY_PENALTY_START_STEP:-0}
TOOL_PENALTY_START_STEP=${TOOL_PENALTY_START_STEP:-0}

# GPU和内存配置
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=4
gpu_memory_utilization=0.8
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=4
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
max_num_batched_tokens=10000

# 实验名称
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
default_base_model="/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-VL-8B-Instruct"
if [ -n "${MODEL_TAG:-}" ]; then
    model_tag_raw="$MODEL_TAG"
else
    if [ "$model_name" = "$default_base_model" ]; then
        model_tag_raw="exp0-base"
    else
        model_tag_raw="model"
        if [[ "$model_name" =~ exp([0-9]+)_ ]]; then
            model_tag_raw="exp${BASH_REMATCH[1]}"
        fi
        if [[ "$model_name" =~ global_step_([0-9]+) ]]; then
            model_tag_raw="${model_tag_raw}-gs${BASH_REMATCH[1]}"
        fi
    fi
fi
model_tag=$(echo "$model_tag_raw" | tr '/: ' '___' | tr -cd '[:alnum:]_.-')
if [ -z "$model_tag" ]; then
    model_tag="model"
fi
run_name_postfix="multimodal-orchestra-2node-adapt-skills-${EVAL_TOOL_VARIANT}-${model_tag}"

if [ "$enable_agent" = "True" ]; then
    run_name="eval-${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-nodes${n_nodes}-${run_name_postfix}"
else
    run_name="eval-${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-nodes${n_nodes}-${run_name_postfix}"
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-${RUN_TS}_${run_name}}
export VERL_RUN_ID=$RUN_ID
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# Eval is fully decoupled from training:
# - trainer.val_only=True runs validation-only and exits.
# - If SKILL_STORE_DIR is not provided, eval starts from an empty skill store.

# =============================================================================
# 工具服务器配置 (只在 head 节点启动)
# =============================================================================
TOOL_SERVER_HOST=${TOOL_SERVER_HOST:-$MASTER_ADDR}
TOOL_SERVER_PORT=$(shuf -i 30000-31000 -n 1)
tool_server_url="http://${TOOL_SERVER_HOST}:${TOOL_SERVER_PORT}/get_observation"
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
        echo "Invalid EVAL_TOOL_VARIANT: ${EVAL_TOOL_VARIANT}. Use all, id, or ood." >&2
        exit 1
        ;;
esac

# =============================================================================
# 日志目录
# =============================================================================
LOG_DIR="$(pwd)/logs/multimodal_orchestra_multinode"
RECORD_ROOT="${ROOT_DIR}/verl_step_records"
TOOL_LOG_DIR="${RECORD_ROOT}/tool_logs/val/${RUN_ID}"
DEFAULT_SKILL_STORE_DIR="${RECORD_ROOT}/skills/val/${RUN_ID}"
SKILL_STORE_DIR="${SKILL_STORE_DIR:-$DEFAULT_SKILL_STORE_DIR}"
VAL_DATA_DIR="${RECORD_ROOT}/validation_data/val/${RUN_ID}"

# =============================================================================
# Head 节点操作
# =============================================================================
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=========================================="
    echo "Starting as HEAD node"
    echo "=========================================="

    mkdir -p "$LOG_DIR" "$TOOL_LOG_DIR" "$SKILL_STORE_DIR" "$VAL_DATA_DIR"
    export VERL_SKILL_STORE_DIR="$SKILL_STORE_DIR"

    action_stop_tokens_file="$(pwd)/tmp/action_stop_tokens_multinode_$$"
    mkdir -p "$(dirname "$action_stop_tokens_file")"
    echo -e -n "$action_stop_tokens" > "$action_stop_tokens_file"
    echo "action_stop_tokens_file=$action_stop_tokens_file"

    # 启动工具服务器 (使用 uvicorn)
    echo "Starting Tool Server..."
    echo "Host: $TOOL_SERVER_HOST"
    echo "Port: $TOOL_SERVER_PORT"
    echo "URL: $tool_server_url"
    echo "Eval Tool Variant: $EVAL_TOOL_VARIANT"
    echo "Tool Type: $tool_type"
    echo "Eval Train Data: $EVAL_TRAIN_DATA_PATH"
    echo "Eval Data: $EVAL_DATA_PATH"

    # =========================================================================
    # 工具服务器启动方式选择
    # =========================================================================
    # 方式1 (默认): 使用 uvicorn --factory 模式 (推荐，支持多 worker)
    # 方式2 (备用): 使用传统 python -m 启动 (如果 uvicorn/uvloop 有问题)
    #
    # 如需切换到传统启动方式，注释掉方式1的代码，取消方式2的注释
    # =========================================================================

    # ----- 方式1: uvicorn --factory 模式 (推荐) -----
    # 设置工具服务器环境变量 (uvicorn --factory 模式通过环境变量传参)
    # export VT_TOOL_TYPE="$tool_type"
    # export VT_HOST="0.0.0.0"
    # export VT_PORT="$TOOL_SERVER_PORT"
    # export VT_WORKERS_PER_TOOL=32
    # export VT_MAX_CONCURRENT_REQUESTS=1024
    # export VT_LOG_LEVEL="info"

    # uvicorn verl_tool.servers.tool_server:create_app \
    #     --factory \
    #     --host 0.0.0.0 \
    #     --port $TOOL_SERVER_PORT \
    #     --workers 32 \
    #     --loop uvloop \
    #     --http httptools \
    #     > $LOG_DIR/tool_server.log 2>&1 &
    # tool_server_pid=$!

    # ----- 方式2: 传统 python -m 启动 (备用) -----
    # 如果 uvicorn/uvloop 启动失败，取消以下注释使用传统方式
    python -m verl_tool.servers.serve \
        --host 0.0.0.0 \
        --port $TOOL_SERVER_PORT \
        --tool_type "$tool_type" \
        --workers_per_tool 16 \
        --max_concurrent_requests 256 \
        > $LOG_DIR/tool_server.log 2>&1 &
    tool_server_pid=$!

    # 等待工具服务器启动
    echo "Waiting for tool server to start..."
    sleep 10

    if ! kill -0 $tool_server_pid 2>/dev/null; then
        echo "ERROR: Tool server failed to start! Check $LOG_DIR/tool_server.log"
        exit 1
    fi
    echo "Tool server (pid=$tool_server_pid) started successfully"

    # 启动 Ray head 节点
    echo "Starting Ray head node..."
    ray stop --force 2>/dev/null || true
    sleep 2

    ray start --head \
        --node-ip-address=$LOCAL_IP \
        --port=$MASTER_PORT \
        --dashboard-port=$DASHBOARD_PORT \
        --num-cpus=$(nproc) \
        --num-gpus=$n_gpus_per_node \
        --disable-usage-stats

    echo "Ray head node started at $LOCAL_IP:$MASTER_PORT"
    echo "Ray dashboard at http://$LOCAL_IP:$DASHBOARD_PORT"

    get_current_gpus() {
        local json_gpus status_output current_gpus

        json_gpus=$(ray status --format=json 2>/dev/null | python3 -c '
import json
import sys

best = 0.0
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)

def walk(value):
    global best
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "GPU" and isinstance(item, (int, float)):
                best = max(best, float(item))
            walk(item)
    elif isinstance(value, list):
        for item in value:
            walk(item)

walk(data)
print(int(best))
' 2>/dev/null || true)

        if [ -n "$json_gpus" ] && [[ "$json_gpus" =~ ^[0-9]+$ ]]; then
            echo "$json_gpus"
            return
        fi

        status_output=$(ray status 2>/dev/null || true)
        current_gpus=$(printf '%s\n' "$status_output" | sed -nE 's/.*GPU:[[:space:]]*[0-9.]+\/([0-9.]+).*/\1/p' | head -1)
        if [ -z "$current_gpus" ]; then
            current_gpus=$(printf '%s\n' "$status_output" | sed -nE 's/.*[[:space:]]([0-9.]+)\/([0-9.]+)[[:space:]]+GPU.*/\2/p' | head -1)
        fi

        if [ -n "$current_gpus" ]; then
            echo "${current_gpus%.*}"
        else
            echo "0"
        fi
    }

    # 等待所有 worker 节点加入
    echo "Waiting for worker nodes to join..."
    expected_gpus=$((n_gpus_per_node * n_nodes))
    max_wait=40
    wait_count=0

    while [ $wait_count -lt $max_wait ]; do
        current_gpus=$(get_current_gpus)

        if [ "$current_gpus" -ge "$expected_gpus" ] 2>/dev/null; then
            echo "All $expected_gpus GPUs are connected!"
            break
        fi

        echo "Waiting for GPUs... Current: $current_gpus / $expected_gpus"
        sleep 5
        wait_count=$((wait_count + 5))
    done

    if [ $wait_count -ge $max_wait ]; then
        echo "WARNING: Timeout waiting for all nodes. Proceeding with available GPUs."
    fi

    ray status

    # 提交评测任务
    echo "=========================================="
    echo "Submitting Eval Job (val-only)"
    echo "Run Name: $run_name"
    echo "Run ID: $RUN_ID"
    echo "Skill Store: $SKILL_STORE_DIR"
    echo "Eval Tool Variant: $EVAL_TOOL_VARIANT"
    echo "Eval Train Data Path: $EVAL_TRAIN_DATA_PATH"
    echo "Eval Data Path: $EVAL_DATA_PATH"
    echo "=========================================="

    JOB_START_EPOCH=$(date +%s)
    job_exit_code=0
    RAY_ADDRESS="http://127.0.0.1:$DASHBOARD_PORT" \
    ray job submit --runtime-env=verl_tool/trainer/runtime_env.yaml \
        -- \
        PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
        algorithm.adv_estimator=$rl_alg \
        data.train_files="$train_data" \
        data.val_files="$val_data" \
        data.train_batch_size=$batch_size \
        data.val_batch_size=512 \
        data.dataloader_num_workers=8 \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.return_raw_chat=True \
        data.filter_overlong_prompts_workers=32 \
        data.filter_overlong_prompts=True \
        data.truncation='right' \
        reward_model.reward_manager=$reward_manager \
        +reward_model.reward_kwargs.latency_penalty_start_step=$LATENCY_PENALTY_START_STEP \
        +reward_model.reward_kwargs.tool_penalty_start_step=$TOOL_PENALTY_START_STEP \
        reward_model.launch_reward_fn_async=True \
        actor_rollout_ref.model.path=$model_name \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=$lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.strategy=$strategy \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
        actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
        actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        actor_rollout_ref.agent.enable_agent=$enable_agent \
        actor_rollout_ref.agent.tool_call_timeout=$TOOL_CALL_TIMEOUT \
        actor_rollout_ref.agent.tool_server_url=$tool_server_url \
        actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
        actor_rollout_ref.agent.max_response_length=$max_response_length \
        actor_rollout_ref.agent.max_start_length=$max_prompt_length \
        actor_rollout_ref.agent.max_obs_length=$max_obs_length \
        actor_rollout_ref.agent.max_turns=$max_turns \
        actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
        actor_rollout_ref.agent.mask_observations=$mask_observations \
        actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
        actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
        actor_rollout_ref.agent.max_action_length=$max_action_length \
        actor_rollout_ref.agent.max_concurrent_trajectories=256 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
        actor_rollout_ref.rollout.temperature=$temperature \
        actor_rollout_ref.rollout.top_p=$top_p \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.n=$n \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.rollout.max_num_seqs=512 \
        actor_rollout_ref.rollout.mode=$rollout_mode \
        actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
        actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        critic.optim.lr=1e-5 \
        critic.strategy=$strategy \
        critic.model.path=$model_name \
        critic.model.fsdp_config.fsdp_size=$fsdp_size \
        critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        trainer.logger=['console','swanlab'] \
        trainer.project_name=$reward_manager \
        trainer.experiment_name=$run_name \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.default_hdfs_dir=null \
        trainer.resume_mode=disable \
        trainer.n_gpus_per_node=$n_gpus_per_node \
        +trainer.tool_log_dir=$TOOL_LOG_DIR \
        +trainer.skill_store_dir=$SKILL_STORE_DIR \
        trainer.validation_data_dir=$VAL_DATA_DIR \
        trainer.nnodes=$n_nodes \
        +trainer.remove_previous_ckpt_in_save=True \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=100
    job_exit_code=$?
    JOB_END_EPOCH=$(date +%s)

    # 清理
    if [ "$job_exit_code" -eq 0 ]; then
        echo "Evaluation completed successfully. Cleaning up..."
    else
        echo "Evaluation failed with exit code $job_exit_code. Cleaning up..."
    fi

    kill -9 $tool_server_pid 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    [ -n "${action_stop_tokens_file:-}" ] && rm -f "$action_stop_tokens_file"
    echo "Head node cleanup done!"
    exit $job_exit_code

# =============================================================================
# Worker 节点操作
# =============================================================================
else
    echo "=========================================="
    echo "Starting as WORKER node (rank=$NODE_RANK)"
    echo "Connecting to head at $MASTER_ADDR:$MASTER_PORT"
    echo "=========================================="

    # 停止已有的 Ray 进程
    ray stop --force 2>/dev/null || true
    sleep 2

    # 加入 Ray 集群
    ray start \
        --address=$MASTER_ADDR:$MASTER_PORT \
        --num-cpus=$(nproc) \
        --num-gpus=$n_gpus_per_node \
        --disable-usage-stats \
        --block &
    ray_pid=$!

    echo "Worker node joined the cluster. Waiting for training to complete..."
    echo "You can monitor the training from the head node."

    # 等待 Ray 进程结束
    wait $ray_pid
    worker_exit_code=$?

    # 清理
    echo "Worker node cleanup..."
    ray stop --force 2>/dev/null || true
    echo "Worker node done!"
    exit $worker_exit_code
fi
