#!/usr/bin/env bash
# =============================================================================
# Multimodal Orchestra Training Script - Single Node (4 GPUs)
# =============================================================================
# 单节点4卡训练脚本，使用 uvicorn 启动工具服务器
# 
# 用法:
#   cd /path/to/verl_duo
#   bash examples/train/multimodal_orchestra/single_node/train_4gpu.sh
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

export HF_DATASETS_CACHE="/dev/shm/hf_cache_$$"
export HF_HOME="/dev/shm/hf_home_$$"
mkdir -p $HF_DATASETS_CACHE $HF_HOME

# =============================================================================
# 基础路径配置
# =============================================================================
ROOT_DIR="$(pwd)"
DATA_ROOT="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category"
export CORRECT_AND_FORMAT_ONLY_ACTOR_STEPS=0
# =============================================================================
# SwanLab 本地配置
# =============================================================================
export SWANLAB_MODE=local
export SWANLAB_LOG_DIR=${ROOT_DIR}/swanlog_output

# =============================================================================
# 训练数据配置 (20个数据集)
# =============================================================================
# ID核心 (10个，训练+测试)
# train_data_id_core=(
#     "${DATA_ROOT}/chart/ChartQA_2000/train.parquet"
#     "${DATA_ROOT}/chart/PlotQA_2000/train.parquet"
#     "${DATA_ROOT}/diagram/AI2D_2000/train.parquet"
#     "${DATA_ROOT}/geospatial/MapQA_2000/train.parquet"
#     "${DATA_ROOT}/math/geos_processed_2000/train.parquet"
#     "${DATA_ROOT}/math/unigeo_calculation_2000/train.parquet"
#     "${DATA_ROOT}/math/GEOQA_2000/train.parquet"
#     "${DATA_ROOT}/math/geometry3k_2000/train.parquet"
#     "${DATA_ROOT}/science/ScienceQA_2000/train.parquet"
#     "${DATA_ROOT}/spatial/CLEVR_2000/train.parquet"
# )

# ID辅助 (10个，仅训练)
# train_data_id_aux=(
#     "${DATA_ROOT}/add/caption/LocalizedNarratives_2000/train.parquet"
#     "${DATA_ROOT}/add/chart/DVQA_2000/train.parquet"
#     "${DATA_ROOT}/add/code/WebSight_2000/train.parquet"
#     "${DATA_ROOT}/add/diagram/DiagramImageToText_2000/train.parquet"
#     "${DATA_ROOT}/general/AOKVQA_2000/train.parquet"
#     "${DATA_ROOT}/add/general/VQAv2_2000/train.parquet"
#     "${DATA_ROOT}/add/math/InterGPS_2000/train.parquet"
#     "${DATA_ROOT}/add/ocr/TextVQA_2000/train.parquet"
#     "${DATA_ROOT}/add/table/TATQA_2000/train.parquet"
#     "${DATA_ROOT}/doc/InfographicVQA_2000/train.parquet"
# )

# 训练数据配置 (运行前直接在这里填写)
# =============================================================================
TRAIN_DATA_PATH='/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/train_merge_v6_tool_id/train_merge_v6_tool_id.parquet'
VAL_DATA_PATH='/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/train_merge_v6_tool_id/train_merge_v6_tool_id.parquet'

if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "TRAIN_DATA_PATH must be set to the training parquet path." >&2
    exit 1
fi

if [ -z "$VAL_DATA_PATH" ]; then
    echo "VAL_DATA_PATH must be set to a non-empty parquet path because the trainer always builds val_dataset." >&2
    exit 1
fi

train_data="$TRAIN_DATA_PATH"

# =============================================================================
# 验证/测试数据配置 (20个数据集)
# =============================================================================
# ID测试 (10个)
# val_data_id=(
#     "${DATA_ROOT}/chart/ChartQA_2000/test.parquet"
#     "${DATA_ROOT}/chart/PlotQA_2000/test.parquet"
#     "${DATA_ROOT}/diagram/AI2D_2000/test.parquet"
#     "${DATA_ROOT}/geospatial/MapQA_2000/test.parquet"
#     "${DATA_ROOT}/math/geos_processed_2000/test.parquet"
#     "${DATA_ROOT}/math/unigeo_calculation_2000/test.parquet"
#     "${DATA_ROOT}/math/GEOQA_2000/test.parquet"
#     "${DATA_ROOT}/math/geometry3k_2000/test.parquet"
#     "${DATA_ROOT}/science/ScienceQA_2000/test.parquet"
#     "${DATA_ROOT}/spatial/CLEVR_2000/test.parquet"
# )

# OOD测试 (8个)
# val_data_ood=(
#     "${DATA_ROOT}/diagram/ICON-QA_2000/test.parquet"
#     "${DATA_ROOT}/math/CLEVR_MATH_2000/addition/test.parquet"
#     "${DATA_ROOT}/math/CLEVR_MATH_2000/subtraction/test.parquet"
#     "${DATA_ROOT}/math/CLEVR_MATH_2000/subtraction_multihop/test.parquet"
#     "${DATA_ROOT}/math/CLEVR_MATH_2000/adversarial/test.parquet"
#     "${DATA_ROOT}/math/MathVision_2000/test.parquet"
#     "${DATA_ROOT}/math/MathVista_2000/testmini.parquet"
#     "${DATA_ROOT}/doc/DocVQA_2000/test.parquet"
# )

# 验证/测试数据配置 (运行前直接在这里填写)
# =============================================================================
val_data="$VAL_DATA_PATH"
# =============================================================================
# 模型和训练参数配置
# =============================================================================
model_name="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/exp13-skill-reward-lr1e-6-step80"
rl_alg=grpo
n_gpus_per_node=4
n_nodes=1
n=8
batch_size=128
ppo_mini_batch_size=128

# 序列长度配置
max_prompt_length=$((8192*3))
max_response_length=$((8192*3))
max_action_length=4096
max_obs_length=$((8192 + 1024))
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# 采样参数
temperature=1.0
top_p=1.0

# Agent配置
enable_agent=True
strategy="fsdp2"
action_stop_tokens='</tool_call>'
max_turns=10

# KL和熵系数
kl_loss_coef=0.002
kl_coef=0
entropy_coeff=0.002
kl_loss_type=low_var_kl
lr=1e-6
correct_and_format_only_actor_steps=${CORRECT_AND_FORMAT_ONLY_ACTOR_STEPS:-0}
clean_up_skill_step=${CLEAN_UP_SKILL_STEP:-20}

# Reward Manager配置 - 使用 multimodal_orchestra
reward_manager=multimodal_orchestra
LATENCY_PENALTY_START_STEP=${LATENCY_PENALTY_START_STEP:-0}
TOOL_PENALTY_START_STEP=${TOOL_PENALTY_START_STEP:-0}

# GPU和内存配置
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=4
gpu_memory_utilization=0.7
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=2
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
max_num_batched_tokens=$((12048))

# 实验名称
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="skill-reward"

if [ "$enable_agent" = "True" ]; then
    run_name="exp15_${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-lr${lr}-${run_name_postfix}-c_and_f_s-${correct_and_format_only_actor_steps}-cu_s-${clean_up_skill_step}-l_s-${LATENCY_PENALTY_START_STEP}-t_s-${TOOL_PENALTY_START_STEP}-kl_l-${kl_loss_coef}"
else
    run_name="exp15_${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-lr${lr}-${run_name_postfix}-c_and_f_s-${correct_and_format_only_actor_steps}-cu_s-${clean_up_skill_step}-l_s-${LATENCY_PENALTY_START_STEP}-t_s-${TOOL_PENALTY_START_STEP}-kl_l-${kl_loss_coef}"
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-${RUN_TS}_${run_name}}
export VERL_RUN_ID=$RUN_ID
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# Train and eval are intentionally decoupled:
# - test_freq=-1 disables validation during training.
# - VAL_BEFORE_TRAIN is optional and only used for baseline probing.
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

RECORD_ROOT="${ROOT_DIR}/verl_step_records"
TOOL_LOG_DIR="${RECORD_ROOT}/tool_logs/train/${RUN_ID}"
DEFAULT_SKILL_STORE_DIR="${RECORD_ROOT}/skills/train/${RUN_ID}"
SKILL_STORE_DIR="${SKILL_STORE_DIR:-$DEFAULT_SKILL_STORE_DIR}"
CHECKPOINT_DIR="${RECORD_ROOT}/checkpoint/train/${RUN_ID}"
ROLLOUT_DATA_DIR="${RECORD_ROOT}/rollout/train/${RUN_ID}"

export VERL_SKILL_STORE_DIR="$SKILL_STORE_DIR"
mkdir -p "$TOOL_LOG_DIR" "$SKILL_STORE_DIR" "$CHECKPOINT_DIR" "$ROLLOUT_DATA_DIR"

# =============================================================================
# 创建 action_stop_tokens 临时文件
# =============================================================================
action_stop_tokens_file="$(pwd)/tmp/action_stop_tokens_$$"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# =============================================================================
# 启动工具服务器 (使用 uvicorn)
# =============================================================================
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
tool_type="multimodal_processor_tool_adapt_skill_id"

echo "=========================================="
echo "Starting Tool Server"
echo "Host: $host"
echo "Port: $port"
echo "URL: $tool_server_url"
echo "Tool Type: $tool_type"
echo "Train Data: $train_data"
echo "Val Data: $val_data"
echo "=========================================="

# =============================================================================
# 工具服务器启动方式选择
# =============================================================================
# 方式1 (默认): 使用 uvicorn --factory 模式 (推荐，支持多 worker)
# 方式2 (备用): 使用传统 python -m 启动 (如果 uvicorn/uvloop 有问题)
#
# 如需切换到传统启动方式，注释掉方式1的代码，取消方式2的注释
# =============================================================================

# ----- 方式1: uvicorn --factory 模式 (推荐) -----
# 设置工具服务器环境变量 (uvicorn --factory 模式通过环境变量传参)
# export VT_TOOL_TYPE="$tool_type"
# export VT_HOST="$host"
# export VT_PORT="$port"
# export VT_WORKERS_PER_TOOL=32
# export VT_MAX_CONCURRENT_REQUESTS=1024
# export VT_LOG_LEVEL="info"

# # 使用 uvicorn 启动工具服务器 (CPU 资源多，使用 32 workers)
# uvicorn verl_tool.servers.tool_server:create_app \
#     --factory \
#     --host $host \
#     --port $port \
#     --workers 32 \
#     --loop uvloop \
#     --http httptools &
# server_pid=$!

# ----- 方式2: 传统 python -m 启动 (备用) -----
# 如果 uvicorn/uvloop 启动失败，取消以下注释使用传统方式
python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "$tool_type" \
    --workers_per_tool 32 &
server_pid=$!

# 等待服务器启动 (HTTP 健康探测，最多等 120 秒)
echo "Waiting for tool server to start..."
MAX_HEALTH_WAIT=120
HEALTH_WAIT=0
while [ $HEALTH_WAIT -lt $MAX_HEALTH_WAIT ]; do
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "ERROR: Tool server process died!"
        exit 1
    fi
    # 尝试 HTTP 健康检查
    if curl -s -o /dev/null -w '%{http_code}' "http://$host:$port/health" 2>/dev/null | grep -q '200'; then
        echo "Tool server (pid=$server_pid) is healthy and ready at $tool_server_url"
        break
    fi
    echo "  Waiting for tool server... (${HEALTH_WAIT}s / ${MAX_HEALTH_WAIT}s)"
    sleep 5
    HEALTH_WAIT=$((HEALTH_WAIT + 5))
done

if [ $HEALTH_WAIT -ge $MAX_HEALTH_WAIT ]; then
    echo "WARNING: Tool server health check timed out after ${MAX_HEALTH_WAIT}s, proceeding anyway..."
fi

# =============================================================================
# 启动训练
# =============================================================================
echo "=========================================="
echo "Starting Training"
echo "Run Name: $run_name"
echo "Run ID: $RUN_ID"
echo "Train Data Path: $train_data"
echo "Validation During Train: disabled (test_freq=-1)"
echo "Optional Baseline Before Train: VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN"
echo "Val Data Path: $val_data"
echo "=========================================="

JOB_START_EPOCH=$(date +%s)
job_exit_code=0
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=512 \
    data.dataloader_num_workers=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.return_raw_chat=True \
    data.filter_overlong_prompts_workers=32 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    +reward_model.reward_kwargs.latency_penalty_start_step=$LATENCY_PENALTY_START_STEP \
    +reward_model.reward_kwargs.tool_penalty_start_step=$TOOL_PENALTY_START_STEP \
    +reward_model.reward_kwargs.skill_reward_enabled=True \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
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
    actor_rollout_ref.agent.tool_call_timeout=160 \
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
    actor_rollout_ref.agent.max_concurrent_trajectories=64 \
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
    +algorithm.correct_and_format_only_actor_steps=$correct_and_format_only_actor_steps \
    +trainer.clean_up_skill_step=$clean_up_skill_step \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    +trainer.tool_log_dir=$TOOL_LOG_DIR \
    +trainer.skill_store_dir=$SKILL_STORE_DIR \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=43
job_exit_code=$?
JOB_END_EPOCH=$(date +%s)

# =============================================================================
# 清理
# =============================================================================
if [ "$job_exit_code" -eq 0 ]; then
    echo "Training completed successfully. Cleaning up..."
else
    echo "Training failed with exit code $job_exit_code. Cleaning up..."
fi
kill -9 $server_pid 2>/dev/null || true
rm -f $action_stop_tokens_file
rm -rf $HF_DATASETS_CACHE $HF_HOME
echo "Done!"
exit $job_exit_code
