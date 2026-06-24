#!/usr/bin/env bash
# =============================================================================
# Multimodal Orchestra Eval Wrapper with Retry (2 Nodes x 8 GPUs = 16 GPUs)
# =============================================================================
# 启动方式（推荐两台机器都运行 wrapper）:
#
# 1) 一体化模式（head 本机起工具服务）:
#   Head:
#     NODE_RANK=0 MODE=bundled \
#       bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_with_retry.sh
#   Worker:
#     NODE_RANK=1 MASTER_ADDR=<head_ip> MODE=bundled \
#       bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_with_retry.sh
#
# 2) 外置工具模式:
#   Head:
#     NODE_RANK=0 MODE=external TOOL_SERVER_URL=http://<tool_host>:<tool_port>/get_observation \
#       bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_with_retry.sh
#   Worker:
#     NODE_RANK=1 MASTER_ADDR=<head_ip> MODE=external TOOL_SERVER_URL=http://<tool_host>:<tool_port>/get_observation \
#       bash examples/train/multimodal_orchestra/multi_node/eval_2node_16gpu_with_retry.sh
#
# 3) 自动模式（默认）:
#   - 有 TOOL_SERVER_URL -> external
#   - 无 TOOL_SERVER_URL -> bundled
#
# 说明:
#   - 两台机器都建议运行 wrapper。
#   - 只有 NODE_RANK=0 做 Ray 全局健康监控并触发失败重试。
#   - NODE_RANK!=0 仅跟随重启（本地脚本退出后按间隔重启）。
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_RANK=${NODE_RANK:-0}
MODE=${MODE:-auto}
RETRY_SLEEP_SEC=${RETRY_SLEEP_SEC:-30}
HEALTH_CHECK_INTERVAL_SEC=${HEALTH_CHECK_INTERVAL_SEC:-30}
HEALTH_FAIL_THRESHOLD=${HEALTH_FAIL_THRESHOLD:-3}
HEALTH_GRACE_SEC=${HEALTH_GRACE_SEC:-180}
RETRY_LOG_DIR=${RETRY_LOG_DIR:-"$(pwd)/logs/multimodal_orchestra_multinode/retry"}

mkdir -p "$RETRY_LOG_DIR"

BUNDLED_SCRIPT="${SCRIPT_DIR}/eval_2node_16gpu.sh"
EXTERNAL_SCRIPT="${SCRIPT_DIR}/eval_2node_16gpu_no_tool.sh"

resolve_mode() {
    case "$MODE" in
        auto)
            if [ -n "${TOOL_SERVER_URL:-}" ]; then
                echo "external"
            else
                echo "bundled"
            fi
            ;;
        bundled|external)
            echo "$MODE"
            ;;
        *)
            echo "ERROR: MODE must be one of auto|bundled|external, got: $MODE"
            exit 1
            ;;
    esac
}

get_target_script() {
    local resolved_mode=$1
    if [ "$resolved_mode" = "external" ]; then
        if [ -z "${TOOL_SERVER_URL:-}" ]; then
            echo "ERROR: MODE=external requires TOOL_SERVER_URL"
            exit 1
        fi
        echo "$EXTERNAL_SCRIPT"
    else
        echo "$BUNDLED_SCRIPT"
    fi
}

ray_cluster_healthy() {
    local status_output current_gpus

    if ! command -v ray >/dev/null 2>&1; then
        return 1
    fi

    status_output=$(ray status 2>/dev/null || true)
    if [ -z "$status_output" ]; then
        return 1
    fi

    current_gpus=$(printf '%s\n' "$status_output" | sed -nE 's/.*GPU:[[:space:]]*[0-9.]+\/([0-9.]+).*/\1/p' | head -1)
    if [ -z "$current_gpus" ]; then
        current_gpus=$(printf '%s\n' "$status_output" | sed -nE 's/.*[[:space:]]([0-9.]+)\/([0-9.]+)[[:space:]]+GPU.*/\2/p' | head -1)
    fi

    if [ -z "$current_gpus" ]; then
        return 1
    fi

    if [ "${current_gpus%.*}" -le 0 ]; then
        return 1
    fi

    return 0
}

RESOLVED_MODE=$(resolve_mode)
TARGET_SCRIPT=$(get_target_script "$RESOLVED_MODE")

echo "=========================================="
echo "Eval Retry Wrapper"
echo "NODE_RANK: $NODE_RANK"
echo "MODE: $MODE -> $RESOLVED_MODE"
echo "TARGET_SCRIPT: $TARGET_SCRIPT"
echo "RETRY_SLEEP_SEC: $RETRY_SLEEP_SEC"
echo "HEALTH_CHECK_INTERVAL_SEC: $HEALTH_CHECK_INTERVAL_SEC"
echo "HEALTH_FAIL_THRESHOLD: $HEALTH_FAIL_THRESHOLD"
echo "HEALTH_GRACE_SEC: $HEALTH_GRACE_SEC"
echo "RETRY_LOG_DIR: $RETRY_LOG_DIR"
echo "=========================================="

attempt=1
while true; do
    attempt_ts=$(date +%Y%m%d_%H%M%S)
    attempt_log="${RETRY_LOG_DIR}/eval_rank${NODE_RANK}_attempt${attempt}_${attempt_ts}.log"
    attempt_start_epoch=$(date +%s)
    consecutive_health_failures=0
    forced_fail_reason=""

    echo "[WRAPPER] Starting attempt=$attempt at $(date '+%F %T')"
    echo "[WRAPPER] Logging child stdout/stderr to: $attempt_log"

    stdbuf -oL -eL bash "$TARGET_SCRIPT" 2>&1 | stdbuf -oL tee -a "$attempt_log" &
    child_pid=$!
    echo "[WRAPPER] Child PID: $child_pid"

    while kill -0 "$child_pid" 2>/dev/null; do
        sleep "$HEALTH_CHECK_INTERVAL_SEC"

        if [ "$NODE_RANK" -ne 0 ]; then
            continue
        fi

        now_epoch=$(date +%s)
        if [ "$now_epoch" -lt $((attempt_start_epoch + HEALTH_GRACE_SEC)) ]; then
            continue
        fi

        if ray_cluster_healthy; then
            consecutive_health_failures=0
        else
            consecutive_health_failures=$((consecutive_health_failures + 1))
            echo "[WRAPPER][HEAD] Ray health check failed ($consecutive_health_failures/$HEALTH_FAIL_THRESHOLD)" | tee -a "$attempt_log"
            if [ "$consecutive_health_failures" -ge "$HEALTH_FAIL_THRESHOLD" ]; then
                forced_fail_reason="ray_unhealthy"
                echo "[WRAPPER][HEAD] Force-stopping child due to repeated Ray health failures" | tee -a "$attempt_log"
                kill -TERM "$child_pid" 2>/dev/null || true
                sleep 10
                if kill -0 "$child_pid" 2>/dev/null; then
                    kill -KILL "$child_pid" 2>/dev/null || true
                fi
                break
            fi
        fi
    done

    wait "$child_pid"
    child_exit_code=$?

    if [ -n "$forced_fail_reason" ] && [ "$child_exit_code" -eq 0 ]; then
        child_exit_code=1
    fi

    if [ "$child_exit_code" -eq 0 ]; then
        echo "[WRAPPER] Attempt=$attempt succeeded, exiting wrapper."
        exit 0
    fi

    echo "[WRAPPER] Attempt=$attempt failed (exit_code=$child_exit_code reason=${forced_fail_reason:-child_exit_nonzero})."
    echo "[WRAPPER] Sleeping ${RETRY_SLEEP_SEC}s before retry..."
    sleep "$RETRY_SLEEP_SEC"
    attempt=$((attempt + 1))
done

