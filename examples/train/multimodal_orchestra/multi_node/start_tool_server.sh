#!/usr/bin/env bash
# =============================================================================
# Tool Server Startup Script for Multimodal Orchestra
# =============================================================================
# 独立的工具服务器启动脚本，用于多节点训练场景
# 
# 工具服务器是无状态的，多节点训练只需要启动一个实例
# 所有节点通过 HTTP 访问同一个工具服务器
#
# 用法:
#   # 方式1: 使用默认配置启动 (uvicorn --factory 模式)
#   bash start_tool_server.sh
#
#   # 方式2: 指定端口和 workers 数量
#   TOOL_SERVER_PORT=5000 TOOL_SERVER_WORKERS=32 bash start_tool_server.sh
#
#   # 方式3: 使用传统 python -m 启动 (如果 uvicorn/uvloop 有问题)
#   USE_TRADITIONAL=1 bash start_tool_server.sh
#
#   # 方式4: 后台启动
#   bash start_tool_server.sh &
#
# 环境变量:
#   TOOL_TYPE: 工具类型 (默认 multimodal_processor_tool)
#   TOOL_SERVER_HOST: 监听地址 (默认 0.0.0.0)
#   TOOL_SERVER_PORT: 监听端口 (默认 5000)
#   TOOL_SERVER_WORKERS: uvicorn worker 数量 (默认 32)
#   TOOL_SERVER_LOG: 日志文件路径 (默认输出到 stdout)
#   USE_TRADITIONAL: 设为 1 使用传统 python -m 启动方式 (默认 0)
# =============================================================================

set -e

# =============================================================================
# 配置参数
# =============================================================================
TOOL_SERVER_HOST=${TOOL_SERVER_HOST:-"0.0.0.0"}
TOOL_SERVER_PORT=${TOOL_SERVER_PORT:-5000}
TOOL_SERVER_WORKERS=${TOOL_SERVER_WORKERS:-32}
TOOL_SERVER_LOG=${TOOL_SERVER_LOG:-""}
TOOL_TYPE=${TOOL_TYPE:-"multimodal_processor_tool_adapt_skill"}

# 设置工具服务器环境变量 (uvicorn --factory 模式通过环境变量传参)
export VT_TOOL_TYPE="$TOOL_TYPE"
export VT_HOST="$TOOL_SERVER_HOST"
export VT_PORT="$TOOL_SERVER_PORT"
export VT_WORKERS_PER_TOOL="$TOOL_SERVER_WORKERS"
export VT_MAX_CONCURRENT_REQUESTS=1024
export VT_LOG_LEVEL="info"

# 获取本机 IP 用于显示
LOCAL_IP=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "localhost")

echo "=========================================="
echo "Multimodal Orchestra Tool Server"
echo "=========================================="
echo "Host: $TOOL_SERVER_HOST"
echo "Port: $TOOL_SERVER_PORT"
echo "Workers: $TOOL_SERVER_WORKERS"
echo "Tool Type: $TOOL_TYPE"
echo "Local IP: $LOCAL_IP"
echo ""
echo "Tool Server URL for training config:"
echo "  http://${LOCAL_IP}:${TOOL_SERVER_PORT}/get_observation"
echo ""
echo "Health check:"
echo "  curl http://${LOCAL_IP}:${TOOL_SERVER_PORT}/health"
echo "=========================================="

# =============================================================================
# 清理函数
# =============================================================================
cleanup() {
    echo ""
    echo "Shutting down tool server..."
    # 发送 SIGTERM 给子进程
    kill -TERM $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    echo "Tool server stopped."
    exit 0
}

# 捕获退出信号
trap cleanup SIGINT SIGTERM

# =============================================================================
# 启动工具服务器
# =============================================================================
# 启动方式选择:
#   USE_TRADITIONAL=0 (默认): 使用 uvicorn --factory 模式 (推荐，支持多 worker)
#   USE_TRADITIONAL=1: 使用传统 python -m 启动 (如果 uvicorn/uvloop 有问题)
# =============================================================================
USE_TRADITIONAL=${USE_TRADITIONAL:-0}

echo "Starting tool server..."

if [ "$USE_TRADITIONAL" -eq 1 ]; then
    # ----- 方式2: 传统 python -m 启动 (备用) -----
    echo "Using traditional python -m startup mode"
    
    if [ -n "$TOOL_SERVER_LOG" ]; then
        echo "Logging to: $TOOL_SERVER_LOG"
        mkdir -p $(dirname "$TOOL_SERVER_LOG")
        
        python -m verl_tool.servers.serve \
            --host $TOOL_SERVER_HOST \
            --port $TOOL_SERVER_PORT \
            --tool_type "$TOOL_TYPE" \
            --workers_per_tool "$TOOL_SERVER_WORKERS" \
            --max_concurrent_requests 1024 \
            > "$TOOL_SERVER_LOG" 2>&1 &
        server_pid=$!
    else
        python -m verl_tool.servers.serve \
            --host $TOOL_SERVER_HOST \
            --port $TOOL_SERVER_PORT \
            --tool_type "$TOOL_TYPE" \
            --workers_per_tool "$TOOL_SERVER_WORKERS" \
            --max_concurrent_requests 1024 &
        server_pid=$!
    fi
else
    # ----- 方式1: uvicorn --factory 模式 (推荐) -----
    echo "Using uvicorn --factory startup mode"
    
    if [ -n "$TOOL_SERVER_LOG" ]; then
        echo "Logging to: $TOOL_SERVER_LOG"
        mkdir -p $(dirname "$TOOL_SERVER_LOG")
        
        uvicorn verl_tool.servers.tool_server:create_app \
            --factory \
            --host $TOOL_SERVER_HOST \
            --port $TOOL_SERVER_PORT \
            --workers $TOOL_SERVER_WORKERS \
            --loop uvloop \
            --http httptools \
            > "$TOOL_SERVER_LOG" 2>&1 &
        server_pid=$!
    else
        uvicorn verl_tool.servers.tool_server:create_app \
            --factory \
            --host $TOOL_SERVER_HOST \
            --port $TOOL_SERVER_PORT \
            --workers $TOOL_SERVER_WORKERS \
            --loop uvloop \
            --http httptools &
        server_pid=$!
    fi
fi

echo "Tool server started with PID: $server_pid"

# =============================================================================
# 等待服务器启动并验证
# =============================================================================
echo "Waiting for server to be ready..."
max_wait=30
wait_count=0

while [ $wait_count -lt $max_wait ]; do
    if curl -s "http://localhost:${TOOL_SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server is ready!"
        break
    fi
    
    # 检查进程是否还在运行
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "ERROR: Tool server process died!"
        if [ -n "$TOOL_SERVER_LOG" ]; then
            echo "Check logs at: $TOOL_SERVER_LOG"
            tail -20 "$TOOL_SERVER_LOG" 2>/dev/null || true
        fi
        exit 1
    fi
    
    sleep 1
    wait_count=$((wait_count + 1))
done

if [ $wait_count -ge $max_wait ]; then
    echo "WARNING: Timeout waiting for server, but process is running. Continuing..."
fi

# =============================================================================
# 保持运行
# =============================================================================
echo ""
echo "=========================================="
echo "Tool server is running"
echo "Press Ctrl+C to stop"
echo "=========================================="

# 等待子进程
wait $server_pid
