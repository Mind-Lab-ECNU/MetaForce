#!/usr/bin/env bash
set -euo pipefail

# Paths
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)"
SERVER_PY="python -m verl_tool.servers.tool_server"
PORT=6665

# 清理函数：使用 fuser 或直接用 ps 杀进程
cleanup_port() {
    echo "Cleaning up port ${PORT}..."
    # 方法1: 使用 fuser (如果有)
    fuser -k ${PORT}/tcp 2>/dev/null || true
    
    # 方法2: 使用 netstat + awk
    netstat -tulpn 2>/dev/null | grep ":${PORT}" | awk '{print $7}' | cut -d'/' -f1 | xargs -r kill -9 2>/dev/null || true
    
    # 方法3: 杀死所有 tool_server 进程
    pkill -9 -f "verl_tool.servers.tool_server" 2>/dev/null || true
    
    sleep 2
}

# 改进的服务器启动检查函数
wait_for_server() {
    local log_file=$1
    local max_wait=30
    local waited=0
    
    echo "Waiting for server to start (max ${max_wait}s)..."
    
    while [ $waited -lt $max_wait ]; do
        # 方法1: 检查端口是否监听
        if netstat -tuln 2>/dev/null | grep -q ":${PORT} "; then
            echo "✓ Server is listening on port ${PORT}"
            sleep 2  # 额外等待确保完全就绪
            return 0
        fi
        
        # 方法2: 检查日志中是否有成功启动的标志
        if grep -q "Uvicorn running on" "$log_file" 2>/dev/null; then
            echo "✓ Server startup detected in logs"
            sleep 2
            return 0
        fi
        
        # 方法3: 尝试连接（不依赖 /health 端点）
        if timeout 1 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
            echo "✓ Server accepting connections on port ${PORT}"
            sleep 2
            return 0
        fi
        
        sleep 1
        waited=$((waited + 1))
        
        # 每5秒显示一次进度
        if [ $((waited % 5)) -eq 0 ]; then
            echo "  Still waiting... (${waited}/${max_wait}s)"
        fi
    done
    
    echo "ERROR: Server failed to start after ${max_wait}s"
    echo "Last 50 lines of log:"
    tail -n 50 "$log_file"
    return 1
}

# 启动前先清理
cleanup_port

# Sync tool, no Ray
echo "========================================"
echo "sync-no-ray: extract_chart_data"
echo "========================================"
${SERVER_PY} --tool_type extract_chart_data --port ${PORT} --workers_per_tool 8 > /tmp/tool_sync.log 2>&1 &
PID_SYNC=$!

if ! wait_for_server /tmp/tool_sync.log; then
    kill ${PID_SYNC} 2>/dev/null || true
    exit 1
fi

python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool model1 --url http://localhost:${PORT}/get_observation
python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool model2 --url http://localhost:${PORT}/get_observation
python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool bad_name --url http://localhost:${PORT}/get_observation || true
python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool bad_function --url http://localhost:${PORT}/get_observation || true

kill ${PID_SYNC} 2>/dev/null || true
cleanup_port

# Sync tool with Ray
echo "========================================"
echo "sync-ray: extract_chart_data"
echo "========================================"
${SERVER_PY} --tool_type extract_chart_data --use_ray True --port ${PORT} --workers_per_tool 8 > /tmp/tool_sync_ray.log 2>&1 &
PID_SYNC_RAY=$!

if ! wait_for_server /tmp/tool_sync_ray.log; then
    kill ${PID_SYNC_RAY} 2>/dev/null || true
    exit 1
fi

python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool model1 --url http://localhost:${PORT}/get_observation
python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool model2 --url http://localhost:${PORT}/get_observation
python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool bad_name --url http://localhost:${PORT}/get_observation || true
python -m verl_tool.servers.tests.test_extract_chart_data_sync_tool bad_function --url http://localhost:${PORT}/get_observation || true

kill ${PID_SYNC_RAY} 2>/dev/null || true
cleanup_port

# Async tool (no Ray)
echo "========================================"
echo "async-no-ray: extract_chart_data_async"
echo "========================================"
${SERVER_PY} --tool_type extract_chart_data_async --port ${PORT} --workers_per_tool 8 > /tmp/tool_async.log 2>&1 &
PID_ASYNC=$!

if ! wait_for_server /tmp/tool_async.log; then
    kill ${PID_ASYNC} 2>/dev/null || true
    exit 1
fi

python -m verl_tool.servers.tests.test_extract_chart_data_async_tool model1 --url http://localhost:${PORT}/get_observation
python -m verl_tool.servers.tests.test_extract_chart_data_async_tool model2 --url http://localhost:${PORT}/get_observation
python -m verl_tool.servers.tests.test_extract_chart_data_async_tool bad_name --url http://localhost:${PORT}/get_observation || true
python -m verl_tool.servers.tests.test_extract_chart_data_async_tool bad_function --url http://localhost:${PORT}/get_observation || true

kill ${PID_ASYNC} 2>/dev/null || true
cleanup_port

echo "========================================"
echo "All tests completed!"
echo "========================================"
