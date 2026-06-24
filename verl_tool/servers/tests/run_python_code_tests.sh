#!/usr/bin/env bash
set -euo pipefail

# Paths
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)"
SERVER_PY="python -m verl_tool.servers.tool_server"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill ${PID_SYNC:-} 2>/dev/null || true
    kill ${PID_SYNC_RAY:-} 2>/dev/null || true
    kill ${PID_ASYNC:-} 2>/dev/null || true
}
trap cleanup EXIT

# Function to wait for server
wait_for_server() {
    local port=$1
    local max_attempts=30
    echo "Waiting for server on port $port to start..."
    for i in $(seq 1 $max_attempts); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        if [ $i -eq $max_attempts ]; then
            echo "Server failed to start after $max_attempts attempts"
            return 1
        fi
        echo "Attempt $i/$max_attempts..."
        sleep 1
    done
}

# Sync tool, no Ray
echo "=========================================="
echo "sync-no-ray: python_code_sync"
echo "=========================================="
${SERVER_PY} --tool_type python_code_sync --port 6661 --workers_per_tool 8 > /tmp/tool_python_code_sync.log 2>&1 &
PID_SYNC=$!
wait_for_server 6661 || exit 1

echo "Testing basic_execution..."
python -m verl_tool.servers.tests.test_python_code_sync_tool basic_execution --url http://localhost:6661/get_observation

echo "Testing math_calculation..."
python -m verl_tool.servers.tests.test_python_code_sync_tool math_calculation --url http://localhost:6661/get_observation

echo "Testing with_imports..."
python -m verl_tool.servers.tests.test_python_code_sync_tool with_imports --url http://localhost:6661/get_observation

echo "Testing syntax_error..."
python -m verl_tool.servers.tests.test_python_code_sync_tool syntax_error --url http://localhost:6661/get_observation

echo "Testing bad_function..."
python -m verl_tool.servers.tests.test_python_code_sync_tool bad_function --url http://localhost:6661/get_observation || true

echo "Testing missing_code..."
python -m verl_tool.servers.tests.test_python_code_sync_tool missing_code --url http://localhost:6661/get_observation || true

echo "Testing markdown_code_blocks"
python -m verl_tool.servers.tests.test_python_code_sync_tool markdown_code_blocks --url http://localhost:6661/get_observation

kill ${PID_SYNC} || true
sleep 2

# Sync tool with Ray
echo "=========================================="
echo "sync-ray: python_code_sync"
echo "=========================================="
${SERVER_PY} --tool_type python_code_sync --use_ray True --port 6661 --workers_per_tool 8 > /tmp/tool_python_code_sync_ray.log 2>&1 &
PID_SYNC_RAY=$!
wait_for_server 6661 || exit 1

echo "Testing basic_execution..."
python -m verl_tool.servers.tests.test_python_code_sync_tool basic_execution --url http://localhost:6661/get_observation

echo "Testing math_calculation..."
python -m verl_tool.servers.tests.test_python_code_sync_tool math_calculation --url http://localhost:6661/get_observation

echo "Testing with_imports..."
python -m verl_tool.servers.tests.test_python_code_sync_tool with_imports --url http://localhost:6661/get_observation

echo "Testing syntax_error..."
python -m verl_tool.servers.tests.test_python_code_sync_tool syntax_error --url http://localhost:6661/get_observation

echo "Testing bad_function..."
python -m verl_tool.servers.tests.test_python_code_sync_tool bad_function --url http://localhost:6661/get_observation || true

echo "Testing missing_code..."
python -m verl_tool.servers.tests.test_python_code_sync_tool missing_code --url http://localhost:6661/get_observation || true

echo "Testing markdown_code_blocks"
python -m verl_tool.servers.tests.test_python_code_sync_tool markdown_code_blocks --url http://localhost:6661/get_observation

kill ${PID_SYNC_RAY} || true
sleep 2

# Async tool (no Ray)
echo "=========================================="
echo "async-no-ray: python_code_async"
echo "=========================================="
${SERVER_PY} --tool_type python_code_async --port 6661 --workers_per_tool 8 > /tmp/tool_python_code_async.log 2>&1 &
PID_ASYNC=$!
wait_for_server 6661 || exit 1

echo "Testing basic_execution..."
python -m verl_tool.servers.tests.test_python_code_async_tool basic_execution --url http://localhost:6661/get_observation

echo "Testing math_calculation..."
python -m verl_tool.servers.tests.test_python_code_async_tool math_calculation --url http://localhost:6661/get_observation

echo "Testing with_imports..."
python -m verl_tool.servers.tests.test_python_code_async_tool with_imports --url http://localhost:6661/get_observation

echo "Testing syntax_error..."
python -m verl_tool.servers.tests.test_python_code_async_tool syntax_error --url http://localhost:6661/get_observation

echo "Testing bad_function..."
python -m verl_tool.servers.tests.test_python_code_async_tool bad_function --url http://localhost:6661/get_observation || true

echo "Testing missing_code..."
python -m verl_tool.servers.tests.test_python_code_async_tool missing_code --url http://localhost:6661/get_observation || true

echo "Testing markdown_code_blocks"
python -m verl_tool.servers.tests.test_python_code_async_tool markdown_code_blocks --url http://localhost:6661/get_observation

kill ${PID_ASYNC} || true

echo "=========================================="
echo "All python_code tests completed!"
echo "=========================================="
