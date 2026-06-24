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
echo "sync-no-ray: analyze_curve_properties"
echo "=========================================="
${SERVER_PY} --tool_type analyze_curve_properties --port 5603 --workers_per_tool 8 > /tmp/tool_sync.log 2>&1 &
PID_SYNC=$!
wait_for_server 5603 || exit 1

echo "Testing model1..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool model1 --url http://localhost:5603/get_observation

echo "Testing model2..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool model2 --url http://localhost:5603/get_observation

echo "Testing bad_name (expected to fail)..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool bad_name --url http://localhost:5603/get_observation || true

echo "Testing bad_function (expected to fail)..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool bad_function --url http://localhost:5603/get_observation || true

kill ${PID_SYNC} || true
sleep 2

# Sync tool with Ray
echo "=========================================="
echo "sync-ray: analyze_curve_properties"
echo "=========================================="
${SERVER_PY} --tool_type analyze_curve_properties --use_ray True --port 5603 --workers_per_tool 8 > /tmp/tool_sync_ray.log 2>&1 &
PID_SYNC_RAY=$!
wait_for_server 5603 || exit 1

echo "Testing model1..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool model1 --url http://localhost:5603/get_observation

echo "Testing model2..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool model2 --url http://localhost:5603/get_observation

echo "Testing bad_name (expected to fail)..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool bad_name --url http://localhost:5603/get_observation || true

echo "Testing bad_function (expected to fail)..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_sync_tool bad_function --url http://localhost:5603/get_observation || true

kill ${PID_SYNC_RAY} || true
sleep 2

# Async tool (no Ray)
echo "=========================================="
echo "async-no-ray: analyze_curve_properties_async"
echo "=========================================="
${SERVER_PY} --tool_type analyze_curve_properties_async --port 5603 --workers_per_tool 8 > /tmp/tool_async.log 2>&1 &
PID_ASYNC=$!
wait_for_server 5603 || exit 1

echo "Testing model1..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_async_tool model1 --url http://localhost:5603/get_observation

echo "Testing model2..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_async_tool model2 --url http://localhost:5603/get_observation

echo "Testing bad_name (expected to fail)..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_async_tool bad_name --url http://localhost:5603/get_observation || true

echo "Testing bad_function (expected to fail)..."
python -m verl_tool.servers.tests.test_analyze_curve_properties_async_tool bad_function --url http://localhost:5603/get_observation || true

kill ${PID_ASYNC} || true

echo "=========================================="
echo "All analyze_curve_properties tests completed!"
echo "=========================================="
