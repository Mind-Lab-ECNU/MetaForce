#!/usr/bin/env bash
set -euo pipefail

# Paths
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)"
SERVER_PY="python -m verl_tool.servers.tool_server"

# Sync tool, no Ray
echo "sync-no-ray: calculate_statistics"
${SERVER_PY} --tool_type calculate_statistics --port 5601 --workers_per_tool 8 > /tmp/tool_sync.log 2>&1 &
PID_SYNC=$!
sleep 3
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool model1 --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool model2 --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool return_info --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool target_label --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool bad_name --url http://localhost:5601/get_observation || true
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool bad_function --url http://localhost:5601/get_observation || true
kill ${PID_SYNC} || true
sleep 2

# Sync tool with Ray
echo "sync-ray: calculate_statistics"
${SERVER_PY} --tool_type calculate_statistics --use_ray True --port 5601 --workers_per_tool 8 > /tmp/tool_sync_ray.log 2>&1 &
PID_SYNC_RAY=$!
sleep 3
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool model1 --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool model2 --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool return_info --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool target_label --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool bad_name --url http://localhost:5601/get_observation || true
python -m verl_tool.servers.tests.test_calculate_statistics_sync_tool bad_function --url http://localhost:5601/get_observation || true
kill ${PID_SYNC_RAY} || true
sleep 2

# Async tool (no Ray)
echo "async-no-ray: calculate_statistics_async"
${SERVER_PY} --tool_type calculate_statistics_async --port 5601 --workers_per_tool 8 > /tmp/tool_async.log 2>&1 &
PID_ASYNC=$!
sleep 3
python -m verl_tool.servers.tests.test_calculate_statistics_async_tool model1 --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_async_tool model2 --url http://localhost:5601/get_observation
python -m verl_tool.servers.tests.test_calculate_statistics_async_tool bad_name --url http://localhost:5601/get_observation || true
python -m verl_tool.servers.tests.test_calculate_statistics_async_tool bad_function --url http://localhost:5601/get_observation || true
kill ${PID_ASYNC} || true
