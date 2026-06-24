#!/usr/bin/env bash
# CPU 占用守护脚本:
# 动态增减 busy 进程，将 CPU 使用率维持在目标区间（默认 50%-70%）

set -u
set -o pipefail

CPU_TARGET_MIN=${CPU_TARGET_MIN:-50}
CPU_TARGET_MAX=${CPU_TARGET_MAX:-70}
CPU_SAMPLE_SEC=${CPU_SAMPLE_SEC:-2}
CPU_MAX_WORKERS=${CPU_MAX_WORKERS:-0}

if [ "$CPU_TARGET_MIN" -lt 1 ] || [ "$CPU_TARGET_MAX" -gt 100 ] || [ "$CPU_TARGET_MIN" -gt "$CPU_TARGET_MAX" ]; then
    echo "ERROR: invalid target range: min=$CPU_TARGET_MIN max=$CPU_TARGET_MAX"
    exit 1
fi

if [ ! -r /proc/stat ]; then
    echo "ERROR: /proc/stat is not readable on this machine. This script supports Linux."
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    CPU_CORES=$(nproc)
else
    CPU_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi

if [ "$CPU_CORES" -lt 1 ]; then
    CPU_CORES=1
fi

if [ "$CPU_MAX_WORKERS" -le 0 ]; then
    CPU_MAX_WORKERS=$CPU_CORES
fi

if [ "$CPU_MAX_WORKERS" -lt 1 ]; then
    CPU_MAX_WORKERS=1
fi

busy_pids=()

spawn_worker() {
    bash -c 'while :; do :; done' &
    busy_pids+=("$!")
}

kill_one_worker() {
    local idx
    if [ "${#busy_pids[@]}" -le 0 ]; then
        return 0
    fi
    idx=$(( ${#busy_pids[@]} - 1 ))
    kill -TERM "${busy_pids[$idx]}" 2>/dev/null || true
    wait "${busy_pids[$idx]}" 2>/dev/null || true
    unset 'busy_pids[idx]'
    busy_pids=("${busy_pids[@]}")
}

kill_all_workers() {
    local pid
    for pid in "${busy_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${busy_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    busy_pids=()
}

cleanup() {
    echo
    echo "Stopping CPU guard workers..."
    kill_all_workers
    echo "Stopped."
}

trap cleanup SIGINT SIGTERM EXIT

read_cpu_totals() {
    local user nice system idle iowait irq softirq steal total idle_all
    read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
    total=$((user + nice + system + idle + iowait + irq + softirq + steal))
    idle_all=$((idle + iowait))
    echo "$total $idle_all"
}

read -r prev_total prev_idle <<< "$(read_cpu_totals)"

cpu_usage_once() {
    local total idle delta_total delta_idle usage
    read -r total idle <<< "$(read_cpu_totals)"
    delta_total=$((total - prev_total))
    delta_idle=$((idle - prev_idle))
    prev_total=$total
    prev_idle=$idle

    if [ "$delta_total" -le 0 ]; then
        echo 0
        return
    fi

    usage=$(( (100 * (delta_total - delta_idle)) / delta_total ))
    if [ "$usage" -lt 0 ]; then
        usage=0
    elif [ "$usage" -gt 100 ]; then
        usage=100
    fi
    echo "$usage"
}

initial_workers=$((CPU_CORES * CPU_TARGET_MIN / 100))
if [ "$initial_workers" -lt 1 ]; then
    initial_workers=1
fi
if [ "$initial_workers" -gt "$CPU_MAX_WORKERS" ]; then
    initial_workers=$CPU_MAX_WORKERS
fi

echo "=========================================="
echo "CPU guard started"
echo "CPU_CORES: $CPU_CORES"
echo "CPU_TARGET_MIN: $CPU_TARGET_MIN"
echo "CPU_TARGET_MAX: $CPU_TARGET_MAX"
echo "CPU_SAMPLE_SEC: $CPU_SAMPLE_SEC"
echo "CPU_MAX_WORKERS: $CPU_MAX_WORKERS"
echo "INITIAL_WORKERS: $initial_workers"
echo "=========================================="

for _ in $(seq 1 "$initial_workers"); do
    spawn_worker
done

while true; do
    sleep "$CPU_SAMPLE_SEC"
    usage=$(cpu_usage_once)
    worker_count=${#busy_pids[@]}

    if [ "$usage" -lt "$CPU_TARGET_MIN" ] && [ "$worker_count" -lt "$CPU_MAX_WORKERS" ]; then
        spawn_worker
        worker_count=${#busy_pids[@]}
    elif [ "$usage" -gt "$CPU_TARGET_MAX" ] && [ "$worker_count" -gt 1 ]; then
        kill_one_worker
        worker_count=${#busy_pids[@]}
    fi

    echo "[CPU] usage=${usage}% workers=${worker_count}"
done

