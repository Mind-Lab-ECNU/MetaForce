#!/usr/bin/env bash
# Wave-like resource guard:
# - Randomly alternates between CPU/GPU load actions.
# - Uses all visible GPUs for concurrent memory/compute waves.
# - Adds anti-pattern guards to avoid obvious periodic behavior.

set -u
set -o pipefail

MODE=${MODE:-auto}                                # auto | cpu_only | gpu_only
CPU_TARGET_MIN=${CPU_TARGET_MIN:-50}
CPU_TARGET_MAX=${CPU_TARGET_MAX:-85}
CPU_PHASE_MIN_SEC=${CPU_PHASE_MIN_SEC:-25}
CPU_PHASE_MAX_SEC=${CPU_PHASE_MAX_SEC:-90}
GPU_FREE_RATIO_MIN=${GPU_FREE_RATIO_MIN:-0.25}
GPU_FREE_RATIO_MAX=${GPU_FREE_RATIO_MAX:-0.55}
GPU_FREE_SEG_1=${GPU_FREE_SEG_1:-0.75}
GPU_FREE_SEG_2=${GPU_FREE_SEG_2:-0.55}
GPU_FREE_SEG_3=${GPU_FREE_SEG_3:-0.35}
GPU_FREE_SEG_4=${GPU_FREE_SEG_4:-0.20}
GPU_ALLOC_SEG_1=${GPU_ALLOC_SEG_1:-0.72}
GPU_ALLOC_SEG_2=${GPU_ALLOC_SEG_2:-0.52}
GPU_ALLOC_SEG_3=${GPU_ALLOC_SEG_3:-0.32}
GPU_ALLOC_SEG_4=${GPU_ALLOC_SEG_4:-0.18}
GPU_ALLOC_SEG_5=${GPU_ALLOC_SEG_5:-0.10}
GPU_PHASE_MIN_SEC=${GPU_PHASE_MIN_SEC:-30}
GPU_PHASE_MAX_SEC=${GPU_PHASE_MAX_SEC:-120}
GPU_IDLE_UTIL_MAX=${GPU_IDLE_UTIL_MAX:-10}
GPU_IDLE_MEM_USED_RATIO_MAX=${GPU_IDLE_MEM_USED_RATIO_MAX:-0.20}
COOLDOWN_MIN_SEC=${COOLDOWN_MIN_SEC:-3}
COOLDOWN_MAX_SEC=${COOLDOWN_MAX_SEC:-20}
LOG_EVERY_SEC=${LOG_EVERY_SEC:-2}

GPU_RATIO_BACKOFF=1.0000
ROUND_ID=0

declare -a ACTION_PIDS=()
declare -a ACTION_HISTORY=()

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

is_int() {
    [[ "${1:-}" =~ ^-?[0-9]+$ ]]
}

float_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'
}

float_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'
}

rand_u32() {
    od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]'
}

rand_int() {
    local min="$1"
    local max="$2"
    local range r

    if [ "$min" -ge "$max" ]; then
        echo "$min"
        return 0
    fi

    range=$((max - min + 1))
    r=$(rand_u32)
    echo $((min + (r % range)))
}

rand_float() {
    local min="$1"
    local max="$2"
    local r
    r=$(rand_u32)
    awk -v mn="$min" -v mx="$max" -v rv="$r" '
        BEGIN {
            s = rv / 4294967295.0;
            printf "%.4f", (mn + (mx - mn) * s);
        }
    '
}

clamp_int() {
    local v="$1"
    local lo="$2"
    local hi="$3"
    if [ "$v" -lt "$lo" ]; then
        echo "$lo"
    elif [ "$v" -gt "$hi" ]; then
        echo "$hi"
    else
        echo "$v"
    fi
}

clamp_float() {
    local v="$1"
    local lo="$2"
    local hi="$3"
    awk -v x="$v" -v l="$lo" -v h="$hi" '
        BEGIN {
            if (x < l) x = l;
            if (x > h) x = h;
            printf "%.4f", x;
        }
    '
}

register_action_pid() {
    ACTION_PIDS+=("$1")
}

kill_one_action_pid() {
    local idx pid
    if [ "${#ACTION_PIDS[@]}" -le 0 ]; then
        return 0
    fi
    idx=$(( ${#ACTION_PIDS[@]} - 1 ))
    pid="${ACTION_PIDS[$idx]}"
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    unset 'ACTION_PIDS[idx]'
    ACTION_PIDS=("${ACTION_PIDS[@]}")
}

stop_current_action() {
    local pid
    if [ "${#ACTION_PIDS[@]}" -le 0 ]; then
        return 0
    fi
    for pid in "${ACTION_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 0.2
    for pid in "${ACTION_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    ACTION_PIDS=()
}

cleanup() {
    log "Stopping all action subprocesses..."
    stop_current_action
    log "Stopped."
}

trap cleanup SIGINT SIGTERM EXIT

has_nvidia_smi() {
    command -v nvidia-smi >/dev/null 2>&1
}

get_cpu_cores() {
    local cores
    if command -v nproc >/dev/null 2>&1; then
        cores="$(nproc)"
    else
        cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
    if ! is_int "$cores" || [ "$cores" -lt 1 ]; then
        cores=1
    fi
    echo "$cores"
}

read_cpu_totals() {
    local user nice system idle iowait irq softirq steal total idle_all
    read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
    total=$((user + nice + system + idle + iowait + irq + softirq + steal))
    idle_all=$((idle + iowait))
    echo "$total $idle_all"
}

PREV_TOTAL=0
PREV_IDLE=0

reset_cpu_meter() {
    read -r PREV_TOTAL PREV_IDLE <<< "$(read_cpu_totals)"
}

cpu_usage_once() {
    local total idle delta_total delta_idle usage
    read -r total idle <<< "$(read_cpu_totals)"
    delta_total=$((total - PREV_TOTAL))
    delta_idle=$((idle - PREV_IDLE))
    PREV_TOTAL=$total
    PREV_IDLE=$idle

    if [ "$delta_total" -le 0 ]; then
        echo 0
        return 0
    fi

    usage=$(( (100 * (delta_total - delta_idle)) / delta_total ))
    usage=$(clamp_int "$usage" 0 100)
    echo "$usage"
}

spawn_cpu_spin_worker() {
    bash -c 'while :; do :; done' &
    register_action_pid "$!"
}

spawn_cpu_burst_worker() {
    bash -c 'while :; do x=0; for ((i=0;i<120000;i++)); do x=$((x+i)); done; done' &
    register_action_pid "$!"
}

push_history() {
    ACTION_HISTORY+=("$1")
    if [ "${#ACTION_HISTORY[@]}" -gt 12 ]; then
        ACTION_HISTORY=("${ACTION_HISTORY[@]:1}")
    fi
}

is_forbidden_candidate() {
    local c="$1"
    local n a b c1 d
    n=${#ACTION_HISTORY[@]}

    # Block 3+ identical rounds in a row.
    if [ "$n" -ge 2 ]; then
        if [ "${ACTION_HISTORY[$((n-1))]}" = "$c" ] && [ "${ACTION_HISTORY[$((n-2))]}" = "$c" ]; then
            return 0
        fi
    fi

    # Block ABAB continuation into another B.
    if [ "$n" -ge 4 ]; then
        a="${ACTION_HISTORY[$((n-4))]}"
        b="${ACTION_HISTORY[$((n-3))]}"
        c1="${ACTION_HISTORY[$((n-2))]}"
        d="${ACTION_HISTORY[$((n-1))]}"
        if [ "$a" = "$c1" ] && [ "$b" = "$d" ] && [ "$a" != "$b" ] && [ "$c" = "$d" ]; then
            return 0
        fi
    fi
    return 1
}

choose_action_from_pool() {
    local -a pool=("$@")
    local pick attempts
    attempts=0
    while true; do
        pick="${pool[$(rand_int 0 $(( ${#pool[@]} - 1 )))]}"
        if ! is_forbidden_candidate "$pick"; then
            echo "$pick"
            return 0
        fi
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 8 ]; then
            echo "$pick"
            return 0
        fi
    done
}

choose_cpu_action() {
    local -a pool=(cpu_spin_wave cpu_spin_wave cpu_compute_burst cpu_compute_burst cpu_compute_burst)
    choose_action_from_pool "${pool[@]}"
}

choose_round_action() {
    local gpu_available="$1"
    local -a pool=()
    case "$MODE" in
        cpu_only)
            pool=(cpu_spin_wave cpu_spin_wave cpu_compute_burst cpu_compute_burst cpu_compute_burst)
            ;;
        gpu_only)
            if [ "$gpu_available" -eq 1 ]; then
                pool=(gpu_tensor_wave gpu_tensor_wave gpu_tensor_wave)
            else
                pool=()
            fi
            ;;
        auto)
            pool=(cpu_spin_wave cpu_spin_wave cpu_compute_burst cpu_compute_burst cpu_compute_burst)
            if [ "$gpu_available" -eq 1 ]; then
                pool+=(gpu_tensor_wave gpu_tensor_wave)
            fi
            ;;
        *)
            fail "invalid MODE=$MODE (expected auto|cpu_only|gpu_only)"
            ;;
    esac

    if [ "${#pool[@]}" -eq 0 ]; then
        echo ""
        return 0
    fi
    choose_action_from_pool "${pool[@]}"
}

list_visible_gpus() {
    local raw idx total used util free out

    raw="$(nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
    [ -z "$raw" ] && return 1

    out=""

    while IFS=',' read -r idx total used util; do
        idx="${idx//[[:space:]]/}"
        total="${total//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        util="${util//[^0-9]/}"
        [ -z "$util" ] && util=0

        if ! is_int "$idx" || ! is_int "$total" || ! is_int "$used" || ! is_int "$util"; then
            continue
        fi
        if [ "$total" -le 0 ] || [ "$used" -lt 0 ] || [ "$used" -gt "$total" ]; then
            continue
        fi

        free=$((total - used))
        out+="${idx},${total},${used},${free},${util}"$'\n'
    done <<< "$raw"

    if [ -z "$out" ]; then
        return 1
    fi
    printf "%s" "$out"
}

calc_segment_alloc_ratio() {
    local free_ratio="$1"
    awk \
        -v fr="$free_ratio" \
        -v s1="$GPU_FREE_SEG_1" -v s2="$GPU_FREE_SEG_2" -v s3="$GPU_FREE_SEG_3" -v s4="$GPU_FREE_SEG_4" \
        -v a1="$GPU_ALLOC_SEG_1" -v a2="$GPU_ALLOC_SEG_2" -v a3="$GPU_ALLOC_SEG_3" -v a4="$GPU_ALLOC_SEG_4" -v a5="$GPU_ALLOC_SEG_5" '
        BEGIN {
            if (fr >= s1) r = a1;
            else if (fr >= s2) r = a2;
            else if (fr >= s3) r = a3;
            else if (fr >= s4) r = a4;
            else r = a5;
            printf "%.4f", r;
        }
    '
}

run_cpu_spin_wave() {
    local phase_sec="$1"
    local end now target next_retarget usage delta worker_count
    local max_workers initial jitter add remove step
    local i

    stop_current_action

    target="$(rand_int "$CPU_TARGET_MIN" "$CPU_TARGET_MAX")"
    jitter="$(rand_int -4 5)"
    target="$(clamp_int $((target + jitter)) "$CPU_TARGET_MIN" "$CPU_TARGET_MAX")"

    max_workers=$((CPU_CORES * 2))
    [ "$max_workers" -lt 2 ] && max_workers=2

    initial=$((CPU_CORES * target / 100))
    [ "$initial" -lt 1 ] && initial=1
    initial=$((initial + $(rand_int -1 2)))
    initial="$(clamp_int "$initial" 1 "$max_workers")"

    log "round=$ROUND_ID action=cpu_spin_wave target=${target}% phase=${phase_sec}s initial_workers=$initial"

    for ((i=0; i<initial; i++)); do
        spawn_cpu_spin_worker
    done

    reset_cpu_meter
    now="$(date +%s)"
    end=$((now + phase_sec))
    next_retarget=$((now + $(rand_int 2 9)))

    while true; do
        now="$(date +%s)"
        if [ "$now" -ge "$end" ]; then
            break
        fi

        sleep "$LOG_EVERY_SEC"
        usage="$(cpu_usage_once)"
        now="$(date +%s)"

        if [ "$now" -ge "$next_retarget" ]; then
            target="$(rand_int "$CPU_TARGET_MIN" "$CPU_TARGET_MAX")"
            target="$(clamp_int $((target + $(rand_int -7 7))) "$CPU_TARGET_MIN" "$CPU_TARGET_MAX")"
            next_retarget=$((now + $(rand_int 2 10)))
        fi

        # Random surge spike to break regular shape.
        if [ "$(rand_int 1 100)" -le 10 ]; then
            add="$(rand_int 1 2)"
            for ((i=0; i<add; i++)); do
                if [ "${#ACTION_PIDS[@]}" -lt "$max_workers" ]; then
                    spawn_cpu_spin_worker
                fi
            done
        fi

        delta=$((target - usage))
        worker_count=${#ACTION_PIDS[@]}

        if [ "$delta" -gt 3 ] && [ "$worker_count" -lt "$max_workers" ]; then
            step=$((delta / 8))
            [ "$step" -lt 1 ] && step=1
            step=$((step + $(rand_int 0 1)))
            for ((i=0; i<step; i++)); do
                if [ "${#ACTION_PIDS[@]}" -lt "$max_workers" ]; then
                    spawn_cpu_spin_worker
                fi
            done
        elif [ "$delta" -lt -3 ] && [ "$worker_count" -gt 1 ]; then
            remove=$(( (-delta) / 8 ))
            [ "$remove" -lt 1 ] && remove=1
            remove=$((remove + $(rand_int 0 1)))
            for ((i=0; i<remove; i++)); do
                if [ "${#ACTION_PIDS[@]}" -gt 1 ]; then
                    kill_one_action_pid
                fi
            done
        fi

        worker_count=${#ACTION_PIDS[@]}
        while [ "$worker_count" -gt "$max_workers" ]; do
            kill_one_action_pid
            worker_count=${#ACTION_PIDS[@]}
        done

        log "round=$ROUND_ID action=cpu_spin_wave usage=${usage}% target=${target}% workers=${worker_count}"
    done

    stop_current_action
    return 0
}

run_cpu_compute_burst() {
    local phase_sec="$1"
    local phase_end now remaining target worker_target max_workers burst_workers
    local burst_len burst_end usage lull step i

    stop_current_action

    target="$(rand_int "$CPU_TARGET_MIN" "$CPU_TARGET_MAX")"
    worker_target=$((CPU_CORES * target / 100))
    [ "$worker_target" -lt 1 ] && worker_target=1
    max_workers=$((CPU_CORES * 2))
    [ "$max_workers" -lt 2 ] && max_workers=2
    worker_target="$(clamp_int "$worker_target" 1 "$max_workers")"

    now="$(date +%s)"
    phase_end=$((now + phase_sec))

    log "round=$ROUND_ID action=cpu_compute_burst target=${target}% phase=${phase_sec}s worker_base=$worker_target"

    while true; do
        now="$(date +%s)"
        remaining=$((phase_end - now))
        if [ "$remaining" -le 0 ]; then
            break
        fi

        burst_len="$(rand_int 2 8)"
        [ "$burst_len" -gt "$remaining" ] && burst_len="$remaining"

        step=$((CPU_CORES / 3))
        [ "$step" -lt 1 ] && step=1
        burst_workers=$((worker_target + $(rand_int "-$step" "$step")))
        burst_workers="$(clamp_int "$burst_workers" 1 "$max_workers")"

        for ((i=0; i<burst_workers; i++)); do
            spawn_cpu_burst_worker
        done

        reset_cpu_meter
        now="$(date +%s)"
        burst_end=$((now + burst_len))

        while true; do
            now="$(date +%s)"
            if [ "$now" -ge "$burst_end" ]; then
                break
            fi

            sleep "$LOG_EVERY_SEC"
            usage="$(cpu_usage_once)"

            # Mid-burst micro-adjustments to add jaggedness.
            if [ "$(rand_int 1 100)" -le 30 ] && [ "${#ACTION_PIDS[@]}" -lt "$max_workers" ]; then
                spawn_cpu_burst_worker
            elif [ "$(rand_int 1 100)" -le 30 ] && [ "${#ACTION_PIDS[@]}" -gt 1 ]; then
                kill_one_action_pid
            fi

            log "round=$ROUND_ID action=cpu_compute_burst usage=${usage}% burst_workers=${#ACTION_PIDS[@]}"
        done

        stop_current_action

        now="$(date +%s)"
        remaining=$((phase_end - now))
        if [ "$remaining" -le 0 ]; then
            break
        fi

        lull="$(rand_float 0.4 2.4)"
        log "round=$ROUND_ID action=cpu_compute_burst lull=${lull}s"
        sleep "$lull"
    done

    stop_current_action
    return 0
}

run_gpu_tensor_wave() {
    local phase_sec="$1"
    local gpu_rows row
    local gpu_idx gpu_total gpu_used gpu_free gpu_util
    local free_ratio base_ratio alloc_ratio
    local pid rc i ok_count fail_count
    local -a child_pids=()
    local -a child_gpu_ids=()

    stop_current_action

    if ! has_nvidia_smi; then
        log "round=$ROUND_ID action=gpu_tensor_wave nvidia-smi not found, skip"
        return 2
    fi

    gpu_rows="$(list_visible_gpus || true)"
    if [ -z "$gpu_rows" ]; then
        log "round=$ROUND_ID action=gpu_tensor_wave no visible gpu found, skip"
        return 2
    fi

    ok_count=0
    fail_count=0

    while IFS= read -r row; do
        [ -z "$row" ] && continue
        IFS=',' read -r gpu_idx gpu_total gpu_used gpu_free gpu_util <<< "$row"
        free_ratio="$(awk -v f="$gpu_free" -v t="$gpu_total" 'BEGIN { if (t <= 0) print 0.0; else printf "%.6f", (f / t) }')"
        base_ratio="$(calc_segment_alloc_ratio "$free_ratio")"
        alloc_ratio="$(awk -v r="$base_ratio" -v b="$GPU_RATIO_BACKOFF" -v lo="$GPU_FREE_RATIO_MIN" -v hi="$GPU_FREE_RATIO_MAX" '
            BEGIN {
                v = r * b;
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                printf "%.4f", v;
            }')"

        log "round=$ROUND_ID action=gpu_tensor_wave gpu=$gpu_idx free=${gpu_free}MiB util=${gpu_util}% free_ratio=${free_ratio} base_ratio=${base_ratio} ratio=${alloc_ratio} phase=${phase_sec}s"

        CUDA_VISIBLE_DEVICES="$gpu_idx" \
        GPU_PHYS_IDX="$gpu_idx" \
        GPU_PHASE_SEC="$phase_sec" \
        GPU_ALLOC_RATIO_INIT="$alloc_ratio" \
        GPU_BACKOFF="$GPU_RATIO_BACKOFF" \
        GPU_RATIO_MIN="$GPU_FREE_RATIO_MIN" \
        GPU_RATIO_MAX="$GPU_FREE_RATIO_MAX" \
        GPU_FREE_SEG_1="$GPU_FREE_SEG_1" \
        GPU_FREE_SEG_2="$GPU_FREE_SEG_2" \
        GPU_FREE_SEG_3="$GPU_FREE_SEG_3" \
        GPU_FREE_SEG_4="$GPU_FREE_SEG_4" \
        GPU_ALLOC_SEG_1="$GPU_ALLOC_SEG_1" \
        GPU_ALLOC_SEG_2="$GPU_ALLOC_SEG_2" \
        GPU_ALLOC_SEG_3="$GPU_ALLOC_SEG_3" \
        GPU_ALLOC_SEG_4="$GPU_ALLOC_SEG_4" \
        GPU_ALLOC_SEG_5="$GPU_ALLOC_SEG_5" \
        LOG_EVERY_SEC="$LOG_EVERY_SEC" \
        python3 - <<'PY' &
import os
import random
import sys
import time

def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [GPUPY][gpu={gpu_phys_idx}] {msg}", flush=True)

phase_sec = float(os.environ.get("GPU_PHASE_SEC", "30"))
init_ratio = float(os.environ.get("GPU_ALLOC_RATIO_INIT", "0.3"))
backoff = float(os.environ.get("GPU_BACKOFF", "1.0"))
ratio_min = float(os.environ.get("GPU_RATIO_MIN", "0.25"))
ratio_max = float(os.environ.get("GPU_RATIO_MAX", "0.55"))
gpu_phys_idx = os.environ.get("GPU_PHYS_IDX", "unknown")
log_every = float(os.environ.get("LOG_EVERY_SEC", "2"))
free_seg_1 = float(os.environ.get("GPU_FREE_SEG_1", "0.75"))
free_seg_2 = float(os.environ.get("GPU_FREE_SEG_2", "0.55"))
free_seg_3 = float(os.environ.get("GPU_FREE_SEG_3", "0.35"))
free_seg_4 = float(os.environ.get("GPU_FREE_SEG_4", "0.20"))
alloc_seg_1 = float(os.environ.get("GPU_ALLOC_SEG_1", "0.72"))
alloc_seg_2 = float(os.environ.get("GPU_ALLOC_SEG_2", "0.52"))
alloc_seg_3 = float(os.environ.get("GPU_ALLOC_SEG_3", "0.32"))
alloc_seg_4 = float(os.environ.get("GPU_ALLOC_SEG_4", "0.18"))
alloc_seg_5 = float(os.environ.get("GPU_ALLOC_SEG_5", "0.10"))

try:
    import torch
except Exception as exc:
    log(f"torch import failed: {exc}")
    sys.exit(3)

if not torch.cuda.is_available():
    log("torch.cuda is unavailable")
    sys.exit(4)

device = torch.device("cuda:0")
torch.cuda.set_device(device)

chunk_bytes = 256 * 1024 * 1024
elem_bytes = 2  # float16
min_target_bytes = 64 * 1024 * 1024
max_consecutive_oom = 5
blocks = []
allocated_bytes = 0
oom_scale = 1.0
oom_count = 0
consecutive_oom = 0

def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def segment_ratio(free_ratio: float) -> float:
    if free_ratio >= free_seg_1:
        return alloc_seg_1
    if free_ratio >= free_seg_2:
        return alloc_seg_2
    if free_ratio >= free_seg_3:
        return alloc_seg_3
    if free_ratio >= free_seg_4:
        return alloc_seg_4
    return alloc_seg_5

def append_block(request_bytes: int) -> None:
    global allocated_bytes
    current = max(4 * 1024 * 1024, request_bytes)
    numel = max(1, current // elem_bytes)
    block = torch.empty((numel,), dtype=torch.float16, device=device)
    blocks.append(block)
    allocated_bytes += block.numel() * block.element_size()

def shrink_pool(target_bytes: int) -> None:
    global allocated_bytes
    while blocks and allocated_bytes > target_bytes:
        blk = blocks.pop()
        allocated_bytes -= blk.numel() * blk.element_size()
        del blk
    torch.cuda.empty_cache()

def rebalance_pool():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_ratio = free_bytes / max(total_bytes, 1)
    base_ratio = segment_ratio(free_ratio)
    target_ratio = clamp(base_ratio * backoff * oom_scale, ratio_min, ratio_max)
    target_bytes = int(max(min_target_bytes, free_bytes * target_ratio))

    while allocated_bytes < target_bytes:
        remain = target_bytes - allocated_bytes
        current = min(chunk_bytes, remain)
        if remain < 96 * 1024 * 1024:
            current = remain
        append_block(current)

    if allocated_bytes > int(target_bytes * 1.15):
        shrink_pool(int(target_bytes * 1.02))

    return free_bytes, total_bytes, free_ratio, target_ratio

start = time.time()
end = start + phase_sec
last_log = 0.0

while time.time() < end:
    target_ratio = init_ratio
    free_bytes = 0
    total_bytes = 0
    n = random.choice([640, 768, 896, 1024, 1152, 1280, 1408])
    reps = random.randint(1, 4)
    try:
        free_bytes, total_bytes, _, target_ratio = rebalance_pool()

        a = torch.randn((n, n), device=device, dtype=torch.float16)
        b = torch.randn((n, n), device=device, dtype=torch.float16)
        for _ in range(reps):
            c = torch.matmul(a, b)
            a, b = b, c

        if blocks and random.random() < 0.40:
            idx = random.randrange(len(blocks))
            blocks[idx].mul_(random.uniform(0.98, 1.02))

        torch.cuda.synchronize(device)
        consecutive_oom = 0
        oom_scale = min(1.0, oom_scale + 0.03)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            oom_count += 1
            consecutive_oom += 1
            oom_scale = max(0.10, oom_scale * 0.70)
            if blocks:
                shrink_pool(int(allocated_bytes * 0.70))
            torch.cuda.empty_cache()
            log(f"oom_event oom_count={oom_count} consecutive={consecutive_oom} target_ratio={target_ratio:.3f} alloc_mb={allocated_bytes / 1024 / 1024:.1f}")
            if consecutive_oom >= max_consecutive_oom:
                log("too many consecutive OOMs, exit")
                sys.exit(8)
            time.sleep(random.uniform(0.3, 1.0))
            continue
        raise

    now = time.time()
    if now - last_log >= log_every:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        log(f"heartbeat free_mb={free_bytes / 1024 / 1024:.0f} target_ratio={target_ratio:.3f} alloc_mb={allocated_bytes / 1024 / 1024:.1f} oom_count={oom_count}")
        last_log = now

    time.sleep(random.uniform(0.05, 0.9))

blocks.clear()
torch.cuda.empty_cache()
log(f"phase done alloc_mb={allocated_bytes / 1024 / 1024:.1f} oom_count={oom_count}")
sys.exit(0)
PY
        pid="$!"
        register_action_pid "$pid"
        child_pids+=("$pid")
        child_gpu_ids+=("$gpu_idx")
    done <<< "$gpu_rows"

    if [ "${#child_pids[@]}" -le 0 ]; then
        log "round=$ROUND_ID action=gpu_tensor_wave no valid gpu worker launched"
        return 2
    fi

    for i in "${!child_pids[@]}"; do
        pid="${child_pids[$i]}"
        if wait "$pid"; then
            rc=0
            ok_count=$((ok_count + 1))
        else
            rc=$?
            fail_count=$((fail_count + 1))
        fi
        log "round=$ROUND_ID action=gpu_tensor_wave gpu=${child_gpu_ids[$i]} rc=$rc"
    done
    ACTION_PIDS=()

    if [ "$ok_count" -le 0 ]; then
        GPU_RATIO_BACKOFF="$(clamp_float "$(awk -v b="$GPU_RATIO_BACKOFF" 'BEGIN { printf "%.4f", b * 0.75 }')" 0.35 1.00)"
        log "round=$ROUND_ID action=gpu_tensor_wave all_failed ok=$ok_count fail=$fail_count -> backoff=${GPU_RATIO_BACKOFF}"
        return 1
    fi

    GPU_RATIO_BACKOFF="$(clamp_float "$(awk -v b="$GPU_RATIO_BACKOFF" 'BEGIN { printf "%.4f", b + 0.05 }')" 0.35 1.00)"
    log "round=$ROUND_ID action=gpu_tensor_wave summary ok=$ok_count fail=$fail_count backoff=${GPU_RATIO_BACKOFF}"
    return 0
}

validate_config() {
    is_int "$CPU_TARGET_MIN" || fail "CPU_TARGET_MIN must be int"
    is_int "$CPU_TARGET_MAX" || fail "CPU_TARGET_MAX must be int"
    is_int "$CPU_PHASE_MIN_SEC" || fail "CPU_PHASE_MIN_SEC must be int"
    is_int "$CPU_PHASE_MAX_SEC" || fail "CPU_PHASE_MAX_SEC must be int"
    is_int "$GPU_PHASE_MIN_SEC" || fail "GPU_PHASE_MIN_SEC must be int"
    is_int "$GPU_PHASE_MAX_SEC" || fail "GPU_PHASE_MAX_SEC must be int"
    is_int "$GPU_IDLE_UTIL_MAX" || fail "GPU_IDLE_UTIL_MAX must be int"
    is_int "$LOG_EVERY_SEC" || fail "LOG_EVERY_SEC must be int"

    [ "$CPU_TARGET_MIN" -ge 1 ] || fail "CPU_TARGET_MIN must be >= 1"
    [ "$CPU_TARGET_MAX" -le 100 ] || fail "CPU_TARGET_MAX must be <= 100"
    [ "$CPU_TARGET_MIN" -le "$CPU_TARGET_MAX" ] || fail "CPU_TARGET_MIN > CPU_TARGET_MAX"
    [ "$CPU_PHASE_MIN_SEC" -ge 1 ] || fail "CPU_PHASE_MIN_SEC must be >= 1"
    [ "$CPU_PHASE_MIN_SEC" -le "$CPU_PHASE_MAX_SEC" ] || fail "CPU_PHASE_MIN_SEC > CPU_PHASE_MAX_SEC"
    [ "$GPU_PHASE_MIN_SEC" -ge 1 ] || fail "GPU_PHASE_MIN_SEC must be >= 1"
    [ "$GPU_PHASE_MIN_SEC" -le "$GPU_PHASE_MAX_SEC" ] || fail "GPU_PHASE_MIN_SEC > GPU_PHASE_MAX_SEC"
    [ "$GPU_IDLE_UTIL_MAX" -ge 0 ] || fail "GPU_IDLE_UTIL_MAX must be >= 0"
    [ "$GPU_IDLE_UTIL_MAX" -le 100 ] || fail "GPU_IDLE_UTIL_MAX must be <= 100"
    [ "$LOG_EVERY_SEC" -ge 1 ] || fail "LOG_EVERY_SEC must be >= 1"

    float_le "$GPU_FREE_RATIO_MIN" "$GPU_FREE_RATIO_MAX" || fail "GPU_FREE_RATIO_MIN > GPU_FREE_RATIO_MAX"
    float_ge "$GPU_FREE_RATIO_MIN" 0.01 || fail "GPU_FREE_RATIO_MIN must be >= 0.01"
    float_le "$GPU_FREE_RATIO_MAX" 0.95 || fail "GPU_FREE_RATIO_MAX must be <= 0.95"
    float_ge "$GPU_FREE_SEG_1" 0.01 || fail "GPU_FREE_SEG_1 must be >= 0.01"
    float_le "$GPU_FREE_SEG_1" 0.99 || fail "GPU_FREE_SEG_1 must be <= 0.99"
    float_ge "$GPU_FREE_SEG_2" 0.01 || fail "GPU_FREE_SEG_2 must be >= 0.01"
    float_le "$GPU_FREE_SEG_2" 0.99 || fail "GPU_FREE_SEG_2 must be <= 0.99"
    float_ge "$GPU_FREE_SEG_3" 0.01 || fail "GPU_FREE_SEG_3 must be >= 0.01"
    float_le "$GPU_FREE_SEG_3" 0.99 || fail "GPU_FREE_SEG_3 must be <= 0.99"
    float_ge "$GPU_FREE_SEG_4" 0.01 || fail "GPU_FREE_SEG_4 must be >= 0.01"
    float_le "$GPU_FREE_SEG_4" 0.99 || fail "GPU_FREE_SEG_4 must be <= 0.99"
    float_ge "$GPU_FREE_SEG_1" "$GPU_FREE_SEG_2" || fail "GPU_FREE_SEG_1 must be >= GPU_FREE_SEG_2"
    float_ge "$GPU_FREE_SEG_2" "$GPU_FREE_SEG_3" || fail "GPU_FREE_SEG_2 must be >= GPU_FREE_SEG_3"
    float_ge "$GPU_FREE_SEG_3" "$GPU_FREE_SEG_4" || fail "GPU_FREE_SEG_3 must be >= GPU_FREE_SEG_4"

    float_ge "$GPU_ALLOC_SEG_1" 0.01 || fail "GPU_ALLOC_SEG_1 must be >= 0.01"
    float_le "$GPU_ALLOC_SEG_1" 0.95 || fail "GPU_ALLOC_SEG_1 must be <= 0.95"
    float_ge "$GPU_ALLOC_SEG_2" 0.01 || fail "GPU_ALLOC_SEG_2 must be >= 0.01"
    float_le "$GPU_ALLOC_SEG_2" 0.95 || fail "GPU_ALLOC_SEG_2 must be <= 0.95"
    float_ge "$GPU_ALLOC_SEG_3" 0.01 || fail "GPU_ALLOC_SEG_3 must be >= 0.01"
    float_le "$GPU_ALLOC_SEG_3" 0.95 || fail "GPU_ALLOC_SEG_3 must be <= 0.95"
    float_ge "$GPU_ALLOC_SEG_4" 0.01 || fail "GPU_ALLOC_SEG_4 must be >= 0.01"
    float_le "$GPU_ALLOC_SEG_4" 0.95 || fail "GPU_ALLOC_SEG_4 must be <= 0.95"
    float_ge "$GPU_ALLOC_SEG_5" 0.01 || fail "GPU_ALLOC_SEG_5 must be >= 0.01"
    float_le "$GPU_ALLOC_SEG_5" 0.95 || fail "GPU_ALLOC_SEG_5 must be <= 0.95"
    float_ge "$GPU_ALLOC_SEG_1" "$GPU_ALLOC_SEG_2" || fail "GPU_ALLOC_SEG_1 must be >= GPU_ALLOC_SEG_2"
    float_ge "$GPU_ALLOC_SEG_2" "$GPU_ALLOC_SEG_3" || fail "GPU_ALLOC_SEG_2 must be >= GPU_ALLOC_SEG_3"
    float_ge "$GPU_ALLOC_SEG_3" "$GPU_ALLOC_SEG_4" || fail "GPU_ALLOC_SEG_3 must be >= GPU_ALLOC_SEG_4"
    float_ge "$GPU_ALLOC_SEG_4" "$GPU_ALLOC_SEG_5" || fail "GPU_ALLOC_SEG_4 must be >= GPU_ALLOC_SEG_5"

    float_ge "$GPU_IDLE_MEM_USED_RATIO_MAX" 0.01 || fail "GPU_IDLE_MEM_USED_RATIO_MAX must be >= 0.01"
    float_le "$GPU_IDLE_MEM_USED_RATIO_MAX" 0.95 || fail "GPU_IDLE_MEM_USED_RATIO_MAX must be <= 0.95"
    float_le "$COOLDOWN_MIN_SEC" "$COOLDOWN_MAX_SEC" || fail "COOLDOWN_MIN_SEC > COOLDOWN_MAX_SEC"
}

validate_config

if [ "$MODE" != "auto" ] && [ "$MODE" != "cpu_only" ] && [ "$MODE" != "gpu_only" ]; then
    fail "MODE must be one of auto|cpu_only|gpu_only"
fi

CPU_CORES="$(get_cpu_cores)"
CPU_AVAILABLE=0
if [ -r /proc/stat ]; then
    CPU_AVAILABLE=1
fi

if [ "$MODE" != "gpu_only" ] && [ "$CPU_AVAILABLE" -ne 1 ]; then
    fail "/proc/stat is not readable. CPU actions require Linux /proc/stat."
fi

log "======================================================"
log "resource wave guard started"
log "MODE=$MODE CPU_CORES=$CPU_CORES CPU_AVAILABLE=$CPU_AVAILABLE"
log "CPU target=${CPU_TARGET_MIN}-${CPU_TARGET_MAX}% phase=${CPU_PHASE_MIN_SEC}-${CPU_PHASE_MAX_SEC}s"
log "GPU mode=all_visible ratio_bounds=${GPU_FREE_RATIO_MIN}-${GPU_FREE_RATIO_MAX} phase=${GPU_PHASE_MIN_SEC}-${GPU_PHASE_MAX_SEC}s"
log "GPU segments free=[${GPU_FREE_SEG_1},${GPU_FREE_SEG_2},${GPU_FREE_SEG_3},${GPU_FREE_SEG_4}] alloc=[${GPU_ALLOC_SEG_1},${GPU_ALLOC_SEG_2},${GPU_ALLOC_SEG_3},${GPU_ALLOC_SEG_4},${GPU_ALLOC_SEG_5}]"
log "GPU idle settings kept for compatibility: util<=${GPU_IDLE_UTIL_MAX}% mem_used_ratio<=${GPU_IDLE_MEM_USED_RATIO_MAX} (not used for selection)"
log "cooldown=${COOLDOWN_MIN_SEC}-${COOLDOWN_MAX_SEC}s log_every=${LOG_EVERY_SEC}s"
log "======================================================"

while true; do
    ROUND_ID=$((ROUND_ID + 1))
    local_gpu_available=0
    if has_nvidia_smi; then
        local_gpu_available=1
    fi

    action="$(choose_round_action "$local_gpu_available")"
    if [ -z "$action" ]; then
        log "round=$ROUND_ID no eligible action in MODE=$MODE (gpu unavailable). sleeping 5s"
        sleep 5
        continue
    fi

    rc=0
    executed_action="$action"

    case "$action" in
        cpu_spin_wave)
            phase="$(rand_int "$CPU_PHASE_MIN_SEC" "$CPU_PHASE_MAX_SEC")"
            run_cpu_spin_wave "$phase"
            rc="$?"
            ;;
        cpu_compute_burst)
            phase="$(rand_int "$CPU_PHASE_MIN_SEC" "$CPU_PHASE_MAX_SEC")"
            run_cpu_compute_burst "$phase"
            rc="$?"
            ;;
        gpu_tensor_wave)
            phase="$(rand_int "$GPU_PHASE_MIN_SEC" "$GPU_PHASE_MAX_SEC")"
            run_gpu_tensor_wave "$phase"
            rc="$?"
            if [ "$rc" -ne 0 ] && [ "$MODE" = "auto" ] && [ "$CPU_AVAILABLE" -eq 1 ]; then
                fallback_action="$(choose_cpu_action)"
                executed_action="$fallback_action"
                log "round=$ROUND_ID gpu phase unavailable/fail, fallback=$fallback_action"
                phase="$(rand_int "$CPU_PHASE_MIN_SEC" "$CPU_PHASE_MAX_SEC")"
                if [ "$fallback_action" = "cpu_spin_wave" ]; then
                    run_cpu_spin_wave "$phase"
                else
                    run_cpu_compute_burst "$phase"
                fi
                rc="$?"
            fi
            ;;
        *)
            log "round=$ROUND_ID unexpected action=$action"
            rc=1
            ;;
    esac

    push_history "$executed_action"

    cooldown_base="$(rand_float "$COOLDOWN_MIN_SEC" "$COOLDOWN_MAX_SEC")"
    cooldown_scale="$(rand_float 0.85 1.30)"
    cooldown="$(awk -v c="$cooldown_base" -v s="$cooldown_scale" -v lo="$COOLDOWN_MIN_SEC" -v hi="$COOLDOWN_MAX_SEC" '
        BEGIN {
            v = c * s;
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            printf "%.4f", v;
        }')"

    log "round=$ROUND_ID done action=$executed_action rc=$rc cooldown=${cooldown}s"
    sleep "$cooldown"
done
