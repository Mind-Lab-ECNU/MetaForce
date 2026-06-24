#!/usr/bin/env bash
# 高频 IO 占用脚本:
# 循环执行: 复制目录 -> 删除目录 -> 复制目录 -> 删除目录

set -u
set -o pipefail

SOURCE_DIR=${SOURCE_DIR:-"/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/model/Qwen3-VL-2B-Instruct"}
IO_WORK_ROOT=${IO_WORK_ROOT:-"$(pwd)"}

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: SOURCE_DIR does not exist: $SOURCE_DIR"
    exit 1
fi

IO_WORK_ROOT_ABS="$(cd "$IO_WORK_ROOT" && pwd)"
host_raw=$(hostname -I 2>/dev/null || hostname -i 2>/dev/null || hostname 2>/dev/null || echo "unknown_host")
host_dir_name=$(echo "$host_raw" | tr '[:space:]' '_' | tr -cd '[:alnum:]_.:-')

if [ -z "$host_dir_name" ]; then
    host_dir_name="unknown_host"
fi

TARGET_BASE="${IO_WORK_ROOT_ABS}/${host_dir_name}"
TARGET_DIR="${TARGET_BASE}/Qwen3-VL-32B-Instruct-copy"

mkdir -p "$TARGET_BASE"

safe_delete_target() {
    case "$TARGET_DIR" in
        "${TARGET_BASE}/"*)
            rm -rf -- "$TARGET_DIR"
            ;;
        *)
            echo "ERROR: Refusing to delete unsafe path: $TARGET_DIR"
            exit 1
            ;;
    esac
}

cleanup() {
    echo
    echo "Cleaning up target directory..."
    safe_delete_target || true
    echo "Done."
}

trap cleanup SIGINT SIGTERM EXIT

echo "=========================================="
echo "IO copy-delete loop started"
echo "SOURCE_DIR: $SOURCE_DIR"
echo "IO_WORK_ROOT_ABS: $IO_WORK_ROOT_ABS"
echo "HOST_DIR_NAME: $host_dir_name"
echo "TARGET_BASE: $TARGET_BASE"
echo "TARGET_DIR: $TARGET_DIR"
echo "=========================================="

iteration=0
while true; do
    iteration=$((iteration + 1))
    echo "[IO] iteration=$iteration copying..."
    safe_delete_target

    if ! cp -a "$SOURCE_DIR" "$TARGET_DIR"; then
        echo "[IO] copy failed on iteration=$iteration, retrying..."
        safe_delete_target
        sleep 1
        continue
    fi

    echo "[IO] iteration=$iteration deleting..."
    safe_delete_target
done

