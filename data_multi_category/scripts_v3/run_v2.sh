#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/run_v2.env"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash ${SCRIPT_DIR}/run_v2.sh [--config /path/to/run_v2.env]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

: "${DATA_ROOT:?DATA_ROOT is required in config}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${DATA_ROOT}}"
TEST_MERGE_DIR_NAME="${TEST_MERGE_DIR_NAME:-test_merge_v3}"

SEED="${SEED:-42}"
THRESHOLD="${THRESHOLD:-5000}"
SAMPLE_OVER_THRESHOLD="${SAMPLE_OVER_THRESHOLD:-2000}"
STRICT="${STRICT:-1}"

TEST_TOOL_JSON="${TEST_TOOL_JSON:-${SCRIPT_DIR}/test/real_tool.json}"
OUT_TEST_OOD_JSON="${OUT_TEST_OOD_JSON:-${OUTPUT_BASE_DIR}/${TEST_MERGE_DIR_NAME}/test_ood_merge_v3.json}"

STRICT_ARGS=()
case "${STRICT}" in
  1|true|TRUE|yes|YES|on|ON)
    STRICT_ARGS=(--strict)
    ;;
  0|false|FALSE|no|NO|off|OFF|"")
    STRICT_ARGS=()
    ;;
  *)
    echo "Invalid STRICT value: ${STRICT}. Use 1/0 or true/false." >&2
    exit 1
    ;;
esac

COMMON_PATH_ARGS=(
  --data-root "${DATA_ROOT}"
  --output-base-dir "${OUTPUT_BASE_DIR}"
  --test-merge-dir-name "${TEST_MERGE_DIR_NAME}"
)

echo "==> [1/3] prepare ood test reuse"
python "${SCRIPT_DIR}/test/prepare_reuse_ood_test_merge_v3.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --seed "${SEED}" \
  --threshold "${THRESHOLD}" \
  --sample-over-threshold "${SAMPLE_OVER_THRESHOLD}"

echo "==> [2/3] update test prompts"
python "${SCRIPT_DIR}/test/update_prompts_test_v3.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --tool-json "${TEST_TOOL_JSON}" \
  "${STRICT_ARGS[@]}"

echo "==> [3/3] merge test"
python "${SCRIPT_DIR}/test/merge_datasets_to_parquet_test_v3.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --out-test-ood-json "${OUT_TEST_OOD_JSON}" \
  "${STRICT_ARGS[@]}"

echo "Pipeline finished."
echo "Test OOD merged JSON: ${OUT_TEST_OOD_JSON}"
