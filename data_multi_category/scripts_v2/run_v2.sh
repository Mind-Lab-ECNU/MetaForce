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
TRAIN_MERGE_DIR_NAME="${TRAIN_MERGE_DIR_NAME:-train_merge_v2}"
TEST_MERGE_DIR_NAME="${TEST_MERGE_DIR_NAME:-test_merge_v2}"

SEED="${SEED:-42}"
TRAIN_SAMPLE="${TRAIN_SAMPLE:-600}"
MAPQA_TRAIN_SAMPLE="${MAPQA_TRAIN_SAMPLE:-600}"
CLEVR_MATH_TRAIN_SAMPLE_PER_SUBSET="${CLEVR_MATH_TRAIN_SAMPLE_PER_SUBSET:-150}"
THRESHOLD="${THRESHOLD:-5000}"
SAMPLE_OVER_THRESHOLD="${SAMPLE_OVER_THRESHOLD:-2000}"
SAMPLE_PER_SUBSET="${SAMPLE_PER_SUBSET:-500}"
STRICT="${STRICT:-1}"

TRAIN_TOOL_JSON="${TRAIN_TOOL_JSON:-${SCRIPT_DIR}/train/real_tool.json}"
TEST_TOOL_JSON="${TEST_TOOL_JSON:-${SCRIPT_DIR}/test/real_tool.json}"

OUT_TRAIN_JSON="${OUT_TRAIN_JSON:-${OUTPUT_BASE_DIR}/${TRAIN_MERGE_DIR_NAME}/train_merge_v2.json}"
OUT_TEST_ALL_JSON="${OUT_TEST_ALL_JSON:-${OUTPUT_BASE_DIR}/${TEST_MERGE_DIR_NAME}/test_merge_v2.json}"
OUT_TEST_ID_JSON="${OUT_TEST_ID_JSON:-${OUTPUT_BASE_DIR}/${TEST_MERGE_DIR_NAME}/test_id_merge_v2.json}"
OUT_TEST_OOD_JSON="${OUT_TEST_OOD_JSON:-${OUTPUT_BASE_DIR}/${TEST_MERGE_DIR_NAME}/test_ood_merge_v2.json}"

RUN_STAGES="${RUN_STAGES:-all}"
case "${RUN_STAGES}" in
  train|test|all) ;;
  *)
    echo "Invalid RUN_STAGES value: ${RUN_STAGES}. Use train / test / all." >&2
    exit 1
    ;;
esac

run_train() { [[ "${RUN_STAGES}" == "train" || "${RUN_STAGES}" == "all" ]]; }
run_test()  { [[ "${RUN_STAGES}" == "test"  || "${RUN_STAGES}" == "all" ]]; }

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
  --train-merge-dir-name "${TRAIN_MERGE_DIR_NAME}"
  --test-merge-dir-name "${TEST_MERGE_DIR_NAME}"
)

run_train && echo "==> [1/8] prepare train reuse"
run_train && python "${SCRIPT_DIR}/train/prepare_reuse_train_merge_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --seed "${SEED}" \
  --train-sample "${TRAIN_SAMPLE}" \
  --mapqa-train-sample "${MAPQA_TRAIN_SAMPLE}" \
  --clevr-math-train-sample-per-subset "${CLEVR_MATH_TRAIN_SAMPLE_PER_SUBSET}"

run_test && echo "==> [2/8] prepare iid test reuse"
run_test && python "${SCRIPT_DIR}/test/prepare_reuse_iid_test_merge_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --seed "${SEED}" \
  --train-sample "${TRAIN_SAMPLE}" \
  --mapqa-train-sample "${MAPQA_TRAIN_SAMPLE}"

run_test && echo "==> [3/8] prepare ood test reuse"
run_test && python "${SCRIPT_DIR}/test/prepare_reuse_ood_test_merge_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --seed "${SEED}" \
  --threshold "${THRESHOLD}" \
  --sample-over-threshold "${SAMPLE_OVER_THRESHOLD}"

run_test && echo "==> [4/8] prepare clevr_math merged test"
run_test && python "${SCRIPT_DIR}/test/prepare_clevr_math_test_merged_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --seed "${SEED}" \
  --sample-per-subset "${SAMPLE_PER_SUBSET}"

run_train && echo "==> [5/8] update train prompts"
run_train && python "${SCRIPT_DIR}/train/update_prompts_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --tool-json "${TRAIN_TOOL_JSON}" \
  "${STRICT_ARGS[@]}"

run_test && echo "==> [6/8] update test prompts"
run_test && python "${SCRIPT_DIR}/test/update_prompts_test_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --tool-json "${TEST_TOOL_JSON}" \
  "${STRICT_ARGS[@]}"

run_train && echo "==> [7/8] merge train"
run_train && python "${SCRIPT_DIR}/train/merge_train_datasets_to_parquet_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --out-train-json "${OUT_TRAIN_JSON}" \
  "${STRICT_ARGS[@]}"

run_test && echo "==> [8/8] merge test"
run_test && python "${SCRIPT_DIR}/test/merge_datasets_to_parquet_test_v2.py" \
  "${COMMON_PATH_ARGS[@]}" \
  --out-test-all-json "${OUT_TEST_ALL_JSON}" \
  --out-test-id-json "${OUT_TEST_ID_JSON}" \
  --out-test-ood-json "${OUT_TEST_OOD_JSON}" \
  "${STRICT_ARGS[@]}"

echo "Pipeline finished."
echo "Train merged JSON: ${OUT_TRAIN_JSON}"
echo "Test merged JSON: ${OUT_TEST_ALL_JSON}"
echo "Test ID merged JSON: ${OUT_TEST_ID_JSON}"
echo "Test OOD merged JSON: ${OUT_TEST_OOD_JSON}"
