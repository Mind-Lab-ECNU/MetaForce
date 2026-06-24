#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/run_v2.env"
USE_CONFIG=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      USE_CONFIG=1
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash ${SCRIPT_DIR}/run_v2.sh [--config /path/to/run_v2.env]" >&2
      exit 1
      ;;
  esac
done

if [[ "${USE_CONFIG}" == "1" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
fi

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${DATA_ROOT}}"
SRC_REL_DIR="${SRC_REL_DIR:-geospatial/MapQA_2000}"
SRC_FILE="${SRC_FILE:-test.json}"
OUTPUT_NAME="${OUTPUT_NAME:-MapQA_test_ID}"
DATA_SOURCE="${DATA_SOURCE:-MapQA_test_ID}"
OUT_JSON_NAME="${OUT_JSON_NAME:-test.json}"
OUT_PARQUET_NAME="${OUT_PARQUET_NAME:-test.parquet}"
TOOL_JSON="${TOOL_JSON:-${SCRIPT_DIR}/../scripts_v2/test/real_tool.json}"
STRICT="${STRICT:-1}"

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

echo "==> [1/1] prepare MapQA test-only reuse"
python "${SCRIPT_DIR}/prepare_mapqa_test_reuse.py" \
  --data-root "${DATA_ROOT}" \
  --output-base-dir "${OUTPUT_BASE_DIR}" \
  --src-rel-dir "${SRC_REL_DIR}" \
  --src-file "${SRC_FILE}" \
  --output-name "${OUTPUT_NAME}" \
  --data-source "${DATA_SOURCE}" \
  --out-json-name "${OUT_JSON_NAME}" \
  --out-parquet-name "${OUT_PARQUET_NAME}" \
  --tool-json "${TOOL_JSON}" \
  "${STRICT_ARGS[@]}"

echo "Pipeline finished."
echo "Output JSON: ${OUTPUT_BASE_DIR}/${OUTPUT_NAME}/${OUT_JSON_NAME}"
echo "Output Parquet: ${OUTPUT_BASE_DIR}/${OUTPUT_NAME}/${OUT_PARQUET_NAME}"
