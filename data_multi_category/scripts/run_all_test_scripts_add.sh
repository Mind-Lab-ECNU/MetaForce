#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running add test/val scripts under ${SCRIPT_DIR}"

echo "Running prepare_datikz_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_datikz_test_2000.py"

# echo "Running prepare_hateful_memes_test_2000.py" no_use
# python3 "${SCRIPT_DIR}/add/prepare_hateful_memes_test_2000.py"

echo "Running prepare_ocrvqa_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_ocrvqa_test_2000.py"

echo "Running prepare_tallyqa_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_tallyqa_test_2000.py"

echo "Running prepare_tat_dqa_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_tat_dqa_test_2000.py"

echo "Running prepare_textcaps_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_textcaps_test_2000.py"

echo "Running prepare_textvqa_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_textvqa_test_2000.py"

echo "Running prepare_vqarad_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vqarad_test_2000.py"

echo "Running prepare_vqav2_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vqav2_test_2000.py"

echo "Running prepare_vsr_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vsr_test_2000.py"

echo "Running prepare_websight_test_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_websight_test_2000.py"
