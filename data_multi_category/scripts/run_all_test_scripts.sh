#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running split test scripts under ${SCRIPT_DIR}"

echo "Running chart/prepare_plotqa_test_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_plotqa_test_2000.py"

echo "Running diagram/prepare_iconqa_val_2000.py"
python3 "${SCRIPT_DIR}/diagram/prepare_iconqa_val_2000.py"

echo "Running diagram/prepare_iconqa_test_2000.py"
python3 "${SCRIPT_DIR}/diagram/prepare_iconqa_test_2000.py"

echo "Running diagram/prepare_ai2d_no_mask_test_2000.py"
python3 "${SCRIPT_DIR}/diagram/prepare_ai2d_no_mask_test_2000.py"

echo "Running geospatial/prepare_mapqa_vs_test_2000.py"
python3 "${SCRIPT_DIR}/geospatial/prepare_mapqa_vs_test_2000.py"

echo "Running math/prepare_geoqa_test_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_geoqa_test_2000.py"

echo "Running chart/prepare_figureqa_validation2_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_figureqa_validation2_2000.py"

echo "Running chart/prepare_figureqa_no_annot_test2_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_figureqa_no_annot_test2_2000.py"
