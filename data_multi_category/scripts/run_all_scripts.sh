#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running scripts under ${SCRIPT_DIR}"

echo "Running chart/prepare_chartqa_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_chartqa_2000.py"

echo "Running chart/prepare_figureqa_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_figureqa_2000.py"

echo "Running chart/prepare_plotqa_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_plotqa_2000.py"

echo "Running chart/prepare_tabmwp_2000.py"
python3 "${SCRIPT_DIR}/chart/prepare_tabmwp_2000.py"

echo "Running diagram/prepare_ai2d_2000.py"
python3 "${SCRIPT_DIR}/diagram/prepare_ai2d_2000.py"

echo "Running diagram/prepare_iconqa_2000.py"
python3 "${SCRIPT_DIR}/diagram/prepare_iconqa_2000.py"

echo "Running doc/prepare_docvqa_2000.py"
python3 "${SCRIPT_DIR}/doc/prepare_docvqa_2000.py"

echo "Running general/prepare_aokvqa_2000.py"
python3 "${SCRIPT_DIR}/general/prepare_aokvqa_2000.py"

echo "Running doc/prepare_infographicvqa_2000.py"
python3 "${SCRIPT_DIR}/doc/prepare_infographicvqa_2000.py"

echo "Running geospatial/prepare_mapqa_2000.py"
python3 "${SCRIPT_DIR}/geospatial/prepare_mapqa_2000.py"

echo "Running math/prepare_geo3k_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_geo3k_2000.py"

echo "Running math/prepare_geoqa_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_geoqa_2000.py"

echo "Running math/prepare_geos_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_geos_2000.py"

echo "Running math/prepare_mathvision_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_mathvision_2000.py"

echo "Running math/prepare_mathvista_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_mathvista_2000.py"

echo "Running math/prepare_unigeon_calculation_2000.py"
python3 "${SCRIPT_DIR}/math/prepare_unigeon_calculation_2000.py"

echo "Running science/prepare_scienceqa_2000.py"
python3 "${SCRIPT_DIR}/science/prepare_scienceqa_2000.py"

# echo "Running science/prepare_thinkvl_2000.py" too big not use
# python3 "${SCRIPT_DIR}/science/prepare_thinkvl_2000.py"

echo "Running science/prepare_vizwiz_2000.py"
python3 "${SCRIPT_DIR}/science/prepare_vizwiz_2000.py"

echo "Running spatial/prepare_clevr_2000.py"
python3 "${SCRIPT_DIR}/spatial/prepare_clevr_2000.py"

echo "Running math/prepare_clevr_math_general_2000.py" # use_existing_cache_file from clevr_2000
python3 "${SCRIPT_DIR}/math/prepare_clevr_math_general_2000.py"