#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running scripts under ${SCRIPT_DIR}"


echo "Running add/prepare_chart2text_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_chart2text_2000.py"

echo "Running add/prepare_cocoqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_cocoqa_2000.py"

echo "Running add/prepare_datikz_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_datikz_2000.py"

echo "Running add/prepare_diagram_image_to_text_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_diagram_image_to_text_2000.py"

echo "Running add/prepare_dvqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_dvqa_2000.py"

echo "Running add/prepare_finqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_finqa_2000.py"

echo "Running add/prepare_geomverse_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_geomverse_2000.py"

echo "Running add/prepare_hateful_memes_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_hateful_memes_2000.py"

echo "Running add/prepare_hitab_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_hitab_2000.py"

echo "Running add/prepare_iam_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_iam_2000.py"

echo "Running add/prepare_intergps_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_intergps_2000.py"

echo "Running add/prepare_localized_narratives_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_localized_narratives_2000.py"

echo "Running add/prepare_ocrvqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_ocrvqa_2000.py"

echo "Running add/prepare_raven_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_raven_2000.py"

echo "Running add/prepare_rendered_text_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_rendered_text_2000.py"

echo "Running add/prepare_robut_sqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_robut_sqa_2000.py"

echo "Running add/prepare_robut_wikisql_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_robut_wikisql_2000.py"

echo "Running add/prepare_robut_wtq_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_robut_wtq_2000.py"

echo "Running add/prepare_screen2words_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_screen2words_2000.py"

echo "Running add/prepare_st_vqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_st_vqa_2000.py"

echo "Running add/prepare_tallyqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_tallyqa_2000.py"

echo "Running add/prepare_tat_qa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_tat_qa_2000.py"

echo "Running add/prepare_textcaps_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_textcaps_2000.py"

echo "Running add/prepare_textvqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_textvqa_2000.py"

echo "Running add/prepare_tqa_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_tqa_2000.py"

echo "Running add/prepare_vistext_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vistext_2000.py"

echo "Running add/prepare_visual7w_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_visual7w_2000.py"

echo "Running add/prepare_visualmrc_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_visualmrc_2000.py"

echo "Running add/prepare_vqarad_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vqarad_2000.py"

echo "Running add/prepare_vqav2_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vqav2_2000.py"

echo "Running add/prepare_vsr_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_vsr_2000.py"

echo "Running add/prepare_websight_2000.py"
python3 "${SCRIPT_DIR}/add/prepare_websight_2000.py"
