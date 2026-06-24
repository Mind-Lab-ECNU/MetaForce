#!/usr/bin/env python3
"""Configuration checks for scripts_v4 OOD-only dataset composition and tool variants."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT_DIR / "test"

def _extract_literal_assignments(path: Path, names: set[str]) -> dict:
    module = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in module.body:
        target = None
        value = None
        if isinstance(node, ast.Assign):
            for item in node.targets:
                if isinstance(item, ast.Name):
                    target = item.id
                    value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        if target in names and value is not None:
            out[target] = ast.literal_eval(value)
    return out


class ConfigurationV4Test(unittest.TestCase):
    def test_ood_configs_match_expected_layout(self):
        prepare_cfg = _extract_literal_assignments(
            TEST_DIR / "prepare_reuse_ood_test_merge_v4.py",
            {"OUTPUT_NAME", "SUBSETS", "SAMPLE_PER_SUBSET"},
        )
        merge_cfg = _extract_literal_assignments(
            TEST_DIR / "merge_datasets_to_parquet_test_v4.py",
            {"OOD_DATASETS"},
        )

        self.assertEqual(prepare_cfg["OUTPUT_NAME"], "CLEVR_MATH_test_OOD")
        self.assertEqual(
            prepare_cfg["SUBSETS"],
            ["addition", "subtraction", "subtraction_multihop", "adversarial"],
        )
        self.assertEqual(prepare_cfg["SAMPLE_PER_SUBSET"], 500)

        self.assertEqual(merge_cfg["OOD_DATASETS"], ["CLEVR_MATH_test_OOD"])

    def test_run_script_and_env_are_test_only(self):
        run_text = (ROOT_DIR / "run_v2.sh").read_text(encoding="utf-8")
        env_text = (ROOT_DIR / "run_v2.env.example").read_text(encoding="utf-8")

        self.assertIn('TEST_MERGE_DIR_NAME="${TEST_MERGE_DIR_NAME:-test_merge_v4}"', run_text)
        self.assertIn('SAMPLE_PER_SUBSET="${SAMPLE_PER_SUBSET:-500}"', run_text)
        self.assertIn('OUT_TEST_OOD_JSON="${OUT_TEST_OOD_JSON:-${OUTPUT_BASE_DIR}/${TEST_MERGE_DIR_NAME}/test_ood_merge_v4.json}"', run_text)
        self.assertIn("[1/3] prepare ood test reuse", run_text)
        self.assertIn("[2/3] update test prompts", run_text)
        self.assertIn("[3/3] merge test", run_text)
        self.assertIn("prepare_reuse_ood_test_merge_v4.py", run_text)
        self.assertIn("update_prompts_test_v4.py", run_text)
        self.assertIn("merge_datasets_to_parquet_test_v4.py", run_text)
        self.assertIn("TEST_MERGE_DIR_NAME=test_merge_v4", env_text)
        self.assertIn("OUT_TEST_OOD_JSON=", env_text)
        self.assertIn("SAMPLE_PER_SUBSET=500", env_text)

    def test_tool_json_variants_exist(self):
        for name in ["real_tool.json", "real_tool_id.json", "real_tool_ood.json"]:
            self.assertTrue((TEST_DIR / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
