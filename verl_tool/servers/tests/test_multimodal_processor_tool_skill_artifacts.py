#!/usr/bin/env python
import asyncio
import os
import sys
import unittest
from types import MethodType

# 方便直接运行测试脚本时导入工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.multimodal_processor_tool_adapt_skill import MultimodalProcessorTool


class TestSkillArtifactGeneration(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_stage2_called_once_per_script_and_canonical_skill_md(self):
        tool = MultimodalProcessorTool()
        call_log = []

        async def fake_call(self, model_name, system_prompt, user_prompt):
            call_log.append(user_prompt)
            if "Stage 1 (planning only)" in user_prompt:
                return (
                    '{"skill_name":"demo-skill","skill_description":"demo desc",'
                    '"skill_overview_md":"### Scripts\\n- `scripts/a.py`: do A\\n- `scripts/b.sh`: do B\\n\\n### Parameters\\n- `--mode` (required): run mode\\n\\n### When to call\\n- Call model only after input is ready.",'
                    '"scripts":[{"path":"scripts/a.py","purpose":"do A","params":[],"notes":"n1","code":"bad"},'
                    '{"path":"scripts/b.sh","purpose":"do B","params":[],"notes":"n2"}]}',
                    {},
                )
            if "Target script path: scripts/a.py" in user_prompt:
                return '{"path":"scripts/a.py","content":"print(\\"a\\")\\n"}', {}
            if "Target script path: scripts/b.sh" in user_prompt:
                return '{"path":"scripts/b.sh","content":"#!/usr/bin/env bash\\necho b\\n"}', {}
            return "{}", {}

        tool._call_text_model_async = MethodType(fake_call, tool)
        artifacts = self._run(tool._generate_skill_artifacts("Do something useful", []))
        self.assertIn("skill_overview_md", call_log[0])
        self.assertIn("when to call each script/model", call_log[0])
        self.assertIn("requires_image", call_log[0])
        self.assertIn("Do not add image contract boilerplate to skill_overview_md", call_log[0])
        self.assertNotIn("unless needed", call_log[0])
        self.assertIn("Skill overview markdown (from Stage 1):", call_log[1])
        self.assertIn("Call model only after input is ready.", call_log[1])
        self.assertIn("if the script needs image input", call_log[1])
        self.assertIn("must read image data from `SKILL_IMAGE_PATH` / `SKILL_IMAGE_DATA_URL`", call_log[1])
        self.assertIn("Do not declare `--image_index` in the script", call_log[1])
        self.assertIn("print key intermediate variables and the final result", call_log[1])

        self.assertEqual(artifacts["name"], "demo-skill")
        self.assertEqual(artifacts["description"], "demo desc")
        self.assertIn("name: demo-skill", artifacts["skill_md"])
        self.assertIn("description: demo desc", artifacts["skill_md"])
        self.assertIn("## Usage", artifacts["skill_md"])
        self.assertIn("## Image Input Contract", artifacts["skill_md"])
        self.assertNotIn("## Runtime Contract", artifacts["skill_md"])
        self.assertNotIn("## Stage1 Spec (Machine Readable)", artifacts["skill_md"])
        self.assertIsInstance(artifacts.get("stage1_spec"), dict)
        self.assertIn("scripts", artifacts["stage1_spec"])
        self.assertEqual(len(artifacts["files"]), 2)
        self.assertEqual(artifacts["files"][0]["path"], "scripts/a.py")
        self.assertEqual(artifacts["files"][1]["path"], "scripts/b.sh")
        self.assertEqual(len(call_log), 3, "Expected 1 Stage-1 call + 2 per-script Stage-2 calls")

    def test_invalid_extension_filtered(self):
        tool = MultimodalProcessorTool()
        stage2_targets = []

        async def fake_call(self, model_name, system_prompt, user_prompt):
            if "Stage 1 (planning only)" in user_prompt:
                return (
                    '{"skill_name":"ext-test","skill_description":"ext desc","scripts":['
                    '{"path":"scripts/x.js","purpose":"bad ext","params":[],"notes":""},'
                    '{"path":"scripts/y.py","purpose":"good ext","params":[],"notes":""}]}',
                    {},
                )
            if "Stage 2 (single script generation)" in user_prompt:
                if "Target script path: scripts/y.py" in user_prompt:
                    stage2_targets.append("scripts/y.py")
                    return '{"path":"scripts/y.py","content":"print(\\"ok\\")\\n"}', {}
            return "{}", {}

        tool._call_text_model_async = MethodType(fake_call, tool)
        artifacts = self._run(tool._generate_skill_artifacts("desc", []))
        self.assertEqual(stage2_targets, ["scripts/y.py"])
        self.assertEqual([f["path"] for f in artifacts["files"]], ["scripts/y.py"])

    def test_mismatched_stage2_path_uses_fallback(self):
        tool = MultimodalProcessorTool()

        async def fake_call(self, model_name, system_prompt, user_prompt):
            if "Stage 1 (planning only)" in user_prompt:
                return (
                    '{"skill_name":"fallback-case","skill_description":"fallback desc","scripts":['
                    '{"path":"scripts/a.py","purpose":"A purpose","params":[{"name":"mode","type":"string","required":true,"description":"mode"}],"notes":""},'
                    '{"path":"scripts/b.sh","purpose":"B purpose","params":[],"notes":""}]}',
                    {},
                )
            if "Target script path: scripts/a.py" in user_prompt:
                return '{"path":"scripts/x.py","content":"print(\\"wrong\\")"}', {}
            if "Target script path: scripts/b.sh" in user_prompt:
                return '{"path":"scripts/b.sh","content":""}', {}
            return "{}", {}

        tool._call_text_model_async = MethodType(fake_call, tool)
        artifacts = self._run(tool._generate_skill_artifacts("desc", []))
        self.assertEqual(len(artifacts["files"]), 2)
        self.assertEqual(artifacts["files"][0]["path"], "scripts/a.py")
        self.assertIn("import argparse", artifacts["files"][0]["content"])
        self.assertEqual(artifacts["files"][1]["path"], "scripts/b.sh")
        self.assertIn("#!/usr/bin/env bash", artifacts["files"][1]["content"])
        self.assertGreaterEqual(len(artifacts.get("generation_warnings", [])), 2)


if __name__ == "__main__":
    unittest.main()
