#!/usr/bin/env python
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import fire

# 方便直接运行测试脚本时导入工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.multimodal_processor_tool_adapt_skill_ood import MultimodalProcessorTool

# 测试用图片路径，请替换为真实图片路径
IMAGE_PATH = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general/HatefulMemes_2000/images/train/train_1.png"


def _build_action(model_name: str, prompt: str, image_index: int) -> str:
    """构建 <tool_call> 格式的 action 字符串。"""
    payload = {
        "name": model_name,
        "arguments": {
            "prompt": prompt,
            "image_index": image_index,
        },
    }
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _build_wolfram_action(query: str) -> str:
    payload = {
        "name": "WolframAlpha",
        "arguments": {
            "query": query,
        },
    }
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _build_generic_action(name: str, arguments=None) -> str:
    payload = {
        "name": name,
        "arguments": arguments,
    }
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _setup_skill_store() -> Path:
    store_dir = os.environ.get("VERL_SKILL_STORE_DIR")
    if store_dir:
        temp_dir = Path(store_dir)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="skill_test_"))
        os.environ["VERL_SKILL_STORE_DIR"] = str(temp_dir)

    md_only_dir = temp_dir / "md-only-skill"
    md_only_dir.mkdir(parents=True, exist_ok=True)
    md_only_md = (
        "---\n"
        "name: md-only-skill\n"
        "description: Return markdown only.\n"
        "---\n\n"
        "## Usage\n"
        "This skill only returns SKILL.md and has no executable scripts.\n"
    )
    (md_only_dir / "SKILL.md").write_text(md_only_md, encoding="utf-8")

    run_dir = temp_dir / "echo-skill"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_md = (
        "---\n"
        "name: echo-skill\n"
        "description: Echo args and required image env.\n"
        "requires_image: true\n"
        "---\n\n"
        "## Usage\n"
        "Use this skill to echo inputs for testing.\n\n"
        "### Image Input Contract\n"
        "- args.image_index is required.\n"
        "- Runtime will inject SKILL_IMAGE_PATH / SKILL_IMAGE_DATA_URL; script should read env only.\n\n"
        "## Executable\n"
        "entrypoints:\n"
        "- scripts/run.py\n\n"
        "args:\n"
        "- note (optional)\n"
        "\n## Stage1 Spec (Machine Readable)\n"
        "```json\n"
        "{\n"
        "  \"requires_image\": true,\n"
        "  \"scripts\": [\n"
        "    {\n"
        "      \"path\": \"scripts/run.py\",\n"
        "      \"params\": [\n"
        "        {\"name\": \"image_index\", \"type\": \"integer\", \"required\": true, \"description\": \"caller image selector\"},\n"
        "        {\"name\": \"note\", \"type\": \"string\", \"required\": false, \"description\": \"optional note\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )
    (run_dir / "SKILL.md").write_text(run_md, encoding="utf-8")
    run_spec = {
        "requires_image": True,
        "scripts": [
            {
                "path": "scripts/run.py",
                "params": [
                    {
                        "name": "image_index",
                        "type": "integer",
                        "required": True,
                        "description": "caller image selector",
                    },
                    {
                        "name": "note",
                        "type": "string",
                        "required": False,
                        "description": "optional note",
                    },
                ],
            }
        ],
    }
    (run_dir / "SKILL_SPEC.json").write_text(json.dumps(run_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    scripts_dir = run_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    run_py = (
        "import argparse\n"
        "import os\n"
        "\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument(\"--note\", default=\"ok\")\n"
        "args = parser.parse_args()\n"
        "\n"
        "print(f\"echo-skill note={args.note}\")\n"
        "print(f\"SKILL_IMAGE_PATH_SET={bool(os.getenv('SKILL_IMAGE_PATH'))}\")\n"
        "print(f\"SKILL_IMAGE_DATA_URL_SET={bool(os.getenv('SKILL_IMAGE_DATA_URL'))}\")\n"
        "if os.getenv(\"SKILL_IMAGE_PATH\"):\n"
        "    print(f\"SKILL_IMAGE_PATH={os.getenv('SKILL_IMAGE_PATH')}\")\n"
        "if os.getenv(\"SKILL_IMAGE_DATA_URL\"):\n"
        "    print(\"SKILL_IMAGE_DATA_URL=present\")\n"
    )
    (scripts_dir / "run.py").write_text(run_py, encoding="utf-8")

    no_image_dir = temp_dir / "no-image-skill"
    no_image_dir.mkdir(parents=True, exist_ok=True)
    no_image_md = (
        "---\n"
        "name: no-image-skill\n"
        "description: No image input required.\n"
        "requires_image: false\n"
        "---\n\n"
        "## Usage\n"
        "This skill does not need images.\n\n"
        "### Image Input Contract\n"
        "- Do not pass image_index.\n"
        "\n## Stage1 Spec (Machine Readable)\n"
        "```json\n"
        "{\n"
        "  \"requires_image\": false,\n"
        "  \"scripts\": [\n"
        "    {\n"
        "      \"path\": \"scripts/run.py\",\n"
        "      \"params\": [\n"
        "        {\"name\": \"note\", \"type\": \"string\", \"required\": false, \"description\": \"optional note\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )
    (no_image_dir / "SKILL.md").write_text(no_image_md, encoding="utf-8")
    no_image_spec = {
        "requires_image": False,
        "scripts": [
            {
                "path": "scripts/run.py",
                "params": [
                    {
                        "name": "note",
                        "type": "string",
                        "required": False,
                        "description": "optional note",
                    }
                ],
            }
        ],
    }
    (no_image_dir / "SKILL_SPEC.json").write_text(
        json.dumps(no_image_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    no_image_scripts = no_image_dir / "scripts"
    no_image_scripts.mkdir(parents=True, exist_ok=True)
    no_image_py = (
        "import argparse\n"
        "import os\n"
        "\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument(\"--note\", default=\"ok\")\n"
        "args = parser.parse_args()\n"
        "print(f\"no-image-skill note={args.note}\")\n"
        "print(f\"SKILL_IMAGE_PATH_SET={bool(os.getenv('SKILL_IMAGE_PATH'))}\")\n"
    )
    (no_image_scripts / "run.py").write_text(no_image_py, encoding="utf-8")

    multi_dir = temp_dir / "multi-entrypoint-skill"
    multi_dir.mkdir(parents=True, exist_ok=True)
    multi_md = (
        "---\n"
        "name: multi-entrypoint-skill\n"
        "description: Skill with image and non-image entrypoints.\n"
        "requires_image: true\n"
        "---\n\n"
        "## Usage\n"
        "Use scripts/a.py for image flow, scripts/b.py for text flow.\n"
        "\n## Stage1 Spec (Machine Readable)\n"
        "```json\n"
        "{\n"
        "  \"requires_image\": true,\n"
        "  \"scripts\": [\n"
        "    {\n"
        "      \"path\": \"scripts/a.py\",\n"
        "      \"params\": [\n"
        "        {\"name\": \"image_index\", \"type\": \"integer\", \"required\": true, \"description\": \"caller image selector\"}\n"
        "      ]\n"
        "    },\n"
        "    {\n"
        "      \"path\": \"scripts/b.py\",\n"
        "      \"params\": [\n"
        "        {\"name\": \"note\", \"type\": \"string\", \"required\": false, \"description\": \"optional\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )
    (multi_dir / "SKILL.md").write_text(multi_md, encoding="utf-8")
    multi_spec = {
        "requires_image": True,
        "scripts": [
            {
                "path": "scripts/a.py",
                "params": [
                    {
                        "name": "image_index",
                        "type": "integer",
                        "required": True,
                        "description": "caller image selector",
                    }
                ],
            },
            {
                "path": "scripts/b.py",
                "params": [
                    {
                        "name": "note",
                        "type": "string",
                        "required": False,
                        "description": "optional",
                    }
                ],
            },
        ],
    }
    (multi_dir / "SKILL_SPEC.json").write_text(json.dumps(multi_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    multi_scripts = multi_dir / "scripts"
    multi_scripts.mkdir(parents=True, exist_ok=True)
    (multi_scripts / "a.py").write_text(
        "import os\n"
        "print(f\"multi-a path_set={bool(os.getenv('SKILL_IMAGE_PATH'))}\")\n",
        encoding="utf-8",
    )
    (multi_scripts / "b.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--note', default='ok')\n"
        "args = parser.parse_args()\n"
        "print(f\"multi-b note={args.note}\")\n",
        encoding="utf-8",
    )

    legacy_dir = temp_dir / "legacy-image-skill"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_md = (
        "---\n"
        "name: legacy-image-skill\n"
        "description: Requires image in frontmatter but has no stage1 spec block.\n"
        "requires_image: true\n"
        "---\n\n"
        "## Usage\n"
        "Legacy skill for compatibility fallback.\n"
    )
    (legacy_dir / "SKILL.md").write_text(legacy_md, encoding="utf-8")
    legacy_scripts = legacy_dir / "scripts"
    legacy_scripts.mkdir(parents=True, exist_ok=True)
    (legacy_scripts / "run.py").write_text(
        "print('legacy skill run ok')\n",
        encoding="utf-8",
    )

    return temp_dir


_SKILL_STORE = _setup_skill_store()


def _encode_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_tool_initialization():
    """测试工具初始化与子工具注册。"""
    print(">>> [START] test_tool_initialization")
    print("  初始化 MultimodalProcessorTool...")
    tool = MultimodalProcessorTool()
    print(f"  tool_type: {tool.tool_type}")
    assert tool.tool_type == "multimodal_processor_tool_adapt_skill_ood"
    print("  验证子工具注册...")
    assert "InternVL3.5-38B-Instruct" in tool.valid_mcp_func_names
    assert "InternVL3.5-14B-Instruct" in tool.valid_mcp_func_names
    assert "WolframAlpha" in tool.valid_mcp_func_names
    assert "UniChart" in tool.valid_mcp_func_names
    assert "Deplot" in tool.valid_mcp_func_names
    assert "ChartMoe" in tool.valid_mcp_func_names
    assert "step3" in tool.valid_mcp_func_names
    assert "python_code" in tool.valid_mcp_func_names
    print(f"  已注册子工具数量: {len(tool.valid_mcp_func_names)}")
    print(">>> [PASSED] test_tool_initialization")


def test_parameter_validation():
    """测试参数校验（缺失 prompt / image_index）。"""
    print(">>> [START] test_parameter_validation")
    tool = MultimodalProcessorTool()
    print("  构建缺失 image_index 的 action...")
    action = "<tool_call>{\"name\": \"InternVL3.5-38B-Instruct\", \"arguments\": {\"prompt\": \"hi\"}}</tool_call>"
    obs, done, valid = tool.conduct_action("t-001", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}, invalid_reason={obs.get('invalid_reason')}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    assert obs.get("tool") == "InternVL3.5-38B-Instruct"
    print(">>> [PASSED] test_parameter_validation")


def test_wolfram_parameter_validation():
    """测试 WolframAlpha 参数校验（缺失 query）。"""
    print(">>> [START] test_wolfram_parameter_validation")
    tool = MultimodalProcessorTool()
    print("  构建缺失 query 的 action...")
    action = "<tool_call>{\"name\": \"WolframAlpha\", \"arguments\": {}}</tool_call>"
    obs, done, valid = tool.conduct_action("t-004", action, {})
    print(f"  obs={obs}, valid={valid}, invalid_reason={obs.get('invalid_reason')}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    assert obs.get("tool") == "WolframAlpha"
    print(">>> [PASSED] test_wolfram_parameter_validation")


def test_model_call():
    """测试 VLM 模型调用流程（真实请求）。"""
    print(">>> [START] test_model_call")
    tool = MultimodalProcessorTool()

    model_names = [
        "InternVL3.5-38B-Instruct",
        "InternVL3.5-14B-Instruct",
    ]
    for idx, model_name in enumerate(model_names, start=2):
        print(f"  调用 {model_name}...")
        action = _build_action(model_name, "请描述图片", 1)
        obs, done, valid = tool.conduct_action(f"t-00{idx}", action, {"images": [IMAGE_PATH]})
        print(f"    obs={obs}, valid={valid}")
        assert valid is True, f"{model_name} call should be valid, got: {obs}"
        assert obs.get("tool") == model_name
        assert model_name in obs.get("obs", "")
        print(f"  - {model_name} passed")
    print(">>> [PASSED] test_model_call")


def test_wolfram_call():
    """测试 WolframAlpha 调用流程（真实请求）。"""
    print(">>> [START] test_wolfram_call")
    tool = MultimodalProcessorTool()

    print("  调用 WolframAlpha...")
    action = _build_wolfram_action("10 densest elemental metals")
    obs, done, valid = tool.conduct_action("t-005", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"WolframAlpha call should be valid, got: {obs}"
    assert obs.get("tool") == "WolframAlpha"
    assert "[Tool: WolframAlpha]" in obs.get("obs", "")
    print(">>> [PASSED] test_wolfram_call")


def test_unichart_call():
    """测试 UniChart 调用流程（真实请求）。"""
    print(">>> [START] test_unichart_call")
    tool = MultimodalProcessorTool()

    print("  调用 UniChart...")
    action = _build_generic_action("UniChart", {"image_index": 1})
    obs, done, valid = tool.conduct_action("t-005-unichart", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"UniChart call should be valid, got: {obs}"
    assert obs.get("tool") == "UniChart"
    assert "[Tool: UniChart]" in obs.get("obs", "")
    print(">>> [PASSED] test_unichart_call")


def test_deplot_call():
    """测试 Deplot 调用流程（真实请求）。"""
    print(">>> [START] test_deplot_call")
    tool = MultimodalProcessorTool()

    print("  调用 Deplot...")
    action = _build_generic_action("Deplot", {"image_index": 1})
    obs, done, valid = tool.conduct_action("t-005-deplot", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"Deplot call should be valid, got: {obs}"
    assert obs.get("tool") == "Deplot"
    assert "[Tool: Deplot]" in obs.get("obs", "")
    print(">>> [PASSED] test_deplot_call")


def test_chartmoe_call():
    """测试 ChartMoe 调用流程（真实请求）。"""
    print(">>> [START] test_chartmoe_call")
    tool = MultimodalProcessorTool()

    print("  调用 ChartMoe...")
    action = _build_generic_action("ChartMoe", {"image_index": 1})
    obs, done, valid = tool.conduct_action("t-005-chartmoe", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"ChartMoe call should be valid, got: {obs}"
    assert obs.get("tool") == "ChartMoe"
    assert "[Tool: ChartMoe]" in obs.get("obs", "")
    print(">>> [PASSED] test_chartmoe_call")


def test_step3_call():
    """测试 step3 通过 _call_model_async 路径调用。"""
    print(">>> [START] test_step3_call")
    tool = MultimodalProcessorTool()

    print("  调用 step3...")
    action = _build_action("step3", "请描述图片", 1)
    obs, done, valid = tool.conduct_action("t-005-step3", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"step3 call should be valid, got: {obs}"
    assert obs.get("tool") == "step3"
    print(">>> [PASSED] test_step3_call")


def test_python_code_call():
    """测试 python_code 调用流程（本地执行）。"""
    print(">>> [START] test_python_code_call")
    tool = MultimodalProcessorTool()

    print("  调用 python_code...")
    action = _build_generic_action("python_code", {"code": "1 + 1"})
    obs, done, valid = tool.conduct_action("t-005-python", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"python_code call should be valid, got: {obs}"
    assert obs.get("tool") == "python_code"
    assert "2" in obs.get("obs", ""), f"Expected auto-printed result, got: {obs}"
    print(">>> [PASSED] test_python_code_call")


def test_python_code_parameter_validation():
    """测试 python_code 参数校验（缺失 code）。"""
    print(">>> [START] test_python_code_parameter_validation")
    tool = MultimodalProcessorTool()

    print("  构建缺失 code 的 action...")
    action = _build_generic_action("python_code", {})
    obs, done, valid = tool.conduct_action("t-005-python-missing", action, {})
    print(f"  obs={obs}, valid={valid}, invalid_reason={obs.get('invalid_reason')}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    print(">>> [PASSED] test_python_code_parameter_validation")


def test_skill_md_call():
    """测试 skill 的 MD 调用流程。"""
    print(">>> [START] test_skill_md_call")
    tool = MultimodalProcessorTool()
    action = _build_generic_action("md-only-skill", {})
    obs, done, valid = tool.conduct_action("t-skill-md-001", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"skill MD call should be valid, got: {obs}"
    assert obs.get("tool") == "md-only-skill"
    assert "md-only-skill" in obs.get("obs", "")
    print(">>> [PASSED] test_skill_md_call")


def test_create_skill_parse_with_arguments():
    """测试 create_skill 仅接受 arguments.description。"""
    print(">>> [START] test_create_skill_parse_with_arguments")
    tool = MultimodalProcessorTool()
    action = _build_generic_action(
        "create_skill",
        {"description": "Extract key data from bar charts and compute summary stats"},
    )
    parsed, is_valid = tool.parse_action(action)
    print(f"  parsed={parsed}, is_valid={is_valid}")
    assert is_valid is True, "create_skill with arguments.description should be valid"
    assert parsed.get("name") == "__create_skill__"
    assert parsed.get("arguments", {}).get("description") == "Extract key data from bar charts and compute summary stats"
    print(">>> [PASSED] test_create_skill_parse_with_arguments")


def test_create_skill_parse_reject_top_level_description():
    """测试 create_skill 顶层 description 不再支持。"""
    print(">>> [START] test_create_skill_parse_reject_top_level_description")
    tool = MultimodalProcessorTool()
    action = (
        "<tool_call>{\"name\": \"create_skill\", "
        "\"description\": \"Extract key data from bar charts and compute summary stats\"}</tool_call>"
    )
    parsed, is_valid = tool.parse_action(action)
    print(f"  parsed={parsed}, is_valid={is_valid}")
    assert is_valid is True, "top-level description format should parse but fail validation later"
    assert parsed.get("name") == "__create_skill__"
    assert parsed.get("arguments", {}).get("description") is None

    obs, done, valid = tool.conduct_action("t-create-skill-legacy-001", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_parameters"
    print(">>> [PASSED] test_create_skill_parse_reject_top_level_description")


def test_create_skill_real_call():
    """测试 create_skill 真实调用模型生成 skill。"""
    print(">>> [START] test_create_skill_real_call")
    tool = MultimodalProcessorTool()
    description = (
        "Create a text-only skill to summarize a paragraph and extract top keywords. "
        "The skill should include clear usage guidance and script entrypoints."
    )
    action = _build_generic_action("create_skill", {"description": description})
    obs, done, valid = tool.conduct_action("t-create-skill-real-001", action, {})
    print(f"  obs={obs}, valid={valid}")

    assert valid is True, f"create_skill real call should be valid, got: {obs}"
    assert obs.get("tool") == "create_skill"
    assert isinstance(obs.get("skill_name"), str) and bool(obs.get("skill_name"))
    assert isinstance(obs.get("skill_path"), str) and bool(obs.get("skill_path"))
    skill_dir = Path(obs["skill_path"])
    assert skill_dir.exists() and skill_dir.is_dir(), f"skill dir not found: {skill_dir}"
    skill_md_path = skill_dir / "SKILL.md"
    assert skill_md_path.exists(), f"SKILL.md not found in: {skill_dir}"
    assert "[Skill:" in obs.get("obs", "")
    print(">>> [PASSED] test_create_skill_real_call")


def test_skill_md_call_with_null_arguments():
    """测试无参 skill 调用使用 arguments=null。"""
    print(">>> [START] test_skill_md_call_with_null_arguments")
    tool = MultimodalProcessorTool()
    action = _build_generic_action("md-only-skill", None)
    obs, done, valid = tool.conduct_action("t-skill-md-null-001", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"skill MD call with null arguments should be valid, got: {obs}"
    assert obs.get("tool") == "md-only-skill"
    assert "md-only-skill" in obs.get("obs", "")
    print(">>> [PASSED] test_skill_md_call_with_null_arguments")


def test_run_skill_call():
    """测试 run_skill 调用流程（图像 skill 必传 image_index）。"""
    print(">>> [START] test_run_skill_call")
    tool = MultimodalProcessorTool()
    action = _build_generic_action(
        "run_skill",
        {
            "skill_name": "echo-skill",
            "entrypoint": "scripts/run.py",
            "args": {"image_index": 1},
        },
    )
    obs, done, valid = tool.conduct_action("t-skill-run-001", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"run_skill call should be valid, got: {obs}"
    assert "echo-skill" in obs.get("obs", "")
    assert "SKILL_IMAGE_PATH_SET=True" in obs.get("obs", "")
    assert isinstance(obs.get("latency"), (int, float)), "latency should be reported"
    print(">>> [PASSED] test_run_skill_call")


def test_run_skill_call_with_raw_base64_image():
    """测试 run_skill 接收裸 base64 图像并通过临时文件注入路径。"""
    print(">>> [START] test_run_skill_call_with_raw_base64_image")
    tool = MultimodalProcessorTool()
    image_base64 = _encode_image_as_base64(IMAGE_PATH)
    action = _build_generic_action(
        "run_skill",
        {
            "skill_name": "echo-skill",
            "entrypoint": "scripts/run.py",
            "args": {"image_index": 1},
        },
    )
    obs, done, valid = tool.conduct_action("t-skill-run-b64-001", action, {"images": [image_base64]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"run_skill raw base64 call should be valid, got: {obs}"
    assert "SKILL_IMAGE_PATH_SET=True" in obs.get("obs", "")
    assert "SKILL_IMAGE_DATA_URL_SET=False" in obs.get("obs", "")
    print(">>> [PASSED] test_run_skill_call_with_raw_base64_image")


def test_run_skill_invalid_long_string_image_input():
    """测试 run_skill 对超长非法图片字符串返回明确错误而不是异常。"""
    print(">>> [START] test_run_skill_invalid_long_string_image_input")
    tool = MultimodalProcessorTool()
    bad_image = "not_base64_" * 800
    action = _build_generic_action(
        "run_skill",
        {
            "skill_name": "echo-skill",
            "entrypoint": "scripts/run.py",
            "args": {"image_index": 1},
        },
    )
    obs, done, valid = tool.conduct_action("t-skill-run-invalid-img-001", action, {"images": [bad_image]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is False
    assert obs.get("invalid_reason") == "invalid_image_input"
    print(">>> [PASSED] test_run_skill_invalid_long_string_image_input")


def test_run_skill_missing_required_image_index():
    """测试图像 skill 缺失 image_index 时直接报错。"""
    print(">>> [START] test_run_skill_missing_required_image_index")
    tool = MultimodalProcessorTool()
    action = _build_generic_action(
        "run_skill",
        {
            "skill_name": "echo-skill",
            "entrypoint": "scripts/run.py",
            "args": {},
        },
    )
    obs, done, valid = tool.conduct_action("t-skill-run-missing-img-001", action, {"images": [IMAGE_PATH]})
    print(f"  obs={obs}, valid={valid}")
    assert valid is False
    assert obs.get("invalid_reason") == "missing_image_index"
    assert obs.get("tool") == "run_skill"
    print(">>> [PASSED] test_run_skill_missing_required_image_index")


def test_run_skill_non_image_skill_without_image_index():
    """测试非图像 skill 不传 image_index 也应成功。"""
    print(">>> [START] test_run_skill_non_image_skill_without_image_index")
    tool = MultimodalProcessorTool()
    action = _build_generic_action(
        "run_skill",
        {
            "skill_name": "no-image-skill",
            "entrypoint": "scripts/run.py",
            "args": {"note": "ok"},
        },
    )
    obs, done, valid = tool.conduct_action("t-skill-run-no-image-001", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True, f"run_skill call should be valid, got: {obs}"
    assert "no-image-skill note=ok" in obs.get("obs", "")
    assert "SKILL_IMAGE_PATH_SET=False" in obs.get("obs", "")
    print(">>> [PASSED] test_run_skill_non_image_skill_without_image_index")


def test_run_skill_entrypoint_level_image_requirement():
    """同一 skill 下按 entrypoint 判断是否需要 image_index。"""
    print(">>> [START] test_run_skill_entrypoint_level_image_requirement")
    tool = MultimodalProcessorTool()
    action_a = _build_generic_action(
        "run_skill",
        {
            "skill_name": "multi-entrypoint-skill",
            "entrypoint": "scripts/a.py",
            "args": {},
        },
    )
    obs_a, _, valid_a = tool.conduct_action("t-skill-multi-a-001", action_a, {"images": [IMAGE_PATH]})
    print(f"  obs_a={obs_a}, valid_a={valid_a}")
    assert valid_a is False
    assert obs_a.get("invalid_reason") == "missing_image_index"

    action_b = _build_generic_action(
        "run_skill",
        {
            "skill_name": "multi-entrypoint-skill",
            "entrypoint": "scripts/b.py",
            "args": {"note": "ok"},
        },
    )
    obs_b, _, valid_b = tool.conduct_action("t-skill-multi-b-001", action_b, {})
    print(f"  obs_b={obs_b}, valid_b={valid_b}")
    assert valid_b is True
    assert "multi-b note=ok" in obs_b.get("obs", "")
    print(">>> [PASSED] test_run_skill_entrypoint_level_image_requirement")


def test_run_skill_missing_stage1_spec_degrades():
    """缺少 Stage1 Spec 时降级放行（不做 entrypoint 级强制）。"""
    print(">>> [START] test_run_skill_missing_stage1_spec_degrades")
    tool = MultimodalProcessorTool()
    action = _build_generic_action(
        "run_skill",
        {
            "skill_name": "legacy-image-skill",
            "entrypoint": "scripts/run.py",
            "args": {},
        },
    )
    obs, _, valid = tool.conduct_action("t-skill-legacy-001", action, {})
    print(f"  obs={obs}, valid={valid}")
    assert valid is True
    assert "legacy skill run ok" in obs.get("obs", "")
    print(">>> [PASSED] test_run_skill_missing_stage1_spec_degrades")


def test_env_image_accumulation():
    """测试 OOD 池下环境状态累计（不依赖图像生成工具）。"""
    print(">>> [START] test_env_image_accumulation")
    tool = MultimodalProcessorTool()
    trajectory_id = "t-env-img-001"

    initial_images = [IMAGE_PATH]

    print("  Skill MD 调用: md-only-skill...")
    action0 = _build_generic_action("md-only-skill", {})
    obs0, done0, valid0 = tool.conduct_action(trajectory_id, action0, {"images": initial_images})
    print(f"    obs0={obs0}, valid0={valid0}")
    assert valid0 is True, f"skill MD call should be valid, got: {obs0}"

    print("  run_skill 调用: echo-skill...")
    action0b = _build_generic_action(
        "run_skill",
        {
            "skill_name": "echo-skill",
            "entrypoint": "scripts/run.py",
            "args": {"image_index": 1},
        },
    )
    obs0b, done0b, valid0b = tool.conduct_action(trajectory_id, action0b, {"images": initial_images})
    print(f"    obs0b={obs0b}, valid0b={valid0b}")
    assert valid0b is True, f"run_skill call should be valid, got: {obs0b}"
    assert isinstance(obs0b.get("latency"), (int, float)), "latency should be reported"

    print("  第一次调用: InternVL3.5-38B-Instruct...")
    action1 = _build_action("InternVL3.5-38B-Instruct", "请描述图片", 1)
    obs1, done1, valid1 = tool.conduct_action(trajectory_id, action1, {"images": initial_images})
    print(f"    obs1={obs1}, valid1={valid1}")
    assert valid1 is True, f"InternVL call should be valid, got: {obs1}"

    print("  第二次调用: InternVL3.5-14B-Instruct...")
    action2 = _build_action("InternVL3.5-14B-Instruct", "请描述图片", 1)
    obs2, done2, valid2 = tool.conduct_action(trajectory_id, action2, {"images": initial_images})
    print(f"    obs2={obs2}, valid2={valid2}")
    assert valid2 is True, f"InternVL call should be valid, got: {obs2}"

    print("  第三次调用: WolframAlpha...")
    action3 = _build_wolfram_action("10 densest elemental metals")
    obs3, done3, valid3 = tool.conduct_action(trajectory_id, action3, {"images": initial_images})
    print(f"    obs3={obs3}, valid3={valid3}")
    assert valid3 is True, f"WolframAlpha call should be valid, got: {obs3}"

    print("  第四次调用: python_code...")
    action4 = _build_generic_action("python_code", {"code": "1 + 1"})
    obs4, done4, valid4 = tool.conduct_action(trajectory_id, action4, {"images": initial_images})
    print(f"    obs4={obs4}, valid4={valid4}")
    assert valid4 is True, f"python_code call should be valid, got: {obs4}"
    assert "2" in obs4.get("obs", "")

    env = tool.load_env(trajectory_id)
    assert len(env["images"]) >= 1, "Original images should remain available in env"
    assert len(env["previous_obs"]) >= 4, f"Expected at least 4 observations, got {len(env['previous_obs'])}"
    assert env["metadata"]["turns"] >= 4, f"Expected at least 4 turns, got {env['metadata']['turns']}"
    print(f"  - env images: {len(env['images'])}, turns: {env['metadata']['turns']}")

    tool.delete_env(trajectory_id)
    assert not tool.has_env(trajectory_id), "Environment should be deleted"
    print("  - 环境清理完成")

    print(">>> [PASSED] test_env_image_accumulation")


def main():
    """主入口，支持命令行运行单个测试。"""
    fire.Fire(
        {
            "init": test_tool_initialization,
            "params": test_parameter_validation,
            "wolfram_params": test_wolfram_parameter_validation,
            "call": test_model_call,
            "wolfram_call": test_wolfram_call,
            "unichart_call": test_unichart_call,
            "deplot_call": test_deplot_call,
            "chartmoe_call": test_chartmoe_call,
            "step3_call": test_step3_call,
            "python_code_call": test_python_code_call,
            "python_code_params": test_python_code_parameter_validation,
            "skill_md_call": test_skill_md_call,
            "create_skill_parse_ok": test_create_skill_parse_with_arguments,
            "create_skill_parse_reject_legacy": test_create_skill_parse_reject_top_level_description,
            "create_skill_real": test_create_skill_real_call,
            "skill_md_call_null_args": test_skill_md_call_with_null_arguments,
            "run_skill_call": test_run_skill_call,
            "run_skill_call_raw_base64": test_run_skill_call_with_raw_base64_image,
            "run_skill_invalid_long_string_image_input": test_run_skill_invalid_long_string_image_input,
            "run_skill_missing_required_image_index": test_run_skill_missing_required_image_index,
            "run_skill_non_image_without_image_index": test_run_skill_non_image_skill_without_image_index,
            "run_skill_entrypoint_level_image_requirement": test_run_skill_entrypoint_level_image_requirement,
            "run_skill_missing_stage1_spec_degrades": test_run_skill_missing_stage1_spec_degrades,
            "env_image_accumulation": test_env_image_accumulation,
        }
    )


if __name__ == "__main__":
    main()
