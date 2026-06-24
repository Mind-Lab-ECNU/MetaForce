#!/usr/bin/env python
import base64
import json
import os
import tempfile
from pathlib import Path

import fire
import requests

IMAGE_PATH = "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general/HatefulMemes_2000/images/train/train_1.png"


def _encode_image_as_data_url(image_path: str) -> str:
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    encoded = base64.b64encode(path_obj.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def _encode_image_as_base64(image_path: str) -> str:
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return base64.b64encode(path_obj.read_bytes()).decode()


def _save_data_url_image(data_url: str, output_path: str) -> None:
    if not data_url.startswith("data:image") or "base64," not in data_url:
        raise ValueError("Invalid data URL image format.")
    b64 = data_url.split("base64,", 1)[1]
    img_data = base64.b64decode(b64)
    with open(output_path, "wb") as f:
        f.write(img_data)


def _send_request(url: str, action: str, image_data_url: str) -> dict:
    payload = {
        "trajectory_ids": ["mm-async-001"],
        "actions": [action],
        "extra_fields": [
            {
                "images": [image_data_url],
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _send_request_no_images(url: str, action: str) -> dict:
    payload = {
        "trajectory_ids": ["mm-async-002"],
        "actions": [action],
        "extra_fields": [{}],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _send_request_with_images(url: str, action: str, images: list) -> dict:
    payload = {
        "trajectory_ids": ["mm-async-b64-001"],
        "actions": [action],
        "extra_fields": [
            {
                "images": images,
            }
        ],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _setup_skill_store() -> Path:
    store_dir = os.environ.get("VERL_SKILL_STORE_DIR")
    if store_dir:
        root = Path(store_dir)
    else:
        root = Path(tempfile.mkdtemp(prefix="skill_test_async_"))
        os.environ["VERL_SKILL_STORE_DIR"] = str(root)

    md_only_dir = root / "md-only-skill"
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

    run_dir = root / "echo-skill"
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

    no_image_dir = root / "no-image-skill"
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

    multi_dir = root / "multi-entrypoint-skill"
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

    legacy_dir = root / "legacy-image-skill"
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

    return root


_SKILL_STORE = _setup_skill_store()


def _assert_success(result: dict):
    assert result.get("valids", [False])[0] is True, f"expected valid=True, got {result.get('valids')}"
    obs = result["observations"][0]
    assert isinstance(obs, dict), f"expect dict obs, got {type(obs)}"


def _assert_invalid(result: dict, reason: str):
    assert result.get("valids", [True])[0] is False, f"expected valid=False, got {result.get('valids')}"
    obs = result["observations"][0]
    assert isinstance(obs, dict)
    assert obs.get("invalid_reason") == reason, f"expect {reason}, got {obs.get('invalid_reason')}"


def test_vlm_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "请描述图片",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "Qwen3-VL-8B-Instruct", "arguments": {"prompt": prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "Qwen3-VL-8B-Instruct"
    return result


def test_vlm_235_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "请描述图片",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "Qwen3-VL-235B-A22B-Instruct", "arguments": {"prompt": prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "Qwen3-VL-235B-A22B-Instruct"
    return result


def test_parameter_validation(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = '<tool_call>{"name": "Qwen3-VL-8B-Instruct", "arguments": {"prompt": "hi"}}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_parameters")
    return result


def test_wolfram_call(
    url: str = "http://localhost:5000/get_observation",
    query: str = "10 densest elemental metals",
):
    action = f'<tool_call>{json.dumps({"name": "WolframAlpha", "arguments": {"query": query}})}</tool_call>'
    payload = {
        "trajectory_ids": ["mm-async-wolfram-001"],
        "actions": [action],
        "extra_fields": [{}],
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "WolframAlpha"
    return result


def test_mineru_content(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "MinerU2.5", "arguments": {"extract_type": "content", "content_type": "text", "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "MinerU2.5"
    return result


def test_paddle_ocr(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "PaddleOCR", "arguments": {"image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "PaddleOCR"
    return result


def test_easyocr_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "EasyOCR", "arguments": {"image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "EasyOCR"
    return result


def test_groundingdino_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "a person.",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "GroundingDINO", "arguments": {"image_index": 1, "prompt": prompt}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "GroundingDINO"
    return result


def test_unichart_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "UniChart", "arguments": {"image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "UniChart"
    return result


def test_deplot_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "Deplot", "arguments": {"image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "Deplot"
    return result


def test_chartmoe_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "ChartMoe", "arguments": {"image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "ChartMoe"
    return result


def test_step3_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "请描述图片",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "step3", "arguments": {"prompt": prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "step3"
    return result


def test_sam3_text(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    text_prompt: str = "person",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "SAM3", "arguments": {"segment_type": "text", "text_prompt": text_prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    if "image" in obs:
        output_path = os.path.join(os.getcwd(), "sam3_text_async_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"saved image to: {output_path}")
    return result


def test_image_edit(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
    prompt: str = "make the background blue",
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "Qwen-Image-Edit", "arguments": {"prompt": prompt, "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    if "image" in obs:
        output_path = os.path.join(os.getcwd(), "image_edit_async_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"saved image to: {output_path}")
    return result


def test_skill_md_call(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "md-only-skill", "arguments": {}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "md-only-skill"
    return result


def test_create_skill_real_call(
    url: str = "http://localhost:5000/get_observation",
):
    action = (
        "<tool_call>"
        + json.dumps(
            {
                "name": "create_skill",
                "arguments": {
                    "description": (
                        "Create a text-only skill that summarizes input text and "
                        "extracts keywords with clear CLI usage."
                    )
                },
            }
        )
        + "</tool_call>"
    )
    payload = {
        "trajectory_ids": ["mm-async-create-skill-001"],
        "actions": [action],
        "extra_fields": [{}],
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "create_skill"
    assert isinstance(obs.get("skill_name"), str) and bool(obs.get("skill_name"))
    assert isinstance(obs.get("skill_path"), str) and bool(obs.get("skill_path"))
    skill_dir = Path(obs["skill_path"])
    assert skill_dir.exists() and skill_dir.is_dir(), f"skill dir not found: {skill_dir}"
    assert (skill_dir / "SKILL.md").exists(), f"SKILL.md not found in: {skill_dir}"
    return result


def test_run_skill_call(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "echo-skill", "entrypoint": "scripts/run.py", "args": {"image_index": 1}}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_success(result)
    obs = result["observations"][0]
    assert "echo-skill" in obs.get("obs", "")
    assert "SKILL_IMAGE_PATH_SET=True" in obs.get("obs", "")
    assert isinstance(obs.get("latency"), (int, float)), "latency should be reported"
    return result


def test_run_skill_call_with_raw_base64_image(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    image_base64 = _encode_image_as_base64(image_path)
    action = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "echo-skill", "entrypoint": "scripts/run.py", "args": {"image_index": 1}}})}</tool_call>'
    result = _send_request_with_images(url, action, [image_base64])
    _assert_success(result)
    obs = result["observations"][0]
    assert "SKILL_IMAGE_PATH_SET=True" in obs.get("obs", "")
    assert "SKILL_IMAGE_DATA_URL_SET=False" in obs.get("obs", "")
    return result


def test_run_skill_invalid_long_string_image_input(
    url: str = "http://localhost:5000/get_observation",
):
    bad_image = "not_base64_" * 800
    action = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "echo-skill", "entrypoint": "scripts/run.py", "args": {"image_index": 1}}})}</tool_call>'
    result = _send_request_with_images(url, action, [bad_image])
    _assert_invalid(result, "invalid_image_input")
    return result


def test_run_skill_missing_required_image_index(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "echo-skill", "entrypoint": "scripts/run.py", "args": {}}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_image_index")
    return result


def test_run_skill_non_image_without_image_index(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "no-image-skill", "entrypoint": "scripts/run.py", "args": {"note": "ok"}}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_success(result)
    obs = result["observations"][0]
    assert "no-image-skill note=ok" in obs.get("obs", "")
    assert "SKILL_IMAGE_PATH_SET=False" in obs.get("obs", "")
    return result


def test_run_skill_entrypoint_level_image_requirement(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action_a = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "multi-entrypoint-skill", "entrypoint": "scripts/a.py", "args": {}}})}</tool_call>'
    result_a = _send_request(url, action_a, img)
    _assert_invalid(result_a, "missing_image_index")

    action_b = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "multi-entrypoint-skill", "entrypoint": "scripts/b.py", "args": {"note": "ok"}}})}</tool_call>'
    result_b = _send_request_no_images(url, action_b)
    _assert_success(result_b)
    obs_b = result_b["observations"][0]
    assert "multi-b note=ok" in obs_b.get("obs", "")
    return {"a": result_a, "b": result_b}


def test_run_skill_missing_stage1_spec_degrades(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "legacy-image-skill", "entrypoint": "scripts/run.py", "args": {}}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_success(result)
    obs = result["observations"][0]
    assert "legacy skill run ok" in obs.get("obs", "")
    return result


def test_python_code_call(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "python_code", "arguments": {"code": "1 + 1"}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "python_code"
    assert "2" in obs.get("obs", ""), f"Expected auto-printed result, got: {obs}"
    return result


def test_python_code_parameter_validation(
    url: str = "http://localhost:5000/get_observation",
):
    action = f'<tool_call>{json.dumps({"name": "python_code", "arguments": {}})}</tool_call>'
    result = _send_request_no_images(url, action)
    _assert_invalid(result, "missing_parameters")
    return result


def test_opencv_operations(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    operations = [
        ("crop", {"operation": "crop", "image_index": 1, "x": 10, "y": 10, "w": 64, "h": 64}),
        ("resize", {"operation": "resize", "image_index": 1, "width": 128, "height": 96}),
        ("rotate", {"operation": "rotate", "image_index": 1, "angle": 30}),
        ("flip", {"operation": "flip", "image_index": 1, "flip_code": 1}),
        ("grayscale", {"operation": "grayscale", "image_index": 1}),
        ("blur", {"operation": "blur", "image_index": 1, "ksize": 4}),
        ("threshold", {"operation": "threshold", "image_index": 1, "thresh": 127, "type": "BINARY"}),
        ("canny", {"operation": "canny", "image_index": 1, "threshold1": 100, "threshold2": 200}),
    ]

    results = {}
    for name, args in operations:
        action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": args})}</tool_call>'
        result = _send_request(url, action, img)
        _assert_success(result)
        obs = result["observations"][0]
        assert obs.get("tool") == "OpenCV"
        if "image" in obs:
            assert obs.get("image", "").startswith("data:image"), f"OpenCV {name} should return image"
            output_path = os.path.join(os.getcwd(), f"opencv_async_{name}_output.png")
            _save_data_url_image(obs["image"], output_path)
            print(f"saved image to: {output_path}")
        results[name] = result
    return results


def test_opencv_parameter_validation(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)

    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "crop", "x": 0, "y": 0, "w": 10, "h": 10}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_parameters")

    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "unknown", "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "invalid_parameters")

    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "threshold", "image_index": 1, "type": "BINARY"}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "missing_parameters")
    return result


def test_opencv_base64_image(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    image_base64 = _encode_image_as_base64(image_path)
    action = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "grayscale", "image_index": 1}})}</tool_call>'
    result = _send_request_with_images(url, action, [image_base64])
    _assert_success(result)
    obs = result["observations"][0]
    assert obs.get("tool") == "OpenCV"
    if "image" in obs:
        assert obs.get("image", "").startswith("data:image")
        output_path = os.path.join(os.getcwd(), "opencv_async_base64_output.png")
        _save_data_url_image(obs["image"], output_path)
        print(f"saved image to: {output_path}")
    return result


def test_bad_function(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    action = f'<tool_call>{json.dumps({"name": "wrong_function", "arguments": {"prompt": "hi", "image_index": 1}})}</tool_call>'
    result = _send_request(url, action, img)
    _assert_invalid(result, "unknown_skill")
    return result


def test_env_image_accumulation(
    url: str = "http://localhost:5000/get_observation",
    image_path: str = IMAGE_PATH,
):
    img = _encode_image_as_data_url(image_path)
    trajectory_id = "mm-async-env-001"

    action1 = f'<tool_call>{json.dumps({"name": "SAM3", "arguments": {"segment_type": "text", "text_prompt": "person", "image_index": 1}})}</tool_call>'
    payload1 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action1],
        "extra_fields": [{"images": [img]}],
    }
    resp1 = requests.post(url, json=payload1, timeout=180)
    resp1.raise_for_status()
    result1 = resp1.json()
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    _assert_success(result1)

    action_md = f'<tool_call>{json.dumps({"name": "md-only-skill", "arguments": {}})}</tool_call>'
    payload_md = {
        "trajectory_ids": [trajectory_id],
        "actions": [action_md],
        "extra_fields": [{}],
    }
    resp_md = requests.post(url, json=payload_md, timeout=180)
    resp_md.raise_for_status()
    result_md = resp_md.json()
    print(json.dumps(result_md, indent=2, ensure_ascii=False))
    _assert_success(result_md)

    action_run = f'<tool_call>{json.dumps({"name": "run_skill", "arguments": {"skill_name": "echo-skill", "entrypoint": "scripts/run.py", "args": {"image_index": 1}}})}</tool_call>'
    payload_run = {
        "trajectory_ids": [trajectory_id],
        "actions": [action_run],
        "extra_fields": [{}],
    }
    resp_run = requests.post(url, json=payload_run, timeout=180)
    resp_run.raise_for_status()
    result_run = resp_run.json()
    print(json.dumps(result_run, indent=2, ensure_ascii=False))
    _assert_success(result_run)

    action2 = f'<tool_call>{json.dumps({"name": "Qwen3-VL-8B-Instruct", "arguments": {"prompt": "请描述图片", "image_index": 1}})}</tool_call>'
    payload2 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action2],
        "extra_fields": [{}],
    }
    resp2 = requests.post(url, json=payload2, timeout=180)
    resp2.raise_for_status()
    result2 = resp2.json()
    print(json.dumps(result2, indent=2, ensure_ascii=False))
    _assert_success(result2)

    action3 = f'<tool_call>{json.dumps({"name": "Qwen-Image-Edit", "arguments": {"prompt": "make the background blue", "image_index": 2}})}</tool_call>'
    payload3 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action3],
        "extra_fields": [{}],
    }
    resp3 = requests.post(url, json=payload3, timeout=180)
    resp3.raise_for_status()
    result3 = resp3.json()
    print(json.dumps(result3, indent=2, ensure_ascii=False))
    _assert_success(result3)
    obs3 = result3["observations"][0]
    if "image" in obs3:
        output_path = os.path.join(os.getcwd(), "env_accumulation_async_edit_1.png")
        _save_data_url_image(obs3["image"], output_path)
        print(f"saved image to: {output_path}")

    action4 = f'<tool_call>{json.dumps({"name": "PaddleOCR", "arguments": {"image_index": 1}})}</tool_call>'
    payload4 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action4],
        "extra_fields": [{}],
    }
    resp4 = requests.post(url, json=payload4, timeout=180)
    resp4.raise_for_status()
    result4 = resp4.json()
    print(json.dumps(result4, indent=2, ensure_ascii=False))
    _assert_success(result4)

    action5 = f'<tool_call>{json.dumps({"name": "WolframAlpha", "arguments": {"query": "10 densest elemental metals"}})}</tool_call>'
    payload5 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action5],
        "extra_fields": [{}],
    }
    resp5 = requests.post(url, json=payload5, timeout=180)
    resp5.raise_for_status()
    result5 = resp5.json()
    print(json.dumps(result5, indent=2, ensure_ascii=False))
    _assert_success(result5)

    action6 = f'<tool_call>{json.dumps({"name": "python_code", "arguments": {"code": "1 + 1"}})}</tool_call>'
    payload6 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action6],
        "extra_fields": [{}],
    }
    resp6 = requests.post(url, json=payload6, timeout=180)
    resp6.raise_for_status()
    result6 = resp6.json()
    print(json.dumps(result6, indent=2, ensure_ascii=False))
    _assert_success(result6)

    action7 = f'<tool_call>{json.dumps({"name": "OpenCV", "arguments": {"operation": "grayscale", "image_index": 3}})}</tool_call>'
    payload7 = {
        "trajectory_ids": [trajectory_id],
        "actions": [action7],
        "extra_fields": [{}],
    }
    resp7 = requests.post(url, json=payload7, timeout=180)
    resp7.raise_for_status()
    result7 = resp7.json()
    print(json.dumps(result7, indent=2, ensure_ascii=False))
    _assert_success(result7)
    obs7 = result7["observations"][0]
    if "image" in obs7:
        output_path = os.path.join(os.getcwd(), "env_accumulation_async_opencv_1.png")
        _save_data_url_image(obs7["image"], output_path)
        print(f"saved image to: {output_path}")

    return {
        "sam3": result1,
        "skill_md": result_md,
        "run_skill": result_run,
        "vlm": result2,
        "edit1": result3,
        "paddle": result4,
        "wolfram": result5,
        "python": result6,
        "opencv": result7,
    }


def main():
    fire.Fire(
        {
            "vlm": test_vlm_call,
            "vlm_235": test_vlm_235_call,
            "params": test_parameter_validation,
            "wolfram_call": test_wolfram_call,
            "mineru_content": test_mineru_content,
            "paddle_call": test_paddle_ocr,
            "easyocr_call": test_easyocr_call,
            "groundingdino_call": test_groundingdino_call,
            "unichart_call": test_unichart_call,
            "deplot_call": test_deplot_call,
            "chartmoe_call": test_chartmoe_call,
            "step3_call": test_step3_call,
            "sam3_text": test_sam3_text,
            "image_edit": test_image_edit,
            "skill_md_call": test_skill_md_call,
            "create_skill_real": test_create_skill_real_call,
            "run_skill_call": test_run_skill_call,
            "run_skill_call_raw_base64": test_run_skill_call_with_raw_base64_image,
            "run_skill_invalid_long_string_image_input": test_run_skill_invalid_long_string_image_input,
            "run_skill_missing_required_image_index": test_run_skill_missing_required_image_index,
            "run_skill_non_image_without_image_index": test_run_skill_non_image_without_image_index,
            "run_skill_entrypoint_level_image_requirement": test_run_skill_entrypoint_level_image_requirement,
            "run_skill_missing_stage1_spec_degrades": test_run_skill_missing_stage1_spec_degrades,
            "python_code_call": test_python_code_call,
            "python_code_params": test_python_code_parameter_validation,
            "opencv_ops": test_opencv_operations,
            "opencv_params": test_opencv_parameter_validation,
            "opencv_base64": test_opencv_base64_image,
            "env_image_accumulation": test_env_image_accumulation,
            "bad_function": test_bad_function,
        }
    )


if __name__ == "__main__":
    main()
