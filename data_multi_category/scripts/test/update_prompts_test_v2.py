#!/usr/bin/env python3
"""
Batch-update prompt fields in test/ID and test/OOD dataset files, and standardize data format.
Only test datasets (ID + OOD) are processed.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================================
# Configuration
# ============================================================================

TOOL_JSON_PATH = "real_tool.json"

NEW_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_xml}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

NEW_GUIDELINES = (
    "Guidelines: Use tools/skills only when helpful. "
    "Call a skill to read its full content of SKILL.md: <tool_call>{\"name\": \"skill_name\", \"arguments\": null}</tool_call>. Then you will get the full content of the skill definition and the following executable scripts and their corresponding parameters. "
    "If you must execute a skill script, call "
    "<tool_call>{\"name\": \"run_skill\", \"arguments\": {\"skill_name\": \"skill_name\", "
    "\"entrypoint\": \"scripts/run.py\", \"args\": {\"...\": \"...\"}}}</tool_call> "
    "For run_skill image input: pass `image_index` inside `args` only when the target entrypoint requires image input; "
    "image_index starts from 1, maximum value is the number of images in the current environment. Choose 1 to operate on the first image. "
    "if the skill does not require image input, do not pass `image_index`. "
    "and follow SKILL.md for parameters. "
    "Create a new skill only if no existing tool/skill fits and it is reusable: "
    "<tool_call>{\"name\": \"create_skill\", \"arguments\": {\"description\": \"...\"}}</tool_call>. "
    "Do not create skills casually; scripts (.py/.sh) only if needed. "
    "Format: tool calls must be wrapped in <tool_call>...</tool_call> JSON. "
    "Reasoning must be inside <thinking>...</thinking>; final answer inside <answer>...</answer>."
)

DATA_DIR = Path(
    "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/test"
)

BACKUP = True

SEED = 42
SAMPLE_SIZE = 0  # 0 means no sampling; process all records


# ============================================================================
# Processing logic
# ============================================================================


def load_tools(tool_json_path: str) -> str:
    """Load tool definitions and format them as XML."""
    script_dir = Path(__file__).parent
    full_path = script_dir / tool_json_path

    if not full_path.exists():
        raise FileNotFoundError(f"Tool JSON file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"<tools>\n{tools_str}\n</tools>"


def build_new_system_prompt(tools_xml: str) -> str:
    """Build the new system prompt."""
    return NEW_SYSTEM_PROMPT_TEMPLATE.format(tools_xml=tools_xml)


def extract_question_from_user_content(user_content: str) -> str:
    guidelines_pos = user_content.find("\n\nGuidelines:")
    if guidelines_pos != -1:
        return user_content[:guidelines_pos]
    return user_content


def build_new_user_content(original_content: str) -> str:
    question_part = extract_question_from_user_content(original_content)
    return f"{question_part}\n\n{NEW_GUIDELINES}"


def normalize_images_field(images: Any) -> List[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, (list, tuple)):
        result = []
        for item in images:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                if "image" in item:
                    result.append(item["image"])
                elif "path" in item:
                    result.append(item["path"])
                elif "url" in item:
                    result.append(item["url"])
            else:
                result.append(str(item))
        return result
    return [str(images)]


def parse_extra_info(extra_info: Any) -> Dict[str, Any]:
    if extra_info is None:
        return {}
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def extract_question_from_prompt(prompt: List[Dict[str, str]]) -> str:
    for message in prompt:
        if message.get("role") == "user":
            content = message.get("content", "")
            return extract_question_from_user_content(content)
    return ""


def standardize_sample(sample: Dict[str, Any], new_system_prompt: str) -> Dict[str, Any]:
    std_sample: Dict[str, Any] = {}

    original_prompt = sample.get("prompt", [])
    if isinstance(original_prompt, str):
        try:
            original_prompt = json.loads(original_prompt)
        except (json.JSONDecodeError, TypeError):
            original_prompt = []

    system_prompt = new_system_prompt
    user_content = ""
    if isinstance(original_prompt, list):
        for message in original_prompt:
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                user_content = content
    if user_content:
        new_user_content = build_new_user_content(user_content)
    else:
        question = sample.get("question") or extract_question_from_prompt(original_prompt)
        new_user_content = f"{question}\n\n{NEW_GUIDELINES}" if question else NEW_GUIDELINES

    std_sample["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": new_user_content},
    ]

    extra_info = parse_extra_info(sample.get("extra_info"))

    image_paths = normalize_images_field(sample.get("images"))
    if not image_paths and "images" in extra_info:
        image_paths = normalize_images_field(extra_info.get("images"))

    extra_info["images"] = image_paths

    question_text = extra_info.get("question")
    if not question_text:
        question_text = extract_question_from_prompt(std_sample["prompt"])
        extra_info["question"] = question_text

    if "split" not in extra_info:
        extra_info["split"] = "test"
    if "index" not in extra_info:
        extra_info["index"] = 0
    if "qid" not in extra_info:
        extra_info["qid"] = f"{extra_info['split']}_{extra_info['index']}"

    std_sample["extra_info"] = extra_info

    std_sample["images"] = [{"image": path} for path in image_paths]

    if "data_source" in sample:
        std_sample["data_source"] = str(sample["data_source"])
    else:
        std_sample["data_source"] = "unknown"

    if "reward_model" in sample:
        reward_model = sample["reward_model"]
        if isinstance(reward_model, dict):
            std_sample["reward_model"] = {
                "style": reward_model.get("style", "rule"),
                "ground_truth": str(reward_model.get("ground_truth", "")),
            }
        else:
            std_sample["reward_model"] = {"style": "rule", "ground_truth": str(reward_model)}
    else:
        std_sample["reward_model"] = {"style": "rule", "ground_truth": ""}

    if "ability" in sample:
        std_sample["ability"] = str(sample["ability"])

    return std_sample


def find_all_json_files(data_dir: Path) -> List[Path]:
    json_files = []
    for json_file in data_dir.rglob("*.json"):
        if json_file.name.endswith(".bak"):
            continue
        if json_file.name in ["real_tool.json", "fake_tool.json"]:
            continue
        if json_file.stem in ["test", "testmini", "val", "validation"]:
            json_files.append(json_file)
    return sorted(json_files)


def normalize_extra_info_for_json(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extra_info, dict):
        return extra_info

    result = {}
    for key, value in extra_info.items():
        if key == "images":
            result[key] = value
        elif isinstance(value, dict):
            result[key] = normalize_extra_info_for_json(value)
        elif isinstance(value, bool):
            result[key] = value
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        elif isinstance(value, list):
            result[key] = [
                float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def save_json(file_path: Path, data: List[Dict]):
    import copy

    processed_data = copy.deepcopy(data)
    for record in processed_data:
        if "extra_info" in record and isinstance(record["extra_info"], dict):
            record["extra_info"] = normalize_extra_info_for_json(record["extra_info"])

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def save_parquet(file_path: Path, data: List[Dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(data)

    def normalize_extra_info(extra_info):
        if not isinstance(extra_info, dict):
            return extra_info
        result = {}
        for key, value in extra_info.items():
            if key == "images":
                result[key] = value
                continue
            if isinstance(value, dict):
                result[key] = normalize_extra_info(value)
            elif isinstance(value, bool):
                result[key] = value
            elif isinstance(value, (int, float)):
                result[key] = float(value)
            elif isinstance(value, list):
                result[key] = [
                    item if isinstance(item, bool) else (float(item) if isinstance(item, (int, float)) else item)
                    for item in value
                ]
            else:
                result[key] = value
        return result

    if "extra_info" in df.columns:
        df["extra_info"] = df["extra_info"].apply(normalize_extra_info)

    df.to_parquet(file_path, index=False)


def process_file(file_path: Path, new_system_prompt: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  Skip non-list file: {file_path.name}")
        return 0

    original_data = data

    if SAMPLE_SIZE and len(data) > SAMPLE_SIZE:
        rng = random.Random(SEED)
        data = rng.sample(data, SAMPLE_SIZE)
        print(f"  Sampled {SAMPLE_SIZE} items (seed={SEED})")

    updated_data = [standardize_sample(sample, new_system_prompt) for sample in data]

    if BACKUP:
        backup_path = file_path.with_suffix(".json.bak")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(f"  Backup created: {backup_path.name}")

    save_json(file_path, updated_data)
    parquet_path = file_path.with_suffix(".parquet")
    save_parquet(parquet_path, updated_data)

    return len(updated_data)


def main() -> None:
    print("=" * 70)
    print("Update test prompts + standardize format (ID + OOD)")
    print("=" * 70)

    tools_xml = load_tools(TOOL_JSON_PATH)
    new_system_prompt = build_new_system_prompt(tools_xml)

    json_files = find_all_json_files(DATA_DIR)
    if not json_files:
        print("No files to process.")
        return

    total_samples = 0
    for i, file_path in enumerate(json_files, 1):
        rel_path = file_path.relative_to(DATA_DIR)
        print(f"[{i}/{len(json_files)}] Processing: {rel_path}")
        try:
            count = process_file(file_path, new_system_prompt)
            total_samples += count
            print(f"  Updated {count} samples")
        except Exception as e:
            print(f"  Failed: {e}")

    print("=" * 70)
    print(f"Done. Updated samples: {total_samples}")
    print("=" * 70)


if __name__ == "__main__":
    main()
