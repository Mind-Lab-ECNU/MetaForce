#!/usr/bin/env python3
"""Common helpers for scripts_v2 pure-reuse data processing."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image


DATA_ROOT = Path(
    "/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data"
)
REPO_ROOT = DATA_ROOT.parents[1]
OLD_REPO_PREFIXES = (
    "/path/to/project1",
    "/path/to/project2",
    "/path/to/verl_duo",
)

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
    "Call a skill to read its full content of SKILL.md: "
    "<tool_call>{\"name\": \"skill_name\", \"arguments\": null}</tool_call>. "
    "Then you will get the full content of the skill definition and the following executable scripts "
    "and their corresponding parameters. "
    "If you must execute a skill script, call "
    "<tool_call>{\"name\": \"run_skill\", \"arguments\": {\"skill_name\": \"skill_name\", "
    "\"entrypoint\": \"scripts/run.py\", \"args\": {\"...\": \"...\"}}}</tool_call> "
    "For run_skill image input: pass `image_index` inside `args` only when the target entrypoint requires "
    "image input; image_index starts from 1, maximum value is the number of images in the current "
    "environment. Choose 1 to operate on the first image. if the skill does not require image input, "
    "do not pass `image_index`. and follow SKILL.md for parameters. Create a new skill only if no "
    "existing tool/skill fits and it is reusable: "
    "<tool_call>{\"name\": \"create_skill\", \"arguments\": {\"description\": \"...\"}}</tool_call>. "
    "Do not create skills casually; scripts (.py/.sh) only if needed. "
    "Format: tool calls must be wrapped in <tool_call>...</tool_call> JSON. "
    "Reasoning must be inside <thinking>...</thinking>; final answer inside <answer>...</answer>."
    # "Reasoning must be inside <think>...</think>; final answer inside <answer>...</answer>."
)


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data)}")
    return data


def save_json(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def save_parquet(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(items)
    df.to_parquet(path, index=False)


def save_json_and_parquet(base_dir: Path, split_name: str, items: List[Dict[str, Any]]) -> None:
    save_json(base_dir / f"{split_name}.json", items)
    save_parquet(base_dir / f"{split_name}.parquet", items)


def load_tools(script_dir: Path, tool_json_name: str = "real_tool.json") -> str:
    path = script_dir / tool_json_name
    if not path.exists():
        raise FileNotFoundError(f"Tool JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    return f"<tools>\n{json.dumps(tools, ensure_ascii=False, indent=2)}\n</tools>"


def build_system_prompt(tools_xml: str) -> str:
    return NEW_SYSTEM_PROMPT_TEMPLATE.format(tools_xml=tools_xml)


def normalize_images_field(images: Any) -> List[str]:
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        result: List[str] = []
        for item in images:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                if "image" in item and item["image"]:
                    result.append(str(item["image"]))
                elif "path" in item and item["path"]:
                    result.append(str(item["path"]))
            elif item is not None:
                result.append(str(item))
        return result
    return [str(images)]


def remap_image_path(image_path: str) -> str:
    for old_prefix in OLD_REPO_PREFIXES:
        if image_path.startswith(old_prefix):
            return str(REPO_ROOT) + image_path[len(old_prefix) :]
    return image_path


def parse_extra_info(extra_info: Any) -> Dict[str, Any]:
    if extra_info is None:
        return {}
    if isinstance(extra_info, dict):
        return copy.deepcopy(extra_info)
    if isinstance(extra_info, str):
        try:
            value = json.loads(extra_info)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def get_ground_truth(sample: Dict[str, Any]) -> str:
    reward_model = sample.get("reward_model")
    if isinstance(reward_model, dict):
        value = reward_model.get("ground_truth", "")
    else:
        value = reward_model if reward_model is not None else ""
    return str(value).strip()


def normalize_record_for_reuse(sample: Dict[str, Any]) -> Dict[str, Any]:
    item = copy.deepcopy(sample)

    extra_info = parse_extra_info(item.get("extra_info"))
    image_paths = normalize_images_field(item.get("images"))
    if not image_paths:
        image_paths = normalize_images_field(extra_info.get("images"))

    image_paths = [remap_image_path(p) for p in image_paths]
    item["images"] = [{"image": p} for p in image_paths]
    extra_info["images"] = image_paths
    if "question" not in extra_info:
        extra_info["question"] = ""
    item["extra_info"] = extra_info

    ground_truth = get_ground_truth(item)
    if "reward_model" not in item or not isinstance(item["reward_model"], dict):
        item["reward_model"] = {"style": "rule", "ground_truth": ground_truth}
    else:
        item["reward_model"] = {
            "style": str(item["reward_model"].get("style", "rule")),
            "ground_truth": ground_truth,
        }

    if "data_source" in item:
        item["data_source"] = str(item["data_source"])
    return item


def resolve_image_path(image_path: str, data_root: Path) -> Path:
    p = Path(remap_image_path(image_path))
    if p.is_absolute():
        return p
    return data_root / p


def is_image_readable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_sample_basic(sample: Dict[str, Any], data_root: Path) -> Tuple[bool, str]:
    item = normalize_record_for_reuse(sample)

    image_paths = normalize_images_field(item.get("images"))
    if len(image_paths) != 1:
        return False, "requires exactly one image"

    ground_truth = get_ground_truth(item)
    if not ground_truth:
        return False, "ground_truth is empty"

    image_path = resolve_image_path(image_paths[0], data_root)
    if not is_image_readable(image_path):
        return False, f"image path unreadable: {image_paths[0]}"

    return True, ""


def filter_valid_samples(
    items: Iterable[Dict[str, Any]],
    data_root: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    valid: List[Dict[str, Any]] = []
    errors: List[str] = []
    for idx, sample in enumerate(items):
        ok, reason = validate_sample_basic(sample, data_root)
        if ok:
            valid.append(normalize_record_for_reuse(sample))
        else:
            errors.append(f"#{idx}: {reason}")
    return valid, errors


def stable_sample_indices(total: int, sample_size: int, seed: int) -> List[int]:
    if sample_size >= total:
        return list(range(total))
    rng = random.Random(seed)
    indices = rng.sample(range(total), sample_size)
    indices.sort()
    return indices


def stable_sample(items: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    indices = stable_sample_indices(len(items), sample_size, seed)
    return [items[i] for i in indices]


def sample_uid(sample: Dict[str, Any]) -> str:
    extra_info = parse_extra_info(sample.get("extra_info"))
    qid = extra_info.get("qid")
    if qid:
        return str(qid)

    images = normalize_images_field(sample.get("images"))
    if not images:
        images = normalize_images_field(extra_info.get("images"))

    payload = {
        "question": str(extra_info.get("question", "")),
        "images": images,
        "ground_truth": get_ground_truth(sample),
        "data_source": str(sample.get("data_source", "")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()


def count_image_tokens(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return text.count("<image>")


def find_user_content(sample: Dict[str, Any]) -> str:
    prompt = sample.get("prompt")
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(prompt, list):
        return ""
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def extract_question_from_user_content(text: str) -> str:
    marker = "\n\nGuidelines:"
    pos = text.find(marker)
    if pos != -1:
        return text[:pos]
    return text


def build_user_content(old_content: str) -> str:
    question = extract_question_from_user_content(old_content)
    if question:
        return f"{question}\n\n{NEW_GUIDELINES}"
    return NEW_GUIDELINES


def standardize_prompt_fields(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    item = normalize_record_for_reuse(sample)

    old_user_content = find_user_content(item)
    user_content = build_user_content(old_user_content)

    item["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    extra_info = parse_extra_info(item.get("extra_info"))
    if not extra_info.get("question"):
        extra_info["question"] = extract_question_from_user_content(old_user_content)
    if "split" not in extra_info:
        extra_info["split"] = "unknown"
    if "index" not in extra_info:
        extra_info["index"] = 0
    if "qid" not in extra_info:
        extra_info["qid"] = f"{extra_info['split']}_{extra_info['index']}"

    image_paths = [remap_image_path(p) for p in normalize_images_field(item.get("images"))]
    extra_info["images"] = image_paths
    item["images"] = [{"image": p} for p in image_paths]
    item["extra_info"] = extra_info

    ground_truth = get_ground_truth(item)
    item["reward_model"] = {"style": "rule", "ground_truth": ground_truth}
    if "data_source" in item:
        item["data_source"] = str(item["data_source"])

    return item


def ensure_jsonable_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ensure_jsonable_types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_jsonable_types(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def ensure_extra_info_jsonable(record: Dict[str, Any]) -> Dict[str, Any]:
    item = copy.deepcopy(record)
    if "extra_info" in item and isinstance(item["extra_info"], dict):
        images = item["extra_info"].get("images")
        normalized = ensure_jsonable_types(item["extra_info"])
        if images is not None:
            normalized["images"] = images
        item["extra_info"] = normalized
    return item


def _is_numeric_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _classify_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "numeric_str" if _is_numeric_string(value) else "str"
    return "other"


def collect_extra_info_stats(
    extra_info: Dict[str, Any],
    stats: Dict[Tuple[str, ...], Dict[str, Any]],
    path: Tuple[str, ...] = (),
) -> None:
    if not isinstance(extra_info, dict):
        return
    for key, value in extra_info.items():
        if key == "images":
            continue
        current_path = path + (key,)
        entry = stats.setdefault(
            current_path,
            {"kinds": set(), "scalar_types": set(), "list_types": set()},
        )

        if isinstance(value, dict):
            entry["kinds"].add("dict")
            collect_extra_info_stats(value, stats, current_path)
        elif isinstance(value, list):
            entry["kinds"].add("list")
            for item in value:
                if isinstance(item, (dict, list)):
                    entry["list_types"].add("complex")
                else:
                    entry["list_types"].add(_classify_scalar(item))
        else:
            entry["kinds"].add("scalar")
            entry["scalar_types"].add(_classify_scalar(value))


def decide_extra_info_policies(
    stats: Dict[Tuple[str, ...], Dict[str, Any]]
) -> Dict[Tuple[str, ...], Dict[str, str]]:
    policies: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for key_path, entry in stats.items():
        kinds = entry["kinds"]
        if len(kinds) > 1:
            policies[key_path] = {"kind": "string"}
            continue
        if "dict" in kinds:
            policies[key_path] = {"kind": "dict"}
            continue
        if "list" in kinds:
            list_types = set(entry["list_types"])
            list_types.discard("none")
            if not list_types:
                policies[key_path] = {"kind": "list", "item": "none"}
            elif list_types == {"bool"}:
                policies[key_path] = {"kind": "list", "item": "bool"}
            elif list_types.issubset({"number", "numeric_str"}):
                policies[key_path] = {"kind": "list", "item": "number"}
            else:
                policies[key_path] = {"kind": "list", "item": "string"}
            continue

        scalar_types = set(entry["scalar_types"])
        scalar_types.discard("none")
        if not scalar_types:
            policies[key_path] = {"kind": "scalar", "type": "none"}
        elif scalar_types == {"bool"}:
            policies[key_path] = {"kind": "scalar", "type": "bool"}
        elif scalar_types.issubset({"number", "numeric_str"}):
            policies[key_path] = {"kind": "scalar", "type": "number"}
        else:
            policies[key_path] = {"kind": "scalar", "type": "string"}
    return policies


def _stringify(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def _normalize_scalar(v: Any, t: str) -> Any:
    if v is None:
        return None
    if t == "bool":
        return bool(v) if not isinstance(v, bool) else v
    if t == "number":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, str) and _is_numeric_string(v):
            return float(v)
        return None
    if t == "string":
        return _stringify(v)
    return v


def _normalize_list(v: Any, t: str) -> Any:
    if v is None:
        return None
    values = v if isinstance(v, list) else [v]
    if t == "number":
        out: List[Any] = []
        for item in values:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append(float(item))
            elif isinstance(item, str) and _is_numeric_string(item):
                out.append(float(item))
            else:
                out.append(None)
        return out
    if t == "bool":
        return [bool(item) if not isinstance(item, bool) else item for item in values]
    if t == "string":
        return [_stringify(item) for item in values]
    return values


def normalize_extra_info_by_policy(
    extra_info: Dict[str, Any],
    policies: Dict[Tuple[str, ...], Dict[str, str]],
    path: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    if not isinstance(extra_info, dict):
        return extra_info

    out: Dict[str, Any] = {}
    for key, value in extra_info.items():
        if key == "images":
            out[key] = value
            continue
        current_path = path + (key,)
        policy = policies.get(current_path)
        if policy is None:
            out[key] = (
                normalize_extra_info_by_policy(value, policies, current_path)
                if isinstance(value, dict)
                else value
            )
            continue

        kind = policy["kind"]
        if kind == "string":
            out[key] = _stringify(value)
        elif kind == "dict":
            out[key] = (
                normalize_extra_info_by_policy(value, policies, current_path)
                if isinstance(value, dict)
                else _stringify(value)
            )
        elif kind == "list":
            out[key] = _normalize_list(value, policy["item"])
        elif kind == "scalar":
            out[key] = _normalize_scalar(value, policy["type"])
        else:
            out[key] = value

    return out


def check_prompt_updated(sample: Dict[str, Any]) -> bool:
    prompt = sample.get("prompt")
    if not isinstance(prompt, list) or len(prompt) < 2:
        return False
    system = prompt[0].get("content", "") if isinstance(prompt[0], dict) else ""
    user = prompt[1].get("content", "") if isinstance(prompt[1], dict) else ""
    return (
        isinstance(system, str)
        and isinstance(user, str)
        and "You are a helpful assistant." in system
        and "Guidelines: Use tools/skills only when helpful." in user
    )


def validate_sample_for_merge(sample: Dict[str, Any], data_root: Path) -> List[str]:
    errors: List[str] = []
    item = normalize_record_for_reuse(sample)

    if not check_prompt_updated(item):
        errors.append("prompt not updated")

    ground_truth = get_ground_truth(item)
    if not ground_truth:
        errors.append("ground_truth empty")

    user_content = find_user_content(item)
    image_paths = normalize_images_field(item.get("images"))

    if len(image_paths) != 1:
        errors.append(f"image count is {len(image_paths)}, expected 1")

    placeholder_count = count_image_tokens(user_content)
    if placeholder_count != len(image_paths):
        errors.append(
            f"<image> count {placeholder_count} does not match image count {len(image_paths)}"
        )

    extra_info = parse_extra_info(item.get("extra_info"))
    extra_images = normalize_images_field(extra_info.get("images"))
    if extra_images != image_paths:
        errors.append("extra_info.images does not match top-level images")

    for p in image_paths + extra_images:
        resolved = resolve_image_path(p, data_root)
        if not is_image_readable(resolved):
            errors.append(f"image path unreadable: {p}")

    return errors


def summarize_errors(errors: Dict[str, List[str]], max_items: int = 30) -> str:
    lines: List[str] = []
    count = 0
    for key, errs in errors.items():
        for err in errs:
            lines.append(f"{key}: {err}")
            count += 1
            if count >= max_items:
                return "\n".join(lines + ["..."])
    return "\n".join(lines)


def build_stats_counter(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        counts[str(rec.get(key, "unknown"))] += 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))
