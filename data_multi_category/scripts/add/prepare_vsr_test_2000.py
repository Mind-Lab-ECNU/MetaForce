#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the VSR dataset to parquet format with local images.
Uniformly sample 2000 examples from val/test splits.
"""

import argparse
import json
import random
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import datasets
from PIL import Image


def _load_tools(tool_json_path: str) -> str:
    """Load tool definitions from JSON and format for system prompt."""
    with open(tool_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"""<tools>
{tools_str}
</tools>"""


def _build_system_prompt(tools_xml: str) -> str:
    """Build system prompt with actual tool definitions."""
    return f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_xml}

For each function call, return a json object with function name and arguments within <tools></tools> XML tags:
<tools>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tools>
"""


guideline = (
    "Guidelines: Understand the given visual information and the user query. "
    "Determine if it is beneficial to employ the given tools. "
    "You must reason and reason with the visual information step by step within <thinking></thinking> tags. "
    "Put your final answer within <answer></answer> tags."
)


def _is_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def _safe_ext(path: str, default_ext: str = ".png") -> str:
    ext = Path(path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return ext
    return default_ext


def _download_url(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path.as_posix())


def _save_pil_image(image: Image.Image, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())


def _resolve_relative_path(path_str: str, base_dirs):
    for base_dir in base_dirs:
        candidate = base_dir / path_str
        if candidate.exists():
            return candidate
    return None


def _copy_or_save_image(image_item, dest_dir: Path, name_prefix: str, base_dirs=None) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_dirs = base_dirs or []

    if isinstance(image_item, str):
        if _is_url(image_item):
            ext = _safe_ext(urllib.parse.urlparse(image_item).path)
            dest_path = dest_dir / f"{name_prefix}{ext}"
            _download_url(image_item, dest_path)
            return dest_path.as_posix()

        src_path = Path(image_item)
        if not src_path.exists() and base_dirs:
            resolved = _resolve_relative_path(image_item, base_dirs)
            if resolved is not None:
                src_path = resolved

        if src_path.exists():
            ext = _safe_ext(src_path.as_posix())
            dest_path = dest_dir / f"{name_prefix}{ext}"
            shutil.copy2(src_path, dest_path)
            return dest_path.as_posix()

        raise FileNotFoundError(f"Image path not found: {image_item}")

    if isinstance(image_item, dict):
        if "path" in image_item and image_item["path"]:
            src_path = Path(image_item["path"])
            if not src_path.exists() and base_dirs:
                resolved = _resolve_relative_path(image_item["path"], base_dirs)
                if resolved is not None:
                    src_path = resolved
            if src_path.exists():
                ext = _safe_ext(src_path.as_posix())
                dest_path = dest_dir / f"{name_prefix}{ext}"
                shutil.copy2(src_path, dest_path)
                return dest_path.as_posix()
        if "bytes" in image_item and image_item["bytes"]:
            dest_path = dest_dir / f"{name_prefix}.png"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path.as_posix(), "wb") as f:
                f.write(image_item["bytes"])
            return dest_path.as_posix()

        raise ValueError("Unsupported image dict format.")

    if isinstance(image_item, Image.Image):
        dest_path = dest_dir / f"{name_prefix}.png"
        _save_pil_image(image_item, dest_path)
        return dest_path.as_posix()

    raise TypeError(f"Unsupported image type: {type(image_item)}")


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _normalize_answer(answer) -> str:
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        for item in answer:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
        return ""
    if isinstance(answer, dict):
        for key in ("answer", "answer_text", "text", "value", "final"):
            if key in answer:
                return _normalize_answer(answer[key])
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def _pick_text(value, default: str = "") -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if value is None:
        return default
    return str(value)


def _format_text_list(texts) -> str:
    if not texts:
        return ""
    if isinstance(texts, str):
        return texts
    if isinstance(texts, list):
        parts = []
        for item in texts:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(texts)


def _format_table(table) -> str:
    if table is None:
        return ""
    if isinstance(table, dict):
        for key in ("table", "table_content", "table_data", "texts", "rows", "cells"):
            if key in table:
                return _format_table(table[key])
        return json.dumps(table, ensure_ascii=False)
    if isinstance(table, list):
        if table and all(isinstance(row, list) for row in table):
            return "\n".join(" | ".join(str(cell) for cell in row) for row in table)
        return "\n".join(str(row) for row in table)
    return str(table)

_QUESTION_TEMPLATES = [
    'Is the statement "{caption}" accurate regarding the image?\nAnswer yes or no.',
    'Does the image validate the caption "{caption}"?\nAnswer yes or no.',
    'Is the given caption "{caption}" fitting for the image?\nAnswer yes or no.',
    'Is "{caption}" an appropriate description for the image?\nAnswer yes or no.',
    'Verify the accuracy of this image caption: "{caption}".\nAnswer yes or no.',
    'Based on the picture, is it true that "{caption}"?\nAnswer yes or no.',
    'Looking at the image, would you say "{caption}" is correct?\nAnswer yes or no.',
    'Is the statement "{caption}" supported by the image?\nAnswer yes or no.',
]


def _build_question(caption: str) -> str:
    if not caption:
        return "Is the statement accurate regarding the image?\nAnswer yes or no."
    template = random.choice(_QUESTION_TEMPLATES)
    return template.format(caption=caption)


def _extract_sample(example: dict) -> tuple[str, str, list, str, dict]:
    caption = _pick_text(example.get("caption") or example.get("sentence") or example.get("text"), default="")
    question = _build_question(caption)

    label = example.get("label")
    if label is None:
        answer = ""
    else:
        if isinstance(label, bool):
            answer = "Yes." if label else "No."
        else:
            answer = "Yes." if int(label) == 1 else "No."

    image_item = example.get("image") or example.get("image_link") or example.get("image_url")
    image_items = [image_item] if image_item is not None else []
    return question, _normalize_answer(answer), image_items, "", {"caption": caption}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="pingzhili/vsr")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/add/general")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        with open(args.tool_json, "r", encoding="utf-8") as f:
            tools = json.load(f)
        tools_xml = _load_tools(args.tool_json)
    except FileNotFoundError:
        print(f"Warning: Tool JSON not found at {args.tool_json}. Using empty tools.")
        tools_xml = ""

    system_prompt = _build_system_prompt(tools_xml)

    try:
        if args.dataset_config is None:
            dataset = datasets.load_dataset(args.dataset_path)
        else:
            dataset = datasets.load_dataset(args.dataset_path, args.dataset_config)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_path} ({args.dataset_config}): {e}")
        return

    dataset_name = "VSR_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    split_aliases = {
        "validation": "val",
        "val": "val",
        "dev": "val",
        "dev_seen": "val",
        "dev_unseen": "val",
        "test": "test",
        "test_seen": "test",
        "test_unseen": "test",
    }
    items_by_split = {"val": [], "test": []}

    for split_name, split_dataset in dataset.items():
        if split_name not in split_aliases:
            continue
        output_split = split_aliases[split_name]
        items = items_by_split[output_split]

        base_dirs = []
        try:
            for cache in split_dataset.cache_files:
                base_dirs.append(Path(cache["filename"]).parent)
        except Exception:
            base_dirs = []

        n_samples = min(args.sample_count, len(split_dataset))
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]
            try:
                question_text, answer, image_items, context_text, extra_fields = _extract_sample(example)
            except Exception as e:
                print(f"Skipping example {orig_idx} due to schema mismatch: {e}")
                print(f"Available keys: {list(example.keys())}")
                continue

            if not isinstance(image_items, list):
                image_items = [image_items] if image_items is not None else []
            image_items = [img for img in image_items if img is not None]

            split_image_dir = images_root / output_split
            local_image_paths = []
            for img_idx, image_item in enumerate(image_items):
                name_prefix = f"{split_name}_{local_idx}"
                try:
                    local_path = _copy_or_save_image(image_item, split_image_dir, name_prefix, base_dirs=base_dirs)
                    local_image_paths.append(local_path)
                except Exception as e:
                    print(f"Error saving image for example {orig_idx}: {e}")

            if image_items and not local_image_paths:
                print(f"Skipping example {orig_idx} due to image save failure")
                continue

            question_with_context = question_text
            if context_text:
                question_with_context = f"{question_text}\n\n{context_text}"

            mm_content = _build_mm_content(question_with_context, args.image_sep, len(local_image_paths))
            mm_content = f"{mm_content}\n\n{guideline}"

            extra_info = {
                "split": output_split,
                "index": orig_idx,
                "qid": f"{split_name}_{local_idx}",
                "images": local_image_paths,
                "question": question_text,
            }
            if extra_fields:
                extra_info.update(extra_fields)

            data = {
                "data_source": dataset_name,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": mm_content},
                ],
                "images": [{"image": path} for path in local_image_paths],
                "ability": "visual_general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer),
                },
                "extra_info": extra_info,
            }
            items.append(data)

    for output_split, items in items_by_split.items():
        if not items:
            continue
        parquet_path = output_root / f"{output_split}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{output_split}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
