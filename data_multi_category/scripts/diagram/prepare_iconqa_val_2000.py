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
Preprocess the ICON-QA validation dataset (lmms-lab/ICON-QA) to parquet format with local images.
Uniformly sample 2000 examples from the validation split.
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


TARGET_SPLITS = ["val", "validation", "dev"]


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


def _copy_or_save_image(image_item, dest_dir: Path, name_prefix: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_item, str):
        if _is_url(image_item):
            ext = _safe_ext(urllib.parse.urlparse(image_item).path)
            dest_path = dest_dir / f"{name_prefix}{ext}"
            _download_url(image_item, dest_path)
            return dest_path.as_posix()

        src_path = Path(image_item)
        if src_path.exists():
            ext = _safe_ext(src_path.as_posix())
            dest_path = dest_dir / f"{name_prefix}{ext}"
            shutil.copy2(src_path, dest_path)
            return dest_path.as_posix()

        raise FileNotFoundError(f"Image path not found: {image_item}")

    if isinstance(image_item, dict):
        if "path" in image_item and image_item["path"]:
            src_path = Path(image_item["path"])
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


def _format_choices(choices) -> str | None:
    if not choices:
        return None
    if isinstance(choices, dict):
        def _sort_key(item):
            key = item[0]
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))
        ordered = sorted(choices.items(), key=_sort_key)
        return "\n".join([f"{key}) {value}" for key, value in ordered])
    if isinstance(choices, list):
        return "\n".join([f"{idx + 1}) {value}" for idx, value in enumerate(choices)])
    return None


def _get_choice_answer(choices, example_answer):
    if choices is None:
        return str(example_answer)
    if isinstance(choices, dict):
        key = str(example_answer)
        if key in choices:
            return str(choices[key])
        return str(example_answer)
    if isinstance(choices, list):
        answer_text = str(example_answer)
        for choice in choices:
            if str(choice) == answer_text:
                return answer_text
        try:
            index = int(example_answer)
            if 0 <= index < len(choices):
                return str(choices[index])
            if 1 <= index <= len(choices):
                return str(choices[index - 1])
        except (TypeError, ValueError):
            pass
    return str(example_answer)


def _normalize_choices(choices):
    if choices is None:
        return None
    if isinstance(choices, str):
        text = choices.strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",")]
        return [part for part in parts if part]
    return choices


def _first_value(example: dict, keys: list[str]):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key], key
    return None, None


def _extract_iconqa_fields(example: dict) -> tuple[str, str, object, object]:
    question, _ = _first_value(
        example,
        ["question", "query", "problem", "question_text", "query_text", "question_str"],
    )
    answer, _ = _first_value(
        example,
        ["answer", "label", "target", "final_answer", "correct_answer"],
    )
    choices, _ = _first_value(
        example,
        ["choices", "options", "candidates", "answer_choices", "answers"],
    )
    image_item, _ = _first_value(
        example,
        ["query_image", "image", "images", "img", "image_path", "image_url"],
    )

    question_text = str(question).strip() if question is not None else None
    answer_text = str(answer).strip() if answer is not None else None

    if not question_text:
        raise ValueError("Missing question field.")
    if answer_text is None or answer_text == "":
        raise ValueError("Missing answer field.")
    if image_item is None:
        raise ValueError("Missing image field.")

    choices = _normalize_choices(choices)

    return question_text, answer_text, choices, image_item


def _has_png_answer(example: dict) -> bool:
    answer, _ = _first_value(
        example,
        ["answer", "label", "target", "final_answer", "correct_answer"],
    )
    if answer is None:
        return False
    return ".png" in str(answer).lower()


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _select_split(dataset_obj):
    if isinstance(dataset_obj, datasets.Dataset):
        return "validation", dataset_obj
    for name in TARGET_SPLITS:
        if name in dataset_obj:
            return name, dataset_obj[name]
    raise KeyError(f"Could not find validation split in dataset. Available: {list(dataset_obj.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="lmms-lab/ICON-QA")
    parser.add_argument(
        "--output_dir",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/diagram",
    )
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    args = parser.parse_args()

    random.seed(args.seed)

    tool_json_path = Path(args.tool_json)
    if not tool_json_path.exists():
        tool_json_path = Path(__file__).parent.parent / "fake_tool.json"

    with open(tool_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_xml = _load_tools(str(tool_json_path))

    system_prompt = _build_system_prompt(tools_xml)

    try:
        dataset = datasets.load_dataset(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_path}: {e}")
        return

    dataset_name = "ICON-QA_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    try:
        split_name, split_dataset = _select_split(dataset)
    except KeyError as e:
        print(str(e))
        return

    items = []
    if len(split_dataset) == 0:
        print(f"Split {split_name} is empty.")
        return

    filtered_indices = [
        idx for idx in range(len(split_dataset)) if not _has_png_answer(split_dataset[idx])
    ]
    if not filtered_indices:
        print("No examples left after filtering out answers containing .png.")
        return

    n_samples = min(args.sample_count, len(filtered_indices))
    sampled_indices = random.sample(filtered_indices, n_samples)
    sampled_indices = sorted(sampled_indices)

    split_image_dir = images_root / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, orig_idx in enumerate(sampled_indices):
        example = split_dataset[orig_idx]
        try:
            question_text, answer, choices, image_item = _extract_iconqa_fields(example)
        except Exception as e:
            print(f"Skipping example {orig_idx} due to schema mismatch: {e}")
            print(f"Available keys: {list(example.keys())}")
            continue

        image_items = image_item if isinstance(image_item, list) else [image_item]
        image_items = [img for img in image_items if img is not None]

        if len(image_items) != 1:
            continue

        local_image_paths = []
        for img_idx, image_item in enumerate(image_items):
            name_prefix = f"{split_name}_{local_idx}"
            try:
                local_path = _copy_or_save_image(image_item, split_image_dir, name_prefix)
                local_image_paths.append(local_path)
            except Exception as e:
                print(f"Error saving image for index {orig_idx}: {e}")
                continue

        if not local_image_paths and image_items:
            continue

        choices_text = _format_choices(choices)
        question_with_choices = question_text
        if choices_text:
            question_with_choices = f"{question_text}\n\nChoices:\n{choices_text}"

        mapped_answer = _get_choice_answer(choices, answer)

        mm_content = _build_mm_content(question_with_choices, args.image_sep, len(local_image_paths))
        mm_content = f"{mm_content}\n\n{guideline}"

        extra_info = {
            "split": split_name,
            "index": orig_idx,
            "qid": f"{split_name}_{local_idx}",
            "images": local_image_paths,
            "question": question_text,
            "choices": choices,
            "question_id": example.get("question_id") or example.get("qid") or example.get("id"),
        }

        data = {
            "data_source": dataset_name,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": mm_content},
            ],
            "images": [{"image": path} for path in local_image_paths],
            "ability": "visual_diagram",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(mapped_answer),
            },
            "extra_info": extra_info,
        }
        items.append(data)

    parquet_path = output_root / f"{split_name}.parquet"
    datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
    print(f"Saved {len(items)} items to {parquet_path}")

    json_path = output_root / f"{split_name}.json"
    with open(json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
