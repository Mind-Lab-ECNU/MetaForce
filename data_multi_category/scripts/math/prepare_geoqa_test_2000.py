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
Preprocess the GeoQA3 test dataset (raw data) to parquet format with local images.
Uniformly sample 2000 examples from the test split.
"""

import argparse
import json
import pickle
import random
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


def _save_pil_image(image: Image.Image, dest_path: Path) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())
    return dest_path.as_posix()


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return str(obj)


def _format_choices(choices) -> str | None:
    if not choices:
        return None
    if isinstance(choices, list):
        return "\n".join([f"{idx + 1}) {value}" for idx, value in enumerate(choices)])
    return None


def _map_choice_answer(choices, label):
    if choices is None:
        return None
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return None
    if 0 <= label_int < len(choices):
        return str(choices[label_int])
    if 1 <= label_int <= len(choices):
        return str(choices[label_int - 1])
    return None


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/scripts/raw_data/GeoQA/data/GeoQA3/test.pk",
    )
    parser.add_argument(
        "--output_dir",
        default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/math",
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

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        print(f"Raw file not found: {raw_path}")
        return

    with open(raw_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list) or not data:
        print("GeoQA3 test data is empty or unsupported format.")
        return

    split_name = "test"
    dataset_name = "GEOQA_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images" / split_name
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    n_samples = min(args.sample_count, len(data))
    sampled_indices = random.sample(range(len(data)), n_samples)
    sampled_indices = sorted(sampled_indices)

    items = []
    for local_idx, orig_idx in enumerate(sampled_indices):
        example = data[orig_idx]
        question = str(example.get("subject", "")).strip()
        choices = example.get("choices") or []
        label = example.get("label")
        image_array = example.get("image")

        if not question:
            continue
        if image_array is None:
            continue

        mapped_answer = _map_choice_answer(choices, label)
        if mapped_answer is None:
            continue

        try:
            image = Image.fromarray(image_array)
        except Exception as e:
            print(f"Skipping example {orig_idx} due to image error: {e}")
            continue

        image_name = f"{split_name}_{local_idx}.png"
        image_path = _save_pil_image(image, images_root / image_name)

        choices_text = _format_choices(choices)
        question_text = question
        if choices_text:
            question_text = f"{question}\n\nChoices:\n{choices_text}"

        mm_content = _build_mm_content(question_text, args.image_sep, 1)
        mm_content = f"{mm_content}\n\n{guideline}"

        extra_info = {
            "split": split_name,
            "index": orig_idx,
            "qid": f"{split_name}_{local_idx}",
            "images": [image_path],
            "question": question,
            "choices": choices,
            "label": label,
            "answer_explanation": example.get("answer"),
            "target_number": example.get("target_number"),
            "numbers": example.get("numbers"),
            "formal_point": example.get("formal_point"),
            "id": example.get("id"),
        }

        data_item = {
            "data_source": dataset_name,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": mm_content},
            ],
            "images": [{"image": image_path}],
            "ability": "visual_math",
            "reward_model": {
                "style": "rule",
                "ground_truth": mapped_answer,
            },
            "extra_info": extra_info,
        }
        items.append(data_item)

    if not items:
        print("No items collected. Skipping save.")
        return

    parquet_path = output_root / f"{split_name}.parquet"
    datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
    print(f"Saved {len(items)} items to {parquet_path}")

    json_path = output_root / f"{split_name}.json"
    with open(json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
