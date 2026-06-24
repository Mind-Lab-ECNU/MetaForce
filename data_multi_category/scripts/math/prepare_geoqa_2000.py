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
Preprocess the GEOQA_R1V_Train_8K dataset to json format with local images.
Uniformly sample 2000 examples from each split.
Tool version with dynamic tool loading.
"""

import argparse
import json
import random
import re
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


def _save_pil_image(image: Image.Image, dest_path: Path) -> None:
    """Save PIL image to destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())


def _extract_answer(solution: str) -> str:
    """Extract answer from <answer>...</answer> format.

    Args:
        solution: Solution string in format <answer> 145° </answer>

    Returns:
        Extracted answer with degree symbol preserved (e.g., "145°")
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", solution)
    if match:
        return match.group(1).strip()
    return solution.strip()


def _get_question(example: dict) -> str:
    """Get question from example."""
    if "problem" in example:
        return example["problem"]
    if "question" in example:
        return example["question"]
    raise KeyError("No question/problem field found in example.")


def _get_answer(example: dict) -> str:
    """Get answer from example."""
    if "solution" in example:
        return _extract_answer(example["solution"])
    if "answer" in example:
        return _extract_answer(example["answer"])
    raise KeyError("No answer/solution field found in example.")


def _get_images(example: dict):
    """Get images from example."""
    if "images" in example:
        return example["images"]
    if "image" in example:
        return example["image"]
    return []


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    """Build multimodal content with image separators."""
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="leonardPKU/GEOQA_R1V_Train_8K")
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/math")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tool_json", default=str(Path(__file__).parent.parent / "fake_tool.json"))
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load tools and build preference vector
    with open(args.tool_json, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_xml = _load_tools(args.tool_json)

    # Build system prompt with actual tools
    system_prompt = _build_system_prompt(tools_xml)

    dataset = datasets.load_dataset(args.dataset_path)

    dataset_name = "GEOQA_2000"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        items = []

        # Uniform sampling: calculate sample count and randomly select indices
        n_samples = min(args.sample_count, len(split_dataset))
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)  # Sort for consistent image naming

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]

            question = _get_question(example)
            answer = _get_answer(example)
            image_item = _get_images(example)

            if image_item is None:
                continue

            split_image_dir = images_root / split_name
            split_image_dir.mkdir(parents=True, exist_ok=True)

            # Handle single image
            if not isinstance(image_item, list):
                image_item = [image_item]

            if len(image_item) != 1:
                continue

            local_image_paths = []
            for img_idx, img in enumerate(image_item):
                if isinstance(img, Image.Image):
                    name_prefix = f"{split_name}_{local_idx}"
                    dest_path = split_image_dir / f"{name_prefix}.png"
                    _save_pil_image(img, dest_path)
                    local_image_paths.append(dest_path.as_posix())

            if not local_image_paths:
                continue

            question_text = question
            mm_content = _build_mm_content(question_text, args.image_sep, len(local_image_paths))
            mm_content = f"{mm_content}\n\n{guideline}"

            data = {
                "data_source": dataset_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": mm_content,
                    },
                ],
                "images": [{"image": path} for path in local_image_paths],
                "ability": "visual_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "split": split_name,
                    "index": orig_idx,
                    "qid": f"{split_name}_{local_idx}",
                    "images": local_image_paths,
                    "question": question_text,
                },
            }
            items.append(data)

        # Save both parquet and JSON with the same sampled items
        parquet_path = output_root / f"{split_name}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{split_name}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
