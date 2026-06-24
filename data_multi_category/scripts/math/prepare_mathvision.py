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
Preprocess MathVision dataset to parquet with local images.
Uniformly sample 2000 examples from each split.
"""

import argparse
import json
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


def _build_mm_content(question: str, image_sep: str, image_count: int) -> str:
    if image_sep and image_sep in question:
        return question
    if image_sep and image_count > 0:
        return (image_sep * image_count) + question
    return question


def _format_choices(options: list[str]) -> str | None:
    if not options:
        return None
    return "\n".join([f"{chr(ord('A') + idx)} {value}" for idx, value in enumerate(options)])


def _save_image(image: Image.Image, dest_path: Path) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())
    return dest_path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="MathLLMs/MathVision")
    parser.add_argument("--output_dir", default="/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data/math")
    parser.add_argument("--image_sep", default="<image>")
    parser.add_argument("--sample_count", type=int, default=None)
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
    dataset_name = "MathVision"
    output_root = Path(args.output_dir) / dataset_name
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        items = []
        split_image_dir = images_root / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)

        # Uniform sampling: calculate sample count and randomly select indices
        n_samples = min(args.sample_count, len(split_dataset)) if args.sample_count else len(split_dataset)
        sampled_indices = random.sample(range(len(split_dataset)), n_samples)
        sampled_indices = sorted(sampled_indices)  # Sort for consistent image naming

        for local_idx, orig_idx in enumerate(sampled_indices):
            example = split_dataset[orig_idx]

            question = example.get("question", "")
            options = example.get("options") or []
            answer = str(example.get("answer", "")).strip()

            question = question.replace("<image1>", args.image_sep)
            choices_text = _format_choices(options)
            question_text = question
            if choices_text:
                question_text = f"{question}\n\nChoices:\n{choices_text}"

            decoded_image = example.get("decoded_image")
            if decoded_image is None:
                raise ValueError("Missing decoded_image for MathVision example.")
            image_name = f"{example.get('id', local_idx)}.jpg"
            image_path = _save_image(decoded_image, split_image_dir / image_name)

            mm_content = _build_mm_content(question_text, args.image_sep, 1)
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
                "images": [{"image": image_path}],
                "ability": "visual_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "split": split_name,
                    "index": orig_idx,
                    "qid": f"{split_name}_{local_idx}",
                    "images": [image_path],
                    "question": question_text,
                    "level": example.get("level"),
                    "subject": example.get("subject"),
                },
            }
            if choices_text:
                data["extra_info"]["choices"] = [
                    f"{chr(ord('A') + idx)} {value}" for idx, value in enumerate(options)
                ]
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
