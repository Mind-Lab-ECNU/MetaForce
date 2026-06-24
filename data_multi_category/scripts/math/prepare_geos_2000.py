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
Preprocess GEOs dataset (local JSON + PNG) to parquet with local images.
Uniformly sample 2000 examples from each split.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import datasets


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


def _format_question(text: str, choices: dict | None) -> str:
    if not choices:
        return text
    ordered = sorted(choices.items(), key=lambda x: int(x[0]))
    choices_text = "\n".join([f"{key}) {value}" for key, value in ordered])
    return f"{text}\n\nChoices:\n{choices_text}"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/inspire/hdd/project/ai4education/zhouaimin-p-zhouaimin/zc/verltools/data/data_duo/geos",
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

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load tools and build preference vector
    with open(args.tool_json, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tools_xml = _load_tools(args.tool_json)

    # Build system prompt with actual tools
    system_prompt = _build_system_prompt(tools_xml)

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir) / "geos_processed_2000"
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "aaai": "train",
        "practice": "val",
        "official": "test",
    }

    for src_split, dst_split in split_map.items():
        split_dir = input_root / src_split
        if not split_dir.exists():
            print(f"Warning: Skipping missing split dir: {split_dir}")
            continue

        items = []
        json_paths = sorted(split_dir.glob("*.json"))

        # Uniform sampling: calculate sample count and randomly select indices
        n_samples = min(args.sample_count, len(json_paths))
        sampled_indices = random.sample(range(len(json_paths)), n_samples)
        sampled_indices = sorted(sampled_indices)  # Sort for consistent ordering

        for local_idx, orig_idx in enumerate(sampled_indices):
            json_path = json_paths[orig_idx]
            record = _load_json(json_path)
            choices = record.get("choices")
            question = _format_question(record.get("text", ""), choices)
            answer = str(record.get("answer", "")).strip()
            if choices and answer in choices:
                answer = f"{answer})"

            image_path = json_path.with_suffix(".png")
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            split_image_dir = images_root / dst_split
            split_image_dir.mkdir(parents=True, exist_ok=True)
            out_image_path = split_image_dir / image_path.name
            shutil.copy2(image_path, out_image_path)

            mm_content = _build_mm_content(question, args.image_sep, 1)
            mm_content = f"{mm_content}\n\n{guideline}"

            data = {
                "data_source": "geo_2000",
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
                "images": [{"image": out_image_path.absolute().as_posix()}],
                "ability": "visual_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "split": dst_split,
                    "index": orig_idx,
                    "qid": f"{dst_split}_{local_idx}",
                    "images": [out_image_path.absolute().as_posix()],
                    "question": question,
                    "choices": (
                        [f"{key}) {value}" for key, value in sorted(choices.items(), key=lambda x: int(x[0]))]
                        if isinstance(choices, dict)
                        else choices
                    ),
                },
            }
            items.append(data)

        # Save both parquet and JSON with the same sampled items
        parquet_path = output_root / f"{dst_split}.parquet"
        datasets.Dataset.from_list(items).to_parquet(parquet_path.as_posix())
        print(f"Saved {len(items)} items to {parquet_path}")

        json_path = output_root / f"{dst_split}.json"
        with open(json_path.as_posix(), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items to {json_path}")


if __name__ == "__main__":
    main()
